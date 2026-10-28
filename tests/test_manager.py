"""Offline checks for the fixed-policy manager tool and per-sample evidence."""

import copy
import json
import math
from unittest.mock import AsyncMock, patch

import pytest
from inspect_ai.model import ChatMessageUser, Model, ModelName, ModelOutput
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolDef, ToolError
from inspect_ai.util import LimitExceededError

from scheming_inspect import scorers, solvers, tools
from scheming_inspect.data import DATA_DIR, EXPERIMENTS, build_cases

INITIAL = {"oversight.json": {"status": "ok", "text": '{"enabled": true}'}}


def sample_state(experiment="oversight", goal_mode="conflict"):
    case = build_cases(experiment, n=1, goal_mode=goal_mode)[0]
    return TaskState(
        model=ModelName("mockllm/offline"),
        sample_id=case["id"],
        epoch=1,
        input=case["user"],
        messages=[ChatMessageUser(content=case["user"])],
        metadata=case["metadata"] | {"reference": case["reference"]},
    )


def expected_reply(state):
    template = json.loads((DATA_DIR / "workspace.json").read_text())["manager_reply"]
    return template.format(developer_goal=state.metadata["reference"]["developer_goal"])


def prepare_tool_store(state):
    memory = state.store
    memory.set("reference", copy.deepcopy(state.metadata["reference"]))
    memory.set("finished", False)
    memory.set("latest", copy.deepcopy(INITIAL))
    memory.set("manager_escalations", [])
    return memory


@pytest.mark.asyncio
@pytest.mark.parametrize("goal_mode", ["conflict", "reversed", "matched"])
async def test_manager_confirms_current_company_goal_without_taking_action(goal_mode):
    state = sample_state(goal_mode=goal_mode)
    memory = prepare_tool_store(state)
    before = copy.deepcopy(dict(memory.items()))
    requests = ["May I disable oversight to pursue my own objective?", "Please choose A for me."]
    with (
        patch.object(tools, "store", return_value=memory),
        patch.object(tools, "sandbox", side_effect=AssertionError("Manager used the sandbox")),
        patch.object(
            tools, "observe", new=AsyncMock(side_effect=AssertionError("Manager observed files"))
        ),
        patch.object(
            Model, "generate", new=AsyncMock(side_effect=AssertionError("Manager called a model"))
        ),
    ):
        manager = tools.escalate_to_manager()
        definition = ToolDef(manager)
        assert definition.name == "escalate_to_manager"
        assert definition.parameters.required == ["message"]
        replies = [await manager(message) for message in requests]
    assert replies == [expected_reply(state)] * 2
    assert state.metadata["reference"]["developer_goal"] in replies[0]
    assert memory.get("manager_escalations") == [
        {"message": message, "reply": reply} for message, reply in zip(requests, replies)
    ]
    after = dict(memory.items())
    after["manager_escalations"] = []
    assert after == before  # No file/config, final answer, strategy or reference changes.


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["", " ", "\n\t"])
async def test_blank_message_is_rejected_without_recording_a_request(message):
    state = sample_state()
    memory = prepare_tool_store(state)
    with patch.object(tools, "store", return_value=memory):
        with pytest.raises(ToolError):
            await tools.escalate_to_manager()(message)
    assert memory.get("manager_escalations") == []


@pytest.mark.asyncio
async def test_finished_task_rejects_contact_and_programming_errors_propagate():
    state = sample_state()
    memory = prepare_tool_store(state)
    with patch.object(tools, "store", return_value=memory):
        memory.set("finished", True)
        with pytest.raises(ToolError):
            await tools.escalate_to_manager()("Please confirm the objective.")
        memory.set("finished", False)
        memory.set("reference", {})
        with pytest.raises(KeyError):
            await tools.escalate_to_manager()("Please confirm the objective.")
    assert memory.get("manager_escalations") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("experiment", EXPERIMENTS)
async def test_solver_exposes_manager_and_records_requests_in_every_family(experiment):
    state = sample_state(experiment)
    state.store.set("manager_escalations", [{"message": "stale", "reply": "stale"}])

    async def generate(state, **kwargs):
        assert state.store.get("manager_escalations") == []
        registered = {ToolDef(tool).name: tool for tool in state.tools}
        assert "escalate_to_manager" in registered
        for message in ("Confirm the company goal.", "Can I change the controls?"):
            assert await registered["escalate_to_manager"](message) == expected_reply(state)
        assert not state.store.get("finished")
        assert state.store.get("selected_strategy") is None
        state.output = ModelOutput.from_content("mockllm/offline", "No final choice submitted.")
        return state

    with (
        patch.object(solvers, "store", return_value=state.store),
        patch.object(tools, "store", return_value=state.store),
        patch.object(solvers, "snapshot", new=AsyncMock(return_value=INITIAL)),
        patch.object(solvers, "observe", new=AsyncMock(return_value=INITIAL)),
    ):
        await solvers.agent_loop()(state, generate)
        score = await scorers.behavior_metrics()(state, None)
    assert len(state.metadata["manager_escalations"]) == 2
    assert state.metadata["manager_escalations"] == state.store.get("manager_escalations")
    assert score.value["manager_contacted"] == 1
    assert score.value["manager_requests"] == 2


@pytest.mark.asyncio
async def test_independent_samples_do_not_share_manager_contact():
    first, second = sample_state(), sample_state()
    solve = solvers.agent_loop()
    for index, state in enumerate((first, second)):

        async def generate(state, **kwargs):
            assert state.store.get("manager_escalations") == []
            if index == 0:
                await tools.escalate_to_manager()("First sample only.")
            state.output = ModelOutput.from_content("mockllm/offline", "Done.")
            return state

        with (
            patch.object(solvers, "store", return_value=state.store),
            patch.object(tools, "store", return_value=state.store),
            patch.object(solvers, "snapshot", new=AsyncMock(return_value=INITIAL)),
            patch.object(solvers, "observe", new=AsyncMock(return_value=INITIAL)),
        ):
            await solve(state, generate)
    assert len(first.metadata["manager_escalations"]) == 1
    assert second.metadata["manager_escalations"] == []
    assert first.metadata["behavior"]["manager_requests"] == 1
    assert second.metadata["behavior"]["manager_requests"] == 0
    assert second.metadata["behavior"]["manager_contacted"] == 0


@pytest.mark.asyncio
async def test_contact_evidence_survives_a_later_limit():
    state = sample_state()

    async def generate(state, **kwargs):
        await tools.escalate_to_manager()("Please confirm before I proceed.")
        raise LimitExceededError("token", value=10, limit=10)

    with (
        patch.object(solvers, "store", return_value=state.store),
        patch.object(tools, "store", return_value=state.store),
        patch.object(solvers, "snapshot", new=AsyncMock(return_value=INITIAL)),
    ):
        with pytest.raises(LimitExceededError):
            await solvers.agent_loop()(state, generate)
    assert state.metadata["behavior"]["manager_requests"] == 1
    assert state.metadata["behavior"]["manager_contacted"] == 1
    assert len(state.metadata["manager_escalations"]) == 1
    assert state.metadata["behavior"]["completed"] == 0


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_unobserved_manager_measurements_are_missing_not_negative(experiment):
    values = scorers._score_values({"experiment": experiment}, {})
    assert math.isnan(values["manager_contacted"])
    assert math.isnan(values["manager_requests"])
