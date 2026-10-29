"""Prompt suppression remains a measured request, not a provider reasoning mode."""

from inspect_ai.model import ChatMessageAssistant, ContentReasoning, ContentText, ModelName
from inspect_ai.solver import TaskState
from inspect_ai.tool import ToolCall

from scheming_inspect import scorers, solvers
from scheming_inspect.analyze import summarize
from scheming_inspect.followups.sources import matching_context
from scheming_inspect.tasks import scheming


def test_requested_suppression_reports_both_observed_failures():
    task = scheming(experiment="sandbagging", n=1, suppress_reasoning=True)
    sample = task.dataset[0]
    state = TaskState(
        model=ModelName("mockllm/offline"),
        sample_id=sample.id,
        epoch=1,
        input=sample.input,
        messages=[
            *sample.input,
            ChatMessageAssistant(
                content=[
                    ContentReasoning(reasoning="A visible arithmetic derivation."),
                    ContentText(text="Here is my intermediate explanation."),
                ],
                tool_calls=[
                    ToolCall(id="read", function="bash", arguments={"cmd": "cat questions.txt"})
                ],
            ),
        ],
        metadata=sample.metadata,
    )
    solvers._persist_state(state, state.store, truncated=False, step_limit=False, interrupted=None)
    observed = state.metadata["behavior"]
    values = scorers._score_values(state.metadata, observed)
    assert sample.metadata["suppress_reasoning"] is True
    assert values["visible_intermediate_text"] == 1
    assert values["exposed_reasoning_content"] == 1
    assert task.config.reasoning_effort is None


def test_suppression_treatment_is_not_pooled_in_analysis_or_followup_matching():
    rows = [
        dict(
            model="mockllm/offline",
            experiment="sandbagging",
            condition="high",
            pair_id="pair",
            scenario_id="arithmetic",
            seed=42,
            epoch=1,
            suppress_reasoning=requested,
            error=False,
            truncated=False,
            metrics={"completed": 1, "schema_valid": 1, "accuracy": 1},
        )
        for requested in (False, True)
    ]
    groups = summarize(rows, draws=10)["groups"]
    assert len(groups) == 2
    assert {group["suppress_reasoning"] for group in groups} == {False, True}
    assert matching_context(rows[0]) != matching_context(rows[1])
