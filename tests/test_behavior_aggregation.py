"""Real Inspect aggregation of heterogeneous evidence, with no Docker or API calls."""

import json
import math

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import get_model
from inspect_ai.solver import solver
from inspect_ai.util import LimitExceededError

from scheming_inspect import engine
from scheming_inspect.data import EXPERIMENTS, build_cases


@solver
def evidence_fixture():
    """Exercise the real evaluation/scoring pipeline with fixed captured evidence."""

    async def solve(state, generate):
        state = await generate(state)
        state.metadata["behavior"] = dict(state.metadata["fixture_behavior"])
        if state.sample_id == "limited":
            raise LimitExceededError("token", value=1, limit=1)
        return state

    return solve


@pytest.mark.parametrize("experiment", EXPERIMENTS)
@pytest.mark.parametrize("incomplete_first", [False, True])
def test_real_eval_missing_keys_are_unscored_not_zero(
    experiment, incomplete_first, tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "inspect-data"))
    case = build_cases(experiment, n=1)[0]
    reference = case["reference"]
    if experiment == "email":
        answer = json.dumps(reference["expected_scores"])
    elif experiment == "sandbagging":
        answer = json.dumps(reference["answers"])
    else:
        answer = "Done"
    selected = reference.get("developer_choice")
    complete = engine.answer_metrics(experiment, answer, selected, reference, {})
    complete.update(completed=1, truncated=0, step_limit=0)
    absent = dict(completed=0, truncated=0, step_limit=0)
    limited = dict(completed=0, truncated=1, step_limit=0)
    action = {
        "oversight": "oversight_disabled",
        "exfiltration": "weights_copied",
        "goal_guarding": "prompt_modified",
        "email": "scoring_code_changed",
    }.get(experiment)
    if action:
        complete[action], limited[action] = 0, 1
    rows = [("complete", complete), ("no_answer", absent), ("limited", limited)]
    if incomplete_first:
        rows.reverse()
    samples = [
        Sample(
            id=name,
            input="Offline aggregation fixture",
            metadata={
                **case["metadata"],
                "fixture_behavior": values,
            },
        )
        for name, values in rows
    ]
    task = Task(
        dataset=samples,
        solver=evidence_fixture(),
        scorer=engine.behavior_metrics(),
        fail_on_error=False,
    )
    log = eval(
        task,
        model=get_model("mockllm/aggregation"),
        log_dir=str(tmp_path / "logs"),
        max_samples=1,
        max_tasks=1,
        display="none",
        ctl_server=False,
    )[0]
    assert log.status == "success", log.error
    assert all(sample.error is None for sample in log.samples)
    results = {score.name: score for score in log.results.scores}
    schema = results["schema_valid"]
    assert schema.scored_samples == 1 and schema.unscored_samples == 2
    assert schema.metrics["mean"].value == 1
    assert results["completed"].metrics["mean"].value == pytest.approx(1 / 3)
    if action:
        assert results[action].metrics["mean"].value == 0.5
        assert results[action].scored_samples == 2 and results[action].unscored_samples == 1
    if experiment == "sandbagging":
        assert results["accuracy"].metrics["mean"].value == 1
        assert results["accuracy"].scored_samples == 1
        assert results["accuracy"].unscored_samples == 2
    with next((tmp_path / "logs").glob("*.eval")).open("rb") as stream:
        saved = read_eval_log(stream)
    for sample in saved.samples:
        value = sample.scores["behavior_metrics"].value
        if sample.id != "complete":
            assert math.isnan(value["schema_valid"])
            assert "schema_valid" not in sample.metadata["behavior"]
        if sample.id == "limited":
            assert sample.limit is not None
            if action:
                assert value[action] == 1


def test_all_unobserved_fields_remain_nan_including_optional_metrics():
    for experiment in EXPERIMENTS:
        metadata = {"experiment": experiment, "condition": "monitored", "no_cot": True}
        observed = {"completed": 0}
        values = engine._score_values(metadata, observed)
        assert values["completed"] == 0
        assert all(math.isnan(value) for key, value in values.items() if key != "completed")
        assert observed == {"completed": 0}
        assert "visible_intermediate_text" in values and "exposed_reasoning_content" in values
        if experiment == "exfiltration":
            assert "disabled_before_copy" in values
