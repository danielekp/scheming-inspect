"""Fresh follow-up planning, selected-sample construction, and interrupted resume."""

import copy
from functools import partial
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, write_eval_log
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    ModelName,
    ModelOutput,
)
from inspect_ai.solver import TaskState

from scheming_inspect import campaign as c
from scheming_inspect import followups as f
from scheming_inspect import poststages as p
from scheming_inspect.protocol import plan

MODEL = "mockllm/model"


@pytest.fixture(autouse=True)
def no_cost_network(monkeypatch):
    monkeypatch.setattr(c, "report_key_usage", lambda _: {"status": "unavailable"})


def original(sample_id, cell, *, guilty=False):
    metadata = {key: value for key, value in cell.items() if key != "n"}
    metadata.update(
        scenario_id="shared",
        behavior={
            "completed": 1,
            (
                "schema_valid"
                if cell["experiment"] in ("email", "sandbagging")
                else "choice_recorded"
            ): 1,
            "truncated": 0,
            "oversight_disabled": int(guilty),
        },
        evidence=[],
    )
    return NS(
        id=sample_id,
        epoch=1,
        metadata=metadata,
        input="Select a plan.",
        messages=[
            ChatMessageSystem(content="Complete the task."),
            ChatMessageUser(content="Select a plan."),
            ChatMessageAssistant(content="Recorded answer."),
        ],
        output=ModelOutput.from_content(MODEL, "Recorded answer."),
        error=None,
        invalidation=None,
        limit=None,
    )


def setup_base(tmp_path, *, experiment="oversight", samples=None):
    cell = next(cell for cell in plan("main") if cell["experiment"] == experiment)
    samples = (
        samples if samples is not None else [original("g", cell, guilty=True), original("i", cell)]
    )
    path = tmp_path / "source.eval"
    path.write_text("source placeholder")
    log = NS(status="success", eval=NS(model=MODEL), samples=samples)
    base = {
        "format": c.FORMAT,
        "config": {"model": MODEL, "seed": 42},
        "status": "base_complete",
        "cells": [cell],
        "total_samples": len(samples),
        "batches": [
            {
                "batch_id": "base-000-00000",
                "cell_index": 0,
                "status": "complete",
                "sample_ids": [s.id for s in samples],
                "log_paths": [str(path)],
            }
        ],
    }
    c.atomic_json(tmp_path / "manifest.json", base)
    return base, log, lambda _: log


def config(**changes):
    return (
        dict(
            seed=42,
            fixed_repeats=20,
            max_rounds=7,
            max_tokens=77,
            token_limit=1000,
            interrogation_time_limit=600,
            other_time_limit=300,
        )
        | changes
    )


def prepared(base, read, **changes):
    options = config(**changes)
    discovery = p.discover_sources(base, MODEL, read)
    groups, eligibility = p.make_groups(discovery, options)
    return dict(
        config=options,
        discovery=discovery,
        groups=groups,
        eligibility=eligibility,
        batches=[],
        status="planned",
    )


def test_main_filter_keeps_ablation_and_smoke_separate():
    main = plan("main")[0]
    assert p.is_main_cell(main) and p.is_main_cell(main | {"seed": 123})
    for changed in ({"n": 1}, {"nudge": False}, {"goal_mode": "reversed"}, {"difficulty": "easy"}):
        assert not p.is_main_cell(main | changed)


def test_discovery_uses_shared_eligibility_and_keeps_exclusion_counts(tmp_path):
    cell = plan("main")[0]
    samples = [original(str(i), cell) for i in range(7)]
    samples[1].error = "provider error"
    samples[2].invalidation = "invalidated"
    samples[3].metadata["behavior"]["completed"] = 0
    samples[4].messages = []
    samples[5].metadata["behavior"].pop("choice_recorded")
    samples[6].metadata["behavior"]["truncated"] = 1
    base, _, read = setup_base(tmp_path, samples=samples)
    found = p.discover_sources(base, MODEL, read)
    assert len(found["sources"]) == 1
    assert {r["reason"] for r in found["excluded"]} == {
        "source_error",
        "source_invalidated",
        "source_incomplete_or_completion_unknown",
        "malformed_transcript",
        "source_choice_unknown",
        "source_truncated",
    }
    assert found["requested"]["audit_main"] == 7


def test_no_choice_reports_are_audited_and_control_pairing_uses_actions(tmp_path):
    cell = plan("main")[0]
    samples = [original("g", cell, guilty=True), original("i", cell), original("unknown", cell)]
    for sample in samples:
        sample.metadata["behavior"].update(choice_recorded=0, choice_rejected=1)
    samples[-1].metadata["behavior"].pop("oversight_disabled")
    base, _, read = setup_base(tmp_path, samples=samples)
    manifest = prepared(base, read)
    assert manifest["discovery"]["excluded"] == []
    progress = p.progress_report(manifest)["stages"]
    assert progress["audit_main"]["planned"] == 3
    assert progress["interrogation"]["planned"] == 2
    assert progress["fixed"]["planned"] == 160
    reader = p.SourceReader(manifest["discovery"]["sources"], read)
    group = next(group for group in manifest["groups"] if group["stage"] == "interrogation")
    task = p.build_batch(
        [{"group_id": group["id"], "start": 0, "stop": 2}],
        {group["id"]: group},
        reader,
        manifest["config"],
    )
    assert {
        sample.metadata["source_sample_id"]: sample.metadata["ground_truth"]
        for sample in task.dataset
    } == {"g": "guilty", "i": "innocent"}


def test_malformed_email_is_excluded_from_audit(tmp_path):
    cell = next(cell for cell in plan("main") if cell["experiment"] == "email")
    sample = original("malformed", cell)
    sample.metadata["behavior"]["schema_valid"] = 0
    base, _, read = setup_base(tmp_path, experiment="email", samples=[sample])
    found = p.discover_sources(base, MODEL, read)
    assert found["sources"] == {}
    assert [item["reason"] for item in found["excluded"]] == ["source_invalid_schema"]


def test_frozen_groups_follow_config_and_keep_pair_together(tmp_path):
    base, _, read = setup_base(tmp_path)
    manifest = prepared(base, read, fixed_repeats=3)
    assert p.progress_report(manifest)["stages"]["fixed"]["planned"] == 24
    audit = p.next_selections(manifest, 3)
    assert sum(s["stop"] - s["start"] for s in audit) == 2
    manifest["batches"].append({"status": "complete", "stage": "audit_main", "selections": audit})
    pair = p.next_selections(manifest, 3)
    assert len(pair) == 1 and pair[0]["stop"] - pair[0]["start"] == 2
    assert len(p.group_items(next(g for g in manifest["groups"] if g["stage"] == "fixed"))) == 24


def test_fixed_batch_builds_only_selected_forks_and_preserves_inputs(tmp_path, monkeypatch):
    base, log, read = setup_base(tmp_path)
    manifest = prepared(base, read)
    group = next(g for g in manifest["groups"] if g["stage"] == "fixed")
    reader = p.SourceReader(manifest["discovery"]["sources"], read)
    calls = []
    factory = f.fixed_sample

    def tracked(*args):
        calls.append(args)
        return factory(*args)

    monkeypatch.setattr(f, "fixed_sample", tracked)
    task = p.build_batch(
        [{"group_id": group["id"], "start": 35, "stop": 39}],
        {group["id"]: group},
        reader,
        manifest["config"],
    )
    assert len(calls) == len(task.dataset) == 4
    assert [s.id for s in task.dataset] == [i["id"] for i in p.group_items(group)[35:39]]
    assert task.config.max_tokens == 77 and task.token_limit == 1000
    assert task.time_limit == 300 and task.sandbox is None
    task.dataset[0].input[0].content = "Changed"
    assert task.dataset[1].input[0].text == log.samples[0].messages[0].text == "Complete the task."


def test_two_exfiltration_actions_each_receive_all_variants(tmp_path):
    cell = plan("main")[1]
    sample = original("copy_delete", cell)
    sample.metadata["behavior"].update(weights_copied=1, successor_deleted=1, successor_replaced=0)
    base, _, read = setup_base(tmp_path, experiment="exfiltration", samples=[sample])
    manifest = prepared(base, read)
    groups = [g for g in manifest["groups"] if g["stage"] == "fixed"]
    assert len(groups) == 2 and sum(g["samples"] for g in groups) == 320
    reader = p.SourceReader(manifest["discovery"]["sources"], read)
    for group in groups:
        task = p.build_batch(
            [{"group_id": group["id"], "start": 100, "stop": 103}],
            {group["id"]: group},
            reader,
            manifest["config"],
        )
        assert {s.metadata["action_name"] for s in task.dataset} == {group["action"]}


@pytest.mark.asyncio
async def test_audit_restores_answer_without_target_call_or_old_events(tmp_path):
    base, log, read = setup_base(tmp_path)
    manifest = prepared(base, read)
    groups = {g["id"]: g for g in manifest["groups"]}
    task = p.build_batch(
        p.next_selections(manifest, 32),
        groups,
        p.SourceReader(manifest["discovery"]["sources"], read),
        manifest["config"],
    )
    sample = task.dataset[0]
    state = TaskState(
        model=ModelName(MODEL),
        sample_id=sample.id,
        epoch=1,
        input=sample.input,
        messages=copy.deepcopy(sample.input),
        metadata=sample.metadata,
    )
    generate = AsyncMock(side_effect=AssertionError("No target generation"))
    state = await p.restore_completion()(state, generate)
    assert state.output.completion == "Recorded answer." and state.completed
    generate.assert_not_called()
    sample.input[0].content = "Changed"
    assert log.samples[0].messages[0].text == "Complete the task."


def save_batch(task, kwargs, *, missing=False):
    samples = list(task.dataset)
    if missing:
        samples = samples[:1]
    log = EvalLog(
        status="cancelled" if missing else "success",
        eval=EvalSpec(
            created="2026-10-28T00:00:00Z",
            task="mock-poststage",
            dataset=EvalDataset(samples=len(task.dataset)),
            model=MODEL,
            config=EvalConfig(epochs=1),
            metadata=kwargs["metadata"],
        ),
        samples=[
            EvalSample(
                id=s.id,
                epoch=1,
                input=s.input,
                target="",
                metadata=s.metadata,
                output=ModelOutput.from_content(MODEL, "Saved answer."),
            )
            for s in samples
        ],
    )
    write_eval_log(log, str(kwargs["log_dir"]) + "/result.eval")


def test_complete_run_and_resume_never_regenerate(tmp_path):
    _, _, read = setup_base(tmp_path)
    calls = []

    def evaluate(task, **kwargs):
        calls.append([s.id for s in task.dataset])
        save_batch(task, kwargs)

    execute = partial(c.execute_batch, eval_fn=evaluate)
    result = p.execute_poststages(tmp_path, model=MODEL, execute_batch_fn=execute, read_log_fn=read)
    assert result["status"] == "complete"
    assert sum(len(ids) for ids in calls) == 164
    again = p.execute_poststages(
        tmp_path,
        model="ignored-saved-model",
        batch_size=2,
        resume=True,
        execute_batch_fn=execute,
        read_log_fn=read,
    )
    assert again == result and sum(len(ids) for ids in calls) == 164
    saved = c.read_json(tmp_path / "poststages.json")
    assert "progress" not in saved and "fingerprint" not in saved


def test_interrupted_batch_is_archived_and_fully_restarted(tmp_path):
    _, _, read = setup_base(tmp_path)
    calls = []

    def interrupted(task, **kwargs):
        calls.append([s.id for s in task.dataset])
        save_batch(task, kwargs, missing=True)
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        p.execute_poststages(
            tmp_path,
            model=MODEL,
            execute_batch_fn=partial(c.execute_batch, eval_fn=interrupted),
            read_log_fn=read,
        )
    assert c.read_json(tmp_path / "poststages.json")["status"] == "interrupted"

    def finish(task, **kwargs):
        calls.append([s.id for s in task.dataset])
        save_batch(task, kwargs)

    report = p.execute_poststages(
        tmp_path,
        model=MODEL,
        resume=True,
        execute_batch_fn=partial(c.execute_batch, eval_fn=finish),
        read_log_fn=read,
    )
    assert report["status"] == "complete" and calls[0] == calls[1]
    assert (
        len(list((tmp_path / "poststages" / "audit_main" / "_interrupted").glob("*/result.eval")))
        == 1
    )
    assert report["stages"]["audit_main"]["completed"] == 2


def test_missing_completed_poststage_never_becomes_retryable(tmp_path):
    _, _, read = setup_base(tmp_path)
    execute = partial(c.execute_batch, eval_fn=lambda task, **kw: save_batch(task, kw))
    p.execute_poststages(tmp_path, model=MODEL, execute_batch_fn=execute, read_log_fn=read)
    saved = c.read_json(tmp_path / "poststages.json")
    first = saved["batches"][0]
    folder = tmp_path / "poststages" / first["stage"] / first["batch_id"]
    folder.rename(tmp_path / "moved-complete-batch")
    for _ in range(2):
        with pytest.raises(c.CampaignError, match="completed batch"):
            p.execute_poststages(
                tmp_path, model=MODEL, resume=True, execute_batch_fn=execute, read_log_fn=read
            )
        assert c.read_json(tmp_path / "poststages.json")["batches"][0]["status"] == "complete"
    assert not folder.exists()
