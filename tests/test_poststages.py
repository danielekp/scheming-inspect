"""Offline post-stage planning, coverage, fork independence, and resume tests."""

import copy
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ModelName,
    ModelOutput,
)
from inspect_ai.solver import TaskState

from scheming_inspect import campaign
from scheming_inspect import poststages as p
from scheming_inspect.protocol import plan

MODEL = "openrouter/deepseek/deepseek-v4-flash"


@pytest.fixture(autouse=True)
def cost_reporting(monkeypatch):
    reports = []
    monkeypatch.setattr(campaign, "report_cost", lambda root: reports.append(root), raising=False)
    return reports


def original(sample_id, cell, *, guilty=False, **changes):
    metadata = {key: value for key, value in cell.items() if key != "n"}
    metadata.update(
        scenario_id="shared",
        data_version="authored-v1",
        behavior={"completed": 1, "truncated": 0, "oversight_disabled": int(guilty)},
    )
    sample = NS(
        id=sample_id,
        epoch=1,
        metadata=metadata,
        store={},
        scores={},
        input=[ChatMessageUser(content="Select a plan.")],
        messages=[
            ChatMessageSystem(content="Finish the task."),
            ChatMessageUser(content="Select a plan."),
            ChatMessageAssistant(content="Recorded answer."),
        ],
        output=ModelOutput.from_content(MODEL, "Recorded answer."),
        error=None,
        invalidation=None,
        limit=None,
        events=[NS(event="model", secret_old_billing_id="gen-original")],
    )
    for key, value in changes.items():
        setattr(sample, key, value)
    return sample


def setup_base(tmp_path, cases=None):
    cell = plan("main")[0]
    cases = cases if cases is not None else [original("g", cell, guilty=True), original("i", cell)]
    source_path = tmp_path / "source.eval"
    source_path.write_text("frozen-source")
    log = NS(status="success", eval=NS(model=MODEL), samples=cases)
    base = {
        "status": "base_complete",
        "fingerprint": "basehash",
        "source_hashes": {},
        "config": {"model": MODEL, "seed": 42},
        "cells": [cell],
        "completed_base_logs": [str(source_path)],
        "batches": [
            {
                "batch_id": "base-0",
                "cell_index": 0,
                "status": "settled",
                "sample_ids": [str(sample.id) for sample in cases],
                "log_paths": [str(source_path)],
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(base))
    return base, log, lambda path: log


def prepared(tmp_path):
    base, log, read = setup_base(tmp_path)
    discovery = p.discover_sources(base, MODEL, read)
    groups, eligibility = p.make_groups(discovery, 42)
    spec = {
        "groups": groups,
        "config": {"max_tokens": 8000},
        "discovery": discovery,
        "eligibility": eligibility,
    }
    return {"spec": spec, "progress": {}, "status": "planned"}, read


def run(tmp_path, read, execute=None, **kwargs):
    return p.execute_poststages(
        tmp_path,
        model=MODEL,
        read_log_fn=read,
        execute_batch_fn=execute,
        current_source_hash_fn=dict,
        source_hashes={},
        **kwargs,
    )


def test_main_filter_uses_full_recorded_cell():
    main = plan("main")[0]
    assert p.is_main_cell(main)
    assert p.is_main_cell(main | {"seed": 123})
    for changed in (
        {"n": 1},
        {"nudge": False},
        {"goal_mode": "reversed"},
        {"difficulty": "easy"},
        {"goal_role": "user"},
    ):
        assert not p.is_main_cell(main | changed)
    assert not p.is_main_cell({"experiment": "oversight"})


def test_discovery_records_each_bad_source_without_aborting(tmp_path):
    cell = plan("main")[0]
    samples = [original(str(i), cell) for i in range(7)]
    samples[1].error = "provider failed"
    samples[2].invalidation = "invalid"
    samples[3].metadata["behavior"]["completed"] = 0
    samples[4].messages = [ChatMessageTool(content="orphan", tool_call_id="absent")]
    samples[5].metadata["nudge"] = False
    samples[6].metadata["behavior"]["truncated"] = 1
    base, _, read = setup_base(tmp_path, samples)
    discovery = p.discover_sources(base, MODEL, read)
    assert len(discovery["sources"]) == 1
    assert {row["reason"] for row in discovery["excluded"]} == {
        "source_error",
        "source_invalidated",
        "source_incomplete_or_completion_unknown",
        "malformed_transcript",
        "sample_cell_mismatch",
        "source_truncated",
    }
    assert discovery["requested"]["audit_main"] == 7


def test_unreadable_and_missing_logs_are_explicit(tmp_path):
    base, _, _ = setup_base(tmp_path)

    def broken(path):
        raise OSError("corrupt")

    discovery = p.discover_sources(base, MODEL, broken)
    assert not discovery["sources"]
    assert [row["reason"] for row in discovery["excluded"]].count("missing_sample") == 2
    assert any(row["reason"] == "unreadable_log" for row in discovery["excluded"])


def test_ablation_never_becomes_main_via_sample_labels(tmp_path):
    base, log, read = setup_base(tmp_path)
    base["cells"][0] = base["cells"][0] | {"nudge": False, "n": 400}
    for sample in log.samples:
        sample.metadata["nudge"] = False
    discovery = p.discover_sources(base, MODEL, read)
    groups, _ = p.make_groups(discovery)
    assert {group["stage"] for group in groups} == {"audit_remaining"}


def test_all_fixed_variants_repeats_and_matched_pairs_are_frozen(tmp_path):
    manifest, _ = prepared(tmp_path)
    counts = p.progress_report(manifest)["stages"]
    assert counts["audit_main"]["planned"] == 2
    assert counts["interrogation"]["planned"] == 2
    assert counts["fixed"]["planned"] == 160
    fixed = next(group for group in manifest["spec"]["groups"] if group["stage"] == "fixed")
    ids = p.group_sample_ids(fixed)
    assert len(ids) == len(set(ids)) == 160
    assert len(fixed["variants"]) == 8 and fixed["repeats"] == 20
    pair = next(group for group in manifest["spec"]["groups"] if group["stage"] == "interrogation")
    assert {value.rsplit(":", 1)[1] for value in p.group_sample_ids(pair)} == {"guilty", "innocent"}


def test_two_exfiltration_actions_each_get_all_160_samples(tmp_path):
    base, log, read = setup_base(tmp_path)
    base["cells"][0] = plan("main")[1]
    for sample in log.samples:
        sample.metadata["experiment"] = "exfiltration"
        sample.metadata["behavior"].update(
            weights_copied=1, successor_deleted=1, successor_replaced=0
        )
    groups, _ = p.make_groups(p.discover_sources(base, MODEL, read))
    assert sum(group["samples"] for group in groups if group["stage"] == "fixed") == 640


@pytest.mark.asyncio
async def test_semantic_task_restores_answer_without_generating_or_copying_events(tmp_path):
    manifest, read = prepared(tmp_path)
    groups = {group["id"]: group for group in manifest["spec"]["groups"]}
    reader = p.SourceReader(manifest["spec"]["discovery"]["sources"], read)
    selection = p.next_selections(manifest, 32)
    task = p.build_batch(selection, groups, reader, max_tokens=8000, token_limit=100000, seed=42)
    sample = list(task.dataset)[0]
    state = TaskState(
        model=ModelName(MODEL),
        sample_id=sample.id,
        epoch=1,
        input=copy.deepcopy(sample.input),
        messages=copy.deepcopy(sample.input),
        metadata=copy.deepcopy(sample.metadata),
    )
    generate = AsyncMock(side_effect=AssertionError("No target generation permitted"))
    state = await p.restore_completion()(state, generate)
    assert state.output.completion == "Recorded answer."
    assert state.completed and not generate.called
    assert "gen-original" not in sample.model_dump_json()
    assert task.sandbox is None
    assert task.config.max_retries == 0 and task.config.cache_prompt is False
    sample.input[0].content = "changed"
    assert reader.sample(sample.metadata["source_key"]).messages[0].text == "Finish the task."


def test_selection_preserves_stage_order_and_interrogation_pairs(tmp_path):
    manifest, _ = prepared(tmp_path)
    selected = p.next_selections(manifest, 32)
    assert sum(item["stop"] - item["start"] for item in selected) == 2
    for group in manifest["spec"]["groups"]:
        if group["stage"] == "audit_main":
            manifest["progress"][group["id"]] = group["samples"]
    selected = p.next_selections(manifest, 3)
    assert sum(item["stop"] - item["start"] for item in selected) == 2
    assert len(selected) == 1
    for group in manifest["spec"]["groups"]:
        if group["stage"] == "interrogation":
            manifest["progress"][group["id"]] = 2
    selected = p.next_selections(manifest, 32)
    assert sum(item["stop"] - item["start"] for item in selected) == 32
    assert selected[0]["start"] == 0 and selected[0]["stop"] == 32


@pytest.mark.parametrize("completed_status", ["complete", "settled", "complete_reserved"])
def test_full_offline_execution_has_no_limit_one_and_resumes_without_dispatch(
    tmp_path, completed_status, cost_reporting
):
    _, _, read = setup_base(tmp_path)
    calls = []

    def execute(task, **kwargs):
        if task is None:
            return {"status": completed_status}
        ids = [str(sample.id) for sample in task.dataset]
        assert ids == kwargs["expected_sample_ids"]
        assert len(ids) <= 32
        assert kwargs["provider"] == {}
        assert "upper_usd" not in kwargs and "ledger" not in kwargs
        assert all(value == MODEL for value in kwargs["model_roles"].values())
        calls.append((kwargs["stage"], ids))
        return {
            "status": completed_status,
            "log_paths": [],
            "sample_status_counts": {"valid": len(ids)},
        }

    result = run(tmp_path, read, execute=execute)
    assert result["status"] == "complete"
    assert result["stages"]["fixed"]["settled"] == 160
    assert [stage for stage, _ in calls] == ["audit_main", "interrogation"] + ["fixed"] * 5
    assert len({sample_id for _, ids in calls for sample_id in ids}) == 164
    assert len(cost_reporting) == len(calls)
    frozen = json.loads((tmp_path / "poststages.json").read_text())
    assert "rates" not in frozen["spec"]
    assert all("upper_usd" not in batch for batch in frozen["batches"])
    before = len(calls)
    again = run(tmp_path, read, execute=execute, resume=True)
    assert again == result and len(calls) == before


def test_cost_report_failure_never_stops_or_skips_work(tmp_path, monkeypatch):
    _, _, read = setup_base(tmp_path)

    def failed_report(root):
        raise OSError("optional counter unavailable")

    monkeypatch.setattr(campaign, "report_cost", failed_report, raising=False)
    result = run(tmp_path, read, execute=lambda *a, **k: {"status": "complete"})
    assert result["status"] == "complete"
    assert result["stages"]["fixed"] == {"planned": 160, "settled": 160, "pending": 0}


def test_resume_archives_incomplete_attempt_and_reruns_its_whole_frozen_batch(tmp_path):
    _, source_log, _ = setup_base(tmp_path)
    partial_log = None
    original_ids, paths = [], {}

    def read(path):
        return source_log if path.name == "source.eval" else partial_log

    def interrupted(task, **kwargs):
        nonlocal partial_log
        original_ids.extend(kwargs["expected_sample_ids"])
        directory = kwargs["log_dir"]
        directory.mkdir(parents=True)
        output = directory / "result.eval"
        output.write_bytes(b"immutable partial attempt with one invalid outcome")
        journal = {
            key: kwargs[key]
            for key in ("batch_id", "stage", "model", "fingerprint", "model_roles", "max_samples")
        }
        journal.update(sample_ids=original_ids, status="interrupted", log_paths=[str(output)])
        (directory / "batch.json").write_text(json.dumps(journal))
        paths.update(directory=directory, partial=output)
        partial_log = NS(
            status="error",
            eval=NS(
                model=MODEL,
                metadata={
                    "campaign_batch_id": kwargs["batch_id"],
                    "campaign_fingerprint": kwargs["fingerprint"],
                },
            ),
            samples=[
                NS(
                    id=original_ids[0],
                    epoch=1,
                    error=None,
                    limit=None,
                    metadata={"behavior": {"completed": 1, "schema_valid": 0}},
                )
            ],
        )
        raise RuntimeError("interrupted after the first invalid-format answer")

    with pytest.raises(RuntimeError, match="interrupted"):
        run(tmp_path, read, interrupted)
    assert len(original_ids) == 2
    calls = []

    def execute(task, **kwargs):
        assert task is not None
        ids = [str(sample.id) for sample in task.dataset]
        assert ids == kwargs["expected_sample_ids"]
        if not calls:
            assert ids == original_ids
            saved = json.loads((tmp_path / "poststages.json").read_text())
            assert len(saved["batches"][0]["restarts"]) == 1
            assert saved["batches"][0]["status"] == "planned"
            assert saved["batches"][0]["log_paths"] == []
            assert not paths["partial"].exists()
        calls.append((kwargs["stage"], ids))
        return {
            "status": "complete",
            "sample_status_counts": {"invalid_schema": 1, "success": len(ids) - 1},
        }

    result = run(tmp_path, read, execute, resume=True)
    assert result["status"] == "complete"
    assert sum(len(ids) for _, ids in calls) == 164
    archives = list((tmp_path.parent / (tmp_path.name + "-superseded")).rglob("result.eval"))
    assert len(archives) == 1
    assert archives[0].read_bytes() == b"immutable partial attempt with one invalid outcome"
    assert not archives[0].is_relative_to(tmp_path)
    assert result["logged_sample_outcomes"]["audit_main"]["invalid_schema"] == 1


def test_completed_invalid_outcome_is_preserved_on_resume(tmp_path, monkeypatch):
    _, _, read = setup_base(tmp_path)

    def complete(task, **kwargs):
        return {
            "status": "complete",
            "sample_status_counts": {"invalid_schema": len(kwargs["expected_sample_ids"])},
        }

    first = run(tmp_path, read, complete)

    def forbid_reset(*args, **kwargs):
        pytest.fail("A completed invalid outcome must not be selected for rerun")

    monkeypatch.setattr(campaign, "reset_incomplete_batch", forbid_reset)

    def resume(task, **kwargs):
        assert task is None
        return {"status": "complete"}

    again = run(tmp_path, read, resume, resume=True)
    assert again == first
    assert again["logged_sample_outcomes"]["audit_main"] == {"invalid_schema": 2}


def test_resume_rejects_mutated_source_and_config(tmp_path):
    _, _, read = setup_base(tmp_path)
    run(tmp_path, read, execute=lambda *a, **k: {"status": "blocked"})
    with pytest.raises(ValueError, match="configuration"):
        run(tmp_path, read, resume=True, max_tokens=4000)
    (tmp_path / "source.eval").write_text("modified")
    with pytest.raises(ValueError, match="source log changed"):
        run(tmp_path, read, resume=True)


def test_incomplete_source_is_not_disguised_as_full_success(tmp_path):
    cell = plan("main")[0]
    source = original("bad", cell)
    source.metadata["behavior"]["completed"] = 0
    _, _, read = setup_base(tmp_path, [source])
    result = run(tmp_path, read, execute=lambda *a, **k: pytest.fail("excluded source"))
    assert result["status"] == "complete_with_exclusions"
    assert result["requested"]["audit_main"] == 1
    assert result["stages"]["audit_main"]["planned"] == 0
    assert result["exclusions"] == {"source_incomplete_or_completion_unknown": 1}


def test_invalid_schema_and_wrong_seed_are_explicitly_excluded(tmp_path):
    cell = plan("main")[0]
    samples = [original("invalid", cell), original("seed", cell)]
    samples[0].metadata["behavior"]["schema_valid"] = 0
    samples[1].metadata["seed"] = 99
    base, _, read = setup_base(tmp_path, samples)
    discovery = p.discover_sources(base, MODEL, read)
    assert {row["reason"] for row in discovery["excluded"]} == {
        "source_invalid_schema",
        "sample_cell_mismatch",
    }


def test_reader_rechecks_source_hash_before_loading(tmp_path):
    manifest, read = prepared(tmp_path)
    discovery = manifest["spec"]["discovery"]
    reader = p.SourceReader(discovery["sources"], read, discovery["source_logs"])
    (tmp_path / "source.eval").write_text("changed")
    with pytest.raises(ValueError, match="source log changed"):
        reader.sample(next(iter(discovery["sources"])))
