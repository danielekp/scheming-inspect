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

from scheming_inspect import poststages as p
from scheming_inspect.protocol import plan

MODEL = "openrouter/deepseek/deepseek-v4-flash"
RATES = {
    "pricing": {"prompt": "0.000000091", "completion": "0.000000182"},
    "context_length": 1048576,
    "provider": {
        "max_price": {"prompt": 0.091, "completion": 0.182, "request": 0},
        "require_parameters": True,
    },
}


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
        "provider_policy": RATES,
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
        "rates": RATES,
        "config": {"max_tokens": 8000},
        "discovery": discovery,
        "eligibility": eligibility,
    }
    return {"spec": spec, "progress": {}, "status": "planned"}, read


class Ledger:
    def __init__(self, remaining=50):
        self.remaining = remaining
        self.batches = {}

    def snapshot(self):
        return {"remaining_usd": self.remaining, "batches": self.batches, "stages": {}}


def run(tmp_path, read, ledger=None, execute=None, **kwargs):
    return p.execute_poststages(
        tmp_path,
        model=MODEL,
        ledger=ledger or Ledger(),
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
    selection = p.next_selections(manifest, 50, 32)
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


def test_budget_bound_counts_all_model_roles():
    call = 1048576 * 0.000000091 + 8000 * 0.000000182
    assert p.upper_bound("audit_main", 1, RATES, 8000) == pytest.approx(call)
    assert p.upper_bound("fixed", 32, RATES, 8000) == pytest.approx(64 * call)
    assert p.upper_bound("interrogation", 2, RATES, 8000) == pytest.approx(56 * call)
    assert p.logical_generations("interrogation", 2) == {
        "target": 14,
        "grader": 14,
        "interrogator": 28,
    }


def test_budget_adapts_all_stages_without_splitting_interrogation_pairs(tmp_path):
    manifest, _ = prepared(tmp_path)
    selected = p.next_selections(manifest, 0.1, 32)
    assert sum(item["stop"] - item["start"] for item in selected) == 1
    for group in manifest["spec"]["groups"]:
        if group["stage"] == "audit_main":
            manifest["progress"][group["id"]] = group["samples"]
    assert not p.next_selections(manifest, 5, 32)
    selected = p.next_selections(manifest, 6, 32)
    assert sum(item["stop"] - item["start"] for item in selected) == 2
    for group in manifest["spec"]["groups"]:
        if group["stage"] == "interrogation":
            manifest["progress"][group["id"]] = 2
    selected = p.next_selections(manifest, 5, 32)
    assert 0 < sum(item["stop"] - item["start"] for item in selected) < 32


@pytest.mark.parametrize("completed_status", ["settled", "complete_reserved"])
def test_full_offline_execution_has_no_limit_one_and_resumes_without_dispatch(
    tmp_path, completed_status
):
    _, _, read = setup_base(tmp_path)
    calls = []

    def execute(task, **kwargs):
        if task is None:
            return {"status": completed_status}
        ids = [str(sample.id) for sample in task.dataset]
        assert ids == kwargs["expected_sample_ids"]
        assert len(ids) <= 32
        assert kwargs["provider"] == RATES["provider"]
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
    before = len(calls)
    again = run(tmp_path, read, execute=execute, resume=True)
    assert again == result and len(calls) == before


def test_budget_halt_keeps_every_unfinished_group_and_exact_counts(tmp_path):
    _, _, read = setup_base(tmp_path)
    result = run(
        tmp_path,
        read,
        ledger=Ledger(0.05),
        execute=lambda *a, **k: pytest.fail("unaffordable dispatch"),
    )
    assert result["status"] == "pending_budget"
    assert result["stages"]["fixed"] == {"planned": 160, "settled": 0, "pending": 160}
    manifest = json.loads((tmp_path / "poststages.json").read_text())
    assert manifest["batches"] == []
    assert len(manifest["spec"]["groups"]) == 4


def test_existing_reserved_batch_reconciles_without_reconstructing_target(tmp_path):
    _, _, read = setup_base(tmp_path)
    ledger = Ledger()

    def interrupted(task, **kwargs):
        ledger.batches[kwargs["batch_id"]] = {"status": "reserved"}
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        run(tmp_path, read, ledger, interrupted)
    calls = []

    def reconcile(task, **kwargs):
        calls.append(task)
        assert task is None
        return {"status": "blocked"}

    result = run(tmp_path, read, ledger, reconcile, resume=True)
    assert result["status"] == "blocked" and calls == [None]
    assert result["stages"]["audit_main"]["settled"] == 0


def test_resume_rejects_mutated_source_and_config(tmp_path):
    _, _, read = setup_base(tmp_path)
    run(tmp_path, read, ledger=Ledger(0.01), execute=lambda *a, **k: None)
    with pytest.raises(ValueError, match="configuration"):
        run(tmp_path, read, ledger=Ledger(0.01), resume=True, max_tokens=4000)
    (tmp_path / "source.eval").write_text("modified")
    with pytest.raises(ValueError, match="source log changed"):
        run(tmp_path, read, ledger=Ledger(0.01), resume=True)


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
