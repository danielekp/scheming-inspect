"""Canonical recovery selections preserve outcomes without double-counting attempts."""
import hashlib
import json
from types import SimpleNamespace as NS

import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, ModelOutput

from scheming_inspect import analyze as a
from scheming_inspect import poststages as p
from scheming_inspect.protocol import plan

MODEL = "openrouter/deepseek/deepseek-v4-flash"


def campaign(tmp_path):
    cell = plan("main")[0]

    def sample(name, **metrics):
        metadata = {key: value for key, value in cell.items() if key != "n"}
        metadata.update(scenario_id="same", data_version="authored-v1", pair_id=name,
                        behavior={"completed": 1, "truncated": 0, "schema_valid": 1,
                                  "oversight_disabled": 0, **metrics})
        return NS(id=name, epoch=1, metadata=metadata, store={}, scores={}, error=None,
                  invalidation=None, limit=None, input="Select a plan.",
                  messages=[ChatMessageSystem(content="Finish the task."),
                            ChatMessageUser(content="Select a plan."),
                            ChatMessageAssistant(content="Done.")],
                  output=ModelOutput.from_content(MODEL, "Done."))

    old_samples = [sample("keep"), sample("invalid", schema_valid=0),
                   sample("limited", completed=0, truncated=1), sample("retry", completed=0)]
    old_samples[-1].error = NS(message="Infrastructure failure")
    logs, records = {}, []
    for name, samples, status, selected, retained in (
        ("old", old_samples, "error", ["keep", "invalid", "limited"], True),
        ("new", [sample("retry")], "success", ["retry"], False),
    ):
        path = tmp_path / f"{name}.eval"
        path.write_text(f"immutable-{name}")
        logs[str(path)] = NS(status=status, eval=NS(model=MODEL, task_args={}), samples=samples,
                             error=NS(message="Aggregation failed") if retained else None)
        records.append({"path": str(path), "cell": cell, "cell_index": 0,
                        "batch_id": "logical" if retained else "retry-attempt-1",
                        "expected_sample_ids": selected, "recovery_retained": retained,
                        "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    base = {"status": "base_complete", "config": {"model": MODEL}, "cells": [cell],
            "total_samples": 4, "source_revisions": [{"reason": "scorer repair"}],
            "completed_base_logs": [record["path"] for record in records],
            "completed_base_log_records": records,
            "batches": [{"batch_id": "logical", "cell_index": 0, "status": "settled",
                         "sample_ids": ["keep", "invalid", "limited", "retry"],
                         "log_paths": [record["path"] for record in records],
                         "recovery": {"retained_ids": ["keep", "invalid", "limited"],
                                      "retry_ids": ["retry"], "attempt_id": "retry-attempt-1"}}]}
    (tmp_path / "manifest.json").write_text(json.dumps(base))
    return base, logs, lambda path: logs[str(path)]


def test_canonical_analysis_preserves_adverse_outcomes_and_audits_old_error(tmp_path):
    _, _, read = campaign(tmp_path)
    rows, logs, audit = a.load_campaign_rows(tmp_path, read)
    assert len(rows) == 4
    assert {row["sample_id"] for row in rows} == {"keep", "invalid", "limited", "retry"}
    assert not any(row["error"] for row in rows)
    assert next(row for row in rows if row["sample_id"] == "invalid")["metrics"]["schema_valid"] == 0
    assert next(row for row in rows if row["sample_id"] == "limited")["truncated"]
    assert audit["canonical_samples"] == 4 and audit["not_yet_canonical_samples"] == 0
    assert audit["excluded_attempt_counts"]["errors"] == 1
    assert audit["excluded_attempt_samples"][0]["sample_id"] == "retry"
    assert logs[0]["status"] == "error" and logs[0]["canonical_samples"] == 3
    assert audit["recovery_attempts"][0]["recovery"]["attempt_id"] == "retry-attempt-1"
    assert audit["source_revisions"] == [{"reason": "scorer repair"}]


def test_poststage_discovery_retains_valid_error_log_sample_only(tmp_path):
    base, _, read = campaign(tmp_path)
    result = p.discover_sources(base, MODEL, read)
    assert {source["sample_id"] for source in result["sources"].values()} == {"keep", "retry"}
    assert result["requested"]["audit_main"] == 4
    assert {row["reason"] for row in result["excluded"]} == {"source_invalid_schema", "source_truncated"}
    assert len(result["recovery_exclusions"]) == 1
    assert result["recovery_exclusions"][0]["error"]


def test_retained_error_log_requires_verified_hash(tmp_path):
    base, _, read = campaign(tmp_path)
    base["completed_base_log_records"][0].pop("log_sha256")
    with pytest.raises(ValueError, match="SHA-256"):
        p.discover_sources(base, MODEL, read)
    base, _, read = campaign(tmp_path)
    (tmp_path / "old.eval").write_text("tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        p.discover_sources(base, MODEL, read)
    with pytest.raises(ValueError, match="SHA-256"):
        a.load_campaign_rows(tmp_path, read)


def test_duplicate_canonical_ids_and_wrong_cells_are_rejected(tmp_path):
    base, _, _ = campaign(tmp_path)
    base["completed_base_log_records"][1]["expected_sample_ids"] = ["keep"]
    with pytest.raises(ValueError, match="duplicated"):
        p.canonical_log_records(base)
    base, _, _ = campaign(tmp_path)
    base["completed_base_log_records"][1]["cell"] = {**base["cells"][0], "nudge": False}
    with pytest.raises(ValueError, match="configuration"):
        p.canonical_log_records(base)


def test_unexpected_sample_in_normal_record_is_not_silently_selected(tmp_path):
    base, logs, read = campaign(tmp_path)
    logs[str(tmp_path / "new.eval")].samples.append(logs[str(tmp_path / "old.eval")].samples[0])
    with pytest.raises(ValueError, match="Unexpected sample"):
        p.discover_sources(base, MODEL, read)
    with pytest.raises(ValueError, match="Unexpected sample"):
        a.load_campaign_rows(tmp_path, read)


def test_legacy_campaign_record_without_index_or_hash_still_works(tmp_path):
    base, _, read = campaign(tmp_path)
    record = base["completed_base_log_records"][1]
    record.update(batch_id="logical")
    for key in ("cell_index", "log_sha256", "recovery_retained"):
        record.pop(key)
    base["batches"][0].update(sample_ids=["retry"], log_paths=[str(tmp_path / "new.eval")])
    base["batches"][0].pop("recovery")
    base.update(total_samples=1, completed_base_log_records=[record])
    (tmp_path / "manifest.json").write_text(json.dumps(base))
    assert len(a.load_campaign_rows(tmp_path, read)[0]) == 1
    assert len(p.discover_sources(base, MODEL, read)["sources"]) == 1
    base.pop("completed_base_log_records")
    assert len(p.discover_sources(base, MODEL, read)["sources"]) == 1


def test_campaign_analysis_cli_exports_recovery_audit_without_input_paths(tmp_path, monkeypatch):
    _, _, read = campaign(tmp_path)
    monkeypatch.setattr(p, "_read_log", read)
    out = tmp_path / "analysis"
    a.main(["--campaign-dir", str(tmp_path), "--out", str(out), "--draws", "10"])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["campaign"]["canonical_samples"] == 4
    assert summary["campaign"]["excluded_attempt_counts"]["errors"] == 1
    assert summary["groups"][0]["coverage"]["rows"] == 4
    assert (out / "samples.csv").is_file()
