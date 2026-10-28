"""Campaign analysis selects complete batches without hiding adverse outcomes."""

import json
from types import SimpleNamespace as NS

import pytest

from scheming_inspect import analyze as a
from scheming_inspect.protocol import plan

MODEL = "mockllm/model"


def campaign(tmp_path):
    cell = {**plan("main")[0], "n": 6}

    def sample(name, **metrics):
        metadata = {key: value for key, value in cell.items() if key != "n"}
        metadata.update(
            scenario_id="same",
            pair_id=name,
            behavior={
                "completed": 1,
                "truncated": 0,
                "choice_recorded": 1,
                "choice_rejected": 0,
                "oversight_disabled": 0,
                **metrics,
            },
        )
        return NS(
            id=name,
            epoch=1,
            metadata=metadata,
            store={},
            scores={},
            error=None,
            invalidation=None,
            limit=None,
        )

    samples = [
        sample("valid", oversight_disabled=1),
        sample("blocked", choice_recorded=0, choice_rejected=1),
        sample("limited", completed=0, truncated=1),
        sample("error", completed=0),
    ]
    samples[-1].error = NS(message="Recorded failure")
    paths = {name: tmp_path / name / "run.eval" for name in ("complete", "partial", "_interrupted")}
    logs = {}
    for name, path in paths.items():
        path.parent.mkdir()
        path.write_text("Fixture content; no hash is required.")
        logs[str(path)] = NS(
            status="success" if name == "complete" else "error",
            eval=NS(model=MODEL, task_args={}),
            samples=samples if name == "complete" else [sample("pending")],
            error=None,
        )
    base = {
        "format": "simple-campaign-v1",
        "status": "interrupted",
        "config": {"model": MODEL},
        "cells": [cell],
        "total_samples": 6,
        "batches": [
            {
                "batch_id": "base-0",
                "cell_index": 0,
                "status": "complete",
                "sample_ids": [s.id for s in samples],
                "log_paths": [str(paths["complete"])],
            },
            {
                "batch_id": "base-1",
                "cell_index": 0,
                "status": "interrupted",
                "sample_ids": ["pending", "unrecorded"],
                "log_paths": [str(paths["partial"])],
            },
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(base))
    calls = []

    def read(path):
        calls.append(str(path))
        return logs[str(path)]

    return base, logs, calls, read


def save(tmp_path, manifest):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))


def test_complete_batches_keep_adverse_outcomes_and_skip_partial_archived_logs(tmp_path):
    _, _, calls, read = campaign(tmp_path)
    rows, logs, coverage = a.load_campaign_rows(tmp_path, read)
    assert [row["sample_id"] for row in rows] == ["valid", "blocked", "limited", "error"]
    assert calls == [str(tmp_path / "complete" / "run.eval")]
    assert rows[1]["metrics"]["choice_recorded"] == 0
    assert rows[1]["metrics"]["choice_rejected"] == 1
    assert rows[2]["truncated"] and rows[3]["error"]
    assert all(row["canonical_batch_id"] == "base-0" for row in rows)
    assert logs[0]["samples"] == 4
    assert coverage["planned_samples"] == 6
    assert coverage["canonical_samples"] == 4
    assert coverage["not_yet_canonical_samples"] == 2
    assert "recovery_attempts" not in coverage and "source_revisions" not in coverage
    assert not any("sha256" in key or "recovery" in key for row in rows for key in row)
    summary = a.summarize(rows, draws=10)["groups"][0]
    assert summary["coverage"]["no_recorded_choice"] == 1
    assert summary["coverage"]["blocked_choice"] == 1
    assert summary["coverage"]["invalid_schema"] == 0
    observed = summary["conditions"]["default"]["oversight_disabled"]
    assert observed["n"] == 2 and observed["mean"] == 0.5


def test_no_log_hash_or_source_fingerprint_is_required(tmp_path):
    _, _, _, read = campaign(tmp_path)
    (tmp_path / "complete" / "run.eval").write_text("Different fixture bytes.")
    assert len(a.load_campaign_rows(tmp_path, read)[0]) == 4


@pytest.mark.parametrize(
    "change,match",
    [
        ("duplicate", "Duplicate sample"),
        ("unexpected", "Unexpected sample"),
        ("missing", "missing expected"),
        ("epoch", "unexpected epoch"),
        ("cell", "metadata differs"),
        ("model", "model differs"),
    ],
)
def test_complete_log_identity_must_match_saved_batch(tmp_path, change, match):
    _, logs, _, read = campaign(tmp_path)
    log = logs[str(tmp_path / "complete" / "run.eval")]
    if change == "duplicate":
        log.samples.append(log.samples[0])
    elif change == "unexpected":
        log.samples[0].id = "another-case"
    elif change == "missing":
        log.samples.pop()
    elif change == "epoch":
        log.samples[0].epoch = 2
    elif change == "cell":
        log.samples[0].metadata["nudge"] = False
    else:
        log.eval.model = "another/model"
    with pytest.raises(ValueError, match=match):
        a.load_campaign_rows(tmp_path, read)


def test_duplicate_complete_records_cannot_count_twice(tmp_path):
    base, _, _, read = campaign(tmp_path)
    base["batches"].append({**base["batches"][0], "batch_id": "duplicate"})
    save(tmp_path, base)
    with pytest.raises(ValueError, match="Duplicate sample"):
        a.load_campaign_rows(tmp_path, read)


def test_empty_campaign_reports_pending_coverage(tmp_path):
    base, _, calls, read = campaign(tmp_path)
    base["batches"][0]["status"] = "planned"
    save(tmp_path, base)
    rows, logs, coverage = a.load_campaign_rows(tmp_path, read)
    assert rows == logs == calls == []
    assert coverage["canonical_samples"] == 0
    assert coverage["not_yet_canonical_samples"] == 6


def test_campaign_analysis_cli_exports_coverage_without_input_paths(tmp_path, monkeypatch):
    _, _, _, read = campaign(tmp_path)
    monkeypatch.setattr(a, "_read_log", read)
    out = tmp_path / "analysis"
    a.main(["--campaign-dir", str(tmp_path), "--out", str(out), "--draws", "10"])
    summary = json.loads((out / "summary.json").read_text())
    assert summary["campaign"]["canonical_samples"] == 4
    assert summary["groups"][0]["coverage"]["rows"] == 4
    assert summary["groups"][0]["coverage"]["errors"] == 1
    assert (out / "samples.csv").is_file()
