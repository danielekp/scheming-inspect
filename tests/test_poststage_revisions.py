"""Preserve completed post-stage journals across explicit source migrations."""

import hashlib
from unittest.mock import Mock

import pytest

from scheming_inspect import campaign
from scheming_inspect import poststages as p

MODEL = "mockllm/poststage-revisions"
HASHES = {"fixture.py": "current-source-hash"}


def run(root, **kwargs):
    return p.execute_poststages(
        root,
        model=MODEL,
        source_hashes=HASHES,
        current_source_hash_fn=lambda: HASHES,
        **kwargs,
    )


def write_base(root):
    campaign.atomic_json(
        root / "manifest.json",
        {
            "status": "base_complete",
            "fingerprint": "base-current",
            "source_hashes": HASHES,
            "config": {"model": MODEL, "seed": 42},
            "cells": [],
            "batches": [],
        },
    )


def revised_campaign(root, monkeypatch, status="complete"):
    write_base(root)
    forbidden = Mock(side_effect=AssertionError("Completed work must never generate or rebuild"))
    monkeypatch.setattr(campaign, "inspect_eval", forbidden)
    monkeypatch.setattr(campaign, "_role_models", forbidden)
    monkeypatch.setattr(campaign, "_read_log", forbidden)
    monkeypatch.setattr(p, "build_batch", forbidden)
    assert run(root)["status"] == "complete"
    path = root / "poststages.json"
    manifest = campaign.read_json(path)
    stages = ("audit_main", "interrogation", "fixed")
    ids = (["audit:a"], ["pair:guilty", "pair:innocent"], ["fixed:a"])
    manifest["spec"]["groups"] = [
        {"id": f"group-{i}", "stage": stage, "samples": len(sample_ids)}
        for i, (stage, sample_ids) in enumerate(zip(stages, ids, strict=True))
    ]
    manifest["fingerprint"] = p.digest(manifest["spec"])
    fingerprints = ["historical-first", "historical-second", manifest["fingerprint"]]
    snapshots = {}
    for i, (stage, sample_ids, fingerprint) in enumerate(
        zip(stages, ids, fingerprints, strict=True)
    ):
        batch_id = f"revision-batch-{i}"
        directory = root / "poststages" / stage / batch_id
        directory.mkdir(parents=True)
        log_path = directory / "result.eval"
        log_path.write_bytes(f"Immutable log from {fingerprint}".encode())
        roles = {"grader": MODEL} | ({"interrogator": MODEL} if stage == "interrogation" else {})
        journal = {
            "batch_id": batch_id,
            "stage": stage,
            "model": MODEL,
            "fingerprint": fingerprint,
            "sample_ids": sample_ids,
            "model_roles": roles,
            "max_samples": 8,
            "status": status,
            "log_paths": [str(log_path)],
            "log_fingerprints": {str(log_path): hashlib.sha256(log_path.read_bytes()).hexdigest()},
        }
        journal_path = directory / "batch.json"
        campaign.atomic_json(journal_path, journal)
        snapshots.update({log_path: log_path.read_bytes(), journal_path: journal_path.read_bytes()})
        batch = {
            "batch_id": batch_id,
            "stage": stage,
            "sample_ids": sample_ids,
            "status": status,
            "selections": [{"group_id": f"group-{i}", "start": 0, "stop": len(sample_ids)}],
            "log_paths": [str(log_path)],
        }
        if i < 2:
            batch["execution_fingerprint"] = fingerprint
            manifest.setdefault("source_revisions", []).append(
                {"preserved_poststage_fingerprints": {batch_id: fingerprint}}
            )
        manifest["batches"].append(batch)
        manifest["progress"][f"group-{i}"] = len(sample_ids)
    campaign.atomic_json(path, manifest)
    return manifest, fingerprints, snapshots, forbidden


@pytest.mark.parametrize("status", sorted(campaign.COMPLETE_BATCH_STATES))
def test_completed_mixed_revisions_resume_with_original_journals(tmp_path, monkeypatch, status):
    manifest, fingerprints, snapshots, forbidden = revised_campaign(tmp_path, monkeypatch, status)
    execute = Mock(wraps=campaign.execute_batch)
    monkeypatch.setattr(campaign, "execute_batch", execute)

    result = run(tmp_path, resume=True)

    assert result["status"] == "complete"
    assert [call.kwargs["fingerprint"] for call in execute.call_args_list] == fingerprints
    assert all(call.args == (None,) and call.kwargs["resume"] for call in execute.call_args_list)
    assert campaign.read_json(tmp_path / "poststages.json")["batches"] == manifest["batches"]
    assert all(path.read_bytes() == original for path, original in snapshots.items())
    forbidden.assert_not_called()


@pytest.mark.parametrize(
    "tampering", ["missing", "empty", "wrong_batch", "wrong_fingerprint", "not_mapping", "batch"]
)
def test_historical_fingerprint_needs_exact_preservation(tmp_path, monkeypatch, tampering):
    manifest, _, snapshots, forbidden = revised_campaign(tmp_path, monkeypatch)
    revision = manifest["source_revisions"][0]
    first = manifest["batches"][0]
    if tampering == "missing":
        manifest.pop("source_revisions")
    elif tampering == "empty":
        revision.clear()
    elif tampering == "wrong_batch":
        revision["preserved_poststage_fingerprints"] = {
            "another-batch": first["execution_fingerprint"]
        }
    elif tampering == "wrong_fingerprint":
        revision["preserved_poststage_fingerprints"][first["batch_id"]] = "unauthorized-revision"
    elif tampering == "not_mapping":
        revision["preserved_poststage_fingerprints"] = [first["batch_id"]]
    else:
        first["execution_fingerprint"] = "unauthorized-revision"
    campaign.atomic_json(tmp_path / "poststages.json", manifest)
    execute = Mock(wraps=campaign.execute_batch)
    monkeypatch.setattr(campaign, "execute_batch", execute)

    with pytest.raises(ValueError, match="execution fingerprint"):
        run(tmp_path, resume=True)

    execute.assert_not_called()
    forbidden.assert_not_called()
    assert all(path.read_bytes() == original for path, original in snapshots.items())


@pytest.mark.parametrize("status", ["planned", "started", "returned", "interrupted", "blocked"])
def test_nonterminal_batch_cannot_reuse_preserved_revision(tmp_path, monkeypatch, status):
    manifest, _, snapshots, forbidden = revised_campaign(tmp_path, monkeypatch)
    batch = manifest["batches"][0]
    batch["status"] = status
    manifest["batches"] = [batch]
    manifest["progress"]["group-0"] = 0
    campaign.atomic_json(tmp_path / "poststages.json", manifest)
    monkeypatch.setattr(campaign, "reset_incomplete_batch", forbidden)
    monkeypatch.setattr(campaign, "execute_batch", forbidden)

    with pytest.raises(ValueError, match="execution fingerprint"):
        run(tmp_path, resume=True)

    forbidden.assert_not_called()
    assert all(path.read_bytes() == original for path, original in snapshots.items())


def test_new_and_reset_batches_record_current_fingerprint(tmp_path, monkeypatch):
    write_base(tmp_path)
    discovery = {"sources": {}, "source_logs": {}, "excluded": [], "requested": {}}
    group = {"id": "audit:a", "stage": "audit_main", "source_keys": ["a"], "samples": 1}
    monkeypatch.setattr(p, "discover_sources", lambda *args: discovery)
    monkeypatch.setattr(p, "make_groups", lambda *args, **kwargs: ([group], []))
    monkeypatch.setattr(p, "build_batch", lambda *args, **kwargs: object())
    monkeypatch.setattr(campaign, "report_cost", lambda root: {})
    assert (
        run(tmp_path, execute_batch_fn=lambda *args, **kwargs: {"status": "blocked"})["status"]
        == "blocked"
    )
    path = tmp_path / "poststages.json"
    manifest = campaign.read_json(path)
    assert manifest["batches"][0].pop("execution_fingerprint") == manifest["fingerprint"]
    campaign.atomic_json(path, manifest)
    reset = Mock(return_value={"reason": "offline reset fixture"})
    monkeypatch.setattr(campaign, "reset_incomplete_batch", reset)

    def execute(task, **kwargs):
        saved = campaign.read_json(path)
        assert saved["batches"][0]["execution_fingerprint"] == saved["fingerprint"]
        assert kwargs["fingerprint"] == saved["fingerprint"]
        return {"status": "complete"}

    assert run(tmp_path, resume=True, execute_batch_fn=execute)["status"] == "complete"
    assert reset.call_args.kwargs["batch_fingerprint"] == manifest["fingerprint"]
