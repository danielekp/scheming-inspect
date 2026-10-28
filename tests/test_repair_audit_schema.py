"""Offline tests for the narrow, resumable semantic-score schema migration."""

import copy
from types import SimpleNamespace as NS

import pytest

from scheming_inspect import campaign as c
from scheming_inspect.poststages import digest
from scripts import repair_audit_schema as r


@pytest.fixture
def stopped(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    source = root / "base" / "base1" / "base.eval"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"immutable base result")
    hashes = {r.JUDGE: "old judge", "scheming_inspect/tasks.py": "unchanged"}
    cells = [{"experiment": "oversight", "n": 2}]
    base = {
        "status": "blocked",
        "config": {"model": c.MODEL, "max_samples": 8},
        "cells": cells,
        "source_hashes": hashes,
        "total_samples": 2,
        "batches": [
            {
                "batch_id": "base1",
                "cell_index": 0,
                "status": "complete",
                "sample_ids": ["s1", "s2"],
                "log_paths": [str(source)],
            }
        ],
        "completed_base_logs": [str(source)],
        "completed_base_log_records": [
            {
                "path": str(source),
                "cell_index": 0,
                "cell": cells[0],
                "batch_id": "base1",
                "expected_sample_ids": ["s1", "s2"],
                "log_sha256": r.sha(source),
            }
        ],
        "poststage_result": {"status": "blocked"},
    }
    base["fingerprint"] = c.fingerprint(
        {key: base[key] for key in ("config", "cells", "source_hashes")}
    )
    spec = {
        "config": {"model": c.MODEL, "max_samples": 8},
        "base_fingerprint": base["fingerprint"],
        "source_hashes": hashes,
        "discovery": {"source_logs": {str(source): r.sha(source)}},
        "groups": [{"id": "g1", "source_keys": ["s1", "s2"]}],
        "eligibility": [],
    }
    post = {
        "spec": spec,
        "fingerprint": digest(spec),
        "progress": {},
        "status": "blocked",
        "blocked_batch": "old-audit",
        "batches": [
            {
                "batch_id": "old-audit",
                "stage": "audit_main",
                "status": "blocked",
                "sample_ids": ["audit:s1", "audit:s2"],
                "selections": [{"group_id": "g1", "start": 0, "stop": 2}],
            }
        ],
    }
    folder = root / "poststages" / "audit_main" / "old-audit"
    folder.mkdir(parents=True)
    partial = folder / "partial.eval"
    partial.write_bytes(b"partial audit with one successful case")
    post["batches"][0]["log_paths"] = [str(partial)]
    journal = {
        "batch_id": "old-audit",
        "stage": "audit_main",
        "status": "blocked",
        "model": c.MODEL,
        "fingerprint": post["fingerprint"],
        "sample_ids": post["batches"][0]["sample_ids"],
        "model_roles": {"grader": c.MODEL},
        "max_samples": 8,
        "log_paths": [str(partial)],
        "log_fingerprints": {str(partial): r.sha(partial)},
    }
    for path, value in [
        (root / "manifest.json", base),
        (root / "poststages.json", post),
        (folder / "batch.json", journal),
    ]:
        c.atomic_json(path, value)
    log = NS(
        status="error",
        eval=NS(
            model=c.MODEL,
            metadata={
                "campaign_batch_id": "old-audit",
                "campaign_fingerprint": post["fingerprint"],
            },
        ),
        samples=[NS(id="audit:s1", epoch=1, metadata={}, error=None, limit=None)],
    )
    return NS(
        root=root,
        base=base,
        post=post,
        folder=folder,
        partial=partial,
        source=source,
        hashes={**hashes, r.JUDGE: "new judge"},
        read=lambda path: log,
        log=log,
    )


def test_dry_run_does_not_change_manifests_or_archive(stopped):
    s = stopped
    before = [(s.root / name).read_bytes() for name in ("manifest.json", "poststages.json")]
    result = r.repair(s.root, hashes=s.hashes, read_log_fn=s.read)
    assert result["status"] == "dry_run" and result["samples_to_rerun"] == 2
    assert [(s.root / name).read_bytes() for name in ("manifest.json", "poststages.json")] == before
    assert s.partial.exists() and not (s.root / "audit-schema-repair").exists()


def test_apply_preserves_base_and_retries_whole_batch_with_old_fingerprint(stopped, monkeypatch):
    s = stopped
    calls = []
    real_reset = c.reset_incomplete_batch

    def reset(*args, **kwargs):
        calls.append(kwargs["batch_fingerprint"])
        return real_reset(*args, **kwargs)

    monkeypatch.setattr(c, "reset_incomplete_batch", reset)
    result = r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read)
    assert calls == [s.post["fingerprint"]]
    base, post = c.read_json(s.root / "manifest.json"), c.read_json(s.root / "poststages.json")
    for key in ("config", "cells", "batches", "completed_base_logs", "completed_base_log_records"):
        assert base[key] == s.base[key]
    for key in ("groups", "eligibility", "discovery", "config"):
        assert post["spec"][key] == s.post["spec"][key]
    batch = post["batches"][0]
    for key in ("batch_id", "sample_ids", "selections"):
        assert batch[key] == s.post["batches"][0][key]
    assert batch["status"] == "planned" and batch["log_paths"] == []
    assert batch["restarts"][0]["discarded_sample_status_counts"] == {"success": 1}
    assert not s.folder.exists() and s.source.read_bytes() == b"immutable base result"
    assert post["fingerprint"] == digest(post["spec"])
    assert post["spec"]["base_fingerprint"] == base["fingerprint"]
    assert base["source_revisions"][0]["changed_files"] == [r.JUDGE]
    assert r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read) == result
    assert calls == [s.post["fingerprint"]]  # no second archival or execution


@pytest.mark.parametrize(
    "issue",
    [
        "unrelated_source",
        "base_incomplete",
        "progress",
        "settled_post",
        "base_fingerprint",
        "post_fingerprint",
        "source_log",
        "attempt_log",
        "full_coverage",
        "running",
        "added_file",
    ],
)
def test_refuses_unsafe_or_out_of_scope_repair(stopped, issue):
    s = stopped
    base, post = copy.deepcopy(s.base), copy.deepcopy(s.post)
    if issue == "unrelated_source":
        s.hashes["scheming_inspect/tasks.py"] = "changed"
    elif issue == "base_incomplete":
        base["batches"][0]["status"] = "blocked"
    elif issue == "progress":
        post["progress"]["g1"] = 1
    elif issue == "settled_post":
        post["batches"][0]["status"] = "complete"
    elif issue == "base_fingerprint":
        base["config"]["model"] = "different model"
    elif issue == "post_fingerprint":
        post["spec"]["groups"][0]["source_keys"] = ["other"]
    elif issue == "source_log":
        s.source.write_bytes(b"changed")
    elif issue == "attempt_log":
        s.partial.write_bytes(b"changed")
    elif issue == "full_coverage":
        s.log.samples.append(NS(id="audit:s2", epoch=1, metadata={}, error=None, limit=None))
    elif issue == "running":
        base["status"] = "running"
    elif issue == "added_file":
        s.hashes["scheming_inspect/extra.py"] = "new file"
    c.atomic_json(s.root / "manifest.json", base)
    c.atomic_json(s.root / "poststages.json", post)
    with pytest.raises((c.CampaignError, ValueError)):
        r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read)
    assert s.folder.exists() and not (s.root / "audit-schema-repair").exists()


def test_resumes_after_only_base_manifest_was_written(stopped, monkeypatch):
    s = stopped
    real_write = c.atomic_json
    failed = False

    def interrupted_write(path, value):
        nonlocal failed
        if path == s.root / "poststages.json" and not failed:
            failed = True
            raise OSError("simulated interruption between manifests")
        return real_write(path, value)

    monkeypatch.setattr(c, "atomic_json", interrupted_write)
    with pytest.raises(OSError):
        r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read)
    assert c.read_json(s.root / "manifest.json")["source_hashes"] == s.hashes
    assert c.read_json(s.root / "poststages.json") == s.post
    assert r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read)["status"] == "complete"
    assert len(list((s.root.parent / "campaign-superseded").iterdir())) == 1


def test_resumes_after_attempt_rename_before_plan_update(stopped, monkeypatch):
    s = stopped
    real_reset = c.reset_incomplete_batch

    def interrupted_reset(*args, **kwargs):
        real_reset(*args, **kwargs)
        raise OSError("simulated interruption after rename")

    monkeypatch.setattr(c, "reset_incomplete_batch", interrupted_reset)
    with pytest.raises(OSError):
        r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read)
    assert not s.folder.exists()
    assert r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read)["status"] == "complete"
    assert len(list((s.root.parent / "campaign-superseded").iterdir())) == 1


def test_active_campaign_lock_prevents_repair(stopped):
    s = stopped
    with c.exclusive_lock(s.root / ".campaign.lock"):
        with pytest.raises(c.CampaignError, match="Another process holds"):
            r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read)
    assert s.folder.exists()


def test_resumes_interrupted_snapshot_before_any_campaign_mutation(stopped, monkeypatch):
    s = stopped
    replace = r.os.replace
    failed = False

    def interrupted_replace(source, destination):
        nonlocal failed
        if destination.name == "manifest-before.json" and not failed:
            failed = True
            raise OSError("interrupted snapshot rename")
        return replace(source, destination)

    monkeypatch.setattr(r.os, "replace", interrupted_replace)
    with pytest.raises(OSError):
        r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read)
    assert c.read_json(s.root / "manifest.json") == s.base and s.partial.exists()
    assert r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read)["status"] == "complete"


@pytest.mark.parametrize("issue", ["manifest", "sources"])
def test_existing_plan_rejects_unrecorded_changes(stopped, issue):
    s = stopped
    r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read)
    if issue == "manifest":
        base = c.read_json(s.root / "manifest.json")
        base["config"]["max_samples"] = 99
        c.atomic_json(s.root / "manifest.json", base)
    else:
        s.hashes[r.JUDGE] = "another revision"
    with pytest.raises(c.CampaignError):
        r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read)
