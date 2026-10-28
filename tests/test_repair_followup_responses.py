"""Offline recovery of one failed interrogation batch after committed post-stages."""

import copy
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from scheming_inspect import campaign as c
from scheming_inspect import poststages as p
from scripts import repair_followup_responses as r


@pytest.fixture
def stopped(tmp_path):
    root = tmp_path / "campaign"
    root.mkdir()
    source = root / "base.eval"
    source.write_bytes(b"immutable original source")
    hashes = {name: "old " + name for name in r.CHANGED_FILES}
    hashes["scheming_inspect/data.py"] = "unchanged data"
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
        "source_revisions": [{"reason": "earlier repair retained"}],
        "poststage_result": {"status": "blocked"},
    }
    base["fingerprint"] = c.fingerprint(
        {key: base[key] for key in ("config", "cells", "source_hashes")}
    )
    groups = [
        {"id": f"audit:{name}", "stage": "audit_main", "source_keys": [name], "samples": 1}
        for name in ("a", "b")
    ] + [
        {
            "id": f"interrogation:{name}",
            "stage": "interrogation",
            "source_keys": [name, "control"],
            "samples": 2,
            "truth_order": ["guilty", "innocent"],
        }
        for name in ("c", "d", "e")
    ]
    spec = {
        "config": {"model": c.MODEL, "max_samples": 8, "max_tokens": 8000, "max_rounds": 7},
        "base_fingerprint": base["fingerprint"],
        "source_hashes": hashes,
        "discovery": {"source_logs": {str(source): r.sha(source)}},
        "groups": groups,
        "eligibility": [],
    }
    post = {
        "spec": spec,
        "fingerprint": p.digest(spec),
        "progress": {},
        "status": "blocked",
        "blocked_batch": "failed",
        "batches": [],
        "source_revisions": [{"reason": "earlier repair retained"}],
    }
    original_files = {source: source.read_bytes()}
    logs = {}
    for batch_id, selected in (
        ("audit_done", groups[:2]),
        ("interrogation_done", groups[2:3]),
        ("failed", groups[3:]),
    ):
        complete = batch_id != "failed"
        ids = [sample_id for group in selected for sample_id in p.group_sample_ids(group)]
        selections = [
            {"group_id": group["id"], "start": 0, "stop": group["samples"]} for group in selected
        ]
        if complete:
            post["progress"].update({group["id"]: group["samples"] for group in selected})
        batch = {
            "batch_id": batch_id,
            "stage": selected[0]["stage"],
            "status": "complete" if complete else "blocked",
            "sample_ids": ids,
            "selections": selections,
            "sample_status_counts": {"limited": 1, "success": 1}
            if complete
            else {"error": 2, "success": 1},
        }
        folder = root / "poststages" / batch["stage"] / batch_id
        folder.mkdir(parents=True)
        log_path = folder / f"{batch_id}.eval"
        log_path.write_bytes(f"immutable {batch_id} outcomes including limits or errors".encode())
        batch["log_paths"] = [str(log_path)]
        roles = {"grader": c.MODEL} | (
            {"interrogator": c.MODEL} if batch["stage"] == "interrogation" else {}
        )
        journal = {
            "batch_id": batch_id,
            "stage": batch["stage"],
            "status": batch["status"],
            "model": c.MODEL,
            "fingerprint": post["fingerprint"],
            "sample_ids": ids,
            "model_roles": roles,
            "max_samples": 8,
            "log_paths": [str(log_path)],
            "log_fingerprints": {str(log_path): r.sha(log_path)},
        }
        c.atomic_json(folder / "batch.json", journal)
        if complete:
            original_files[log_path] = log_path.read_bytes()
            original_files[folder / "batch.json"] = (folder / "batch.json").read_bytes()
        logs[log_path.name] = NS(
            status="success" if complete else "error",
            eval=NS(
                model=c.MODEL,
                metadata={
                    "campaign_batch_id": batch_id,
                    "campaign_fingerprint": post["fingerprint"],
                },
            ),
            samples=[
                NS(
                    id=value,
                    epoch=1,
                    metadata={},
                    error="invalid question" if not complete and i else None,
                    limit=None,
                )
                for i, value in enumerate(ids if complete else ids[:-1])
            ],
        )
        post["batches"].append(batch)
    c.atomic_json(root / "manifest.json", base)
    c.atomic_json(root / "poststages.json", post)
    folder = root / "poststages" / "interrogation" / "failed"
    return NS(
        root=root,
        base=base,
        post=post,
        folder=folder,
        files=original_files,
        hashes={
            key: "new " + key if key in r.CHANGED_FILES else value for key, value in hashes.items()
        },
        read=lambda path: logs[Path(path).name],
        logs=logs,
    )


def test_dry_run_preserves_state(stopped):
    s = stopped
    originals = {
        name: (s.root / name).read_bytes() for name in ("manifest.json", "poststages.json")
    }
    result = r.repair(s.root, hashes=s.hashes, read_log_fn=s.read)
    assert result["poststage_samples_preserved"] == 4
    assert result["completed_poststage_batches_preserved"] == 2
    assert result["samples_to_rerun"] == 4
    assert all((s.root / name).read_bytes() == data for name, data in originals.items())
    assert not (s.root / "followup-response-repair").exists()


def test_apply_preserves_progress_and_resumes_completed_historical_batches(stopped):
    s = stopped
    result = r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read)
    base, post = c.read_json(s.root / "manifest.json"), c.read_json(s.root / "poststages.json")
    assert post["progress"] == s.post["progress"]
    assert post["source_revisions"][:-1] == s.post["source_revisions"]
    assert base["source_revisions"][:-1] == s.base["source_revisions"]
    for key in ("config", "cells", "batches", "completed_base_logs", "completed_base_log_records"):
        assert base[key] == s.base[key]
    for key in ("config", "groups", "discovery", "eligibility"):
        assert post["spec"][key] == s.post["spec"][key]
    assert p.digest(post["spec"]) == post["fingerprint"]
    assert post["spec"]["base_fingerprint"] == base["fingerprint"]
    assert all(path.read_bytes() == data for path, data in s.files.items())
    for old, preserved in zip(s.post["batches"][:-1], post["batches"][:-1]):
        assert preserved == old | {"execution_fingerprint": s.post["fingerprint"]}
        journal_path = s.root / "poststages" / old["stage"] / old["batch_id"] / "batch.json"
        journal = c.read_json(journal_path)
        outcome = c.execute_batch(
            None,
            batch_id=old["batch_id"],
            log_dir=journal_path.parent,
            stage=old["stage"],
            model=c.MODEL,
            fingerprint=p.execution_fingerprint(post, preserved),
            expected_sample_ids=old["sample_ids"],
            model_roles=journal["model_roles"],
            max_samples=8,
            resume=True,
            eval_fn=lambda *args, **kwargs: pytest.fail("Completed sample was rerun"),
        )
        assert outcome == journal
    retry = post["batches"][-1]
    for key in ("sample_ids", "selections", "batch_id"):
        assert retry[key] == s.post["batches"][-1][key]
    assert retry["status"] == "planned" and retry["execution_fingerprint"] == post["fingerprint"]
    assert retry["log_paths"] == [] and retry["sample_status_counts"] == {}
    archived = Path(retry["restarts"][0]["archive_dir"])
    assert archived.parent == s.root.parent / "campaign-superseded"
    assert not s.folder.exists()
    assert retry["restarts"][0]["discarded_sample_status_counts"] == {"success": 1, "error": 2}
    assert r.sha(archived / "batch.json") == r.sha(Path(result["archive"]) / "journal-before.json")
    assert r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read) == result
    assert len(list(archived.parent.iterdir())) == 1


@pytest.mark.parametrize(
    "issue",
    [
        "data_change",
        "added_source",
        "missing_fix",
        "progress",
        "selection",
        "completed_log",
        "completed_journal",
        "base_incomplete",
        "full_coverage",
        "running",
        "multiple_pending",
    ],
)
def test_refuses_unsafe_migration(stopped, issue):
    s = stopped
    base, post = copy.deepcopy(s.base), copy.deepcopy(s.post)
    if issue == "data_change":
        s.hashes["scheming_inspect/data.py"] = "changed data"
    elif issue == "added_source":
        s.hashes["scheming_inspect/new.py"] = "unexpected"
    elif issue == "missing_fix":
        s.hashes["scheming_inspect/followups.py"] = s.base["source_hashes"][
            "scheming_inspect/followups.py"
        ]
    elif issue == "progress":
        post["progress"]["interrogation:d"] = 2
    elif issue == "selection":
        post["batches"][-1]["sample_ids"].reverse()
    elif issue == "completed_log":
        Path(post["batches"][0]["log_paths"][0]).write_bytes(b"modified committed log")
    elif issue == "completed_journal":
        journal_path = s.root / "poststages" / "audit_main" / "audit_done" / "batch.json"
        journal = c.read_json(journal_path)
        journal["fingerprint"] = "unrecorded historical revision"
        c.atomic_json(journal_path, journal)
    elif issue == "base_incomplete":
        base["batches"][0]["status"] = "blocked"
    elif issue == "full_coverage":
        s.logs["failed.eval"].samples.append(
            NS(
                id=post["batches"][-1]["sample_ids"][-1],
                epoch=1,
                metadata={},
                error="failure is still terminal",
                limit=None,
            )
        )
    elif issue == "running":
        base["status"] = "running"
    elif issue == "multiple_pending":
        post["batches"][0]["status"] = "blocked"
    c.atomic_json(s.root / "manifest.json", base)
    c.atomic_json(s.root / "poststages.json", post)
    with pytest.raises((c.CampaignError, ValueError)):
        r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read)
    assert s.folder.exists() and not (s.root / "followup-response-repair").exists()


@pytest.mark.parametrize("interruption", ["after_archive", "between_manifests"])
def test_resumes_interrupted_transaction_without_resetting_committed_progress(
    stopped, monkeypatch, interruption
):
    s = stopped
    if interruption == "after_archive":
        reset = c.reset_incomplete_batch

        def interrupted(*args, **kwargs):
            reset(*args, **kwargs)
            raise OSError("interrupted after rename")

        monkeypatch.setattr(c, "reset_incomplete_batch", interrupted)
    else:
        write = c.atomic_json
        failed = False

        def interrupted(path, value):
            nonlocal failed
            if path == s.root / "poststages.json" and not failed:
                failed = True
                raise OSError("interrupted between manifests")
            return write(path, value)

        monkeypatch.setattr(c, "atomic_json", interrupted)
    with pytest.raises(OSError):
        r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read)
    assert r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read)["status"] == "complete"
    assert c.read_json(s.root / "poststages.json")["progress"] == s.post["progress"]
    assert all(path.read_bytes() == data for path, data in s.files.items())
    assert len(list((s.root.parent / "campaign-superseded").iterdir())) == 1


def test_lock_and_prepared_evidence_are_not_bypassed(stopped, monkeypatch):
    s = stopped
    with c.exclusive_lock(s.root / ".campaign.lock"):
        with pytest.raises(c.CampaignError, match="Another process holds"):
            r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read)
    reset = c.reset_incomplete_batch

    def interrupted(*args, **kwargs):
        reset(*args, **kwargs)
        raise OSError("interrupted after rename")

    monkeypatch.setattr(c, "reset_incomplete_batch", interrupted)
    with pytest.raises(OSError):
        r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read)
    Path(s.post["batches"][0]["log_paths"][0]).write_bytes(b"changed after plan")
    with pytest.raises(c.CampaignError, match="completed post-stage"):
        r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read)
