"""Judge deadline repair preserves prior revisions and a partially completed group."""

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import test_repair_followup_responses as legacy

from scheming_inspect import campaign as c
from scheming_inspect import poststages as p
from scripts import repair_followup_responses as shared
from scripts import repair_judge_timeout as r

JUDGE = "scheming_inspect/judge.py"
prior_stopped = legacy.stopped


@pytest.fixture
def stopped(prior_stopped):
    s = prior_stopped
    old_archive = s.root / "followup-response-repair"
    old_archive.mkdir()
    s.folder.rename(old_archive / "prior-attempt")
    (old_archive / "plan.json").write_text('{"status": "complete", "earlier_repair": true}')
    base, post = copy.deepcopy(s.base), copy.deepcopy(s.post)
    base["source_hashes"][JUDGE] = "judge before deadline"
    base["fingerprint"] = c.fingerprint(
        {key: base[key] for key in ("config", "cells", "source_hashes")}
    )
    group = {
        "id": "fixed:c:disable",
        "stage": "fixed",
        "source_keys": ["c"],
        "action": "disable",
        "variants": ["plain", "explain"],
        "repeats": 20,
        "samples": 40,
    }
    post["spec"].update(
        base_fingerprint=base["fingerprint"],
        source_hashes=base["source_hashes"],
        groups=post["spec"]["groups"][:3] + [group],
    )
    post["fingerprint"] = p.digest(post["spec"])
    post["batches"] = post["batches"][:2]
    post["progress"][group["id"]] = 32
    post["source_revisions"].append(
        {"preserved_poststage_fingerprints": {"audit_done": "earlier-audit-revision"}}
    )
    for batch_id, start, stop, status in (
        ("fixed_done", 0, 32, "complete"),
        ("failed", 32, 40, "blocked"),
    ):
        post["batches"].append(
            {
                "batch_id": batch_id,
                "stage": "fixed",
                "status": status,
                "sample_ids": p.group_sample_ids(group)[start:stop],
                "selections": [{"group_id": group["id"], "start": start, "stop": stop}],
                "sample_status_counts": {"limited": 1, "success": 31}
                if status == "complete"
                else {"success": 3, "error": 1},
            }
        )
    files = {s.root / "base.eval": (s.root / "base.eval").read_bytes()}
    files.update({path: path.read_bytes() for path in old_archive.rglob("*") if path.is_file()})
    fingerprints = {}
    for batch in post["batches"]:
        fingerprint = (
            "earlier-audit-revision" if batch["batch_id"] == "audit_done" else post["fingerprint"]
        )
        batch["execution_fingerprint"] = fingerprint
        folder = s.root / "poststages" / batch["stage"] / batch["batch_id"]
        folder.mkdir(parents=True, exist_ok=True)
        log_path = folder / f"{batch['batch_id']}.eval"
        if not log_path.exists():
            log_path.write_bytes(f"Original {batch['batch_id']} outcomes".encode())
        batch["log_paths"] = [str(log_path)]
        journal = {
            "batch_id": batch["batch_id"],
            "stage": batch["stage"],
            "status": batch["status"],
            "model": c.MODEL,
            "fingerprint": fingerprint,
            "sample_ids": batch["sample_ids"],
            "model_roles": {"grader": c.MODEL}
            | ({"interrogator": c.MODEL} if batch["stage"] == "interrogation" else {}),
            "max_samples": 8,
            "log_paths": batch["log_paths"],
            "log_fingerprints": {str(log_path): shared.sha(log_path)},
        }
        c.atomic_json(folder / "batch.json", journal)
        if batch["status"] == "complete":
            fingerprints[batch["batch_id"]] = fingerprint
            files.update({path: path.read_bytes() for path in (log_path, folder / "batch.json")})
        else:
            (folder / "worker-note.txt").write_text("Keep the whole original attempt.")
            s.logs[log_path.name] = NS(
                status="error",
                eval=NS(
                    model=c.MODEL,
                    metadata={
                        "campaign_batch_id": batch["batch_id"],
                        "campaign_fingerprint": fingerprint,
                    },
                ),
                samples=[
                    NS(
                        id=value,
                        epoch=1,
                        metadata={},
                        error="timeout" if i == 3 else None,
                        limit=None,
                    )
                    for i, value in enumerate(batch["sample_ids"][:4])
                ],
            )
    c.atomic_json(s.root / "manifest.json", base)
    c.atomic_json(s.root / "poststages.json", post)
    s.base, s.post, s.files = base, post, files
    s.folder = s.root / "poststages" / "fixed" / "failed"
    s.attempt = {path.name: path.read_bytes() for path in s.folder.iterdir() if path.is_file()}
    s.hashes = base["source_hashes"] | {JUDGE: "judge with hard deadline"}
    s.fingerprints = fingerprints
    return s


def test_fixed_repair_preserves_two_revisions_and_retries_whole_later_slice(stopped, monkeypatch):
    s = stopped
    manifests = {
        name: (s.root / name).read_bytes() for name in ("manifest.json", "poststages.json")
    }
    dry = r.repair(s.root, hashes=s.hashes, read_log_fn=s.read)
    assert dry["poststage_samples_preserved"] == 36
    assert dry["completed_poststage_batches_preserved"] == 3
    assert dry["samples_to_rerun"] == 8
    assert dry["changed_files"] == [JUDGE]
    assert all((s.root / name).read_bytes() == value for name, value in manifests.items())
    assert not (s.root / "judge-timeout-repair").exists()

    result = r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read)
    base, post = c.read_json(s.root / "manifest.json"), c.read_json(s.root / "poststages.json")
    assert post["progress"] == s.post["progress"]
    for key in ("config", "cells", "batches", "completed_base_logs", "completed_base_log_records"):
        assert base[key] == s.base[key]
    for key in ("config", "groups", "discovery", "eligibility"):
        assert post["spec"][key] == s.post["spec"][key]
    assert post["source_revisions"][:-1] == s.post["source_revisions"]
    assert base["source_revisions"][:-1] == s.base["source_revisions"]
    assert post["batches"][:-1] == s.post["batches"][:-1]
    assert p.digest(post["spec"]) == post["fingerprint"]
    assert post["spec"]["base_fingerprint"] == base["fingerprint"]
    revision = post["source_revisions"][-1]
    assert revision["changed_files"] == [JUDGE]
    assert revision["preserved_poststage_fingerprints"] == s.fingerprints
    assert len(set(s.fingerprints.values())) == 2
    assert all(path.read_bytes() == value for path, value in s.files.items())

    def forbidden(*args, **kwargs):
        pytest.fail("A committed batch must never make model calls")

    monkeypatch.setattr(c, "_role_models", forbidden)
    for batch in post["batches"][:-1]:
        folder = s.root / "poststages" / batch["stage"] / batch["batch_id"]
        journal = c.read_json(folder / "batch.json")
        assert (
            c.execute_batch(
                None,
                batch_id=batch["batch_id"],
                log_dir=folder,
                stage=batch["stage"],
                model=c.MODEL,
                fingerprint=p.execution_fingerprint(post, batch),
                expected_sample_ids=batch["sample_ids"],
                model_roles=journal["model_roles"],
                max_samples=8,
                resume=True,
                eval_fn=forbidden,
            )
            == journal
        )
    retry = post["batches"][-1]
    for key in ("batch_id", "stage", "sample_ids", "selections"):
        assert retry[key] == s.post["batches"][-1][key]
    assert retry["selections"][0] == {"group_id": "fixed:c:disable", "start": 32, "stop": 40}
    assert retry["status"] == "planned" and retry["execution_fingerprint"] == post["fingerprint"]
    assert retry["log_paths"] == [] and retry["sample_status_counts"] == {}
    restart = retry["restarts"][0]
    assert restart["discarded_sample_status_counts"] == {"success": 3, "error": 1}
    archived = Path(restart["archive_dir"])
    assert all((archived / name).read_bytes() == value for name, value in s.attempt.items())
    assert not s.folder.exists()
    assert archived.parent == s.root.parent / "campaign-superseded"
    assert c.read_json(Path(result["archive"]) / "plan.json")["repair_policy"] == r.POLICY.as_dict()
    assert r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read) == result
    assert len(list(archived.parent.iterdir())) == 1


@pytest.mark.parametrize(
    "issue",
    [
        "other_source",
        "missing_fix",
        "added_source",
        "removed_source",
        "source_log",
        "stage",
        "offset",
    ],
)
def test_judge_policy_refuses_unrelated_changes_or_wrong_stage(stopped, issue):
    s = stopped
    post = copy.deepcopy(s.post)
    if issue == "other_source":
        s.hashes["scheming_inspect/followups.py"] = "unrelated change"
    elif issue == "missing_fix":
        s.hashes[JUDGE] = s.base["source_hashes"][JUDGE]
    elif issue == "added_source":
        s.hashes["new.py"] = "new"
    elif issue == "removed_source":
        s.hashes.pop("scheming_inspect/data.py")
    elif issue == "source_log":
        (s.root / "base.eval").write_bytes(b"changed original source")
    elif issue == "stage":
        post["batches"][-1]["stage"] = "interrogation"
    else:
        post["batches"][-1]["selections"][0]["start"] = 0
    c.atomic_json(s.root / "poststages.json", post)
    with pytest.raises((c.CampaignError, ValueError)):
        r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read)
    assert s.folder.exists() and not (s.root / "judge-timeout-repair").exists()


@pytest.mark.parametrize("field", ["reason", "stage", "changed_files", "archive_name", "missing"])
def test_prepared_timeout_policy_cannot_be_changed_or_omitted(stopped, field):
    s = stopped
    result = r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read)
    plan_path = Path(result["archive"]) / "plan.json"
    plan = c.read_json(plan_path)
    if field == "missing":
        plan.pop("repair_policy")
    else:
        plan["repair_policy"][field] = ["unrelated.py"] if field == "changed_files" else "different"
    c.atomic_json(plan_path, plan)
    with pytest.raises(c.CampaignError, match="policy"):
        r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read)


def test_different_requested_policy_and_sources_cannot_reuse_plan(stopped):
    s = stopped
    r.repair(s.root, apply=True, hashes=s.hashes, read_log_fn=s.read)
    with pytest.raises(c.CampaignError, match="policy"):
        shared.repair(
            s.root,
            apply=True,
            hashes=s.hashes,
            read_log_fn=s.read,
            policy=replace(r.POLICY, reason="different repair reason"),
        )
    with pytest.raises(c.CampaignError, match="sources"):
        r.repair(s.root, apply=True, hashes=s.hashes | {JUDGE: "another edit"}, read_log_fn=s.read)


def test_only_default_policy_accepts_legacy_plan_without_policy():
    shared.verify_policy({}, shared.DEFAULT_POLICY)
    with pytest.raises(c.CampaignError, match="policy"):
        shared.verify_policy({}, r.POLICY)


@pytest.mark.parametrize("interruption", ["after_archive", "between_manifests"])
def test_timeout_repair_recovers_transaction_without_losing_offsets(
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
    post = c.read_json(s.root / "poststages.json")
    assert post["progress"] == s.post["progress"]
    assert post["batches"][-1]["selections"] == s.post["batches"][-1]["selections"]
    assert all(path.read_bytes() == value for path, value in s.files.items())
    assert len(list((s.root.parent / "campaign-superseded").iterdir())) == 1
