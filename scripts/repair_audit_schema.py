"""Dry-run or apply the one-time judge schema repair to a stopped campaign.

Stop the campaign and any restart wrapper first. This command takes the campaign
lock, makes no model/billing requests, and preserves every completed base result.
Only judge.py may have changed. The entire first incomplete audit batch is
archived and retried; its successful subset is not cherry-picked.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

from scheming_inspect import campaign as c
from scheming_inspect.poststages import canonical_log_records, digest, verify_record_hash

JUDGE = "scheming_inspect/judge.py"
REASON = "Repair semantic-audit score dictionaries without changing judge rubrics or base results"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(path, content):
    """Persist exact original bytes before the repair plan authorizes mutations."""
    if path.exists():
        if path.read_bytes() != content:
            raise c.CampaignError("An existing repair snapshot differs")
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def frozen_logs(base, post):
    records = canonical_log_records(base, require_complete=True)
    if sum(len(record["expected_sample_ids"]) for record in records) != base["total_samples"]:
        raise c.CampaignError("Canonical coverage differs from the frozen case count")
    known = {}
    for record in records:
        if not record.get("log_sha256"):
            raise c.CampaignError("Every canonical source requires a frozen log hash")
        known[record["path"]] = verify_record_hash(record)
    for path, expected in post["spec"]["discovery"]["source_logs"].items():
        if known.get(path) != expected:
            raise c.CampaignError("Post-stage source hashes differ from canonical base logs")


def initial_state(root, current, read_log_fn):
    base, post = c.read_json(root / "manifest.json"), c.read_json(root / "poststages.json")
    if base["status"] not in {"blocked", "base_complete"} or post["status"] != "blocked":
        raise c.CampaignError("Repair requires a stopped, blocked post-stage campaign")
    if not base["batches"] or any(
        batch["status"] not in c.COMPLETE_BATCH_STATES for batch in base["batches"]
    ):
        raise c.CampaignError("All base batches must already be complete")
    identity = {key: base[key] for key in ("config", "cells", "source_hashes")}
    if c.fingerprint(identity) != base["fingerprint"]:
        raise c.CampaignError("Base fingerprint is inconsistent")
    spec = post["spec"]
    if (
        digest(spec) != post["fingerprint"]
        or spec["base_fingerprint"] != base["fingerprint"]
        or spec["source_hashes"] != base["source_hashes"]
        or spec["config"]["model"] != base["config"]["model"]
    ):
        raise c.CampaignError("Post-stage fingerprint/source identity is inconsistent")
    changed = {
        key
        for key in current.keys() | base["source_hashes"].keys()
        if current.get(key) != base["source_hashes"].get(key)
    }
    if changed != {JUDGE} or current.keys() != base["source_hashes"].keys():
        raise c.CampaignError("Exactly judge.py may change; no added/removed source files")
    if any(value != 0 for value in post["progress"].values()):
        raise c.CampaignError("Committed post-stage progress cannot be regraded by this repair")
    if len(post["batches"]) != 1:
        raise c.CampaignError("This repair handles only the first incomplete audit batch")
    batch = post["batches"][0]
    if batch["status"] != "blocked" or batch["stage"] != "audit_main":
        raise c.CampaignError("A completed or non-audit batch cannot be reset by this repair")
    frozen_logs(base, post)
    folder = root / "poststages" / batch["stage"] / batch["batch_id"]
    journal = c.read_json(folder / "batch.json")
    expected = {
        "batch_id": batch["batch_id"],
        "stage": batch["stage"],
        "model": base["config"]["model"],
        "fingerprint": post["fingerprint"],
        "sample_ids": batch["sample_ids"],
        "model_roles": {"grader": base["config"]["model"]},
        "max_samples": spec["config"]["max_samples"],
    }
    if journal["status"] != "blocked" or any(journal.get(k) != v for k, v in expected.items()):
        raise c.CampaignError("Blocked audit journal identity differs from its frozen batch")
    c._verify_settled_logs(journal, journal.get("log_paths", []))
    coverage = c.inspect_batch_logs(
        folder,
        expected_sample_ids=batch["sample_ids"],
        model=expected["model"],
        batch_id=batch["batch_id"],
        batch_fingerprint=post["fingerprint"],
        read_log_fn=read_log_fn,
    )
    if coverage["complete"]:
        raise c.CampaignError("The audit batch has full terminal coverage; do not rerun it")
    return base, post, folder, journal


def read_snapshots(archive, plan, label):
    result = {}
    for name, expected in plan[label].items():
        path = archive / f"{name}-{label}.json"
        if sha(path) != expected:
            raise c.CampaignError("A repair snapshot was changed")
        result[name] = c.read_json(path)
    return result


def verify_attempt(folder, plan):
    if sha(folder / "batch.json") != plan["journal_sha256"]:
        raise c.CampaignError("The failed attempt journal changed during repair")
    paths = {p.name: sha(p) for p in folder.glob("*.eval")}
    if paths != plan["attempt_logs"]:
        raise c.CampaignError("The failed attempt logs changed during repair")


def transaction_state(root, archive, plan):
    before = read_snapshots(archive, plan, "before")
    after = read_snapshots(archive, plan, "after") if plan.get("after") else None
    for name in before:
        actual = c.read_json(root / f"{name}.json")
        if actual != before[name] and (after is None or actual != after[name]):
            raise c.CampaignError("Campaign changed outside the recorded repair transaction")
    frozen_logs(before["manifest"], before["poststages"])
    return before, after


def finish(root, archive, plan, read_log_fn):
    before, after = transaction_state(root, archive, plan)
    old_post = before["poststages"]
    batch = old_post["batches"][0]
    folder = root / "poststages" / batch["stage"] / batch["batch_id"]
    archive_root = root.parent / (root.name + "-superseded")
    if folder.exists():
        verify_attempt(folder, plan)
        restart = c.reset_incomplete_batch(
            folder,
            archive_root=archive_root,
            expected_sample_ids=batch["sample_ids"],
            model=before["manifest"]["config"]["model"],
            batch_id=batch["batch_id"],
            batch_fingerprint=old_post["fingerprint"],
            read_log_fn=read_log_fn,
        )
        if restart is None:
            raise c.CampaignError("The recorded incomplete attempt was not archived")
    else:
        # Handles interruption after reset's atomic rename but before saving its result.
        candidates = []
        for candidate in archive_root.glob(batch["batch_id"] + "-*"):
            if str(candidate) in plan["previous_archives"]:
                continue
            if (candidate / "batch.json").exists() and sha(candidate / "batch.json") == plan[
                "journal_sha256"
            ]:
                candidates.append(candidate)
        if len(candidates) != 1:
            raise c.CampaignError("Cannot uniquely recover the archived failed attempt")
        verify_attempt(candidates[0], plan)
        restart = c.read_json(candidates[0] / "restart-record.json")
        if (
            restart.get("archive_dir") != str(candidates[0])
            or restart.get("batch_id") != batch["batch_id"]
            or restart.get("sample_ids") != batch["sample_ids"]
        ):
            raise c.CampaignError("Archived restart identity differs from the repair plan")
    if after is None:
        base, post = copy.deepcopy(before["manifest"]), copy.deepcopy(old_post)
        base["source_hashes"] = plan["new_source_hashes"]
        base["fingerprint"] = c.fingerprint(
            {key: base[key] for key in ("config", "cells", "source_hashes")}
        )
        post["spec"]["base_fingerprint"] = base["fingerprint"]
        post["spec"]["source_hashes"] = plan["new_source_hashes"]
        post["fingerprint"] = digest(post["spec"])
        revised = post["batches"][0]
        revised.setdefault("restarts", []).append(restart)
        revised.update(status="planned", log_paths=[], sample_status_counts={})
        revised.pop("error_type", None)
        post.update(status="planned")
        post.pop("blocked_batch", None)
        revision = {
            "at": plan["at"],
            "reason": REASON,
            "changed_files": [JUDGE],
            "old_fingerprint": before["manifest"]["fingerprint"],
            "new_fingerprint": base["fingerprint"],
            "old_poststage_fingerprint": old_post["fingerprint"],
            "new_poststage_fingerprint": post["fingerprint"],
            "old_source_hashes": before["manifest"]["source_hashes"],
            "new_source_hashes": plan["new_source_hashes"],
            "archive": str(archive),
            "superseded_attempt": restart["archive_dir"],
        }
        base.setdefault("source_revisions", []).append(revision)
        post.setdefault("source_revisions", []).append(revision)
        base.update(status="base_complete", updated_at=plan["at"])
        base.pop("poststage_result", None)  # preserved in the before snapshot
        after = {"manifest": base, "poststages": post}
        for name, value in after.items():
            c.atomic_json(archive / f"{name}-after.json", value)
        plan.update(
            after={name: sha(archive / f"{name}-after.json") for name in after},
            status="ready",
            superseded_attempt=restart["archive_dir"],
        )
        c.atomic_json(archive / "plan.json", plan)
    for name, value in after.items():
        if c.read_json(root / f"{name}.json") != value:
            c.atomic_json(root / f"{name}.json", value)
    plan["status"] = "complete"
    c.atomic_json(archive / "plan.json", plan)
    return {"status": "complete", "archive": str(archive), "batch_id": batch["batch_id"]}


def repair(directory, *, apply=False, hashes=None, read_log_fn=None):
    root = Path(directory).resolve()
    archive = root / "audit-schema-repair"
    with c.exclusive_lock(root / ".campaign.lock"):
        current = c.source_hashes() if hashes is None else hashes
        plan_path = archive / "plan.json"
        if plan_path.exists():
            plan = c.read_json(plan_path)
            if plan["new_source_hashes"] != current:
                raise c.CampaignError("Current sources differ from the prepared repair")
            if apply:
                return finish(root, archive, plan, read_log_fn)
            transaction_state(root, archive, plan)
            return {"status": "dry_run", "repair_status": plan["status"], "archive": str(archive)}
        base, post, folder, journal = initial_state(root, current, read_log_fn)
        result = {
            "status": "dry_run",
            "changed_files": [JUDGE],
            "base_samples_preserved": base["total_samples"],
            "batch_id": post["batches"][0]["batch_id"],
            "samples_to_rerun": len(post["batches"][0]["sample_ids"]),
        }
        if not apply:
            return result
        archive.mkdir(exist_ok=True)
        names = ("manifest", "poststages")
        for name in names:
            snapshot(archive / f"{name}-before.json", (root / f"{name}.json").read_bytes())
        snapshot(archive / "journal-before.json", (folder / "batch.json").read_bytes())
        archive_root = root.parent / (root.name + "-superseded")
        plan = {
            "version": 1,
            "at": c.utc_now(),
            "status": "prepared",
            "new_source_hashes": current,
            "before": {name: sha(archive / f"{name}-before.json") for name in names},
            "journal_sha256": sha(folder / "batch.json"),
            "attempt_logs": {
                Path(path).name: value for path, value in journal["log_fingerprints"].items()
            },
            "previous_archives": [
                str(path) for path in archive_root.glob(result["batch_id"] + "-*")
            ],
        }
        c.atomic_json(plan_path, plan)
        return finish(root, archive, plan, read_log_fn)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Apply after stopping all workers")
    args = parser.parse_args(argv)
    print(json.dumps(repair(args.campaign_dir, apply=args.apply), indent=2))


if __name__ == "__main__":
    main()
