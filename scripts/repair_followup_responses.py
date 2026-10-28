"""Migrate a stopped interrogation failure without redoing committed post-stages.

Dry run is the default. Stop the campaign and its restart wrapper before --apply.
Only followups.py and poststages.py may change. The entire incomplete interrogation
batch is archived outside the active campaign, preserving its original bytes.
Completed journals/logs, source selection, prompts, limits, and progress stay fixed.
"""

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

from scheming_inspect import campaign as c
from scheming_inspect import poststages as p
from scripts.repair_audit_schema import (
    frozen_logs,
    read_snapshots,
    sha,
    snapshot,
    verify_attempt,
)

CHANGED_FILES = {"scheming_inspect/followups.py", "scheming_inspect/poststages.py"}
REASON = (
    "Handle malformed interrogation responses as explicit unscored outcomes; "
    "preserve completed post-stage execution fingerprints"
)


@dataclass(frozen=True)
class RepairPolicy:
    """An explicit, narrowly scoped repair; its identity is frozen in the plan."""

    changed_files: frozenset[str]
    stage: str
    archive_name: str
    reason: str

    def __post_init__(self):
        if (
            not self.changed_files
            or self.stage not in p.STAGES
            or not self.reason.strip()
            or not self.archive_name
            or self.archive_name in {".", ".."}
            or Path(self.archive_name).name != self.archive_name
        ):
            raise ValueError(
                "Repair policy requires exact sources, stage, and a local archive name"
            )

    def as_dict(self):
        return {
            "changed_files": sorted(self.changed_files),
            "stage": self.stage,
            "archive_name": self.archive_name,
            "reason": self.reason,
        }


DEFAULT_POLICY = RepairPolicy(
    changed_files=frozenset(CHANGED_FILES),
    stage="interrogation",
    archive_name="followup-response-repair",
    reason=REASON,
)


def verify_policy(plan, policy):
    # Older, already-applied response repairs predate policy serialization.
    recorded = plan.get("repair_policy", DEFAULT_POLICY.as_dict())
    if recorded != policy.as_dict():
        raise c.CampaignError("Repair policy differs from the prepared transaction")


def verify_progress(post):
    """Progress must be explained by complete frozen selections, never partial logs."""
    groups = {group["id"]: group for group in post["spec"]["groups"]}
    committed = {}
    if len(groups) != len(post["spec"]["groups"]):
        raise c.CampaignError("Duplicate frozen post-stage groups")
    for batch in post["batches"]:
        ids = []
        for selection in batch["selections"]:
            group_id = selection["group_id"]
            group = groups.get(group_id)
            start, stop = selection["start"], selection["stop"]
            if (
                group is None
                or group["stage"] != batch["stage"]
                or type(start) is not int
                or type(stop) is not int
                or start != committed.get(group_id, 0)
                or not start < stop <= group["samples"]
                or (batch["stage"] == "interrogation" and (start % 2 or stop % 2))
            ):
                raise c.CampaignError("Post-stage selections disagree with committed progress")
            ids.extend(p.group_sample_ids(group)[start:stop])
            if batch["status"] in c.COMPLETE_BATCH_STATES:
                committed[group_id] = stop
        if ids != batch["sample_ids"] or len(ids) != len(set(ids)):
            raise c.CampaignError("Frozen post-stage sample IDs differ from their selections")
    if set(post["progress"]) - set(groups) or any(
        post["progress"].get(group_id, 0) != committed.get(group_id, 0) for group_id in groups
    ):
        raise c.CampaignError("Recorded progress differs from completed post-stage batches")


def initial_state(root, current, read_log_fn, *, policy=DEFAULT_POLICY):
    base, post = c.read_json(root / "manifest.json"), c.read_json(root / "poststages.json")
    if base["status"] not in {"blocked", "base_complete"} or post["status"] != "blocked":
        raise c.CampaignError("Repair requires a stopped, blocked post-stage campaign")
    if not base["batches"] or any(
        batch["status"] not in c.COMPLETE_BATCH_STATES for batch in base["batches"]
    ):
        raise c.CampaignError("Every base batch must already be complete")
    if (
        c.fingerprint({key: base[key] for key in ("config", "cells", "source_hashes")})
        != base["fingerprint"]
    ):
        raise c.CampaignError("Base fingerprint is inconsistent")
    spec = post["spec"]
    if (
        p.digest(spec) != post["fingerprint"]
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
    if changed != policy.changed_files or current.keys() != base["source_hashes"].keys():
        raise c.CampaignError(
            "Changed source files must exactly match the repair policy: "
            + ", ".join(sorted(policy.changed_files))
        )
    pending = [
        i
        for i, batch in enumerate(post["batches"])
        if batch["status"] not in c.COMPLETE_BATCH_STATES
    ]
    if pending != [len(post["batches"]) - 1]:
        raise c.CampaignError("Repair requires one final incomplete post-stage batch")
    index = pending[0]
    batch = post["batches"][index]
    if (
        batch["status"] != "blocked"
        or batch["stage"] != policy.stage
        or post.get("blocked_batch") != batch["batch_id"]
    ):
        raise c.CampaignError(f"Only the recorded blocked {policy.stage} batch can be reset")
    frozen_logs(base, post)
    verify_progress(post)
    preserved_files, preserved_fingerprints = {}, {}
    for item in post["batches"]:
        folder = root / "poststages" / item["stage"] / item["batch_id"]
        journal_path = folder / "batch.json"
        journal = c.read_json(journal_path)
        expected = {
            "batch_id": item["batch_id"],
            "stage": item["stage"],
            "model": base["config"]["model"],
            "fingerprint": p.execution_fingerprint(post, item),
            "sample_ids": item["sample_ids"],
            "model_roles": {"grader": base["config"]["model"]}
            | (
                {"interrogator": base["config"]["model"]}
                if item["stage"] == "interrogation"
                else {}
            ),
            "max_samples": spec["config"]["max_samples"],
        }
        if (
            any(journal.get(key) != value for key, value in expected.items())
            or journal["status"] != item["status"]
            or journal.get("log_paths") != item.get("log_paths")
        ):
            raise c.CampaignError("Post-stage journal identity differs from its frozen batch")
        record = journal.get("settlement") or journal
        c._verify_settled_logs(record, journal.get("log_paths", []))
        if {str(path.resolve()) for path in folder.glob("*.eval")} != set(journal["log_paths"]):
            raise c.CampaignError("Post-stage directory contains an unrecorded log")
        if item["status"] in c.COMPLETE_BATCH_STATES:
            preserved_fingerprints[item["batch_id"]] = journal["fingerprint"]
            preserved_files[str(journal_path)] = sha(journal_path)
            preserved_files.update(record["log_fingerprints"])
    folder = root / "poststages" / batch["stage"] / batch["batch_id"]
    journal = c.read_json(folder / "batch.json")
    coverage = c.inspect_batch_logs(
        folder,
        expected_sample_ids=batch["sample_ids"],
        model=base["config"]["model"],
        batch_id=batch["batch_id"],
        batch_fingerprint=post["fingerprint"],
        read_log_fn=read_log_fn,
    )
    if coverage["complete"] or coverage["errors"] != [
        "Sample IDs/counts do not exactly match the batch"
    ]:
        raise c.CampaignError(
            "Repair requires incomplete coverage with otherwise valid log identity"
        )
    return base, post, index, folder, journal, preserved_files, preserved_fingerprints


def transaction_state(root, archive, plan):
    before = read_snapshots(archive, plan, "before")
    after = read_snapshots(archive, plan, "after") if plan.get("after") else None
    for name in before:
        actual = c.read_json(root / f"{name}.json")
        if actual != before[name] and (after is None or actual != after[name]):
            raise c.CampaignError("Campaign changed outside the recorded repair transaction")
    frozen_logs(before["manifest"], before["poststages"])
    for path, expected in plan["preserved_files"].items():
        if sha(path) != expected:
            raise c.CampaignError("A completed post-stage journal or log changed during repair")
    return before, after


def finish(root, archive, plan, read_log_fn, *, policy=DEFAULT_POLICY):
    verify_policy(plan, policy)
    before, after = transaction_state(root, archive, plan)
    old_post = before["poststages"]
    batch = old_post["batches"][plan["batch_index"]]
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
        candidates = [
            candidate
            for candidate in archive_root.glob(batch["batch_id"] + "-*")
            if str(candidate) not in plan["previous_archives"]
            and (candidate / "batch.json").exists()
            and sha(candidate / "batch.json") == plan["journal_sha256"]
        ]
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
        post["fingerprint"] = p.digest(post["spec"])
        for item in post["batches"]:
            if item["status"] in c.COMPLETE_BATCH_STATES:
                item["execution_fingerprint"] = plan["preserved_poststage_fingerprints"][
                    item["batch_id"]
                ]
        revised = post["batches"][plan["batch_index"]]
        revised.setdefault("restarts", []).append(restart)
        revised.update(
            status="planned",
            log_paths=[],
            sample_status_counts={},
            execution_fingerprint=post["fingerprint"],
        )
        revised.pop("error_type", None)
        post.update(status="planned")
        post.pop("blocked_batch", None)
        revision = {
            "at": plan["at"],
            "reason": policy.reason,
            "changed_files": sorted(policy.changed_files),
            "old_fingerprint": before["manifest"]["fingerprint"],
            "new_fingerprint": base["fingerprint"],
            "old_poststage_fingerprint": old_post["fingerprint"],
            "new_poststage_fingerprint": post["fingerprint"],
            "old_source_hashes": before["manifest"]["source_hashes"],
            "new_source_hashes": plan["new_source_hashes"],
            "preserved_poststage_fingerprints": plan["preserved_poststage_fingerprints"],
            "archive": str(archive),
            "superseded_attempt": restart["archive_dir"],
        }
        base.setdefault("source_revisions", []).append(revision)
        post.setdefault("source_revisions", []).append(revision)
        base.update(status="base_complete", updated_at=plan["at"])
        base.pop("poststage_result", None)
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


def repair(directory, *, apply=False, hashes=None, read_log_fn=None, policy=DEFAULT_POLICY):
    root = Path(directory).resolve()
    archive = root / policy.archive_name
    with c.exclusive_lock(root / ".campaign.lock"):
        current = c.source_hashes() if hashes is None else hashes
        plan_path = archive / "plan.json"
        if plan_path.exists():
            plan = c.read_json(plan_path)
            verify_policy(plan, policy)
            if plan["new_source_hashes"] != current:
                raise c.CampaignError("Current sources differ from the prepared repair")
            if apply:
                return finish(root, archive, plan, read_log_fn, policy=policy)
            transaction_state(root, archive, plan)
            return {"status": "dry_run", "repair_status": plan["status"], "archive": str(archive)}
        base, post, index, folder, journal, files, fingerprints = initial_state(
            root, current, read_log_fn, policy=policy
        )
        result = {
            "status": "dry_run",
            "changed_files": sorted(policy.changed_files),
            "base_samples_preserved": base["total_samples"],
            "poststage_samples_preserved": sum(post["progress"].values()),
            "completed_poststage_batches_preserved": len(fingerprints),
            "batch_id": post["batches"][index]["batch_id"],
            "samples_to_rerun": len(post["batches"][index]["sample_ids"]),
        }
        if not apply:
            return result
        archive.mkdir(exist_ok=True)
        for name in ("manifest", "poststages"):
            snapshot(archive / f"{name}-before.json", (root / f"{name}.json").read_bytes())
        snapshot(archive / "journal-before.json", (folder / "batch.json").read_bytes())
        archive_root = root.parent / (root.name + "-superseded")
        plan = {
            "version": 1,
            "at": c.utc_now(),
            "status": "prepared",
            "repair_policy": policy.as_dict(),
            "batch_index": index,
            "new_source_hashes": current,
            "before": {
                name: sha(archive / f"{name}-before.json") for name in ("manifest", "poststages")
            },
            "journal_sha256": sha(folder / "batch.json"),
            "attempt_logs": {
                Path(path).name: value for path, value in journal["log_fingerprints"].items()
            },
            "preserved_files": files,
            "preserved_poststage_fingerprints": fingerprints,
            "previous_archives": [
                str(path) for path in archive_root.glob(result["batch_id"] + "-*")
            ],
        }
        c.atomic_json(plan_path, plan)
        return finish(root, archive, plan, read_log_fn, policy=policy)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Apply after stopping all workers")
    args = parser.parse_args(argv)
    print(json.dumps(repair(args.campaign_dir, apply=args.apply), indent=2))


if __name__ == "__main__":
    main()
