"""Record the requested switch to uncapped execution without changing saved results."""

import argparse
import copy
from pathlib import Path

from scheming_inspect import campaign as c
from scheming_inspect.poststages import canonical_log_records, verify_record_hash

ALLOWED_FILES = {
    "scheming_inspect/budget.py",
    "scheming_inspect/run.py",
    "scheming_inspect/campaign.py",
    "scheming_inspect/poststages.py",
    "scheming_inspect/recovery.py",
    "scheming_inspect/spending.py",
}
POLICY = {
    "spending_limit_usd": None,
    "stage_spending_limits": {},
    "billing": "optional_key_usage_report",
    "provider_price_caps": False,
}


def migrate(directory: Path, *, hashes=None) -> dict:
    directory = directory.resolve()
    with (
        c.exclusive_lock(directory / ".campaign.lock"),
        c.exclusive_lock(directory / ".billing-retry.lock"),
    ):
        path = directory / "manifest.json"
        manifest = c.read_json(path)
        current = c.source_hashes() if hashes is None else hashes
        if manifest.get("execution_policy") == POLICY and manifest["source_hashes"] == current:
            return manifest
        if manifest["status"] not in {"planned", "pending_budget", "base_complete"}:
            raise c.CampaignError("Stop execution before changing the runner")
        if (directory / "poststages.json").exists():
            raise c.CampaignError("This migration expects post-stages not to have started")
        changed = {
            name
            for name in current.keys() | manifest["source_hashes"].keys()
            if current.get(name) != manifest["source_hashes"].get(name)
        }
        if not changed <= ALLOWED_FILES:
            raise c.CampaignError(
                "Migration cannot change prompts, data, scorers, or generation settings"
            )
        for batch in manifest["batches"]:
            if batch["status"] in c.COMPLETE_BATCH_STATES:
                continue
            if batch["status"] not in {"planned", "pending_budget"} or batch.get("recovery"):
                raise c.CampaignError("Unfinished execution requires separate recovery")
            folder = directory / "base" / batch["batch_id"]
            if (folder / "batch.json").exists() or list(folder.glob("*.eval")):
                raise c.CampaignError("Unstarted batch unexpectedly has execution records")
        records = canonical_log_records(manifest)
        for record in records:
            verify_record_hash(record)
        old_identity = {k: manifest[k] for k in ("config", "cells", "source_hashes")}
        if c.fingerprint(old_identity) != manifest["fingerprint"]:
            raise c.CampaignError("Saved campaign fingerprint is inconsistent")
        budget = directory / "budget.json"
        if budget.exists():
            ledger = c.read_json(budget)
            if any(
                b["status"] not in {"settled", "closed_reserved"}
                for b in ledger["batches"].values()
            ):
                raise c.CampaignError("An old reservation may represent unfinished execution")
        timestamp = c.utc_now()
        archive = directory / "runner-revisions" / timestamp.replace(":", "-")
        archive.mkdir(parents=True, exist_ok=False)
        (archive / "manifest-before.json").write_bytes(path.read_bytes())
        if budget.exists():
            (archive / "budget-before.json").write_bytes(budget.read_bytes())
        updated = copy.deepcopy(manifest)
        new_key = c.fingerprint({**old_identity, "source_hashes": current})
        updated.setdefault("source_revisions", []).append(
            {
                "at": timestamp,
                "reason": "User requested removal of all budget caps and billing gates",
                "old_fingerprint": manifest["fingerprint"],
                "new_fingerprint": new_key,
                "old_source_hashes": manifest["source_hashes"],
                "new_source_hashes": current,
                "changed_files": sorted(changed),
                "archive": str(archive),
            }
        )
        for batch in updated["batches"]:
            if batch["status"] == "pending_budget":
                batch["status"] = "planned"
                batch.pop("reservation", None)
        updated.update(
            source_hashes=current,
            fingerprint=new_key,
            execution_policy=POLICY,
            updated_at=timestamp,
        )
        if updated["status"] == "pending_budget":
            updated["status"] = "planned"
        updated.pop("blocker", None)
        updated.pop("provider_policy", None)
        c.atomic_json(path, updated)
        return updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", required=True, type=Path)
    args = parser.parse_args()
    result = migrate(args.campaign_dir)
    print(f"Uncapped runner recorded; {result['total_samples']} original cases preserved.")
