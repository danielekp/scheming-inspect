"""Archive and apply an accounting-only source revision; never dispatch or reconcile.

Run only after independently confirming that the campaign worker has stopped.
Both nonblocking campaign and billing-wrapper locks must be available. The only
live file replaced is manifest.json; original logs, journals, and ledger bytes
are retained unchanged. Repeating an identical interrupted migration is safe.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from scheming_inspect import campaign as c

ALLOWED_FILES = frozenset(
    {
        "scheming_inspect/budget.py",
        "scheming_inspect/campaign.py",
        "scheming_inspect/recovery.py",
        "scheming_inspect/poststages.py",
    }
)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def identity(manifest: dict, hashes: dict | None = None) -> dict:
    return {
        "config": manifest["config"],
        "cells": manifest["cells"],
        "source_hashes": manifest["source_hashes"] if hashes is None else hashes,
    }


def archive_bytes(path: Path, data: bytes) -> None:
    """Write exact snapshot bytes once; never replace a conflicting archive."""
    if path.exists():
        if path.read_bytes() != data:
            raise c.CampaignError(f"Archived snapshot differs: {path.name}")
        return
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def verify_archive(directory: Path, plan: dict) -> None:
    core = {key: value for key, value in plan.items() if key not in {"created_at", "migration_id"}}
    if c.fingerprint(core) != plan.get("migration_id"):
        raise c.CampaignError("Accounting revision plan checksum is inconsistent.")
    if set(plan["snapshot_sha256"]) != {
        "manifest-before.json",
        "budget-before.json",
        "journal-before.json",
    }:
        raise c.CampaignError("Accounting revision snapshot inventory is inconsistent.")
    for name, expected in plan["snapshot_sha256"].items():
        path = directory / name
        if not path.is_file() or sha(path.read_bytes()) != expected:
            raise c.CampaignError("Accounting revision archive is missing or changed.")


def validate_blocked(directory: Path, manifest: dict, ledger: dict, *, read_log_fn=None):
    """Read-only validation against the physical attempt's original fingerprint."""
    config = manifest["config"]
    if (
        config.get("retries") != 0
        or config.get("epochs") != 1
        or config.get("cache_prompt") is not False
    ):
        raise c.CampaignError("Frozen execution retry/cache bounds are inconsistent.")
    try:
        total = Decimal(str(config["budget_usd"]))
        stage = Decimal(str(config["stage_budget_usd"]))
        if (
            not total.is_finite()
            or not stage.is_finite()
            or not 0 < stage <= total <= 50
            or total != Decimal(str(ledger["limit_usd"]))
            or stage != Decimal(str(ledger["stage_limits"]["base"]))
            or Decimal(str(config["starting_spend_usd"]))
            != Decimal(str(ledger["starting_spend_usd"]))
        ):
            raise ValueError("caps differ")
    except (KeyError, ValueError, ArithmeticError) as exc:
        raise c.CampaignError("Existing ledger and frozen campaign budget caps differ.") from exc
    active = [
        (key, value)
        for key, value in ledger["batches"].items()
        if value["status"] in {"reserved", "blocked"}
    ]
    if manifest["status"] != "blocked" or len(active) != 1 or active[0][1]["status"] != "blocked":
        raise c.CampaignError("Expected exactly one blocked, stopped physical batch.")
    batch_id, reservation = active[0]
    matches = [
        batch
        for batch in manifest["batches"]
        if batch["batch_id"] == batch_id or batch.get("recovery", {}).get("attempt_id") == batch_id
    ]
    if len(matches) != 1 or matches[0]["status"] != "blocked":
        raise c.CampaignError("Blocked ledger attempt has no matching blocked logical batch.")
    batch = matches[0]
    expected_ids = (
        batch["sample_ids"]
        if batch["batch_id"] == batch_id
        else batch["recovery"]["retry_sample_ids"]
    )
    metadata = reservation["metadata"]
    log_dir = Path(metadata["log_dir"]).resolve()
    if not log_dir.is_relative_to(directory):
        raise c.CampaignError("Blocked journal must be inside the campaign directory.")
    journal_path = log_dir / "batch.json"
    journal_bytes = journal_path.read_bytes()
    journal = json.loads(journal_bytes)
    keys = ("batch_id", "stage", "model", "fingerprint", "sample_ids", "max_samples", "model_roles")
    if (
        journal.get("status") != "blocked"
        or journal.get("complete") is not True
        or journal.get("errors") != []
        or any(journal.get(key) != metadata.get(key) for key in keys)
        or journal["batch_id"] != batch_id
        or journal["stage"] != "base"
        or journal["sample_ids"] != expected_ids
        or journal["model"] != config["model"]
        or journal["max_samples"] != config["max_samples"]
        or Decimal(str(journal["upper_usd"])) != Decimal(str(reservation["upper_usd"]))
    ):
        raise c.CampaignError(
            "Blocked journal identity or frozen bounds do not match its reservation."
        )
    known_fingerprints = {manifest["fingerprint"]}
    for revision in manifest.get("source_revisions", []):
        for side in ("old", "new"):
            historical = c.fingerprint(identity(manifest, revision[f"{side}_source_hashes"]))
            if historical != revision[f"{side}_fingerprint"]:
                raise c.CampaignError(
                    "Historical fingerprint does not preserve the frozen configuration."
                )
            known_fingerprints.add(historical)
    if journal["fingerprint"] not in known_fingerprints:
        raise c.CampaignError(
            "Blocked journal fingerprint is absent from the recorded source history."
        )
    if batch.get("reservation", {}).get("provider") != journal.get("provider"):
        raise c.CampaignError("Blocked batch provider caps differ from its frozen reservation.")
    coverage = c.inspect_batch_logs(
        log_dir,
        expected_sample_ids=expected_ids,
        model=config["model"],
        batch_id=batch_id,
        batch_fingerprint=journal["fingerprint"],
        read_log_fn=read_log_fn,
    )
    if not coverage["complete"] or coverage["log_paths"] != journal["log_paths"]:
        raise c.CampaignError("Blocked batch logs do not have complete original sample coverage.")
    fingerprints = reservation.get("log_fingerprints", {})
    if not fingerprints or set(fingerprints) != set(coverage["log_paths"]):
        raise c.CampaignError("Blocked ledger is missing the original log fingerprints.")
    for name, expected_hash in fingerprints.items():
        if sha(Path(name).read_bytes()) != expected_hash:
            raise c.CampaignError("A blocked batch log changed after reconciliation.")
    return batch_id, journal_path, journal_bytes, fingerprints


def migrate(directory: Path | str, *, reason: str, read_log_fn=None) -> dict:
    if not reason.strip():
        raise c.CampaignError("An explicit accounting migration reason is required.")
    directory = Path(directory).expanduser().resolve()
    if not (directory / "manifest.json").is_file():
        raise c.CampaignError("Campaign manifest does not exist.")
    with (
        c.exclusive_lock(directory / ".campaign.lock"),
        c.exclusive_lock(directory / ".billing-retry.lock"),
    ):
        manifest_path, budget_path = directory / "manifest.json", directory / "budget.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        current_hashes = c.source_hashes()
        if c.fingerprint(identity(manifest)) != manifest["fingerprint"]:
            raise c.CampaignError("Frozen manifest configuration/fingerprint is inconsistent.")
        changed = sorted(
            key
            for key in set(current_hashes) | set(manifest["source_hashes"])
            if current_hashes.get(key) != manifest["source_hashes"].get(key)
        )
        if not changed:
            revisions = manifest.get("source_revisions", [])
            revision = revisions[-1] if revisions else {}
            if (
                revision.get("accounting_only") is not True
                or revision.get("reason") != reason
                or revision.get("new_fingerprint") != manifest["fingerprint"]
                or revision.get("new_source_hashes") != current_hashes
            ):
                raise c.CampaignError("No unapplied accounting-only source changes were found.")
            archive = Path(revision["accounting_revision"])
            if not archive.resolve().is_relative_to(directory / "accounting-revisions"):
                raise c.CampaignError("Accounting revision archive is outside the campaign.")
            plan = c.read_json(archive / "plan.json")
            verify_archive(archive, plan)
            if any(
                plan.get(key) != revision.get(key)
                for key in (
                    "migration_id",
                    "reason",
                    "old_fingerprint",
                    "new_fingerprint",
                    "old_source_hashes",
                    "new_source_hashes",
                    "changed_files",
                    "accounting_only",
                )
            ):
                raise c.CampaignError("Applied source revision does not match its archived plan.")
            return {
                "status": "already_applied",
                "archive": str(archive),
                "fingerprint": manifest["fingerprint"],
                "changed_files": revision["changed_files"],
            }
        if not set(changed) <= ALLOWED_FILES or set(current_hashes) != set(
            manifest["source_hashes"]
        ):
            raise c.CampaignError(
                "Only existing budget/campaign/recovery/poststages files may change; task/data/model changes are forbidden."
            )
        budget_bytes = budget_path.read_bytes()
        ledger = json.loads(budget_bytes)
        batch_id, journal_path, journal_bytes, log_hashes = validate_blocked(
            directory, manifest, ledger, read_log_fn=read_log_fn
        )
        new_fingerprint = c.fingerprint(identity(manifest, current_hashes))
        snapshots = {
            "manifest-before.json": manifest_bytes,
            "budget-before.json": budget_bytes,
            "journal-before.json": journal_bytes,
        }
        core = {
            "version": 1,
            "accounting_only": True,
            "reason": reason,
            "campaign_dir": str(directory),
            "batch_id": batch_id,
            "journal_path": str(journal_path),
            "old_fingerprint": manifest["fingerprint"],
            "new_fingerprint": new_fingerprint,
            "old_source_hashes": manifest["source_hashes"],
            "new_source_hashes": current_hashes,
            "changed_files": changed,
            "snapshot_sha256": {name: sha(raw) for name, raw in snapshots.items()},
            "log_sha256": log_hashes,
            "process_precondition": "Caller independently verified no active worker; both campaign and billing-wrapper locks acquired.",
        }
        migration_id = c.fingerprint(core)
        candidates = sorted((directory / "accounting-revisions").glob(f"*-{migration_id[:16]}"))
        if len(candidates) > 1:
            raise c.CampaignError("Multiple archives match the same accounting revision.")
        archive = (
            candidates[0]
            if candidates
            else directory
            / "accounting-revisions"
            / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + migration_id[:16])
        )
        plan_path = archive / "plan.json"
        if plan_path.exists():
            plan = c.read_json(plan_path)
            if (
                any(plan.get(key) != value for key, value in core.items())
                or plan.get("migration_id") != migration_id
            ):
                raise c.CampaignError("An interrupted accounting revision has different inputs.")
        else:
            plan = core | {"migration_id": migration_id, "created_at": c.utc_now()}
            c.atomic_json(plan_path, plan)
        for name, raw in snapshots.items():
            archive_bytes(archive / name, raw)
        verify_archive(archive, plan)
        # Detect concurrent edits even though compliant runner/wrapper processes are locked out.
        if (
            manifest_path.read_bytes() != manifest_bytes
            or budget_path.read_bytes() != budget_bytes
            or journal_path.read_bytes() != journal_bytes
            or c.source_hashes() != current_hashes
            or any(sha(Path(name).read_bytes()) != value for name, value in log_hashes.items())
        ):
            raise c.CampaignError(
                "Campaign records or source changed while preparing the accounting revision."
            )
        updated = copy.deepcopy(manifest)
        updated.setdefault("source_revisions", []).append(
            {
                "at": plan["created_at"],
                "reason": reason,
                "accounting_only": True,
                "migration_id": migration_id,
                "accounting_revision": str(archive),
                "old_fingerprint": manifest["fingerprint"],
                "new_fingerprint": new_fingerprint,
                "old_source_hashes": manifest["source_hashes"],
                "new_source_hashes": current_hashes,
                "changed_files": changed,
            }
        )
        updated.update(
            source_hashes=current_hashes, fingerprint=new_fingerprint, updated_at=c.utc_now()
        )
        c.atomic_json(manifest_path, updated)
        return {
            "status": "applied",
            "archive": str(archive),
            "fingerprint": new_fingerprint,
            "changed_files": changed,
            "batch_id": batch_id,
        }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    try:
        result = migrate(args.campaign_dir, reason=args.reason)
    except (c.CampaignError, OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"Accounting migration refused: {exc}") from exc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
