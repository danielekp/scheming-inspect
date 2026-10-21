"""Explicit recovery of a stopped campaign, preserving original attempts and cost bounds.

This module never starts a model. Preparation archives the old manifest, closes
the uncertain attempt at its full reserved ceiling, and records an exact retry
allowlist. The ordinary campaign runner dispatches that allowlist on --resume.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

from inspect_ai.dataset import MemoryDataset

from . import campaign as c
from .budget import CampaignLedger

# These repairs affect scoring, accounting, and analysis. Task/data/model
# definitions are deliberately not eligible for a recovery-time revision.
REPAIR_FILES = {
    "scheming_inspect/engine.py",
    "scheming_inspect/budget.py",
    "scheming_inspect/campaign.py",
    "scheming_inspect/recovery.py",
    "scheming_inspect/poststages.py",
    "scheming_inspect/analyze.py",
}


def sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_status(sample) -> str:
    behavior = (sample.metadata or {}).get("behavior", {})
    if sample.error:
        return "error"
    if getattr(sample, "limit", None):
        return "limited"
    if behavior.get("completed") == 0:
        return "incomplete"
    if behavior.get("schema_valid") == 0:
        return "invalid_schema"
    return "success"


def recovery_spec(
    manifest: dict, batch: dict, journal: dict, *, read_log_fn=None, task_factory=c.scheming
) -> dict:
    """Retain every non-error outcome, including invalid answers and limits."""
    paths = journal.get("log_paths", [])
    if len(paths) != 1:
        raise c.CampaignError("Recovery requires exactly one original terminal log")
    path = str(Path(paths[0]).resolve())
    log = (read_log_fn or c._read_log)(Path(path))
    meta = log.eval.metadata or {}
    if (
        log.status != "error"
        or str(log.eval.model) != manifest["config"]["model"]
        or meta.get("campaign_batch_id") != batch["batch_id"]
        or meta.get("campaign_fingerprint") != manifest["fingerprint"]
    ):
        raise c.CampaignError("Original failed log identity does not match the frozen campaign")
    expected = batch["sample_ids"]
    samples = {str(s.id): s for s in log.samples or []}
    if len(samples) != len(log.samples or []) or not set(samples) <= set(expected):
        raise c.CampaignError("Original failed log has duplicate or unexpected sample IDs")
    config = manifest["config"]
    limits = {name: config[name] for name in ("max_steps", "max_tokens", "token_limit")}
    original_task = task_factory(**manifest["cells"][batch["cell_index"]], **limits)
    fixtures = {str(s.id): s for s in original_task.dataset}
    if not set(expected) <= set(fixtures):
        raise c.CampaignError("Recovery fixtures do not cover the original batch")
    for key, sample in samples.items():
        if sample.epoch != 1 or (sample.metadata or {}).get("fixture_sha256") != (
            fixtures[key].metadata or {}
        ).get("fixture_sha256"):
            raise c.CampaignError("Original fixture/epoch changed; recovery is not comparable")
    retained = [
        key
        for key in expected
        if key in samples
        and not samples[key].error
        and not getattr(samples[key], "invalidation", None)
    ]
    retry = [key for key in expected if key not in retained]
    if not retry:
        raise c.CampaignError("No failed or unrecorded samples require recovery")
    return {
        "version": 1,
        "original_batch_id": batch["batch_id"],
        "attempt_id": batch["batch_id"] + "-recovery-01",
        "original_fingerprint": manifest["fingerprint"],
        "original_log": path,
        "original_log_sha256": sha256(path),
        "original_journal_sha256": c.fingerprint(journal),
        "retained_sample_ids": retained,
        "retry_sample_ids": retry,
        "retained_status_counts": dict(Counter(sample_status(samples[key]) for key in retained)),
        "excluded_original_attempts": [
            {
                "sample_id": key,
                "reason": "unrecorded"
                if key not in samples
                else "invalidated"
                if getattr(samples[key], "invalidation", None)
                else "sample_error",
                "error": getattr(samples[key].error, "message", None) if key in samples else None,
            }
            for key in retry
        ],
        "selection_policy": "Keep all non-error outcomes; retry only errors or unrecorded cases.",
    }


def retained_record(manifest: dict, batch: dict) -> dict:
    recovery = batch["recovery"]
    return {
        "path": recovery["original_log"],
        "cell": manifest["cells"][batch["cell_index"]],
        "cell_index": batch["cell_index"],
        "batch_id": batch["batch_id"],
        "expected_sample_ids": recovery["retained_sample_ids"],
        "log_sha256": recovery["original_log_sha256"],
        "recovery_retained": True,
    }


def prepare_recovery(
    directory: Path,
    *,
    reason: str,
    process_evidence: dict,
    hashes: dict | None = None,
    read_log_fn=None,
    task_factory=c.scheming,
) -> dict:
    """Prepare once under the campaign lock; no model or billing requests."""
    if not reason.strip() or process_evidence.get("process_stopped") is not True:
        raise c.CampaignError("Recovery requires a reason and verified stopped-process evidence")
    directory = directory.resolve()
    with c.exclusive_lock(directory / ".campaign.lock"):
        manifest = c.read_json(directory / "manifest.json")
        current_hashes = hashes if hashes is not None else c.source_hashes()
        if manifest.get("source_revisions") and manifest["source_hashes"] == current_hashes:
            if any(batch.get("recovery") for batch in manifest["batches"]):
                return manifest
        changed = {
            key
            for key in set(current_hashes) | set(manifest["source_hashes"])
            if current_hashes.get(key) != manifest["source_hashes"].get(key)
        }
        if not changed or not changed <= REPAIR_FILES:
            raise c.CampaignError("Recovery permits only recorded scorer/accounting repairs")
        config = manifest["config"]
        if config["retries"] != 0 or config["epochs"] != 1 or config["cache_prompt"] is not False:
            raise c.CampaignError("Original execution bounds do not support reserved-cost recovery")
        blocked = [batch for batch in manifest["batches"] if batch["status"] == "blocked"]
        if manifest["status"] != "blocked" or len(blocked) != 1:
            raise c.CampaignError("Expected one stopped, blocked base batch")
        batch = blocked[0]
        journal = c.read_json(directory / "base" / batch["batch_id"] / "batch.json")
        if journal["status"] != "blocked":
            raise c.CampaignError("The physical batch journal is not blocked")
        ledger = CampaignLedger(
            directory / "budget.json",
            limit_usd=config["budget_usd"],
            starting_spend_usd=config["starting_spend_usd"],
            stage_limits={"base": config["stage_budget_usd"]},
        )
        record = c._ledger_batch(ledger, batch["batch_id"])
        if not record or record["status"] not in {"blocked", "closed_reserved"}:
            raise c.CampaignError("Original attempt has no blocked/retained cost reservation")
        c._verify_settled_logs(record, journal["log_paths"])
        for old in manifest["batches"]:
            if old["status"] == "settled":
                c._verify_settled_logs(c._ledger_batch(ledger, old["batch_id"]), old["log_paths"])
        spec = recovery_spec(
            manifest, batch, journal, read_log_fn=read_log_fn, task_factory=task_factory
        )
        recovery_dir = directory / "recoveries" / spec["attempt_id"]
        plan_path = recovery_dir / "plan.json"
        if plan_path.exists():
            plan = c.read_json(plan_path)
            if (
                plan["recovery"] != spec
                or plan["new_source_hashes"] != current_hashes
                or plan["old_fingerprint"] != manifest["fingerprint"]
            ):
                raise c.CampaignError("An existing recovery plan differs; refusing to replace it")
        else:
            plan = {
                "status": "prepared",
                "created_at": c.utc_now(),
                "reason": reason,
                "old_fingerprint": manifest["fingerprint"],
                "recovery": spec,
                "new_source_hashes": current_hashes,
                "process_evidence": process_evidence,
            }
            c.atomic_json(recovery_dir / "manifest-before.json", manifest)
            c.atomic_json(plan_path, plan)
        if record["status"] == "blocked":
            ledger.close_with_reservation(
                batch["batch_id"],
                journal["log_paths"],
                reason=reason,
                process_stopped=True,
                evidence={
                    "bounded_execution_confirmed": True,
                    "process": process_evidence,
                    "recovery_plan": str(plan_path),
                    "config": config,
                    "provider": journal["provider"],
                },
            )
        new_key = c.fingerprint(
            {"config": config, "cells": manifest["cells"], "source_hashes": current_hashes}
        )
        manifest.setdefault("source_revisions", []).append(
            {
                "at": plan["created_at"],
                "reason": reason,
                "old_fingerprint": manifest["fingerprint"],
                "new_fingerprint": new_key,
                "old_source_hashes": manifest["source_hashes"],
                "new_source_hashes": current_hashes,
                "changed_files": sorted(changed),
                "recovery_plan": str(plan_path),
            }
        )
        batch.update(recovery=spec, status="recovery_ready")
        manifest.update(
            source_hashes=current_hashes,
            fingerprint=new_key,
            status="recovery_ready",
            updated_at=c.utc_now(),
        )
        manifest.pop("blocker", None)
        c.refresh_base_records(manifest)
        c.atomic_json(directory / "manifest.json", manifest)
        plan.update(status="applied", new_fingerprint=new_key)
        c.atomic_json(plan_path, plan)
        return manifest


def run_recovery_batch(
    directory: Path,
    manifest: dict,
    batch: dict,
    ledger,
    *,
    resume: bool,
    bound_fn,
    task_factory=c.scheming,
    eval_fn=None,
    read_log_fn=None,
) -> dict:
    """Dispatch only the frozen recovery IDs and combine canonical log selections."""
    spec, config = batch["recovery"], manifest["config"]
    retained, retry = spec["retained_sample_ids"], spec["retry_sample_ids"]
    if set(retained) & set(retry) or Counter(retained + retry) != Counter(batch["sample_ids"]):
        raise c.CampaignError("Recovery selection changed or duplicates original work")
    original = c._ledger_batch(ledger, spec["original_batch_id"])
    if not original or original["status"] != "closed_reserved":
        raise c.CampaignError("The original uncertain cost reservation is not retained")
    c._verify_settled_logs(original, [spec["original_log"]])
    if sha256(spec["original_log"]) != spec["original_log_sha256"]:
        raise c.CampaignError("Original recovery evidence changed")
    log_dir = directory / "recovery-attempts" / spec["attempt_id"]
    journal_path = log_dir / "batch.json"
    journal = c.read_json(journal_path) if journal_path.exists() else None
    reserved = c._ledger_batch(ledger, spec["attempt_id"])
    task = None
    if reserved or (journal and journal["status"] != "planned"):
        upper, provider = (journal or reserved)["upper_usd"], (journal or {})["provider"]
    else:
        bound = bound_fn(config["model"], len(retry), config["token_limit"], config["max_tokens"])
        upper, provider = bound["conservative_max_usd"], bound["provider"]
        limits = {name: config[name] for name in ("max_steps", "max_tokens", "token_limit")}
        task = task_factory(**manifest["cells"][batch["cell_index"]], **limits)
        dataset = task.dataset
        selected = [sample for sample in dataset if str(sample.id) in set(retry)]
        if [str(sample.id) for sample in selected] != retry:
            raise c.CampaignError("Recovery generator does not match the frozen ID order")
        task.dataset = MemoryDataset(
            selected,
            name=getattr(dataset, "name", None),
            location=getattr(dataset, "location", None),
            shuffled=False,
        )
    result = c.execute_batch(
        task,
        batch_id=spec["attempt_id"],
        log_dir=log_dir,
        ledger=ledger,
        upper_usd=upper,
        stage="base",
        model=config["model"],
        provider=provider,
        fingerprint=manifest["fingerprint"],
        expected_sample_ids=retry,
        max_samples=config["max_samples"],
        resume=resume,
        eval_fn=eval_fn,
        read_log_fn=read_log_fn,
    )
    if result["status"] != "settled":
        return result
    counts = Counter(spec["retained_status_counts"])
    counts.update(result["sample_status_counts"])
    canonical = [retained_record(manifest, batch)] if retained else []
    canonical.extend(
        {
            "path": path,
            "cell": manifest["cells"][batch["cell_index"]],
            "cell_index": batch["cell_index"],
            "batch_id": spec["attempt_id"],
            "expected_sample_ids": retry,
            "log_sha256": sha256(path),
            "recovery_retained": False,
        }
        for path in result["log_paths"]
    )
    return {
        "status": "settled",
        "sample_status_counts": dict(counts),
        "log_paths": [record["path"] for record in canonical],
        "canonical_log_records": canonical,
        "recovery_attempt": spec["attempt_id"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--process-evidence", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare_recovery(
        args.campaign_dir, reason=args.reason, process_evidence=c.read_json(args.process_evidence)
    )
    c.emit(
        "recovery_prepared",
        fingerprint=manifest["fingerprint"],
        recoveries=[batch["recovery"] for batch in manifest["batches"] if batch.get("recovery")],
    )


if __name__ == "__main__":
    main()
