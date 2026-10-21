"""Resumable, budget-reserved batches for the complete reconstruction.

Nothing runs without --execute. A durable reservation or started batch is never
automatically rerun: --resume only reconciles it from existing terminal logs.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.log import read_eval_log
from inspect_ai.model import GenerateConfig, get_model

from .protocol import PROFILES, plan
from .run import openrouter_bound
from .tasks import ROOT, scheming

MODEL = "openrouter/deepseek/deepseek-v4-flash"
MANIFEST_VERSION = 1


class CampaignError(RuntimeError):
    """A campaign needs explicit reconciliation or corrected configuration."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path | str, value: Any) -> None:
    """Replace a JSON record only after flushing its complete new contents."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignError(f"Another process holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def read_json(path: Path | str) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise CampaignError(f"Expected a JSON object: {path}")
    return value


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_hashes(root: Path = ROOT) -> dict[str, str]:
    files = sorted(
        {
            *root.glob("scheming_inspect/*.py"),
            *root.glob("data/*.json"),
            root / "uv.lock",
            root / "pyproject.toml",
            root / "Dockerfile",
            root / "compose.yaml",
            root / ".dockerignore",
        }
    )
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
        if path.is_file()
    }


def emit(event: str, **values: Any) -> None:
    print(json.dumps({"time": utc_now(), "event": event, **values}, allow_nan=False), flush=True)


def _ledger_batch(ledger, batch_id: str) -> dict | None:
    batches = ledger.snapshot().get("batches", {})
    if isinstance(batches, dict):
        return batches.get(batch_id)
    return next((batch for batch in batches if batch.get("batch_id") == batch_id), None)


def _read_log(path: Path):
    # Binary streams avoid filesystem-provider discovery on local .eval files.
    with path.open("rb") as stream:
        return read_eval_log(stream)


def inspect_batch_logs(
    log_dir: Path,
    *,
    expected_sample_ids: list[str],
    model: str,
    batch_id: str,
    batch_fingerprint: str,
    read_log_fn=None,
) -> dict:
    """Check complete coverage before allowing a reservation to be released."""
    reader = read_log_fn or _read_log
    paths = sorted(log_dir.glob("*.eval"))
    errors, outcomes, seen = [], [], Counter()
    if len(paths) != 1:
        errors.append(f"Expected exactly one eval log; found {len(paths)}")
    for path in paths:
        try:
            log = reader(path)
        except Exception as exc:
            errors.append(f"Unreadable log {path.name}: {type(exc).__name__}")
            continue
        if log.status not in {"success", "error", "cancelled"}:
            errors.append(f"Nonterminal log status: {log.status}")
        if str(log.eval.model) != model:
            errors.append("Log model does not match the reserved model")
        metadata = getattr(log.eval, "metadata", None) or {}
        if metadata.get("campaign_batch_id") != batch_id:
            errors.append("Log batch ID does not match its journal")
        if metadata.get("campaign_fingerprint") != batch_fingerprint:
            errors.append("Log fingerprint does not match its journal")
        for sample in log.samples or []:
            sample_id = str(sample.id)
            seen[sample_id] += 1
            if sample.epoch != 1:
                errors.append(f"Unexpected epoch for sample {sample_id}: {sample.epoch}")
            behavior = (sample.metadata or {}).get("behavior", {})
            if sample.error:
                status = "error"
            elif getattr(sample, "limit", None):
                status = "limited"
            elif behavior.get("completed") == 0:
                status = "incomplete"
            elif behavior.get("schema_valid") == 0:
                status = "invalid_schema"
            else:
                status = "success"
            outcomes.append(
                {
                    "id": sample_id,
                    "epoch": sample.epoch,
                    "status": status,
                    "log": str(path.resolve()),
                }
            )
    expected = Counter(expected_sample_ids)
    if seen != expected:
        errors.append("Sample IDs/counts do not exactly match the reserved batch")
    return {
        "complete": not errors,
        "errors": errors,
        "samples": outcomes,
        "sample_status_counts": dict(Counter(row["status"] for row in outcomes)),
        "log_paths": [str(path.resolve()) for path in paths],
    }


def _role_models(roles: dict[str, str] | None, model: str, provider: dict) -> dict:
    result = {}
    for role, name in (roles or {}).items():
        if not isinstance(name, str) or name != model:
            raise CampaignError("Campaign role models must use the same authorized model string")
        result[role] = get_model(
            name, provider=provider, config=GenerateConfig(max_retries=0, cache_prompt=False)
        )
    return result


def _verify_settled_logs(record: dict, paths: list[str]) -> None:
    recorded = record.get("log_fingerprints", {})
    if set(recorded) != set(paths):
        raise CampaignError("Settled ledger log inventory differs from the batch journal")
    for raw in paths:
        path = Path(raw)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != recorded[raw]:
            raise CampaignError("A settled batch log was removed or changed")


def execute_batch(
    task,
    *,
    batch_id: str,
    log_dir: str | Path,
    ledger,
    upper_usd: float,
    stage: str,
    model: str,
    provider: dict,
    fingerprint: str,
    expected_sample_ids: list[str] | None = None,
    model_roles: dict[str, str] | None = None,
    max_samples: int = 8,
    resume: bool = False,
    eval_fn=None,
    read_log_fn=None,
) -> dict:
    """Execute once, then reconcile; reusable by the optional post stages.

    Returns ``settled``, ``blocked``, or ``pending_budget``. The last status means
    no reservation/start occurred, so a caller may safely choose a smaller batch.
    Existing started/reserved batches are reconciliation-only, even on resume.
    Roles are role-to-model strings; each gets the same provider price caps.
    """
    from .budget import BudgetError, BudgetExceeded

    if not math.isfinite(upper_usd) or upper_usd < 0 or max_samples < 1:
        raise CampaignError("Invalid batch budget or concurrency")
    log_dir = Path(log_dir).resolve()
    ids = (
        [str(sample.id) for sample in task.dataset]
        if expected_sample_ids is None and task is not None
        else [str(value) for value in expected_sample_ids or []]
    )
    if not ids or len(set(ids)) != len(ids):
        raise CampaignError("A batch requires distinct original sample IDs")
    if task is not None and [str(sample.id) for sample in task.dataset] != ids:
        raise CampaignError("Task slice does not match the frozen sample IDs")
    roles = dict(model_roles or {})
    if any(not isinstance(name, str) or name != model for name in roles.values()):
        raise CampaignError("All campaign roles must use the same authorized model")
    identity = {
        "batch_id": batch_id,
        "stage": stage,
        "model": model,
        "fingerprint": fingerprint,
        "sample_ids": ids,
        "model_roles": roles,
        "max_samples": max_samples,
    }
    journal_path = log_dir / "batch.json"
    with exclusive_lock(log_dir / ".batch.lock"):
        previous = read_json(journal_path) if journal_path.exists() else None
        reserved = _ledger_batch(ledger, batch_id)
        if previous and any(previous.get(key) != value for key, value in identity.items()):
            raise CampaignError("Existing batch identity/configuration does not match")
        if reserved and any(
            reserved.get("metadata", {}).get(key) != value for key, value in identity.items()
        ):
            raise CampaignError("Ledger reservation identity differs from the batch")
        if (previous or reserved) and not resume:
            raise CampaignError("Batch already exists; use --resume to reconcile, never rerun")
        if previous and previous["status"] == "settled":
            if not reserved or reserved.get("status") != "settled":
                raise CampaignError("Settled journal and budget ledger disagree")
            _verify_settled_logs(reserved, previous.get("log_paths", []))
            return previous
        journal = previous or {
            **identity,
            "version": MANIFEST_VERSION,
            "created_at": utc_now(),
            "status": "planned",
            "log_dir": str(log_dir),
            "upper_usd": upper_usd,
            "provider": provider,
        }
        # A durable ledger reservation is enough to forbid redispatch, even if
        # the process died before its journal could be marked started.
        reconcile_only = reserved is not None or journal["status"] in {
            "reserved",
            "started",
            "returned",
            "interrupted",
            "blocked",
        }
        if not reconcile_only:
            if task is None:
                raise CampaignError("An unstarted batch needs its original task")
            if list(log_dir.glob("*.eval")):
                raise CampaignError("Unstarted batch directory already contains eval logs")
            # Save planned identity first. An unsuccessful budget check cannot
            # leave an ambiguous started record or charge a model call.
            atomic_json(journal_path, journal)
            try:
                ledger.reserve(
                    batch_id, upper_usd, stage=stage, metadata={**identity, "log_dir": str(log_dir)}
                )
            except BudgetExceeded as exc:
                if _ledger_batch(ledger, batch_id) is not None:
                    raise CampaignError(
                        "Reservation outcome uncertain; resume reconciliation"
                    ) from exc
                journal.update(status="planned", budget_blocker=str(exc), updated_at=utc_now())
                # No reservation and no dispatch occurred. Keep an audit note,
                # but no batch identity lockout: post stages may safely shrink
                # this not-yet-started batch to fit the remaining reservation.
                atomic_json(log_dir / "budget-denial.json", journal)
                journal_path.unlink(missing_ok=True)
                return {**journal, "status": "pending_budget"}
            except BudgetError as exc:
                raise CampaignError(
                    "Ledger refuses new work; reconcile its outstanding blocker"
                ) from exc
            journal.update(
                status="reserved", upper_usd=upper_usd, provider=provider, reserved_at=utc_now()
            )
            atomic_json(journal_path, journal)
            # The started record is durable before the first paid request.
            journal.update(status="started", started_at=utc_now())
            atomic_json(journal_path, journal)
            emit(
                "batch_started",
                batch_id=batch_id,
                stage=stage,
                samples=len(ids),
                upper_usd=upper_usd,
            )
            evaluator = eval_fn or inspect_eval
            # DeepSeek's available capped providers do not advertise this
            # optional parameter. Omit it; the solver executes tools serially.
            task.config.parallel_tool_calls = None
            try:
                evaluator(
                    task,
                    model=model,
                    model_args={"provider": provider},
                    model_roles=_role_models(roles, model, provider),
                    log_dir=str(log_dir),
                    max_samples=max_samples,
                    max_tasks=1,
                    epochs=1,
                    max_retries=0,
                    retry_on_error=0,
                    cache_prompt=False,
                    log_model_api=True,
                    ctl_server=False,
                    metadata={
                        "campaign_batch_id": batch_id,
                        "campaign_fingerprint": fingerprint,
                        "campaign_stage": stage,
                        "spending_reservation_usd": upper_usd,
                    },
                )
            except BaseException as exc:
                journal.update(
                    status="interrupted",
                    interruption=type(exc).__name__,
                    updated_at=utc_now(),
                    log_paths=[str(path.resolve()) for path in sorted(log_dir.glob("*.eval"))],
                )
                atomic_json(journal_path, journal)
                # The reservation remains in the ledger. Never launch another
                # batch or guess whether a cancelled request was billed.
                raise
            journal.update(status="returned", returned_at=utc_now())
            atomic_json(journal_path, journal)
        coverage = inspect_batch_logs(
            log_dir,
            expected_sample_ids=ids,
            model=model,
            batch_id=batch_id,
            batch_fingerprint=fingerprint,
            read_log_fn=read_log_fn,
        )
        journal.update(coverage, updated_at=utc_now())
        atomic_json(journal_path, journal)
        try:
            if reserved and reserved.get("status") == "settled":
                # A crash after ledger settlement but before journal replacement
                # must neither repay nor lose the completed batch.
                if not coverage["complete"]:
                    raise CampaignError("Settled ledger has incomplete/mismatching batch logs")
                _verify_settled_logs(reserved, coverage["log_paths"])
                settlement = reserved
            else:
                settlement = ledger.settle(
                    batch_id,
                    coverage["log_paths"],
                    complete=coverage["complete"],
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                )
        except BaseException as exc:
            journal.update(
                status="blocked", settlement_error=type(exc).__name__, updated_at=utc_now()
            )
            atomic_json(journal_path, journal)
            raise
        journal.update(status=settlement["status"], settlement=settlement, updated_at=utc_now())
        atomic_json(journal_path, journal)
        emit(
            "batch_" + journal["status"],
            batch_id=batch_id,
            stage=stage,
            samples=len(ids),
            sample_status_counts=coverage["sample_status_counts"],
            settlement=settlement,
        )
        return journal


def make_schedule(
    cells: list[dict],
    batch_size: int,
    *,
    campaign_fingerprint: str,
    task_factory=scheming,
    limits: dict | None = None,
) -> list[dict]:
    """First batch is one integrated preflight; all original IDs are retained."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise CampaignError("batch_size must be a positive integer")
    schedule = []
    for cell_index, cell in enumerate(cells):
        original = task_factory(**cell, **(limits or {}))
        ids = [str(sample.id) for sample in original.dataset]
        if len(ids) != cell["n"] or len(set(ids)) != len(ids):
            raise CampaignError("Original cell dataset has missing/duplicate sample IDs")
        offset = 0
        while offset < len(ids):
            size = 1 if not schedule else batch_size
            end = min(offset + size, len(ids))
            schedule.append(
                {
                    "batch_id": f"{campaign_fingerprint[:12]}-base-{cell_index:03d}-{offset:05d}",
                    "cell_index": cell_index,
                    "start": offset,
                    "stop": end,
                    "sample_ids": ids[offset:end],
                    "status": "planned",
                }
            )
            offset = end
    return schedule


def sliced_task(cell: dict, batch: dict, *, limits: dict, task_factory=scheming):
    # Always construct the ORIGINAL full cell, then slice. Calling the generator
    # with n=batch_size would repeat its initial IDs/questions for every batch.
    task = task_factory(**cell, **limits)
    dataset = task.dataset
    samples = list(dataset)[batch["start"] : batch["stop"]]
    if [str(sample.id) for sample in samples] != batch["sample_ids"]:
        raise CampaignError("Generated sample IDs changed from the frozen manifest")
    task.dataset = MemoryDataset(
        samples,
        name=getattr(dataset, "name", None),
        location=getattr(dataset, "location", None),
        shuffled=False,
    )
    task.config.parallel_tool_calls = None
    return task


def prepare_manifest(
    directory: Path,
    *,
    config: dict,
    cells: list[dict],
    hashes: dict,
    resume: bool,
    task_factory=scheming,
) -> dict:
    path = directory / "manifest.json"
    identity = {"config": config, "cells": cells, "source_hashes": hashes}
    key = fingerprint(identity)
    if path.exists():
        if not resume:
            raise CampaignError("Campaign exists; --resume is required")
        existing = read_json(path)
        if existing.get("fingerprint") != key or any(
            existing.get(k) != v for k, v in identity.items()
        ):
            raise CampaignError(
                "Resume configuration or source hashes differ from the frozen campaign"
            )
        return existing
    if resume:
        raise CampaignError("--resume requires an existing manifest")
    limits = {name: config[name] for name in ("max_steps", "max_tokens", "token_limit")}
    schedule = make_schedule(
        cells,
        config["batch_size"],
        campaign_fingerprint=key,
        task_factory=task_factory,
        limits=limits,
    )
    manifest = {
        "version": MANIFEST_VERSION,
        "fingerprint": key,
        **identity,
        "created_at": utc_now(),
        "status": "planned",
        "stage": "base",
        "total_samples": sum(cell["n"] for cell in cells),
        "batches": schedule,
        "completed_base_logs": [],
    }
    atomic_json(path, manifest)
    return manifest


def refresh_base_records(manifest: dict) -> None:
    """One canonical log/sample selection per original case, including recovery."""
    records = []
    for batch in manifest["batches"]:
        if batch.get("canonical_log_records") and batch["status"] == "settled":
            records.extend(batch["canonical_log_records"])
        elif batch.get("recovery"):
            from .recovery import retained_record

            if batch["recovery"]["retained_sample_ids"]:
                records.append(retained_record(manifest, batch))
        elif batch["status"] == "settled":
            records.extend(
                {
                    "path": path,
                    "cell": manifest["cells"][batch["cell_index"]],
                    "cell_index": batch["cell_index"],
                    "batch_id": batch["batch_id"],
                    "expected_sample_ids": batch["sample_ids"],
                    "log_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    "recovery_retained": False,
                }
                for path in batch.get("log_paths", [])
            )
    manifest["completed_base_log_records"] = records
    manifest["completed_base_logs"] = list(dict.fromkeys(record["path"] for record in records))


def run_base(
    directory: Path,
    manifest: dict,
    ledger,
    *,
    resume: bool,
    bound_fn=openrouter_bound,
    task_factory=scheming,
    eval_fn=None,
    read_log_fn=None,
    source_hash_fn=source_hashes,
) -> dict:
    config = manifest["config"]
    limits = {name: config[name] for name in ("max_steps", "max_tokens", "token_limit")}
    for batch in manifest["batches"]:
        if source_hash_fn() != manifest["source_hashes"]:
            raise CampaignError("Source changed during the campaign; refusing further model calls")
        log_dir = directory / "base" / batch["batch_id"]
        journal_path = log_dir / "batch.json"
        journal = read_json(journal_path) if journal_path.exists() else None
        if batch.get("recovery"):
            from .recovery import run_recovery_batch

            result = run_recovery_batch(
                directory,
                manifest,
                batch,
                ledger,
                resume=resume,
                bound_fn=bound_fn,
                task_factory=task_factory,
                eval_fn=eval_fn,
                read_log_fn=read_log_fn,
            )
        elif journal and journal.get("status") == "settled":
            reservation = _ledger_batch(ledger, batch["batch_id"])
            if reservation is None or reservation.get("status") != "settled":
                raise CampaignError("Settled batch and ledger disagree")
            _verify_settled_logs(reservation, journal.get("log_paths", []))
            result = journal
        else:
            # Resume a reservation without querying new prices or constructing a
            # fresh agent task. The only allowed action is cost reconciliation.
            reserved = _ledger_batch(ledger, batch["batch_id"])
            if reserved or (journal and journal["status"] not in {"planned"}):
                if not resume:
                    raise CampaignError("A started/reserved batch needs explicit --resume")
                upper = (journal or reserved).get("upper_usd", 0.0)
                provider = (journal or {}).get("provider", {})
                task = None
            else:
                bound = bound_fn(
                    config["model"],
                    len(batch["sample_ids"]),
                    config["token_limit"],
                    config["max_tokens"],
                )
                upper, provider = bound["conservative_max_usd"], bound["provider"]
                batch["reservation"] = bound
                manifest["provider_policy"] = bound
                atomic_json(directory / "manifest.json", manifest)
                task = sliced_task(
                    manifest["cells"][batch["cell_index"]],
                    batch,
                    limits=limits,
                    task_factory=task_factory,
                )
            result = execute_batch(
                task,
                batch_id=batch["batch_id"],
                log_dir=log_dir,
                ledger=ledger,
                upper_usd=upper,
                stage="base",
                model=config["model"],
                provider=provider,
                fingerprint=manifest["fingerprint"],
                expected_sample_ids=batch["sample_ids"],
                max_samples=config["max_samples"],
                resume=resume,
                eval_fn=eval_fn,
                read_log_fn=read_log_fn,
            )
        batch.update(
            status=result["status"],
            log_paths=result.get("log_paths", []),
            sample_status_counts=result.get("sample_status_counts", {}),
        )
        if result.get("canonical_log_records"):
            batch["canonical_log_records"] = result["canonical_log_records"]
        refresh_base_records(manifest)
        manifest.update(updated_at=utc_now(), status="running", current_batch=batch["batch_id"])
        if result["status"] != "settled":
            manifest.update(status=result["status"], blocker=result)
            atomic_json(directory / "manifest.json", manifest)
            return manifest
        manifest.pop("blocker", None)
        atomic_json(directory / "manifest.json", manifest)
        emit(
            "base_progress",
            completed_samples=sum(
                len(item["sample_ids"])
                for item in manifest["batches"]
                if item["status"] == "settled"
            ),
            total_samples=manifest["total_samples"],
            budget=budget_summary(ledger.snapshot()),
        )
    manifest.update(status="base_complete", updated_at=utc_now(), current_batch=None)
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def budget_summary(snapshot: dict) -> dict:
    return {key: value for key, value in snapshot.items() if key != "batches"}


def _local_budget_summary(state: dict) -> dict:
    starting = Decimal(str(state["starting_spend_usd"]))
    costs = Counter()
    for entry in state.get("generations", {}).values():
        if entry.get("owner_batch") is not None:
            costs[entry["owner_batch"]] += Decimal(str(entry["cost_usd"]))
    verified = starting + sum(costs.values(), Decimal(0))
    retained = sum(
        (
            max(Decimal(str(batch["upper_usd"])) - costs[key], Decimal(0))
            for key, batch in state["batches"].items()
            if batch["status"] != "settled"
        ),
        Decimal(0),
    )
    return {
        "limit_usd": float(state["limit_usd"]),
        "verified_usd": float(verified),
        "retained_reservation_usd": float(retained),
        "committed_usd": float(verified + retained),
        "remaining_usd": float(
            max(Decimal(str(state["limit_usd"])) - verified - retained, Decimal(0))
        ),
        "pending_batches": [
            {"batch_id": key, "status": value["status"], "errors": value.get("errors", [])}
            for key, value in state["batches"].items()
            if value["status"] not in {"settled", "closed_reserved"}
        ],
        "closed_reserved_batches": [
            key for key, value in state["batches"].items() if value["status"] == "closed_reserved"
        ],
    }


def campaign_status(directory: Path) -> dict:
    """Read-only local status; never call a provider or mutate a ledger."""
    manifest = read_json(directory / "manifest.json")
    counts, completed, outcomes, current = Counter(), 0, Counter(), []
    for batch in manifest["batches"]:
        if batch.get("recovery"):
            recovery = batch["recovery"]
            path = directory / "recovery-attempts" / recovery["attempt_id"] / "batch.json"
            item = read_json(path) if path.exists() else {"status": "recovery_ready"}
            counts[item["status"]] += 1
            completed += len(recovery["retained_sample_ids"])
            outcomes.update(recovery["retained_status_counts"])
            outcomes.update(item.get("sample_status_counts", {}))
            if item["status"] == "settled":
                completed += len(recovery["retry_sample_ids"])
            elif item["status"] in {"reserved", "started", "returned", "interrupted", "blocked"}:
                current.append(recovery["attempt_id"])
            continue
        path = directory / "base" / batch["batch_id"] / "batch.json"
        item = read_json(path) if path.exists() else batch
        counts[item["status"]] += 1
        outcomes.update(item.get("sample_status_counts", {}))
        if item["status"] == "settled":
            completed += len(batch["sample_ids"])
        elif item["status"] in {"reserved", "started", "returned", "interrupted", "blocked"}:
            current.append(batch["batch_id"])
    ledger_path = directory / "budget.json"
    poststage_path = directory / "poststages.json"
    poststages = None
    if poststage_path.exists():
        from .poststages import progress_report

        poststages = progress_report(read_json(poststage_path))
    return {
        "campaign": str(directory),
        "status": manifest["status"],
        "base_batches": dict(counts),
        "base_samples_settled": completed,
        "base_samples_total": manifest["total_samples"],
        "sample_outcomes": dict(outcomes),
        "current_batches": current,
        "blocker": manifest.get("blocker"),
        "budget": _local_budget_summary(read_json(ledger_path)) if ledger_path.exists() else None,
        "poststages": poststages,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=ROOT / "logs" / "full-campaign")
    parser.add_argument("--profile", choices=PROFILES, default="full")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--budget-usd", type=float, default=50.0)
    parser.add_argument("--stage-budget-usd", type=float, default=35.0)
    parser.add_argument("--starting-spend-usd", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--token-limit", type=int, default=100000)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poststages", choices=("none", "all"), default="none")
    args = parser.parse_args(argv)
    directory = args.campaign_dir.expanduser().resolve()
    if args.status:
        status = campaign_status(directory)
        print(
            f"Campaign: {status['status']}\nBase samples settled: "
            f"{status['base_samples_settled']}/{status['base_samples_total']}\n"
            f"Batch states: {status['base_batches']}\nSample outcomes: {status['sample_outcomes']}"
        )
        if status["budget"]:
            budget = status["budget"]
            print(
                f"Verified spend: ${budget['verified_usd']:.6f}; retained reservation: "
                f"${budget['retained_reservation_usd']:.6f}; total committed: "
                f"${budget['committed_usd']:.6f}/${budget['limit_usd']:.2f}"
            )
            if budget["pending_batches"]:
                print("Pending billing: " + json.dumps(budget["pending_batches"]))
            if budget["closed_reserved_batches"]:
                print(
                    "Stopped attempts with permanently retained reservations: "
                    + ", ".join(budget["closed_reserved_batches"])
                )
        if status["current_batches"]:
            print("Current batches: " + ", ".join(status["current_batches"]))
        if status["blocker"]:
            print(
                "Blocker: "
                + json.dumps(
                    status["blocker"].get("settlement", status["blocker"].get("budget_blocker"))
                )
            )
        if status["poststages"]:
            post = status["poststages"]
            print(f"Post stages: {post['status']}")
            for stage, counts in post["stages"].items():
                print(
                    f"  {stage}: {counts['settled']}/{counts['planned']} settled; {counts['pending']} pending"
                )
            if post.get("blocked_batch"):
                print("Blocked post-stage batch: " + str(post["blocked_batch"]))
            if post.get("exclusions"):
                print("Excluded sources: " + json.dumps(post["exclusions"]))
        return
    if args.model != MODEL:
        parser.error("This authorized campaign uses the same DeepSeek model as validation")
    if any(
        not math.isfinite(value)
        for value in (args.budget_usd, args.stage_budget_usd, args.starting_spend_usd)
    ):
        parser.error("Budget values must be finite")
    if (
        not 0 < args.stage_budget_usd <= args.budget_usd <= 50
        or args.starting_spend_usd < 0
        or args.starting_spend_usd >= args.budget_usd
        or min(args.max_tokens, args.max_steps, args.token_limit, args.max_samples, args.batch_size)
        < 1
    ):
        parser.error(
            "Require positive limits, 0 < stage budget <= total budget <= $50, and a valid prior balance"
        )
    cells = [{**cell, "seed": args.seed} for cell in plan(args.profile)]
    config = {
        key: value
        for key, value in vars(args).items()
        if key not in {"execute", "resume", "status", "campaign_dir"}
    }
    config.update(
        first_batch_size=1,
        max_tasks=1,
        retries=0,
        epochs=1,
        cache_prompt=False,
        ctl_server=False,
        parallel_tool_calls=None,
        python_version=platform.python_version(),
    )
    if not args.execute:
        emit(
            "plan_only",
            cells=len(cells),
            samples=sum(cell["n"] for cell in cells),
            config=config,
            campaign_dir=str(directory),
        )
        return
    load_dotenv(ROOT / ".env", override=False)
    if not os.getenv("OPENROUTER_API_KEY"):
        parser.error("OPENROUTER_API_KEY is required for execution")
    from .budget import CampaignLedger

    os.environ.setdefault("XDG_DATA_HOME", str(ROOT / "inspect-data"))
    with exclusive_lock(directory / ".campaign.lock"):
        manifest = prepare_manifest(
            directory, config=config, cells=cells, hashes=source_hashes(), resume=args.resume
        )
        ledger = CampaignLedger(
            directory / "budget.json",
            limit_usd=args.budget_usd,
            starting_spend_usd=args.starting_spend_usd,
            stage_limits={"base": args.stage_budget_usd},
        )
        manifest = run_base(directory, manifest, ledger, resume=args.resume)
        if manifest["status"] != "base_complete":
            raise SystemExit(
                f"Campaign stopped: {manifest['status']}; inspect --status before resuming"
            )
        if args.poststages == "all":
            from .poststages import execute_poststages

            result = execute_poststages(
                directory,
                model=args.model,
                ledger=ledger,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
                max_tokens=args.max_tokens,
                token_limit=args.token_limit,
                resume=args.resume,
                source_hashes=manifest["source_hashes"],
            )
            manifest.update(poststage_result=result, status=result["status"], updated_at=utc_now())
            atomic_json(directory / "manifest.json", manifest)
            emit("poststages_result", result=result)
            if result["status"] not in {"complete", "complete_with_exclusions"}:
                raise SystemExit(f"Post stages stopped: {result['status']}; inspect --status")
        emit(
            "campaign_finished",
            budget=budget_summary(ledger.snapshot()),
            base_logs=manifest["completed_base_logs"],
        )


if __name__ == "__main__":
    try:
        main()
    except CampaignError as exc:
        print(f"Campaign stopped: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
