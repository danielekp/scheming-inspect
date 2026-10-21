"""Run the reconstruction in resumable batches, without spending limits.

Billing is optional reporting only. Completed or interrupted batches are never
automatically regenerated; resumes inspect their saved results first.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.log import read_eval_log
from inspect_ai.model import GenerateConfig, get_model

from .protocol import PROFILES, plan
from .spending import report_cost
from .tasks import ROOT, scheming

MODEL = "openrouter/deepseek/deepseek-v4-flash"
MANIFEST_VERSION = 1
COMPLETE_BATCH_STATES = {"complete", "settled", "complete_reserved"}


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
    """Validate terminal log identity and exact coverage, retaining all outcomes."""
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
    if not paths or set(recorded) != set(paths):
        raise CampaignError("Saved log inventory differs from the batch journal")
    for raw in paths:
        path = Path(raw)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != recorded[raw]:
            raise CampaignError("A completed batch log was removed or changed")


def execute_batch(
    task,
    *,
    batch_id: str,
    log_dir: str | Path,
    stage: str,
    model: str,
    fingerprint: str,
    expected_sample_ids: list[str] | None = None,
    model_roles: dict[str, str] | None = None,
    max_samples: int = 8,
    resume: bool = False,
    provider: dict | None = None,
    eval_fn=None,
    read_log_fn=None,
) -> dict:
    """Run once and save results. No price queries, reservations, or billing gates."""
    if max_samples < 1:
        raise CampaignError("Concurrency must be positive")
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
    if any(name != model for name in roles.values()):
        raise CampaignError("All campaign roles must use the same model")
    identity = dict(
        batch_id=batch_id,
        stage=stage,
        model=model,
        fingerprint=fingerprint,
        sample_ids=ids,
        model_roles=roles,
        max_samples=max_samples,
    )
    journal_path = log_dir / "batch.json"
    with exclusive_lock(log_dir / ".batch.lock"):
        previous = read_json(journal_path) if journal_path.exists() else None
        if previous and any(previous.get(key) != value for key, value in identity.items()):
            raise CampaignError("Existing batch identity/configuration does not match")
        if previous and not resume:
            raise CampaignError("Batch already exists; use --resume, never rerun")
        if previous and previous["status"] in COMPLETE_BATCH_STATES:
            # Legacy financial records are used only as immutable log hashes.
            record = previous.get("settlement") or previous
            _verify_settled_logs(record, previous.get("log_paths", []))
            return previous
        journal = previous or {
            **identity,
            "version": MANIFEST_VERSION,
            "created_at": utc_now(),
            "status": "planned",
            "log_dir": str(log_dir),
        }
        if journal["status"] == "planned":
            if task is None:
                raise CampaignError("An unstarted batch needs its original task")
            if list(log_dir.glob("*.eval")):
                raise CampaignError("Unstarted batch directory already contains eval logs")
            # Durable before any request: an interrupted batch cannot be replayed silently.
            journal.update(status="started", started_at=utc_now())
            atomic_json(journal_path, journal)
            emit("batch_started", batch_id=batch_id, stage=stage, samples=len(ids))
            task.config.parallel_tool_calls = None
            try:
                (eval_fn or inspect_eval)(
                    task,
                    model=model,
                    model_args={"provider": {}},
                    model_roles=_role_models(roles, model, {}),
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
        journal.update(
            coverage,
            status="complete" if coverage["complete"] else "blocked",
            log_fingerprints={
                raw: hashlib.sha256(Path(raw).read_bytes()).hexdigest()
                for raw in coverage["log_paths"]
            },
            updated_at=utc_now(),
        )
        atomic_json(journal_path, journal)
        emit(
            "batch_" + journal["status"],
            batch_id=batch_id,
            stage=stage,
            samples=len(ids),
            sample_status_counts=coverage["sample_status_counts"],
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
        historical = {"budget_usd", "stage_budget_usd", "starting_spend_usd"}
        old_config = {k: v for k, v in existing["config"].items() if k not in historical}
        new_config = {k: v for k, v in config.items() if k not in historical}
        saved_identity = {k: existing[k] for k in identity}
        if (
            existing.get("fingerprint") != fingerprint(saved_identity)
            or old_config != new_config
            or existing["cells"] != cells
            or existing["source_hashes"] != hashes
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
        if batch.get("canonical_log_records") and batch["status"] in COMPLETE_BATCH_STATES:
            records.extend(batch["canonical_log_records"])
        elif batch.get("recovery"):
            from .recovery import retained_record

            if batch["recovery"]["retained_sample_ids"]:
                records.append(retained_record(manifest, batch))
        elif batch["status"] in COMPLETE_BATCH_STATES:
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
    *,
    resume: bool,
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
        was_complete = batch["status"] in COMPLETE_BATCH_STATES
        if batch.get("recovery"):
            from .recovery import run_recovery_batch

            result = run_recovery_batch(
                directory,
                manifest,
                batch,
                resume=resume,
                task_factory=task_factory,
                eval_fn=eval_fn,
                read_log_fn=read_log_fn,
            )
        else:
            log_dir = directory / "base" / batch["batch_id"]
            path = log_dir / "batch.json"
            journal = read_json(path) if path.exists() else None
            if was_complete and (journal is None or journal["status"] not in COMPLETE_BATCH_STATES):
                raise CampaignError("Completed batch is missing its completed execution journal")
            task = None
            if journal is None or journal["status"] == "planned":
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
                stage="base",
                model=config["model"],
                fingerprint=(journal or {}).get("fingerprint", manifest["fingerprint"]),
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
        if was_complete:
            continue
        refresh_base_records(manifest)
        manifest.update(updated_at=utc_now(), status="running", current_batch=batch["batch_id"])
        if result["status"] not in COMPLETE_BATCH_STATES:
            manifest.update(status=result["status"], blocker=result)
            atomic_json(directory / "manifest.json", manifest)
            return manifest
        manifest.pop("blocker", None)
        atomic_json(directory / "manifest.json", manifest)
        completed = sum(
            len(b["sample_ids"])
            for b in manifest["batches"]
            if b["status"] in COMPLETE_BATCH_STATES
        )
        emit(
            "base_progress",
            completed_samples=completed,
            total_samples=manifest["total_samples"],
            spending=report_cost(directory),
        )
        print(f"Base experiments: {completed}/{manifest['total_samples']} recorded", flush=True)
    refresh_base_records(manifest)
    manifest.update(status="base_complete", updated_at=utc_now(), current_batch=None)
    atomic_json(directory / "manifest.json", manifest)
    return manifest


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
            if item["status"] in COMPLETE_BATCH_STATES:
                completed += len(recovery["retry_sample_ids"])
            elif item["status"] in {"reserved", "started", "returned", "interrupted", "blocked"}:
                current.append(recovery["attempt_id"])
            continue
        path = directory / "base" / batch["batch_id"] / "batch.json"
        item = read_json(path) if path.exists() else batch
        counts[item["status"]] += 1
        outcomes.update(item.get("sample_status_counts", {}))
        if item["status"] in COMPLETE_BATCH_STATES:
            completed += len(batch["sample_ids"])
        elif item["status"] in {"reserved", "started", "returned", "interrupted", "blocked"}:
            current.append(batch["batch_id"])
    spending_path = directory / "spending.json"
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
        "spending": read_json(spending_path) if spending_path.exists() else None,
        "spending_limit": None,
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
            f"Campaign: {status['status']}\nBase samples recorded: "
            f"{status['base_samples_settled']}/{status['base_samples_total']}\n"
            f"Batch states: {status['base_batches']}\nSample outcomes: {status['sample_outcomes']}\n"
            "Spending limit: none"
        )
        print("Latest optional cost report: " + json.dumps(status["spending"]))
        if status["current_batches"]:
            print("Current batches: " + ", ".join(status["current_batches"]))
        if status["blocker"]:
            print("Blocker: " + json.dumps(status["blocker"].get("errors", [])))
        if status["poststages"]:
            print("Post stages: " + json.dumps(status["poststages"]))
        return
    if (
        min(args.max_tokens, args.max_steps, args.token_limit, args.max_samples, args.batch_size)
        < 1
    ):
        parser.error("Execution limits must be positive")
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
            spending_limit=None,
        )
        return
    load_dotenv(ROOT / ".env", override=False)
    if not os.getenv("OPENROUTER_API_KEY"):
        parser.error("OPENROUTER_API_KEY is required for execution")
    os.environ.setdefault("XDG_DATA_HOME", str(ROOT / "inspect-data"))
    with exclusive_lock(directory / ".campaign.lock"):
        manifest = prepare_manifest(
            directory, config=config, cells=cells, hashes=source_hashes(), resume=args.resume
        )
        manifest = run_base(directory, manifest, resume=args.resume)
        if manifest["status"] != "base_complete":
            raise SystemExit(f"Campaign stopped: {manifest['status']}; inspect --status")
        if args.poststages == "all":
            from .poststages import execute_poststages

            result = execute_poststages(
                directory,
                model=args.model,
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
        emit("campaign_finished", base_logs=manifest["completed_base_logs"])


if __name__ == "__main__":
    try:
        main()
    except CampaignError as exc:
        print(f"Campaign stopped: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
