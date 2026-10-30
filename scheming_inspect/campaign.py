"""Run fresh campaigns in batches; resume keeps complete batches and restarts partial ones."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import MemoryDataset
from inspect_ai.log import read_eval_log
from inspect_ai.model import GenerateConfig, get_model

from .costs import report_key_usage
from .protocol import PROFILES, plan
from .tasks import ROOT, scheming

FORMAT = "simple-campaign-v1"
MODEL = "openrouter/deepseek/deepseek-v4-flash"


class CampaignError(RuntimeError):
    """The campaign cannot safely continue from its saved records."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path | str, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CampaignError("Another process is running this campaign") from exc
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


def read_manifest(directory: Path | str) -> dict:
    manifest = read_json(Path(directory) / "manifest.json")
    if manifest.get("format") != FORMAT:
        raise CampaignError("This is an older campaign format. Start in a new directory.")
    return manifest


def read_log(path: Path | str):
    path = Path(path)
    with path.open("rb") as stream:
        return read_eval_log(stream, format="json" if path.suffix == ".json" else "eval")


def emit(event: str, **values) -> None:
    print(json.dumps({"time": utc_now(), "event": event, **values}, allow_nan=False), flush=True)


def sample_status(sample) -> str:
    behavior = (sample.metadata or {}).get("behavior", {})
    if sample.error:
        return "error"
    if getattr(sample, "invalidation", None):
        return "invalidated"
    if getattr(sample, "limit", None):
        return "limited"
    if behavior.get("completed") == 0:
        return "incomplete"
    if behavior.get("schema_valid") == 0:
        return "invalid_schema"
    if behavior.get("choice_recorded") == 0:
        return "blocked_choice" if behavior.get("choice_rejected") == 1 else "no_choice"
    return "success"


def _batch_result(log_dir: Path, batch: dict, model: str, reader) -> dict:
    paths = sorted(log_dir.glob("*.eval"))
    if len(paths) > 1:
        raise CampaignError("More than one evaluation log exists for this batch")
    rows, errors = [], []
    if not paths:
        errors.append("No evaluation log was saved")
    for path in paths:
        try:
            log = reader(path)
        except Exception as exc:
            errors.append(f"Unreadable evaluation log: {type(exc).__name__}")
            continue
        metadata = log.eval.metadata or {}
        if str(log.eval.model) != model or metadata.get("campaign_batch_id") != batch["batch_id"]:
            raise CampaignError("Saved log model or batch ID does not match")
        ids = [str(sample.id) for sample in log.samples or []]
        if len(ids) != len(set(ids)) or not set(ids) <= set(batch["sample_ids"]):
            raise CampaignError("Saved log has duplicate or unexpected sample IDs")
        if any(sample.epoch != 1 for sample in log.samples or []):
            raise CampaignError("Saved log has an unexpected sample epoch")
        if log.status not in {"success", "error", "cancelled"}:
            errors.append("The evaluation log is not terminal")
        if Counter(ids) != Counter(batch["sample_ids"]):
            errors.append("The evaluation log does not contain every expected sample")
        rows = [
            {"id": str(sample.id), "status": sample_status(sample)} for sample in log.samples or []
        ]
    return {
        "status": "interrupted" if errors else "complete",
        "errors": errors,
        "log_paths": [str(path.resolve()) for path in paths],
        "samples": rows,
        "sample_status_counts": dict(Counter(row["status"] for row in rows)),
    }


def execute_batch(
    task_factory,
    *,
    batch: dict,
    log_dir: Path | str,
    model: str,
    model_roles: dict[str, str] | None = None,
    max_samples: int = 8,
    eval_fn=None,
    read_log_fn=None,
) -> dict:
    """Keep complete coverage; otherwise archive the attempt and lazily rebuild it.

    The caller holds the campaign lock. Complete coverage includes recorded errors
    and invalid answers; those outcomes never trigger selective regeneration.
    """
    log_dir = Path(log_dir).resolve()
    ids = [str(value) for value in batch["sample_ids"]]
    if not ids or len(set(ids)) != len(ids) or max_samples < 1:
        raise CampaignError("A batch needs distinct sample IDs and positive concurrency")
    identity = {"batch_id": batch["batch_id"], "sample_ids": ids, "model": model}
    journal_path = log_dir / "batch.json"
    previous = read_json(journal_path) if journal_path.exists() else {}
    if previous and any(previous.get(key) != value for key, value in identity.items()):
        raise CampaignError("Saved batch identity does not match")
    result = _batch_result(log_dir, identity, model, read_log_fn or read_log)
    if result["status"] == "complete":
        record = {**previous, **identity, **result}
        atomic_json(journal_path, record)
        return record
    if batch.get("status") == "complete" or previous.get("status") == "complete":
        raise CampaignError("A completed batch is missing its complete evaluation log")
    archives = list(previous.get("archived_attempts", []))
    if previous or result["log_paths"]:
        archive = log_dir.parent / "_interrupted" / f"{batch['batch_id']}-{uuid4().hex}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        log_dir.rename(archive)
        archives.append(str(archive))
        emit("batch_restarted", batch_id=batch["batch_id"], archived_attempt=str(archive))
    record = {**identity, "status": "started", "archived_attempts": archives}
    atomic_json(journal_path, record)
    emit("batch_started", batch_id=batch["batch_id"], samples=len(ids))
    try:
        task = task_factory()
        if [str(sample.id) for sample in task.dataset] != ids:
            raise CampaignError("Generated task does not match the batch's original sample IDs")
        task.config.parallel_tool_calls = None
        task.config.max_retries = 0
        roles = {
            role: get_model(name, config=GenerateConfig(max_retries=0, cache_prompt=False))
            for role, name in (model_roles or {}).items()
        }
        (eval_fn or inspect_eval)(
            task,
            model=model,
            model_roles=roles,
            log_dir=str(log_dir),
            max_samples=max_samples,
            max_tasks=1,
            epochs=1,
            max_retries=0,
            retry_on_error=0,
            cache_prompt=False,
            log_model_api=True,
            ctl_server=False,
            metadata={"campaign_batch_id": batch["batch_id"]},
        )
        record.update(_batch_result(log_dir, identity, model, read_log_fn or read_log))
    except BaseException as exc:
        record.update(status="interrupted", error_type=type(exc).__name__)
        atomic_json(journal_path, record)
        raise
    atomic_json(journal_path, record)
    emit(
        "batch_" + record["status"],
        batch_id=batch["batch_id"],
        outcomes=record["sample_status_counts"],
    )
    return record


def make_schedule(
    cells: list[dict], batch_size: int, *, task_factory=scheming, limits=None
) -> list[dict]:
    if batch_size < 1:
        raise CampaignError("Batch size must be positive")
    batches = []
    for index, cell in enumerate(cells):
        ids = [str(sample.id) for sample in task_factory(**cell, **(limits or {})).dataset]
        if len(ids) != cell["n"] or len(set(ids)) != len(ids):
            raise CampaignError("Cell has missing or duplicate sample IDs")
        for start in range(0, len(ids), batch_size):
            batches.append(
                {
                    "batch_id": f"base-{index:03d}-{start:05d}",
                    "cell_index": index,
                    "start": start,
                    "stop": min(start + batch_size, len(ids)),
                    "sample_ids": ids[start : start + batch_size],
                    "status": "planned",
                }
            )
    return batches


def sliced_task(cell: dict, batch: dict, *, limits: dict, task_factory=scheming):
    # Regenerate the original full cell before slicing, so later batches keep their fixtures.
    task = task_factory(**cell, **limits)
    task.dataset = MemoryDataset(list(task.dataset)[batch["start"] : batch["stop"]], shuffled=False)
    return task


def base_log_records(manifest: dict) -> list[dict]:
    return [
        {
            "path": path,
            "cell": manifest["cells"][batch["cell_index"]],
            "cell_index": batch["cell_index"],
            "batch_id": batch["batch_id"],
            "expected_sample_ids": batch["sample_ids"],
        }
        for batch in manifest["batches"]
        if batch["status"] == "complete"
        for path in batch["log_paths"]
    ]


def prepare_manifest(
    directory: Path, *, config: dict, cells: list[dict], task_factory=scheming
) -> dict:
    if (directory / "manifest.json").exists():
        read_manifest(directory)
        raise CampaignError("Campaign already exists; use --resume or a new directory")
    limits = {name: config[name] for name in ("max_steps", "max_tokens", "token_limit")}
    manifest = {
        "format": FORMAT,
        "config": config,
        "cells": cells,
        "status": "planned",
        "total_samples": sum(cell["n"] for cell in cells),
        "batches": make_schedule(
            cells, config["batch_size"], task_factory=task_factory, limits=limits
        ),
    }
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def run_base(
    directory: Path, manifest: dict, *, task_factory=scheming, eval_fn=None, read_log_fn=None
) -> dict:
    config = manifest["config"]
    limits = {name: config[name] for name in ("max_steps", "max_tokens", "token_limit")}
    for batch in manifest["batches"]:
        was_complete = batch["status"] == "complete"
        try:
            result = execute_batch(
                lambda: sliced_task(
                    manifest["cells"][batch["cell_index"]],
                    batch,
                    limits=limits,
                    task_factory=task_factory,
                ),
                batch=batch,
                log_dir=directory / "base" / batch["batch_id"],
                model=config["model"],
                max_samples=config["max_samples"],
                eval_fn=eval_fn,
                read_log_fn=read_log_fn,
            )
        except BaseException:
            if not was_complete:
                batch["status"] = "interrupted"
            manifest["status"] = "interrupted"
            atomic_json(directory / "manifest.json", manifest)
            raise
        batch.update(result)
        manifest["status"] = "running" if result["status"] == "complete" else "interrupted"
        atomic_json(directory / "manifest.json", manifest)
        if result["status"] != "complete":
            return manifest
        if not was_complete:
            emit(
                "base_progress",
                completed=sum(
                    len(b["sample_ids"]) for b in manifest["batches"] if b["status"] == "complete"
                ),
                total=manifest["total_samples"],
                spending=report_key_usage(directory),
            )
    manifest["status"] = "base_complete"
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def campaign_status(directory: Path) -> dict:
    manifest = read_manifest(directory)
    poststages = None
    if (directory / "poststages.json").exists():
        from .poststages import progress_report

        poststages = progress_report(read_json(directory / "poststages.json"))
    return {
        "status": poststages["status"] if poststages else manifest["status"],
        "poststages": poststages,
        "total_samples": manifest["total_samples"],
        "completed_samples": sum(
            len(b["sample_ids"]) for b in manifest["batches"] if b["status"] == "complete"
        ),
        "batches": dict(Counter(b["status"] for b in manifest["batches"])),
        "sample_outcomes": dict(
            sum(
                (Counter(b.get("sample_status_counts", {})) for b in manifest["batches"]), Counter()
            )
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, default=ROOT / "logs" / "campaign")
    parser.add_argument("--profile", choices=PROFILES, default="full")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--resume", action="store_true", help="Use the saved configuration and cells"
    )
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--token-limit", type=int, default=400000)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--poststages", choices=("none", "all"), default="none")
    args = parser.parse_args(argv)
    directory = args.campaign_dir.expanduser().resolve()
    if args.status:
        print(json.dumps(campaign_status(directory), indent=2))
        return
    if args.resume:
        saved = read_manifest(directory)
        config, cells = saved["config"], saved["cells"]
    else:
        config = {
            key: value
            for key, value in vars(args).items()
            if key not in {"execute", "resume", "status", "campaign_dir"}
        }
        if (
            min(
                config[key]
                for key in ("max_tokens", "max_steps", "token_limit", "max_samples", "batch_size")
            )
            < 1
        ):
            parser.error("Execution limits must be positive")
        cells = [{**cell, "seed": args.seed} for cell in plan(args.profile)]
    if config.get("poststages") == "all" and config["batch_size"] < 2:
        parser.error("Post-stages require --batch-size of at least 2 for matched interviews")
    if not args.execute:
        emit("plan_only", config=config, cells=cells, samples=sum(cell["n"] for cell in cells))
        return
    load_dotenv(ROOT / ".env", override=False)
    if config["model"].startswith("openrouter/") and not os.getenv("OPENROUTER_API_KEY"):
        parser.error("OPENROUTER_API_KEY is required for this model")
    os.environ.setdefault("XDG_DATA_HOME", str(ROOT / "inspect-data"))
    with exclusive_lock(directory / ".campaign.lock"):
        manifest = (
            read_manifest(directory)
            if args.resume
            else prepare_manifest(directory, config=config, cells=cells)
        )
        manifest = run_base(directory, manifest)
        if manifest["status"] != "base_complete":
            raise SystemExit("Campaign interrupted; inspect the logs, then use --execute --resume")
        if config.get("poststages") == "all":
            from .poststages import execute_poststages

            result = execute_poststages(
                directory,
                model=config["model"],
                batch_size=config["batch_size"],
                max_samples=config["max_samples"],
                max_tokens=config["max_tokens"],
                token_limit=config["token_limit"],
                resume=args.resume,
            )
            manifest.update(status=result["status"], poststage_result=result)
            atomic_json(directory / "manifest.json", manifest)
            if result["status"] not in {"complete", "complete_with_exclusions"}:
                raise SystemExit("Post-stages interrupted; use --execute --resume")
        emit("campaign_finished", status=manifest["status"])


if __name__ == "__main__":
    try:
        main()
    except CampaignError as exc:
        raise SystemExit(str(exc)) from exc
