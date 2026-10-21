"""Bounded reconciliation retries for fresh, complete, billing-only stops.

This wrapper never calls a model or billing API itself. Every subsequent child
uses the original campaign arguments plus --resume, whose ledger gate forbids
regenerating previously dispatched batches. Scripts are outside experiment source
hashes. No evaluation settings or spending limits are changed here.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_WAIT_SECONDS = 900
DELAYS = (15, 30, 60, 120, 120, 120, 120, 120, 120, 60)  # At most 885 seconds asleep.
BILLING_ERROR = re.compile(r"Billing unavailable for (gen-[A-Za-z0-9_-]+) \(BillingError\)\.")


class Stopped(BaseException):
    def __init__(self, signum):
        self.signum = signum


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, allow_nan=False), flush=True)


def file_digest(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def source_hashes(root):
    """Same frozen inventory as campaign.source_hashes, without importing Inspect."""
    paths = {
        *root.glob("scheming_inspect/*.py"),
        *root.glob("data/*.json"),
        *(
            root / name
            for name in ("uv.lock", "pyproject.toml", "Dockerfile", "compose.yaml", ".dockerignore")
        ),
    }
    return {str(path.relative_to(root)): file_digest(path) for path in paths if path.is_file()}


def campaign_directory(args, root):
    # The last exact option wins, as in the campaign's argparse parser.
    result = root / "logs" / "full-campaign"
    for index, value in enumerate(args):
        if value.startswith("--campaign-dir="):
            result = Path(value.split("=", 1)[1])
        elif value == "--campaign-dir" and index + 1 < len(args):
            result = Path(args[index + 1])
    result = result.expanduser()
    if not result.is_absolute():
        result = root / result
    return result.resolve()


def billing_only_block(directory, root, before_digest):
    """Trust the executor's complete journal, but reject stale or mixed failures."""
    try:
        path = directory / "budget.json"
        after_digest = file_digest(path)
        if after_digest is None or after_digest == before_digest:
            return None
        ledger = json.loads(path.read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["source_hashes"] != source_hashes(root):
            return None
        active = [
            (key, row)
            for key, row in ledger["batches"].items()
            if row["status"] in {"reserved", "blocked"}
        ]
        if len(active) != 1 or active[0][1]["status"] != "blocked":
            return None
        batch_id, batch = active[0]
        errors = batch["errors"]
        if not errors or not all(isinstance(error, str) for error in errors):
            return None
        matches = [BILLING_ERROR.fullmatch(error) for error in errors]
        if not all(matches):
            return None
        missing = set(batch["generation_ids"]) - set(ledger["generations"])
        if not missing or missing != {match.group(1) for match in matches}:
            return None
        log_dir = Path(batch["metadata"]["log_dir"]).resolve()
        if not log_dir.is_relative_to(directory):
            return None
        journal = json.loads((log_dir / "batch.json").read_text())
        settlement = journal["settlement"]
        if (
            journal["batch_id"] != batch_id
            or journal["status"] != "blocked"
            or journal["complete"] is not True
            or journal["errors"] != []
            or journal.get("sample_status_counts", {}).get("error", 0)
            or settlement["status"] != "blocked"
            or settlement["errors"] != errors
            or not batch.get("reconciled_at")
            or settlement.get("reconciled_at") != batch["reconciled_at"]
        ):
            return None
        if journal["fingerprint"] != batch["metadata"]["fingerprint"]:
            return None
        # The executor already checked complete IDs/epochs and parsed each log.
        # Keep the exact recorded log bytes as the evidence for this retry.
        fingerprints = batch["log_fingerprints"]
        if not fingerprints or set(journal["log_paths"]) != set(fingerprints):
            return None
        if any(file_digest(path) != expected for path, expected in fingerprints.items()):
            return None
        return {"batch_id": batch_id, "missing_billing_records": len(missing)}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def run_child(command, *, cwd):
    """Forward interruption to the child's process group; never resume on a signal."""
    child = subprocess.Popen(command, cwd=cwd, start_new_session=True)
    try:
        return child.wait()
    except (Stopped, KeyboardInterrupt) as exc:
        signum = exc.signum if isinstance(exc, Stopped) else signal.SIGINT
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=15)
        except (subprocess.TimeoutExpired, Stopped, KeyboardInterrupt):
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        raise


def run_with_billing_retries(
    args, *, root=ROOT, run=run_child, sleep=time.sleep, monotonic=time.monotonic, report=emit
):
    directory = campaign_directory(args, root)
    original = [sys.executable, "-m", "scheming_inspect.campaign", *args]
    resumed = original if "--resume" in args else [*original, "--resume"]
    command = original
    episode, deadline, attempts, slept = None, None, 0, 0
    while True:
        before = file_digest(directory / "budget.json")
        returncode = run(command, cwd=root)
        if returncode != 1:
            return returncode if returncode >= 0 else 128 - returncode
        block = billing_only_block(directory, root, before)
        if block is None:
            report("billing_retry_not_applicable", exit_code=returncode)
            return returncode
        if block["batch_id"] != episode:
            episode, deadline, attempts, slept = (
                block["batch_id"],
                monotonic() + MAX_WAIT_SECONDS,
                0,
                0,
            )
        remaining = deadline - monotonic()
        if attempts == len(DELAYS) or remaining <= 0 or DELAYS[attempts] >= remaining:
            report(
                "billing_retry_exhausted",
                **block,
                attempts=attempts,
                waited_seconds=slept,
                max_window_seconds=MAX_WAIT_SECONDS,
            )
            return returncode
        delay = DELAYS[attempts]
        report(
            "billing_retry_wait",
            **block,
            attempt=attempts + 1,
            delay_seconds=delay,
            window_remaining_seconds=round(remaining, 1),
        )
        # Short interruptible sleeps also enforce the deadline if a sleep overshoots.
        left = delay
        while left > 0:
            chunk = min(left, 30, max(0, deadline - monotonic()))
            if chunk <= 0:
                break
            sleep(chunk)
            slept += chunk
            left -= chunk
        if left > 0 or monotonic() >= deadline:
            report(
                "billing_retry_exhausted",
                **block,
                attempts=attempts,
                waited_seconds=slept,
                max_window_seconds=MAX_WAIT_SECONDS,
            )
            return returncode
        attempts += 1
        report(
            "billing_retry_resume",
            **block,
            attempt=attempts,
            message="Reconcile the existing batch; continue only after its billing settles.",
        )
        command = resumed


def main(args=None):
    args = list(sys.argv[1:] if args is None else args)

    def stop(signum, _frame):
        raise Stopped(signum)

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        directory = campaign_directory(args, ROOT)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".billing-retry.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                emit("billing_retry_already_running")
                return 2
            return run_with_billing_retries(args)
    except Stopped as exc:
        emit("billing_retry_interrupted", signal=exc.signum)
        return 128 + exc.signum
    except KeyboardInterrupt:
        emit("billing_retry_interrupted", signal=signal.SIGINT)
        return 130
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
