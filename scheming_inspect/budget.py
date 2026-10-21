"""Atomic campaign reservations and exact, deduplicated OpenRouter billing.

Reserve before invoking any model. A process crash leaves the whole reservation
in place. Reconcile existing logs on resume; never rerun an existing batch ID.
The caller must explicitly assert ``complete=True`` only after matching terminal
logs to its durable manifest of expected tasks/samples. This module cannot prove
that an unlogged remote request did not happen.

Unknown charges, incomplete logs, missing IDs, or failed billing checks block all new
reservations. Known costs are still saved, but the full reservation remains until
the batch is reconciled successfully. An explicitly audited, stopped batch may
instead be closed with its full reservation permanently retained. Such closure
does not establish its actual cost and cannot be silently settled or rerun.
JSON writes use fsync + atomic replacement;
an advisory file lock serializes independent campaign processes on this host.
No API key or raw provider response is persisted. Billing GETs never generate
model output and have bounded retries for eventually available metadata.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import io
import json
import math
import os
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import httpx
from inspect_ai.log import read_eval_log

VERSION = 1
BILLING_URL = "https://openrouter.ai/api/v1/generation"


class BudgetError(RuntimeError):
    """A spending invariant prevents a campaign operation."""


class BudgetExceeded(BudgetError):
    pass


class BatchAlreadyExists(BudgetError):
    pass


class BillingError(BudgetError):
    pass


def _money(value: Any, name: str = "amount") -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise BudgetError(f"{name} must be a finite nonnegative number.")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise BudgetError(f"{name} must be a finite nonnegative number.") from exc
    if not amount.is_finite() or amount < 0:
        raise BudgetError(f"{name} must be a finite nonnegative number.")
    return amount


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generation_cost(data: Any, response_id: str) -> Decimal:
    if not isinstance(data, dict) or data.get("id") != response_id:
        raise BillingError("Billing metadata has a missing or mismatched generation ID.")
    cost = data.get("total_cost")
    # Provider JSON specifies a numeric amount; strings/nulls are not settled.
    if isinstance(cost, bool) or not isinstance(cost, (int, float, Decimal)):
        raise BillingError("Billing cost is unavailable or malformed.")
    try:
        return _money(cost, "billing cost")
    except BudgetError as exc:
        raise BillingError("Billing cost is unavailable or malformed.") from exc


def fetch_generation(
    response_id: str,
    *,
    api_key: str,
    attempts: int = 4,
    timeout: float = 10,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Get one exact generation's bill, retrying only bounded read requests."""
    if not isinstance(response_id, str) or not response_id.startswith("gen-"):
        raise BillingError("Invalid OpenRouter generation ID.")
    if not api_key:
        raise BillingError("Billing API key is unavailable.")
    if not 1 <= attempts <= 6 or not math.isfinite(timeout) or not 0 < timeout <= 30:
        raise BillingError("Billing retry/timeout bounds are invalid.")
    failure = "billing metadata unavailable"
    with httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout) as client:
        for attempt in range(attempts):
            if attempt:
                sleep(min(2 ** (attempt - 1), 4))
            try:
                response = client.get(BILLING_URL, params={"id": response_id})
            except httpx.RequestError as exc:
                # Never include exception text: HTTP errors can embed headers.
                failure = type(exc).__name__
                continue
            if response.status_code != 200:
                failure = f"HTTP {response.status_code}"
                if response.status_code not in {404, 408, 429} and response.status_code < 500:
                    break
                continue
            try:
                data = response.json()["data"]
                _generation_cost(data, response_id)
            except (ValueError, KeyError, TypeError, BillingError):
                failure = "invalid or unavailable billing metadata"
                continue
            # Return only the fields needed for accounting, not raw response data.
            return {"id": response_id, "total_cost": data["total_cost"]}
    raise BillingError(f"Billing lookup failed after bounded retries: {failure}.")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            state = json.load(stream)
    except (OSError, ValueError) as exc:
        raise BudgetError(
            "Campaign ledger is missing, unreadable, or corrupt; refusing to reset it."
        ) from exc
    if not isinstance(state, dict) or state.get("version") != VERSION:
        raise BudgetError("Unsupported or corrupt campaign ledger.")
    return state


def _atomic_write(path: Path, state: dict[str, Any]) -> None:
    temporary: str | None = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(state, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def _owned_cost(state: dict[str, Any], batch_id: str) -> Decimal:
    return sum(
        (
            _money(row["cost_usd"])
            for row in state["generations"].values()
            if row["owner_batch"] == batch_id
        ),
        Decimal(0),
    )


def _batch_committed(state: dict[str, Any], batch_id: str) -> Decimal:
    batch = state["batches"][batch_id]
    known = _owned_cost(state, batch_id)
    return known if batch["status"] == "settled" else max(known, _money(batch["upper_usd"]))


def _committed(state: dict[str, Any], stage: str | None = None) -> Decimal:
    total = _money(state["starting_spend_usd"]) if stage is None else Decimal(0)
    return total + sum(
        (
            _batch_committed(state, key)
            for key, batch in state["batches"].items()
            if stage is None or batch["stage"] == stage
        ),
        Decimal(0),
    )


def _validate_state(state: dict[str, Any]) -> None:
    try:
        limit = _money(state["limit_usd"])
        starting = _money(state["starting_spend_usd"])
        if limit <= 0 or starting > limit:
            raise BudgetError("Invalid lifetime budget in ledger.")
        if not isinstance(state["stage_limits"], dict) or not isinstance(state["batches"], dict):
            raise BudgetError("Invalid ledger collections.")
        for amount in state["stage_limits"].values():
            if _money(amount) <= 0:
                raise BudgetError("Invalid stage budget.")
        for batch_id, batch in state["batches"].items():
            if batch["id"] != batch_id or batch["status"] not in {
                "reserved",
                "blocked",
                "settled",
                "closed_reserved",
            }:
                raise BudgetError("Invalid batch identity/status.")
            if not isinstance(batch["stage"], str):
                raise BudgetError("Invalid batch stage.")
            _money(batch["upper_usd"])
            if batch["status"] == "closed_reserved":
                closure = batch.get("closure", {})
                if (
                    closure.get("process_stopped") is not True
                    or not isinstance(closure.get("reason"), str)
                    or not closure["reason"].strip()
                    or closure.get("evidence", {}).get("bounded_execution_confirmed") is not True
                    or closure.get("log_fingerprints") != batch.get("log_fingerprints")
                    or not closure.get("log_fingerprints")
                ):
                    raise BudgetError("Closed reservation has missing or invalid audit evidence.")
        covered = Decimal(0)
        for response_id, row in state["generations"].items():
            if not response_id.startswith("gen-"):
                raise BudgetError("Invalid recorded generation ID.")
            cost = _money(row["cost_usd"])
            if row["owner_batch"] is None:
                covered += cost
            elif row["owner_batch"] not in state["batches"]:
                raise BudgetError("Generation has no owning batch.")
        if covered > starting:
            raise BudgetError("Initial generation costs exceed the initial balance.")
    except (KeyError, TypeError, AttributeError) as exc:
        raise BudgetError("Corrupt campaign ledger; refusing to reset it.") from exc


def _events(events: Sequence[Any]) -> Iterator[Any]:
    for event in events:
        yield event
        children = getattr(event, "events", None)
        if isinstance(children, list):
            yield from _events(children)


def collect_generation_ids(log_paths: Sequence[str | Path]) -> dict[str, Any]:
    """Read explicitly supplied terminal logs and expose any accounting gaps."""
    ids: set[str] = set()
    errors: list[str] = []
    warnings: list[str] = []
    cached = 0
    event_count = 0
    samples = 0
    logs_read = 0
    retries_observed = False
    fingerprints = {}
    paths = sorted({Path(raw).expanduser().resolve() for raw in log_paths})
    if not paths:
        errors.append("No logs supplied; unlogged requests cannot be ruled out.")
    for path in paths:
        try:
            raw = path.read_bytes()
            fingerprints[str(path)] = hashlib.sha256(raw).hexdigest()
            # Parse exactly the bytes fingerprinted above, with no fsspec path lookup.
            with io.BytesIO(raw) as stream:
                log = read_eval_log(stream, format="json" if path.suffix == ".json" else "eval")
        except Exception as exc:
            errors.append(f"Log unreadable ({type(exc).__name__}); accounting is incomplete.")
            continue
        logs_read += 1
        if getattr(log, "status", None) not in {"success", "error", "cancelled"}:
            errors.append("A source evaluation has no terminal status.")
        elif log.status == "cancelled":
            errors.append(
                "A source evaluation was cancelled; in-flight charges cannot be excluded."
            )
        elif log.status != "success":
            warnings.append("A source evaluation ended with an error or cancellation.")
        if not log.samples:
            errors.append("A source log has no sample records.")
        for sample in log.samples or []:
            samples += 1
            if getattr(sample, "error", None) or getattr(sample, "invalidation", None):
                warnings.append("A source sample has an error or invalidation.")
            if getattr(sample, "error_retries", None):
                retries_observed = True
                errors.append(
                    "A source sample was retried; earlier attempt charges may be unrepresented."
                )
            model_events = [
                event for event in _events(sample.events or []) if event.event == "model"
            ]
            if not model_events and getattr(sample, "model_usage", None):
                errors.append("Sample usage exists without model-event billing records.")
            for event in model_events:
                if getattr(event, "cache", None) == "read":
                    cached += 1
                    continue
                event_count += 1
                if getattr(event, "error", None) or getattr(event, "pending", False):
                    errors.append(
                        "A model request failed or remains pending; charges may be unknown."
                    )
                if (getattr(event, "retries", 0) or 0) > 0:
                    retries_observed = True
                    errors.append(
                        "A model request retried; unrepresented attempt charges are possible."
                    )
                response_id = getattr(getattr(event, "output", None), "response_id", None)
                if isinstance(response_id, str) and response_id.startswith("gen-"):
                    ids.add(response_id)
                else:
                    errors.append("A noncached model event has no exact OpenRouter generation ID.")
    return {
        "ids": sorted(ids),
        "errors": list(dict.fromkeys(errors)),
        "warnings": list(dict.fromkeys(warnings)),
        "cached_events_excluded": cached,
        "model_events": event_count,
        "samples": samples,
        "logs_read": logs_read,
        "retries_observed": retries_observed,
        "log_fingerprints": fingerprints,
    }


class CampaignLedger:
    def __init__(
        self,
        path: str | Path,
        limit_usd: float | str = 50,
        starting_spend_usd: float | str = 0,
        stage_limits: Mapping[str, float | str] | None = None,
        starting_generations: Mapping[str, float | str] | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        lock_existed = self.lock_path.exists()
        limit, starting = (
            _money(limit_usd, "limit_usd"),
            _money(starting_spend_usd, "starting_spend_usd"),
        )
        stages = {str(key): str(_money(value)) for key, value in (stage_limits or {}).items()}
        generations = {
            key: {"cost_usd": str(_money(value)), "owner_batch": None}
            for key, value in (starting_generations or {}).items()
        }
        with self._lock():
            if self.path.exists():
                state = self._load()
                if (
                    _money(state["limit_usd"]) != limit
                    or _money(state["starting_spend_usd"]) != starting
                    or {key: _money(value) for key, value in state["stage_limits"].items()}
                    != {key: _money(value) for key, value in stages.items()}
                ):
                    raise BudgetError(
                        "Existing ledger budget configuration differs; refusing to reset or widen it."
                    )
                for key, row in generations.items():
                    if state["generations"].get(key) != row:
                        raise BudgetError("Existing ledger initial billing IDs differ.")
            else:
                if lock_existed:
                    raise BudgetError(
                        "Ledger is missing beside an existing lock; refusing to reset prior spending."
                    )
                state = {
                    "version": VERSION,
                    "created_at": _now(),
                    "updated_at": _now(),
                    "limit_usd": str(limit),
                    "starting_spend_usd": str(starting),
                    "stage_limits": stages,
                    "generations": generations,
                    "batches": {},
                }
                _validate_state(state)
                self._save(state)

    @contextmanager
    def _lock(self) -> Iterator[None]:
        descriptor = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        with os.fdopen(descriptor, "a+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _load(self) -> dict[str, Any]:
        state = _read_json(self.path)
        _validate_state(state)
        return state

    def _save(self, state: dict[str, Any]) -> None:
        state["updated_at"] = _now()
        _atomic_write(self.path, state)

    def _batch_view(self, state: dict[str, Any], batch_id: str) -> dict[str, Any]:
        batch = copy.deepcopy(state["batches"][batch_id])
        batch["upper_usd"] = float(_money(batch["upper_usd"]))
        batch["verified_usd"] = float(_owned_cost(state, batch_id))
        batch["actual_usd"] = batch["verified_usd"] if batch["status"] == "settled" else None
        batch["committed_usd"] = float(_batch_committed(state, batch_id))
        return batch

    def snapshot(self) -> dict[str, Any]:
        with self._lock():
            state = self._load()
            committed = _committed(state)
            verified = _money(state["starting_spend_usd"]) + sum(
                (
                    _money(row["cost_usd"])
                    for row in state["generations"].values()
                    if row["owner_batch"] is not None
                ),
                Decimal(0),
            )
            return {
                "limit_usd": float(_money(state["limit_usd"])),
                "starting_spend_usd": float(_money(state["starting_spend_usd"])),
                "committed_usd": float(committed),
                "verified_usd": float(verified),
                "retained_reservation_usd": float(committed - verified),
                "remaining_usd": float(max(Decimal(0), _money(state["limit_usd"]) - committed)),
                "stages": {
                    key: {
                        "limit_usd": float(_money(limit)),
                        "committed_usd": float(_committed(state, key)),
                        "remaining_usd": float(
                            max(Decimal(0), _money(limit) - _committed(state, key))
                        ),
                    }
                    for key, limit in state["stage_limits"].items()
                },
                "batches": {key: self._batch_view(state, key) for key in state["batches"]},
                "verified_generation_count": len(state["generations"]),
            }

    def pending(self) -> list[dict[str, Any]]:
        """Active accounting blockers; closed reservations still consume budget."""
        return [
            row
            for row in self.snapshot()["batches"].values()
            if row["status"] in {"reserved", "blocked"}
        ]

    def reserve(
        self,
        batch_id: str,
        upper_usd: float | str,
        *,
        stage: str = "base",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        upper = _money(upper_usd, "upper_usd")
        if upper <= 0 or not isinstance(batch_id, str) or not batch_id.strip():
            raise BudgetError("Batch ID and a positive reservation are required.")
        if not isinstance(stage, str) or not stage.strip():
            raise BudgetError("A nonempty stage name is required.")
        # Ensure durable metadata is JSON-safe before taking any reservation.
        try:
            metadata = json.loads(json.dumps(metadata or {}, allow_nan=False))
        except (ValueError, TypeError) as exc:
            raise BudgetError("Batch metadata must be JSON-serializable.") from exc
        with self._lock():
            state = self._load()
            if batch_id in state["batches"]:
                raise BatchAlreadyExists(
                    "Batch ID already exists; reconcile it instead of rerunning it."
                )
            if any(
                batch["status"] in {"reserved", "blocked"} for batch in state["batches"].values()
            ):
                raise BudgetError(
                    "An unresolved batch retains its reservation; reconcile it before more work."
                )
            if _committed(state) + upper > _money(state["limit_usd"]):
                raise BudgetExceeded("Reservation exceeds the remaining lifetime campaign budget.")
            if stage in state["stage_limits"] and _committed(state, stage) + upper > _money(
                state["stage_limits"][stage]
            ):
                raise BudgetExceeded("Reservation exceeds the remaining stage budget.")
            state["batches"][batch_id] = {
                "id": batch_id,
                "stage": stage,
                "upper_usd": str(upper),
                "status": "reserved",
                "reserved_at": _now(),
                "metadata": metadata,
                "errors": [],
                "generation_ids": [],
            }
            self._save(state)
            return self._batch_view(state, batch_id)

    def close_with_reservation(
        self,
        batch_id: str,
        logs: Sequence[str | Path],
        *,
        reason: str,
        process_stopped: bool,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Close stopped work without releasing a cent of its original ceiling.

        This is an explicit audited recovery operation, never an automatic error
        handler. The caller must verify the worker has stopped and include
        ``bounded_execution_confirmed=True`` in evidence after checking that the
        original rate/token caps and no-retry policy covered every request.
        Logs must match the last blocked reconciliation exactly. The closure is
        permanent in this API: ``settle`` cannot convert it to a known actual cost.
        Existing generation IDs, verified amounts, and error diagnostics remain.
        """
        if process_stopped is not True or not isinstance(reason, str) or not reason.strip():
            raise BudgetError(
                "Closure requires an explicit stopped-process confirmation and reason."
            )
        if (
            not isinstance(evidence, dict)
            or evidence.get("bounded_execution_confirmed") is not True
        ):
            raise BudgetError("Closure requires evidence confirming the original execution bound.")
        try:
            audit_evidence = json.loads(json.dumps(evidence, allow_nan=False))
        except (ValueError, TypeError) as exc:
            raise BudgetError("Closure evidence must be JSON-serializable.") from exc
        with self._lock():
            state = self._load()
            if batch_id not in state["batches"]:
                raise BudgetError("Cannot close an unreserved batch.")
            batch = state["batches"][batch_id]
            if batch["status"] != "blocked":
                raise BudgetError("Only a blocked, previously reconciled batch may be closed.")
            observed = collect_generation_ids(logs)
            fingerprints = observed["log_fingerprints"]
            if (
                not fingerprints
                or fingerprints != batch.get("log_fingerprints")
                or observed["logs_read"] != len(fingerprints)
            ):
                raise BudgetError(
                    "Closure logs are missing, unreadable, or changed since reconciliation."
                )
            if set(observed["ids"]) != set(batch.get("generation_ids", [])):
                raise BudgetError(
                    "Closure generation IDs differ from recorded reconciliation evidence."
                )
            if observed["retries_observed"]:
                raise BudgetError(
                    "Recorded retries undermine the original bound; closure is refused."
                )
            if _owned_cost(state, batch_id) > _money(batch["upper_usd"]):
                raise BudgetError("Known charges exceed the original bound; closure is refused.")
            batch["closure"] = {
                "closed_at": _now(),
                "reason": reason.strip(),
                "process_stopped": True,
                "evidence": audit_evidence,
                "log_fingerprints": dict(fingerprints),
                "generation_ids": list(batch.get("generation_ids", [])),
                "accounting_errors": list(observed["errors"]),
                "retention_policy": "permanent_full_original_reservation",
            }
            batch["status"] = "closed_reserved"
            self._save(state)
            return self._batch_view(state, batch_id)

    def settle(
        self,
        batch_id: str,
        logs: Sequence[str | Path],
        *,
        complete: bool = False,
        api_key: str | None = None,
        lookup: Callable[[str], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Reconcile a reserved batch, never silently release uncertain charges."""
        with self._lock():
            state = self._load()
            if batch_id not in state["batches"]:
                raise BudgetError("Cannot settle a batch that was never reserved.")
            batch = state["batches"][batch_id]
            if batch["status"] in {"settled", "closed_reserved"}:
                raise BatchAlreadyExists(
                    "Batch is terminal; its settlement/retained reservation cannot be changed."
                )
            evidence = collect_generation_ids(logs)
            errors = list(evidence["errors"])
            if complete is not True:
                errors.append(
                    "Caller has not confirmed all expected work and terminal logs are present."
                )
            old_ids = set(batch.get("generation_ids", []))
            if old_ids - set(evidence["ids"]):
                errors.append(
                    "Previously observed generation IDs disappeared from reconciliation logs."
                )
            batch.update(
                generation_ids=sorted(old_ids | set(evidence["ids"])),
                log_fingerprints=evidence["log_fingerprints"],
                cached_events_excluded=evidence["cached_events_excluded"],
                model_events=evidence["model_events"],
                samples=evidence["samples"],
                warnings=evidence["warnings"],
            )
            # Persist before billing requests: a crash here still retains full reserve.
            batch.update(status="blocked", errors=errors or ["Billing reconciliation in progress."])
            self._save(state)
            for response_id in evidence["ids"]:
                if response_id in state["generations"]:
                    continue
                try:
                    data = (
                        lookup(response_id)
                        if lookup
                        else fetch_generation(response_id, api_key=api_key or "")
                    )
                    amount = _generation_cost(data, response_id)
                except Exception as exc:
                    # Exception class only: never leak keys/raw HTTP responses.
                    errors.append(f"Billing unavailable for {response_id} ({type(exc).__name__}).")
                    continue
                state["generations"][response_id] = {
                    "cost_usd": str(amount),
                    "owner_batch": batch_id,
                    "verified_at": _now(),
                }
                self._save(state)
            known = _owned_cost(state, batch_id)
            if known > _money(batch["upper_usd"]):
                errors.append("Verified charges exceed this batch's reserved upper bound.")
            if _committed(state) > _money(state["limit_usd"]):
                errors.append("Verified/reserved campaign spending exceeds its lifetime cap.")
            if batch["stage"] in state["stage_limits"] and _committed(
                state, batch["stage"]
            ) > _money(state["stage_limits"][batch["stage"]]):
                errors.append("Verified/reserved stage spending exceeds its stage cap.")
            batch["errors"] = list(dict.fromkeys(errors))
            batch["status"] = "blocked" if errors else "settled"
            batch["reconciled_at"] = _now()
            self._save(state)
            return self._batch_view(state, batch_id)
