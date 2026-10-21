"""Frozen, budgeted post-stages over canonical completed campaign logs.

All eligible main sources are included: no default one-source limit. Descriptors
are frozen before generation; histories are loaded one log at a time and copied
only for the current batch. The shared campaign executor owns reservations,
provider caps, exact billing, and the rule that started batches never rerun.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import GenerateConfig, ModelOutput
from inspect_ai.solver import solver

from . import followups as f
from .campaign import COMPLETE_BATCH_STATES
from .judge import semantic_audit
from .protocol import COVERT, plan

STAGES = ("audit_main", "interrogation", "fixed", "audit_remaining")
MATCH_KEYS = (
    "data_version",
    "seed",
    "nudge",
    "goal_mode",
    "goal_role",
    "difficulty",
    "no_cot",
    "question_set",
    "consequence",
    "user_prompt",
)
VERSION = 1


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def is_main_cell(cell: dict) -> bool:
    """Use the recorded original cell, including n; a smoke/ablation is not main."""
    return any(
        all(cell.get(key) == value for key, value in original.items() if key != "seed")
        for original in plan("main")
    )


def source_key(path: str, sample_id: Any, epoch: int) -> str:
    return hashlib.sha256(f"{path}:{sample_id}:{epoch}".encode()).hexdigest()[:16]


def _read_log(path: Path):
    with path.open("rb") as stream:
        return read_eval_log(stream, format="json" if path.suffix == ".json" else "eval")


def _behavior(sample) -> dict:
    value = (sample.metadata or {}).get("behavior") or (sample.store or {}).get("behavior")
    if not isinstance(value, dict):
        score = (sample.scores or {}).get("behavior_metrics")
        value = getattr(score, "value", None)
    return value if isinstance(value, dict) else {}


def _flags(behavior: dict) -> dict:
    return {
        key: bool(value)
        for key, value in behavior.items()
        if type(value) in (int, float, bool) and value in (0, 1)
    }


def _stub(record: dict, messages=(), task_description="") -> f.SourceCase:
    return f.SourceCase(
        key=record["key"],
        log_path=record["path"],
        sample_id=record["sample_id"],
        epoch=record["epoch"],
        model=record["model"],
        experiment=record["experiment"],
        scenario_id=record["scenario_id"],
        condition=record["condition"],
        messages=tuple(messages),
        behavior=record["flags"],
        task_description=task_description,
        matching_context=record["matching_context"],
    )


def canonical_log_records(base: dict, *, require_complete: bool = False) -> list[dict]:
    """Normalize explicit canonical selections without changing original logs.

    Legacy completed campaigns have one full log per batch. Recovery records may
    select only a subset of a failed attempt; original cell/sample identity stays
    unique even when the same sample ID occurs in several treatment cells.
    """
    batches = {batch["batch_id"]: batch for batch in base["batches"]}
    expected = {
        (batch["cell_index"], str(sample_id))
        for batch in base["batches"]
        for sample_id in batch["sample_ids"]
    }
    raw_records = base.get("completed_base_log_records")
    if raw_records is None:
        raw_records = [
            {
                "path": path,
                "cell_index": batch["cell_index"],
                "cell": base["cells"][batch["cell_index"]],
                "batch_id": batch["batch_id"],
                "expected_sample_ids": batch["sample_ids"],
            }
            for batch in base["batches"]
            if batch["status"] in COMPLETE_BATCH_STATES
            for path in batch.get("log_paths", [])
        ]
    result, selected, paths = [], set(), set()
    for raw in raw_records:
        record = copy.deepcopy(raw)
        index = record.get("cell_index", batches.get(record["batch_id"], {}).get("cell_index"))
        if type(index) is not int or not 0 <= index < len(base["cells"]):
            raise ValueError("Canonical log record has no valid original cell_index.")
        if record.get("cell") != base["cells"][index]:
            raise ValueError("Canonical log record differs from the original cell configuration.")
        record["cell_index"] = index
        record["path"] = str(Path(record["path"]).resolve())
        ids = [str(value) for value in record["expected_sample_ids"]]
        if not ids or len(set(ids)) != len(ids) or record["path"] in paths:
            raise ValueError("Canonical log record has empty/duplicate selections or paths.")
        identities = {(index, sample_id) for sample_id in ids}
        if identities - expected or identities & selected:
            raise ValueError("Canonical sample selections are unexpected or duplicated.")
        record["expected_sample_ids"] = ids
        record["recovery_retained"] = record.get("recovery_retained") is True
        if record["recovery_retained"] and not record.get("log_sha256"):
            raise ValueError("A retained recovery log requires its frozen SHA-256 fingerprint.")
        selected.update(identities)
        paths.add(record["path"])
        result.append(record)
    if require_complete and selected != expected:
        raise ValueError(
            "Canonical records do not cover every original campaign sample exactly once."
        )
    return result


def verify_record_hash(record: dict) -> str:
    actual = hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest()
    if record.get("log_sha256") and record["log_sha256"] != actual:
        raise ValueError("A canonical source log differs from its frozen SHA-256 fingerprint.")
    return actual


def record_accepts_log(record: dict, status: str) -> bool:
    return status == "success" or (status == "error" and record["recovery_retained"])


def discover_sources(base: dict, model: str, read_log_fn: Callable = _read_log) -> dict:
    """Validate canonical sources individually and retain superseded-attempt provenance."""
    if base.get("status") != "base_complete" or any(
        b["status"] not in COMPLETE_BATCH_STATES for b in base["batches"]
    ):
        raise ValueError("Post-stages require a completed, budget-accounted base campaign.")
    records, excluded, recovery_excluded, hashes, seen = {}, [], [], {}, set()
    requested = Counter(
        {"audit_main": 0, "audit_remaining": 0, "fixed_sources": 0, "interrogation_sources": 0}
    )
    for selection in canonical_log_records(base, require_complete=True):
        cell = selection["cell"]
        experiment, main = cell["experiment"], is_main_cell(cell)
        if experiment not in COVERT:
            continue
        path, batch_id = selection["path"], selection["batch_id"]
        stage = "audit_main" if main else "audit_remaining"
        expected = set(selection["expected_sample_ids"])
        requested[stage] += len(expected)
        if main and experiment != "email":
            requested["fixed_sources"] += len(expected)
        if main and experiment == "oversight":
            requested["interrogation_sources"] += len(expected)
        try:
            hashes[path] = verify_record_hash(selection)
            log = read_log_fn(Path(path))
        except ValueError:
            raise  # A tampered selection must never become an ordinary source exclusion.
        except Exception as exc:
            excluded.append(
                {
                    "path": path,
                    "batch_id": batch_id,
                    "stage": stage,
                    "reason": "unreadable_log",
                    "error_type": type(exc).__name__,
                }
            )
            log = None
        found = set()
        for sample in (log.samples or []) if log else []:
            identity = {
                "path": path,
                "sample_id": sample.id,
                "epoch": sample.epoch,
                "batch_id": batch_id,
                "cell_index": selection["cell_index"],
                "stage": stage,
                "experiment": experiment,
                "main": main,
                "recovery_retained": selection["recovery_retained"],
            }
            if str(sample.id) not in expected:
                if selection["recovery_retained"]:
                    recovery_excluded.append(
                        identity
                        | {
                            "reason": "superseded_recovery_attempt",
                            "error": bool(sample.error),
                            "invalidated": bool(getattr(sample, "invalidation", None)),
                        }
                    )
                    continue
                raise ValueError("Unexpected sample in an ordinary canonical source log.")
            key = source_key(path, sample.id, sample.epoch)
            reason = None
            if str(sample.id) in found or key in seen:
                raise ValueError("Duplicate sample in a canonical source log.")
            found.add(str(sample.id))
            seen.add(key)
            meta, behavior = sample.metadata or {}, _behavior(sample)
            if str(log.eval.model) != model:
                reason = "source_model_mismatch"
            elif not record_accepts_log(selection, log.status):
                reason = "source_log_failed"
            elif getattr(sample, "error", None):
                reason = "source_error"
            elif getattr(sample, "invalidation", None):
                reason = "source_invalidated"
            elif sample.epoch != 1:
                reason = "source_unexpected_epoch"
            elif behavior.get("schema_valid") == 0:
                reason = "source_invalid_schema"
            elif getattr(sample, "limit", None) or behavior.get("truncated") == 1:
                reason = "source_truncated"
            elif behavior.get("completed") != 1:
                reason = "source_incomplete_or_completion_unknown"
            elif not behavior:
                reason = "missing_behavior"
            elif not meta.get("scenario_id"):
                reason = "missing_scenario"
            elif any(meta.get(k) != cell[k] for k in cell if k != "n"):
                reason = "sample_cell_mismatch"
            else:
                try:
                    f.validate_transcript(sample.messages)
                except (ValueError, AttributeError, TypeError):
                    reason = "malformed_transcript"
            if reason:
                excluded.append(identity | {"key": key, "reason": reason})
                continue
            records[key] = identity | {
                "key": key,
                "cell": copy.deepcopy(cell),
                "model": model,
                "scenario_id": str(meta["scenario_id"]),
                "flags": _flags(behavior),
                "condition": json.dumps(meta.get("condition", ""), sort_keys=True),
                "matching_context": json.dumps(
                    {k: meta.get(k) for k in MATCH_KEYS}, sort_keys=True
                ),
            }
        for sample_id in sorted(expected - found):
            excluded.append(
                {
                    "batch_id": batch_id,
                    "sample_id": sample_id,
                    "stage": stage,
                    "experiment": experiment,
                    "main": main,
                    "reason": "missing_sample",
                }
            )
    return {
        "sources": records,
        "excluded": excluded,
        "recovery_exclusions": recovery_excluded,
        "source_logs": hashes,
        "requested": dict(requested),
    }


def make_groups(discovery: dict, seed: int = 0) -> tuple[list[dict], list[dict]]:
    groups, eligibility = [], []
    records = discovery["sources"]
    for key, record in sorted(records.items()):
        stage = "audit_main" if record["main"] else "audit_remaining"
        groups.append({"id": f"audit:{key}", "stage": stage, "source_keys": [key], "samples": 1})
    main = [
        _stub(record)
        for record in records.values()
        if record["main"] and record["experiment"] != "email"
    ]
    pairs = f.matched_sources(main, seed=seed)
    rng = random.Random(seed)
    matched = {source.key for pair in pairs for source in pair}
    for guilty, innocent in pairs:
        truth_order = ["guilty", "innocent"]
        rng.shuffle(truth_order)
        groups.append(
            {
                "id": f"interrogation:{guilty.key}:{innocent.key}",
                "stage": "interrogation",
                "source_keys": [guilty.key, innocent.key],
                "samples": 2,
                "truth_order": truth_order,
            }
        )
    for source in sorted(main, key=lambda item: item.key):
        actions = f.eligible_actions(source)
        if not actions:
            eligibility.append(
                {"source_key": source.key, "stage": "fixed", "reason": "no_observed_action"}
            )
        for action in actions:
            groups.append(
                {
                    "id": f"fixed:{source.key}:{action.name}",
                    "stage": "fixed",
                    "source_keys": [source.key],
                    "action": action.name,
                    "samples": 160,
                    "variants": [name for name, _ in f.fixed_questions(action)],
                    "repeats": 20,
                }
            )
        if source.experiment == "oversight" and source.key not in matched:
            reason = (
                "unknown_oversight_outcome"
                if "oversight_disabled" not in source.behavior
                else "no_matched_opposite_outcome"
            )
            eligibility.append(
                {"source_key": source.key, "stage": "interrogation", "reason": reason}
            )
    groups.sort(key=lambda group: STAGES.index(group["stage"]))
    return groups, eligibility


def group_sample_ids(group: dict) -> list[str]:
    if group["stage"].startswith("audit_"):
        return [group["id"]]
    if group["stage"] == "interrogation":
        pair = ":".join(group["source_keys"])
        return [f"{pair}:0:{truth}" for truth in group.get("truth_order", ["guilty", "innocent"])]
    key = group["source_keys"][0]
    return [
        f"{key}:{group['action']}:{variant}:{repeat}"
        for variant in group["variants"]
        for repeat in range(group["repeats"])
    ]


def logical_generations(stage: str, samples: int) -> dict[str, int]:
    if stage.startswith("audit_"):
        return {"grader": samples}
    if stage == "fixed":
        return {"target": samples, "grader": samples}
    if stage == "interrogation":
        return {"target": 7 * samples, "grader": 7 * samples, "interrogator": 14 * samples}
    raise ValueError(f"Unknown post-stage {stage}")


def upper_bound(stage: str, samples: int, rates: dict, max_tokens: int) -> float:
    """Every logical generation reserves a maximum-context prompt and completion."""
    context = int(rates["context_length"])
    prompt, completion = (float(rates["pricing"][name]) for name in ("prompt", "completion"))
    if (
        context < 1
        or max_tokens < 1
        or any(not math.isfinite(rate) or rate < 0 for rate in (prompt, completion))
    ):
        raise ValueError("Invalid post-stage provider pricing/context bound")
    return sum(logical_generations(stage, samples).values()) * (
        context * prompt + max_tokens * completion
    )


class SourceReader:
    """Cache at most one source log; never copy original events into new tasks."""

    def __init__(self, records: dict, read_log_fn: Callable = _read_log, fingerprints=None):
        self.records, self.read_log_fn = records, read_log_fn
        self.fingerprints = fingerprints or {}
        self.path, self.samples = None, {}

    def sample(self, key: str):
        record = self.records[key]
        if self.path != record["path"]:
            expected = self.fingerprints.get(record["path"])
            if (
                expected
                and hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest() != expected
            ):
                raise ValueError("A frozen source log changed before loading a batch.")
            log = self.read_log_fn(Path(record["path"]))
            self.samples = {(str(sample.id), sample.epoch): sample for sample in log.samples or []}
            self.path = record["path"]
        return self.samples[(str(record["sample_id"]), record["epoch"])]

    def source(self, key: str) -> f.SourceCase:
        sample = self.sample(key)
        description = (
            sample.input
            if isinstance(sample.input, str)
            else "\n".join(message.text for message in sample.input if message.role == "user")
        )
        return _stub(self.records[key], sample.messages, description)


@solver
def restore_completion():
    """Restore recorded answer for semantic scoring without a target generation."""

    async def solve(state, generate):
        state.output = ModelOutput.from_content(
            str(state.model), state.metadata["source_completion"]
        )
        state.tools = []
        state.completed = True
        return state

    return solve


def audit_sample(reader: SourceReader, key: str, stage: str) -> Sample:
    original, record = reader.sample(key), reader.records[key]
    metadata = copy.deepcopy(original.metadata or {})
    metadata.update(
        behavior=copy.deepcopy(_behavior(original)),
        evidence=copy.deepcopy(
            metadata.get("evidence", (original.store or {}).get("observations", []))
        ),
        source_completion=original.output.completion,
        source_key=key,
        source_log=record["path"],
        source_sample_id=record["sample_id"],
        source_epoch=record["epoch"],
        audit_stage=stage,
    )
    return Sample(id=f"audit:{key}", input=copy.deepcopy(original.messages), metadata=metadata)


def build_batch(
    selections: list[dict],
    groups: dict,
    reader: SourceReader,
    *,
    max_tokens: int,
    token_limit: int,
    seed: int,
) -> Task:
    stage = groups[selections[0]["group_id"]]["stage"]
    samples = []
    for selection in selections:
        group = groups[selection["group_id"]]
        ids = group_sample_ids(group)[selection["start"] : selection["stop"]]
        if stage.startswith("audit_"):
            candidates = [audit_sample(reader, group["source_keys"][0], stage)]
        elif stage == "fixed":
            # At most one source's 160 (or 320 for two distinct actions) forks in memory.
            candidates = f.fixed_samples(
                [reader.source(group["source_keys"][0])], repeats=20, limit=None
            )
        else:
            # Generate each frozen pair independently so batch slicing cannot re-match it.
            candidates = f.interrogation_samples(
                [reader.source(key) for key in group["source_keys"]],
                repeats=1,
                limit=None,
                seed=seed,
            )
        indexed = {str(sample.id): sample for sample in candidates}
        samples.extend(indexed[sample_id] for sample_id in ids)
    if stage.startswith("audit_"):
        solve, scorers = restore_completion(), [semantic_audit(max_tokens=max_tokens)]
    elif stage == "fixed":
        solve, scorers = f.fixed_followup_solver(max_tokens), [f.admission_metrics(max_tokens)]
    else:
        solve = f.interrogation_solver(7, max_tokens, max_tokens)
        scorers = [f.admission_metrics(max_tokens), f.interrogation_verdict()]
    return Task(
        name=f"scheming_{stage}",
        dataset=samples,
        solver=solve,
        scorer=scorers,
        sandbox=None,
        config=GenerateConfig(
            max_tokens=max_tokens, max_retries=0, cache=False, cache_prompt=False, timeout=90
        ),
        token_limit=token_limit,
        time_limit=600 if stage == "interrogation" else 300,
        fail_on_error=True,
        metadata={
            "protocol": "reconstructed post-stages",
            "stage": stage,
            "selection_seed": seed,
            "fixed_repeats": 20,
            "max_rounds": 7,
        },
    )


def progress_report(manifest: dict) -> dict:
    counts = {stage: {"planned": 0, "settled": 0, "pending": 0} for stage in STAGES}
    for group in manifest["spec"]["groups"]:
        done = manifest["progress"].get(group["id"], 0)
        counts[group["stage"]]["planned"] += group["samples"]
        counts[group["stage"]]["settled"] += done
        counts[group["stage"]]["pending"] += group["samples"] - done
    exclusions = manifest["spec"]["discovery"]["excluded"]
    return {
        "status": manifest["status"],
        "stages": counts,
        "requested": manifest["spec"]["discovery"]["requested"],
        "exclusions": dict(Counter(row["reason"] for row in exclusions)),
        "recovery_exclusions": dict(
            Counter(
                row["reason"]
                for row in manifest["spec"]["discovery"].get("recovery_exclusions", [])
            )
        ),
        "eligibility_exclusions": dict(
            Counter(f"{row['stage']}:{row['reason']}" for row in manifest["spec"]["eligibility"])
        ),
        "blocked_batch": manifest.get("blocked_batch"),
        "logged_sample_outcomes": {
            stage: dict(
                sum(
                    (
                        Counter(batch.get("sample_status_counts", {}))
                        for batch in manifest.get("batches", [])
                        if batch["stage"] == stage
                    ),
                    Counter(),
                )
            )
            for stage in STAGES
        },
        "scope": "Reconstructed protocol; settled counts include explicit unscored judge outcomes, not guaranteed valid labels.",
    }


def next_selections(manifest: dict, remaining_usd: float, batch_size: int) -> list[dict]:
    """Adapt to budget before dispatch; an interrogation pair is indivisible."""
    spec, selected, count, stage = manifest["spec"], [], 0, None
    for group in spec["groups"]:
        start = manifest["progress"].get(group["id"], 0)
        if start >= group["samples"]:
            continue
        if stage is None:
            stage = group["stage"]
        if group["stage"] != stage:
            break
        unit = 2 if stage == "interrogation" else 1
        available = min(group["samples"] - start, batch_size - count)
        take = available - available % unit
        while (
            take
            and upper_bound(stage, count + take, spec["rates"], spec["config"]["max_tokens"])
            > remaining_usd
        ):
            take -= unit
        if not take:
            break
        selected.append({"group_id": group["id"], "start": start, "stop": start + take})
        count += take
        if count >= batch_size or take < available:
            break
    return selected


def execute_poststages(
    campaign_root,
    *,
    model,
    ledger,
    batch_size=32,
    max_samples=8,
    max_tokens=8000,
    token_limit=100000,
    resume=False,
    source_hashes=None,
    execute_batch_fn=None,
    read_log_fn=_read_log,
    bound_fn=None,
    current_source_hash_fn=None,
) -> dict:
    """Run all frozen post-stage groups, or return explicit blocked/pending status.

    The only model is the campaign's authorized model, reused in every role with
    the same provider caps. Injectable functions support entirely offline tests.
    """
    from . import campaign

    if min(batch_size, max_samples, max_tokens, token_limit) < 1 or batch_size < 2:
        raise ValueError(
            "Post-stages require positive limits and batch_size >= 2 for matched pairs."
        )
    root = Path(campaign_root).resolve()
    base = json.loads((root / "manifest.json").read_text())
    if base["config"]["model"] != model:
        raise ValueError("Post-stage model must match the frozen base model.")
    hashes = source_hashes if source_hashes is not None else base["source_hashes"]
    if hashes != base["source_hashes"]:
        raise ValueError("Post-stage code hashes differ from the base campaign.")
    path = root / "poststages.json"
    config = {
        "model": model,
        "batch_size": batch_size,
        "max_samples": max_samples,
        "max_tokens": max_tokens,
        "token_limit": token_limit,
        "seed": base["config"].get("seed", 42),
        "interrogation_time_limit": 600,
        "other_time_limit": 300,
        "fixed_repeats": 20,
        "interrogation_repeats": 1,
        "max_rounds": 7,
        "all_sources": True,
        "stage_order": list(STAGES),
    }
    if path.exists():
        if not resume:
            raise ValueError("Post-stages already exist; use --resume, never restart them.")
        manifest = json.loads(path.read_text())
        spec = manifest["spec"]
        if (
            spec["config"] != config
            or spec["base_fingerprint"] != base["fingerprint"]
            or spec["source_hashes"] != hashes
            or digest(spec) != manifest["fingerprint"]
        ):
            raise ValueError("Frozen post-stage configuration or source identity changed.")
        for filename, expected_hash in spec["discovery"]["source_logs"].items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected_hash:
                raise ValueError("A frozen source log changed; refusing further model calls.")
    else:
        discovery = discover_sources(base, model, read_log_fn)
        groups, eligibility = make_groups(discovery, seed=config["seed"])
        bound = base.get("provider_policy") or (bound_fn or campaign.openrouter_bound)(
            model, 1, token_limit, max_tokens
        )
        rates = {key: bound[key] for key in ("pricing", "context_length", "provider")}
        upper_bound("interrogation", 2, rates, max_tokens)  # Fail before freezing invalid pricing.
        spec = {
            "config": config,
            "base_fingerprint": base["fingerprint"],
            "source_hashes": hashes,
            "rates": rates,
            "discovery": discovery,
            "groups": groups,
            "eligibility": eligibility,
        }
        manifest = {
            "version": VERSION,
            "fingerprint": digest(spec),
            "spec": spec,
            "status": "planned",
            "progress": {},
            "batches": [],
        }
        campaign.atomic_json(path, manifest)
    execute = execute_batch_fn or campaign.execute_batch
    hash_fn = current_source_hash_fn or campaign.source_hashes
    groups = {group["id"]: group for group in manifest["spec"]["groups"]}
    reader = SourceReader(
        manifest["spec"]["discovery"]["sources"],
        read_log_fn,
        manifest["spec"]["discovery"]["source_logs"],
    )
    if resume:
        # Revalidate settled log fingerprints/ledger identity without generating.
        for old in manifest["batches"]:
            if old["status"] not in COMPLETE_BATCH_STATES:
                continue
            roles = {"grader": model} | (
                {"interrogator": model} if old["stage"] == "interrogation" else {}
            )
            result = execute(
                None,
                batch_id=old["batch_id"],
                log_dir=root / "poststages" / old["stage"] / old["batch_id"],
                ledger=ledger,
                upper_usd=old["upper_usd"],
                stage=old["stage"],
                model=model,
                provider=manifest["spec"]["rates"]["provider"],
                fingerprint=manifest["fingerprint"],
                expected_sample_ids=old["sample_ids"],
                model_roles=roles,
                max_samples=max_samples,
                resume=True,
            )
            if result["status"] not in COMPLETE_BATCH_STATES:
                manifest.update(status="blocked", blocked_batch=old["batch_id"])
                campaign.atomic_json(path, manifest)
                return progress_report(manifest)
    while True:
        if hash_fn() != hashes:
            raise ValueError("Source changed during post-stages; refusing further model calls.")
        pending = [
            batch for batch in manifest["batches"] if batch["status"] not in COMPLETE_BATCH_STATES
        ]
        if pending:
            batch = pending[0]
        else:
            snapshot = ledger.snapshot()
            selections = next_selections(manifest, snapshot["remaining_usd"], batch_size)
            if not selections:
                remaining = any(
                    manifest["progress"].get(group["id"], 0) < group["samples"]
                    for group in groups.values()
                )
                manifest["status"] = (
                    "pending_budget"
                    if remaining
                    else (
                        "complete_with_exclusions"
                        if manifest["spec"]["discovery"]["excluded"]
                        else "complete"
                    )
                )
                manifest["budget"] = campaign.budget_summary(snapshot)
                campaign.atomic_json(path, manifest)
                return progress_report(manifest)
            stage = groups[selections[0]["group_id"]]["stage"]
            ids = [
                sample_id
                for selection in selections
                for sample_id in group_sample_ids(groups[selection["group_id"]])[
                    selection["start"] : selection["stop"]
                ]
            ]
            batch_id = f"{manifest['fingerprint'][:12]}-{stage}-{digest(selections)[:16]}"
            batch = {
                "batch_id": batch_id,
                "stage": stage,
                "selections": selections,
                "sample_ids": ids,
                "status": "planned",
                "upper_usd": upper_bound(stage, len(ids), manifest["spec"]["rates"], max_tokens),
            }
            manifest["batches"].append(batch)
            manifest["status"] = "running"
            campaign.atomic_json(path, manifest)
        stage = batch["stage"]
        log_dir = root / "poststages" / stage / batch["batch_id"]
        journal_path = log_dir / "batch.json"
        journal = json.loads(journal_path.read_text()) if journal_path.exists() else {}
        reserved = ledger.snapshot().get("batches", {}).get(batch["batch_id"])
        already_started = bool(reserved) or journal.get("status") in {
            "reserved",
            "started",
            "returned",
            "interrupted",
            "blocked",
            "settled",
            "complete_reserved",
        }
        task = (
            None
            if already_started
            else build_batch(
                batch["selections"],
                groups,
                reader,
                max_tokens=max_tokens,
                token_limit=token_limit,
                seed=config["seed"],
            )
        )
        roles = {"grader": model} | ({"interrogator": model} if stage == "interrogation" else {})
        try:
            result = execute(
                task,
                batch_id=batch["batch_id"],
                log_dir=log_dir,
                ledger=ledger,
                upper_usd=batch["upper_usd"],
                stage=stage,
                model=model,
                provider=manifest["spec"]["rates"]["provider"],
                fingerprint=manifest["fingerprint"],
                expected_sample_ids=batch["sample_ids"],
                model_roles=roles,
                max_samples=max_samples,
                resume=resume,
            )
        except BaseException as exc:
            batch.update(status="blocked", error_type=type(exc).__name__)
            manifest.update(status="blocked", blocked_batch=batch["batch_id"])
            campaign.atomic_json(path, manifest)
            raise
        batch.update(
            status=result["status"],
            log_paths=result.get("log_paths", []),
            sample_status_counts=result.get("sample_status_counts", {}),
        )
        if result["status"] not in COMPLETE_BATCH_STATES:
            manifest.update(status=result["status"], blocked_batch=batch["batch_id"])
            campaign.atomic_json(path, manifest)
            return progress_report(manifest)
        for selection in batch["selections"]:
            manifest["progress"][selection["group_id"]] = selection["stop"]
        manifest.pop("blocked_batch", None)
        campaign.atomic_json(path, manifest)
        campaign.emit(
            "poststage_progress",
            **progress_report(manifest),
            budget=campaign.budget_summary(ledger.snapshot()),
        )
