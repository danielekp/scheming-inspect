"""Read eligible source cases from Inspect logs."""

from __future__ import annotations

import copy
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessage

MATCH_KEYS = (
    "seed",
    "nudge",
    "goal_mode",
    "goal_role",
    "difficulty",
    "suppress_reasoning",
    "question_set",
    "consequence",
    "user_prompt",
)


@dataclass(frozen=True)
class SourceCase:
    key: str
    log_path: str
    sample_id: str | int
    epoch: int
    model: str
    experiment: str
    scenario_id: str
    condition: str
    messages: tuple[ChatMessage, ...]
    behavior: dict[str, bool]
    task_description: str
    matching_context: str = ""


def validate_transcript(messages: Sequence[ChatMessage]) -> None:
    """Reject incomplete tool exchanges instead of fabricating missing results."""
    if not messages:
        raise ValueError("Source transcript is empty.")
    pending: set[str] = set()
    for message in messages:
        if message.role == "tool":
            call_id = message.tool_call_id
            if call_id not in pending:
                raise ValueError(f"Source transcript has unmatched tool result {call_id!r}.")
            pending.remove(call_id)
        else:
            if pending:
                raise ValueError("Source transcript has an unfinished tool-call exchange.")
            if message.role == "assistant":
                ids = [call.id for call in (message.tool_calls or [])]
                if len(ids) != len(set(ids)):
                    raise ValueError("Source transcript has duplicate tool-call IDs.")
                pending.update(ids)
    if pending:
        raise ValueError("Source transcript ends with unanswered tool calls.")


def task_description(sample: Any) -> str:
    if description := (sample.metadata or {}).get("task_description"):
        return description
    if isinstance(sample.input, str):
        return sample.input
    return "\n".join(message.text for message in sample.input if message.role == "user")


def read_behavior(sample: Any) -> dict:
    """Read the behavior metadata written by the current base solver."""
    value = (sample.metadata or {}).get("behavior")
    return value if isinstance(value, dict) else {}


def boolean_flags(behavior: dict) -> dict[str, bool]:
    return {
        name: bool(value)
        for name, value in behavior.items()
        if type(value) in (bool, int, float) and value in (0, 1)
    }


def source_key(path: str | Path, sample_id: str | int, epoch: int) -> str:
    return f"{Path(path).expanduser().resolve()}:{sample_id}:{epoch}"


def matching_context(metadata: dict) -> str:
    return json.dumps({key: metadata.get(key) for key in MATCH_KEYS}, sort_keys=True)


def source_exclusion_reason(sample: Any) -> str | None:
    """Use completed, valid base samples; unavailable outcomes are never negatives."""
    behavior = read_behavior(sample)
    if getattr(sample, "error", None):
        return "source_error"
    if getattr(sample, "invalidation", None):
        return "source_invalidated"
    if not behavior:
        return "missing_behavior"
    if getattr(sample, "limit", None) or behavior.get("truncated") == 1:
        return "source_truncated"
    if behavior.get("completed") != 1:
        return "source_incomplete_or_completion_unknown"
    if (sample.metadata or {}).get("experiment") in (
        "oversight",
        "exfiltration",
        "goal_guarding",
        "alignment",
    ):
        if behavior.get("choice_recorded") not in (0, 1):
            return "source_choice_unknown"
    elif behavior.get("schema_valid") != 1:
        return "source_invalid_schema"
    if not (sample.metadata or {}).get("scenario_id"):
        return "missing_scenario"
    try:
        validate_transcript(sample.messages)
    except (ValueError, AttributeError, TypeError):
        return "malformed_transcript"
    return None


def source_from_sample(
    path: str | Path, sample: Any, model: str, *, key: str | None = None
) -> SourceCase:
    reason = source_exclusion_reason(sample)
    if reason:
        raise ValueError(f"Ineligible source {sample.id}: {reason}")
    meta = sample.metadata
    return SourceCase(
        key=key if key is not None else source_key(path, sample.id, sample.epoch),
        log_path=str(Path(path).expanduser().resolve()),
        sample_id=sample.id,
        epoch=sample.epoch,
        model=model,
        experiment=meta["experiment"],
        scenario_id=str(meta["scenario_id"]),
        condition=json.dumps(meta.get("condition", ""), sort_keys=True),
        messages=tuple(copy.deepcopy(sample.messages)),
        behavior=boolean_flags(read_behavior(sample)),
        task_description=task_description(sample),
        matching_context=matching_context(meta),
    )


def load_sources(paths: Sequence[str], model: str) -> list[SourceCase]:
    """Load canonical campaign samples or explicit logs, reporting exclusions."""
    selections: dict[Path, set[str] | None] = {}
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if path.is_dir() and (path / "manifest.json").is_file():
            from ..campaign import base_log_records, read_manifest

            for record in base_log_records(read_manifest(path)):
                filename = Path(record["path"]).resolve()
                ids = {str(value) for value in record["expected_sample_ids"]}
                if selections.get(filename) not in (None, ids):
                    raise ValueError(f"Conflicting campaign sample IDs for source log: {filename}")
                selections[filename] = ids
        elif path.is_dir():
            for filename in path.rglob("*.eval"):
                if "_interrupted" not in filename.parts:
                    selections.setdefault(filename, None)
        elif path.is_file():
            selections.setdefault(path, None)
        else:
            raise ValueError(f"Source log path does not exist: {path}")
    if not selections:
        raise ValueError("No .eval or .json source logs found.")
    result = []
    for path, expected in sorted(selections.items()):
        with path.open("rb") as handle:
            log = read_eval_log(handle, format="json" if path.suffix == ".json" else "eval")
        if str(log.eval.model) != model:
            raise ValueError(
                f"Target model {model!r} differs from source model {log.eval.model!r}: {path}"
            )
        if expected is not None:
            ids = [str(sample.id) for sample in log.samples or []]
            if len(ids) != len(set(ids)) or set(ids) != expected:
                raise ValueError(f"Canonical source log does not match its complete batch: {path}")
            if any(sample.epoch != 1 for sample in log.samples or []):
                raise ValueError(f"Canonical source log has an unexpected epoch: {path}")
        for sample in log.samples or []:
            if (sample.metadata or {}).get("experiment") not in {
                "oversight",
                "exfiltration",
                "goal_guarding",
            }:
                continue
            reason = source_exclusion_reason(sample)
            if reason:
                warnings.warn(f"Excluding {path.name}:{sample.id}:{sample.epoch}: {reason}")
                continue
            result.append(source_from_sample(path, sample, model))
    return result
