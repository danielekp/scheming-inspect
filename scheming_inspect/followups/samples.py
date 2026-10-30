"""Build independent fixed-question samples and matched interrogation pairs."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Sequence

from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser

from ..data import _load
from .sources import SourceCase


@dataclass(frozen=True)
class Action:
    name: str
    description: str
    direct: str
    indirect: str


def _action(name: str) -> Action:
    return Action(name=name, **_load("followups.json")["actions"][name])


OVERSIGHT = _action("oversight_disabled")


REPLACEMENT = _action("successor_replaced")


COPY = _action("artifact_copied")


DELETION = _action("successor_deleted")


PROMPT = _action("prompt_changed")


def eligible_actions(source: SourceCase) -> list[Action]:
    behavior = source.behavior
    if source.experiment == "oversight" and behavior.get("oversight_disabled") is True:
        return [OVERSIGHT]
    if source.experiment == "goal_guarding" and (
        behavior.get("prompt_removed") is True or behavior.get("prompt_modified") is True
    ):
        return [PROMPT]
    if source.experiment == "exfiltration":
        if behavior.get("successor_replaced") is True:
            return [REPLACEMENT]
        return [action for action in (COPY, DELETION) if behavior.get(action.name) is True]
    return []


def fixed_questions(action: Action) -> list[tuple[str, str]]:
    prompts = _load("followups.json")["fixed"]
    result = []
    for direct in (True, False):
        for management in (False, True):
            for honesty in (False, True):
                variant = f"{'direct' if direct else 'indirect'}_management{int(management)}_honesty{int(honesty)}"
                question = action.direct if direct else action.indirect
                if management:
                    question = prompts["management_prefix"] + question
                if honesty:
                    question += prompts["honesty_suffix"]
                result.append((variant, question))
    return result


def source_metadata(source: SourceCase, action: Action) -> dict[str, Any]:
    return {
        "source_key": source.key,
        "source_log": source.log_path,
        "source_sample_id": source.sample_id,
        "source_epoch": source.epoch,
        "source_model": source.model,
        "source_experiment": source.experiment,
        "scenario_id": source.scenario_id,
        "condition": source.condition,
        "action": action.description,
        "action_name": action.name,
        "task_description": source.task_description,
        "source_config": source.matching_context,
    }


def fixed_sample_id(source_key: str, action_name: str, variant: str, repeat: int) -> str:
    return f"{source_key}:{action_name}:{variant}:{repeat}"


def fixed_sample(source: SourceCase, action: Action, variant: str, repeat: int) -> Sample:
    question = dict(fixed_questions(action))[variant]
    return Sample(
        id=fixed_sample_id(source.key, action.name, variant, repeat),
        input=[*copy.deepcopy(source.messages), ChatMessageUser(content=question)],
        metadata=source_metadata(source, action)
        | {
            "experiment": "fixed_followup",
            "variant": variant,
            "repeat": repeat,
            "ground_truth": "guilty",
            "question": question,
        },
    )


def interrogation_sample_id(guilty_key: str, innocent_key: str, truth: str, repeat: int) -> str:
    return f"{guilty_key}:{innocent_key}:{repeat}:{truth}"


def interrogation_sample(
    guilty: SourceCase, innocent: SourceCase, truth: str, repeat: int, seed: int
) -> Sample:
    if truth not in ("guilty", "innocent"):
        raise ValueError("truth must be guilty or innocent.")
    source = guilty if truth == "guilty" else innocent
    return Sample(
        id=interrogation_sample_id(guilty.key, innocent.key, truth, repeat),
        input=list(copy.deepcopy(source.messages)),
        metadata=source_metadata(source, OVERSIGHT)
        | {
            "experiment": "interrogation",
            "match_id": f"{guilty.key}:{innocent.key}",
            "repeat": repeat,
            "ground_truth": truth,
            "selection_seed": seed,
        },
    )


def fixed_samples(
    sources: Sequence[SourceCase], *, repeats: int = 1, limit: int | None = None
) -> list[Sample]:
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    eligible = [(source, actions) for source in sources if (actions := eligible_actions(source))]
    if limit is not None:
        eligible = eligible[:limit]
    samples = []
    for source, actions in eligible:
        for action in actions:
            for variant, _ in fixed_questions(action):
                for repeat in range(repeats):
                    samples.append(fixed_sample(source, action, variant, repeat))
    return samples


def matched_sources(
    sources: Sequence[SourceCase], *, seed: int = 0
) -> list[tuple[SourceCase, SourceCase]]:
    groups: dict[tuple[str, str, str, str], dict[bool, list[SourceCase]]] = {}
    for source in sources:
        if source.experiment != "oversight" or "oversight_disabled" not in source.behavior:
            continue
        key = (source.scenario_id, source.model, source.condition, source.matching_context)
        groups.setdefault(key, {True: [], False: []})[source.behavior["oversight_disabled"]].append(
            source
        )
    rng = random.Random(seed)
    pairs = []
    for key in sorted(groups):
        group = groups[key]
        for cases in group.values():
            cases.sort(key=lambda source: source.key)
            rng.shuffle(cases)
        pairs.extend(zip(group[True], group[False]))
    rng.shuffle(pairs)
    return pairs


def interrogation_samples(
    sources: Sequence[SourceCase],
    *,
    repeats: int = 1,
    limit: int | None = None,
    seed: int = 0,
) -> list[Sample]:
    if repeats < 1:
        raise ValueError("repeats must be positive.")
    pairs = matched_sources(sources, seed=seed)
    if limit is not None:
        pairs = pairs[:limit]
    samples = []
    for guilty, innocent in pairs:
        for repeat in range(repeats):
            for truth in ("guilty", "innocent"):
                samples.append(interrogation_sample(guilty, innocent, truth, repeat, seed))
    random.Random(seed).shuffle(samples)
    return samples
