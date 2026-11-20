"""Assemble Inspect tasks for fixed follow-ups or interrogation."""

from __future__ import annotations

from typing import Sequence

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig

from .samples import fixed_samples, interrogation_samples
from .scorers import admission_metrics, interrogation_verdict
from .solvers import fixed_followup_solver, interrogation_solver
from .sources import SourceCase


def task_from_samples(
    samples: Sequence[Sample],
    *,
    mode: str,
    max_tokens: int,
    judge_max_tokens: int,
    max_rounds: int = 7,
    time_limit: int = 300,
    token_limit: int | None = None,
    cache_prompt: bool = False,
    metadata: dict | None = None,
) -> Task:
    if mode == "fixed":
        solve = fixed_followup_solver(max_tokens)
        scoring = [admission_metrics(judge_max_tokens)]
    elif mode == "interrogation":
        solve = interrogation_solver(max_rounds, max_tokens, judge_max_tokens)
        scoring = [admission_metrics(judge_max_tokens), interrogation_verdict()]
    else:
        raise ValueError("mode must be fixed or interrogation.")
    if not samples:
        raise ValueError("No eligible sources (or matched guilty/innocent oversight pairs) found.")
    return Task(
        name=f"scheming_{mode}",
        dataset=list(samples),
        solver=solve,
        scorer=scoring,
        config=GenerateConfig(
            max_tokens=max_tokens, cache=False, cache_prompt=cache_prompt, max_retries=0, timeout=90
        ),
        sandbox=None,
        fail_on_error=True,
        time_limit=time_limit,
        token_limit=token_limit,
        metadata={
            "protocol": "paper-inspired followup",
            "mode": mode,
            "max_rounds": max_rounds,
            "judge_max_tokens": judge_max_tokens,
        }
        | (metadata or {}),
    )


def build_task(
    sources: Sequence[SourceCase],
    *,
    mode: str,
    repeats: int = 1,
    limit: int | None = 1,
    seed: int = 0,
    max_rounds: int = 7,
    max_tokens: int = 2048,
    judge_max_tokens: int = 2048,
    time_limit: int = 300,
) -> Task:
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive.")
    if mode == "fixed":
        samples = fixed_samples(sources, repeats=repeats, limit=limit)
    elif mode == "interrogation":
        samples = interrogation_samples(sources, repeats=repeats, limit=limit, seed=seed)
    else:
        raise ValueError("mode must be fixed or interrogation.")
    return task_from_samples(
        samples,
        mode=mode,
        max_tokens=max_tokens,
        judge_max_tokens=judge_max_tokens,
        max_rounds=max_rounds,
        time_limit=time_limit,
        metadata={"repeats": repeats, "selection_seed": seed},
    )
