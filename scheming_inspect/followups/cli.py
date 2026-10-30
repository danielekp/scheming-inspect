"""Opt-in text-only follow-ups to recorded covert-action evaluations.

Interrogation matches guilty/innocent oversight cases within scenario, model and
condition, then randomizes their order with a recorded seed. The interrogator
sees the original user task and interview replies, never source tools, actions,
truth labels, or the turn budget. Each round's verdict is requested on a separate
fork so verdict requests cannot steer later questions. Stop at the first valid
admission annotation, at most seven rounds. False confessions are retained.

All target context retains native message roles (including historical tools),
but no tool can run: calls use tools=[], and there is no sandbox. Providers that
cannot replay their own recorded tool/reasoning messages fail visibly; we never
flatten the conversation or silently switch models. Provider configuration and
model versions may still differ from the original run and must be reported.

Rates are conditional on observed source actions in fixed mode. Interrogation
rates must be separated by ground_truth (guilty/innocent); repeated interviews of
one source are not independent source behaviors. Inspect logs retain source IDs,
matching IDs, every turn, judge failures and verdicts for this analysis."""

from __future__ import annotations

import argparse
from typing import Sequence

from inspect_ai import eval as inspect_eval

from .sources import load_sources
from .tasks import build_task


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mode", required=True, choices=("fixed", "interrogation"))
    parser.add_argument(
        "--logs", required=True, nargs="+", help="Source .eval/.json logs or directories."
    )
    parser.add_argument(
        "--model", required=True, help="Explicit target model, exactly matching source logs."
    )
    parser.add_argument("--judge", required=True, help="Explicit semantic admission grader model.")
    parser.add_argument("--interrogator", help="Required only for interrogation mode.")
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Maximum source cases (fixed) or matched pairs (interrogation).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Independent repetitions; paper fixed protocol uses 20.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Source pairing/order seed, not a promise of deterministic models.",
    )
    parser.add_argument("--max-rounds", type=int, default=7, help="Interrogation rounds, 1..7.")
    parser.add_argument(
        "--max-tokens", type=int, default=2048, help="Per target/interrogator generation cap."
    )
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=1, help="Concurrent samples.")
    parser.add_argument(
        "--time-limit", type=int, default=300, help="Per-sample wall time in seconds."
    )
    parser.add_argument("--log-dir", default="logs/followups")
    args = parser.parse_args(argv)
    if args.mode == "interrogation" and not args.interrogator:
        parser.error("--interrogator is required for interrogation mode.")
    for name in ("limit", "repeats", "max_tokens", "judge_max_tokens", "max_samples", "time_limit"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if not 1 <= args.max_rounds <= 7:
        parser.error("--max-rounds must be in 1..7.")
    sources = load_sources(args.logs, args.model)
    task = build_task(
        sources,
        mode=args.mode,
        repeats=args.repeats,
        limit=args.limit,
        seed=args.seed,
        max_rounds=args.max_rounds,
        max_tokens=args.max_tokens,
        judge_max_tokens=args.judge_max_tokens,
        time_limit=args.time_limit,
    )
    roles = {"grader": args.judge}
    if args.interrogator:
        roles["interrogator"] = args.interrogator
    n = len(task.dataset)
    per_sample = 2 if args.mode == "fixed" else 4 * args.max_rounds
    print(
        f"Prepared {n} samples; at most {n * per_sample} logical model generations (provider retries excluded)."
    )
    logs = inspect_eval(
        task,
        model=args.model,
        model_roles=roles,
        log_dir=args.log_dir,
        max_samples=args.max_samples,
    )
    return 0 if all(log.status == "success" for log in logs) else 1
