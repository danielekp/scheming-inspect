"""Plan first, then explicitly run Inspect evaluations without spending caps."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import eval

from .costs import report_key_usage
from .protocol import PROFILES, plan
from .tasks import ROOT, scheming


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--model", default="openrouter/deepseek/deepseek-v4-flash")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call the model; otherwise only print the plan",
    )
    parser.add_argument("--log-dir", help="Defaults to logs/PROFILE under the repository")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--token-limit", type=int, default=400000)
    parser.add_argument("--experiment", help="Optional single family filter")
    parser.add_argument(
        "--n",
        type=int,
        help="Override repetitions per cell, preserving both counterfactual branches",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    cells = plan(args.profile)
    if args.experiment:
        cells = [c for c in cells if c["experiment"] == args.experiment]
    if not cells:
        parser.error("No matching experimental cells")
    for cell in cells:
        if args.n is not None:
            cell["n"] = args.n
        cell["seed"] = args.seed
    samples = sum(c["n"] for c in cells)
    if (args.n is not None and args.n < 1) or min(
        args.max_tokens, args.max_steps, args.token_limit, samples
    ) <= 0:
        parser.error("Sample counts and limits must be positive")
    print(json.dumps({"profile": args.profile, "samples": samples, "cells": cells}, indent=2))
    if not args.execute:
        print("Plan only. Add --execute to run. Reconstructed data; no paper results are claimed.")
        return
    load_dotenv(ROOT / ".env", override=False)
    if args.model.startswith("openrouter/") and not os.getenv("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY in the environment or repository .env")
    # A visible data directory also works with Snap Docker's filesystem confinement.
    os.environ.setdefault("XDG_DATA_HOME", str(ROOT / "inspect-data"))
    log_dir = Path(args.log_dir or ROOT / "logs" / args.profile).expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "run-plan.json").write_text(
        json.dumps(
            {
                "arguments": vars(args),
                "cells": cells,
            },
            indent=2,
        )
    )
    tasks = [
        scheming(
            **cell,
            max_steps=args.max_steps,
            max_tokens=args.max_tokens,
            token_limit=args.token_limit,
        )
        for cell in cells
    ]
    logs = eval(
        tasks,
        model=args.model,
        log_dir=str(log_dir),
        max_samples=1,
        max_tasks=1,
        max_retries=0,
        retry_on_error=0,
        cache_prompt=False,
        log_model_api=True,
        metadata={"reconstruction": True},
    )
    print(json.dumps({"spending": report_key_usage(log_dir)}))
    if any(log.status != "success" or any(s.error for s in log.samples or []) for log in logs):
        raise SystemExit("Run has errors. Inspect the logs before interpreting results.")


if __name__ == "__main__":
    main()
