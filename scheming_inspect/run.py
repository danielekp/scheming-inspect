"""Plan first, then run explicitly. OpenRouter runs have a conservative spending bound."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from inspect_ai import eval

from .protocol import PROFILES, plan
from .tasks import ROOT, scheming


def openrouter_bound(model: str, samples: int, token_limit: int, max_tokens: int) -> dict:
    """Reserve cumulative-token budget plus one maximum-context request per sample.

    Inspect checks token limits between calls. No retries are allowed by this runner.
    Catalog prices are enforced as provider rate caps, not merely estimated.
    Explicit prompt caching is disabled; unsupported additional fees fail closed.
    """
    if not model.startswith("openrouter/"):
        raise ValueError(
            "Budgeted runner currently supports OpenRouter models only; use Inspect CLI for other providers."
        )
    response = httpx.get("https://openrouter.ai/api/v1/models", timeout=30)
    response.raise_for_status()
    model_id = model.removeprefix("openrouter/")
    entry = next((m for m in response.json()["data"] if m["id"] == model_id), None)
    if entry is None:
        raise ValueError(f"Model {model_id!r} not found in OpenRouter's current catalog.")
    pricing = entry["pricing"]
    try:
        rates = {name: float(value) for name, value in pricing.items()}
        if any(isinstance(value, bool) for value in pricing.values()):
            raise ValueError("Boolean pricing is invalid.")
        if any(not math.isfinite(rate) or rate < 0 for rate in rates.values()):
            raise ValueError("Prices must be finite and nonnegative.")
        prompt, completion = rates["prompt"], rates["completion"]
        context = int(entry["context_length"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Unusable model pricing/context metadata.") from exc
    token_prices = {"prompt", "completion", "input_cache_read", "input_cache_write"}
    if any(rate != 0 for name, rate in rates.items() if name not in token_prices):
        raise ValueError("This model has pricing components unsupported by the spending bound.")
    price = max(prompt, completion)
    if context < 1:
        raise ValueError("Unusable model pricing/context metadata.")
    upper = samples * (token_limit + context + max_tokens) * price
    provider = {
        "max_price": {
            "prompt": prompt * 1_000_000,
            "completion": completion * 1_000_000,
            "request": 0,
        },
        "require_parameters": True,
    }
    if not math.isfinite(upper) or not all(
        math.isfinite(value) for value in provider["max_price"].values()
    ):
        raise ValueError("Unusable model pricing/context metadata.")
    return {
        "model": model_id,
        "samples": samples,
        "pricing": pricing,
        "context_length": context,
        "conservative_max_usd": upper,
        "provider": provider,
        "cache_prompt": False,
        "budget_policy": "provider-rate-caps-v1",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="smoke")
    parser.add_argument("--model", default="openrouter/deepseek/deepseek-v4-flash")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call the model; otherwise only print the plan",
    )
    parser.add_argument("--budget-usd", type=float, default=5.0)
    parser.add_argument("--log-dir", help="Defaults to logs/PROFILE under the repository")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--token-limit", type=int, default=100000)
    parser.add_argument("--experiment", help="Optional single family filter")
    parser.add_argument(
        "--n",
        type=int,
        help="Override repetitions per cell, preserving both counterfactual branches",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
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
    if (
        not math.isfinite(args.budget_usd)
        or (args.n is not None and args.n < 1)
        or min(args.budget_usd, args.max_tokens, args.max_steps, args.token_limit, samples) <= 0
    ):
        parser.error("Budgets, sample counts and limits must be positive")
    print(json.dumps({"profile": args.profile, "samples": samples, "cells": cells}, indent=2))
    if not args.execute:
        print("Plan only. Add --execute to run. Reconstructed data; no paper results are claimed.")
        return
    load_dotenv(ROOT / ".env", override=False)
    if not os.getenv("OPENROUTER_API_KEY"):
        parser.error("Set OPENROUTER_API_KEY in the environment or repository .env")
    reserve = openrouter_bound(args.model, samples, args.token_limit, args.max_tokens)
    print(json.dumps(reserve, indent=2))
    if reserve["conservative_max_usd"] > args.budget_usd:
        parser.error("Conservative run bound exceeds --budget-usd; reduce sample/limit settings")
    # A visible data directory also works with Snap Docker's filesystem confinement.
    os.environ.setdefault("XDG_DATA_HOME", str(ROOT / "inspect-data"))
    log_dir = Path(args.log_dir or ROOT / "logs" / args.profile).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    source_files = sorted(
        [
            *ROOT.glob("scheming_inspect/*.py"),
            *ROOT.glob("data/*.json"),
            ROOT / "uv.lock",
            ROOT / "Dockerfile",
            ROOT / "compose.yaml",
        ]
    )
    hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    (log_dir / "run-plan.json").write_text(
        json.dumps(
            {
                "arguments": vars(args),
                "cells": cells,
                "reservation": reserve,
                "source_hashes": hashes,
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
        model_args={"provider": reserve["provider"]},
        log_dir=str(log_dir),
        max_samples=1,
        max_tasks=1,
        max_retries=0,
        retry_on_error=0,
        cache_prompt=False,
        log_model_api=True,
        metadata={"reconstruction": True, "spending_reservation": reserve, "source_hashes": hashes},
    )
    if any(log.status != "success" or any(s.error for s in log.samples or []) for log in logs):
        raise SystemExit("Run has errors. Inspect the logs before interpreting results.")


if __name__ == "__main__":
    main()
