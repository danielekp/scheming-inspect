"""Read OpenRouter generation costs for logged requests. Never generates model output."""

import argparse
import json
import math
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv
from inspect_ai.log import read_eval_log

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--out", default="results/costs.json")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        parser.error("OPENROUTER_API_KEY is required to read generation metadata")
    paths = set()
    for raw in args.logs:
        path = Path(raw)
        paths.update(path.rglob("*.eval") if path.is_dir() else [path])
    ids, missing, cached = set(), 0, 0
    for path in sorted(paths):
        with path.open("rb") as stream:
            log = read_eval_log(stream)
        for sample in log.samples or []:
            for event in sample.events or []:
                if event.event == "model":
                    if event.cache == "read":
                        cached += 1
                        continue
                    response_id = event.output.response_id
                    if response_id and response_id.startswith("gen-"):
                        ids.add(response_id)
                    else:
                        missing += 1
    entries, errors = [], []
    with httpx.Client(headers={"Authorization": f"Bearer {key}"}, timeout=30) as client:
        for response_id in sorted(ids):
            try:
                response = client.get(
                    "https://openrouter.ai/api/v1/generation", params={"id": response_id}
                )
            except httpx.RequestError as exc:
                errors.append({"id": response_id, "status": type(exc).__name__})
                continue
            if response.status_code != 200:
                errors.append({"id": response_id, "status": response.status_code})
                continue
            try:
                data = response.json()["data"]
                cost = data["total_cost"]
                if (
                    not isinstance(cost, (int, float))
                    or isinstance(cost, bool)
                    or not math.isfinite(cost)
                    or cost < 0
                ):
                    raise ValueError("Invalid cost")
            except (ValueError, KeyError, TypeError):
                errors.append({"id": response_id, "status": "invalid or unavailable cost"})
                continue
            entries.append(
                {
                    name: data.get(name)
                    for name in (
                        "id",
                        "model",
                        "provider_name",
                        "total_cost",
                        "native_tokens_prompt",
                        "native_tokens_completion",
                        "native_tokens_cached",
                        "native_tokens_reasoning",
                    )
                }
            )
    report = {
        "source": "https://openrouter.ai/api/v1/generation",
        "unique_generations": len(ids),
        "verified_generations": len(entries),
        "missing_event_ids": missing,
        "cached_events_excluded": cached,
        "verified_total_usd": sum(item["total_cost"] for item in entries),
        "errors": errors,
        "generations": entries,
        "scope": "Only requests found in supplied logs; duplicated source events counted once.",
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "generations"}, indent=2))
    if errors or missing:
        raise SystemExit("Cost verification is incomplete; inspect the saved report.")


if __name__ == "__main__":
    main()
