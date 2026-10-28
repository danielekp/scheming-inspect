"""Report OpenRouter key usage and logged request costs without enforcing budget caps."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import httpx
from dotenv import load_dotenv
from inspect_ai.log import read_eval_log

ROOT = Path(__file__).resolve().parents[1]
KEY_URL = "https://openrouter.ai/api/v1/key"


def report_key_usage(directory: Path) -> dict[str, Any]:
    """Observe cumulative API-key usage once, with nonfatal local persistence."""
    report: dict[str, Any] = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "status": "unavailable",
        "scope": "openrouter_api_key_total",
        "key_total_usd": None,
    }
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        report["reason"] = "missing_api_key"
    else:
        try:
            with httpx.Client(timeout=3, follow_redirects=False) as client:
                response = client.get(KEY_URL, headers={"Authorization": f"Bearer {api_key}"})
                response.raise_for_status()
                body = response.json()
            data = body.get("data") if isinstance(body, dict) else None
            usage = data.get("usage") if isinstance(data, dict) else None
            if (
                isinstance(usage, bool)
                or not isinstance(usage, (int, float))
                or not math.isfinite(usage)
                or usage < 0
            ):
                report["reason"] = "invalid_usage"
            else:
                report.update(status="available", key_total_usd=usage)
        except Exception:
            report["reason"] = "request_failed"
    try:
        # Imported lazily so campaign can import the reporter without a cycle.
        from .campaign import atomic_json

        atomic_json(Path(directory) / "spending.json", report)
    except Exception:
        report["save_status"] = "unavailable"
    return report


def report_log_costs(
    logs: Sequence[str | Path], destination: str | Path, *, api_key: str
) -> dict[str, Any]:
    """Save and return charges for logged requests, retaining partial results."""
    paths = set()
    for raw in logs:
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
    with httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=30) as client:
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
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    return report


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--out", default="results/costs.json")
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        parser.error("OPENROUTER_API_KEY is required to read generation metadata")
    report = report_log_costs(args.logs, args.out, api_key=key)
    print(json.dumps({k: v for k, v in report.items() if k != "generations"}, indent=2))
    if report["errors"] or report["missing_event_ids"]:
        raise SystemExit("Cost verification is incomplete; inspect the saved report.")


if __name__ == "__main__":
    main()
