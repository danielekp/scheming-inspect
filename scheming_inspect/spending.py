"""Optional key-wide spending observations; never an experiment execution gate."""

from __future__ import annotations

import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

KEY_URL = "https://openrouter.ai/api/v1/key"


def report_cost(directory: Path) -> dict[str, Any]:
    """Observe cumulative API-key usage once, with nonfatal local persistence.

    ``key_total_usd`` is OpenRouter's total usage for this key, including other
    activity and earlier runs. It is not this campaign's exact cost. A missing
    key, unavailable endpoint, invalid response, or failed write never stops work.
    """
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
            # httpx's default transport does not retry. Never follow redirects
            # with credentials, and never inspect or persist raw error bodies.
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
