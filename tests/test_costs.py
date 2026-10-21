import json
from types import SimpleNamespace

import httpx
import pytest

from scheming_inspect import costs


def setup_cli(monkeypatch, tmp_path, events, transport):
    source = tmp_path / "run.eval"
    source.touch()
    destination = tmp_path / "costs.json"
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setattr("sys.argv", ["costs", str(source), "--out", str(destination)])
    monkeypatch.setattr(
        costs, "read_eval_log", lambda _: SimpleNamespace(samples=[SimpleNamespace(events=events)])
    )
    client = httpx.Client(transport=httpx.MockTransport(transport))
    monkeypatch.setattr(costs.httpx, "Client", lambda **_: client)
    return destination


def event(response_id, cache=None):
    return SimpleNamespace(
        event="model", output=SimpleNamespace(response_id=response_id), cache=cache
    )


def test_costs_deduplicate_source_events_and_skip_cache_reads(monkeypatch, tmp_path):
    seen = []

    def transport(request):
        seen.append(request.url.params["id"])
        return httpx.Response(200, json={"data": {"id": "gen-paid", "total_cost": 0.01}})

    destination = setup_cli(
        monkeypatch,
        tmp_path,
        [event("gen-paid"), event("gen-paid"), event("gen-cached", "read")],
        transport,
    )
    costs.main()
    report = json.loads(destination.read_text())
    assert seen == ["gen-paid"]
    assert report["verified_total_usd"] == 0.01
    assert report["cached_events_excluded"] == 1


@pytest.mark.parametrize("failure", ["timeout", "malformed", "null", "http"])
def test_partial_cost_report_survives_metadata_failures(monkeypatch, tmp_path, failure):
    def transport(request):
        if request.url.params["id"] == "gen-good":
            return httpx.Response(200, json={"data": {"id": "gen-good", "total_cost": 0.02}})
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        if failure == "malformed":
            return httpx.Response(200, text="not JSON")
        if failure == "null":
            return httpx.Response(200, json={"data": {"total_cost": None}})
        return httpx.Response(404)

    destination = setup_cli(monkeypatch, tmp_path, [event("gen-good"), event("gen-bad")], transport)
    with pytest.raises(SystemExit, match="incomplete"):
        costs.main()
    report = json.loads(destination.read_text())
    assert report["verified_total_usd"] == 0.02
    assert report["verified_generations"] == 1
    assert len(report["errors"]) == 1
