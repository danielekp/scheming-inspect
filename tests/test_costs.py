"""Offline checks for generation-cost reports and optional API-key usage."""

import json
from types import SimpleNamespace

import httpx
import pytest

from scheming_inspect import campaign, costs


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


def mock_key_transport(monkeypatch, handler):
    original_client = httpx.Client
    options = []

    def client(**kwargs):
        options.append(kwargs)
        return original_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(costs.httpx, "Client", client)
    return options


def test_key_total_is_labeled_and_saved_after_one_bounded_get(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer secret-test-key"
        return httpx.Response(200, json={"data": {"usage": 12.5, "private": "omit"}})

    options = mock_key_transport(monkeypatch, handler)
    result = costs.report_key_usage(tmp_path)
    assert result["status"] == "available"
    assert result["key_total_usd"] == 12.5
    assert result["scope"] == "openrouter_api_key_total"
    assert result["checked_at"]
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert str(requests[0].url) == costs.KEY_URL
    assert options == [{"timeout": 3, "follow_redirects": False}]
    saved = (tmp_path / "spending.json").read_text()
    assert json.loads(saved) == result
    assert "secret" not in saved and "private" not in saved


def test_missing_key_is_nonfatal_and_does_not_call_network(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(costs.httpx, "Client", lambda **_: pytest.fail("network"))
    result = costs.report_key_usage(tmp_path)
    assert result["status"] == "unavailable"
    assert result["key_total_usd"] is None
    assert result["reason"] == "missing_api_key"
    assert json.loads((tmp_path / "spending.json").read_text()) == result


@pytest.mark.parametrize(
    "body",
    [
        {},
        [],
        {"data": None},
        {"data": {"usage": None}},
        {"data": {"usage": True}},
        {"data": {"usage": "1.0"}},
        {"data": {"usage": -1}},
    ],
)
def test_malformed_usage_is_unavailable_without_raising(monkeypatch, tmp_path, body):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    mock_key_transport(monkeypatch, lambda _: httpx.Response(200, json=body))
    result = costs.report_key_usage(tmp_path)
    assert result["status"] == "unavailable"
    assert result["key_total_usd"] is None


@pytest.mark.parametrize("failure", ["timeout", "http", "json", "nonfinite"])
def test_network_and_response_failures_never_retry_or_expose_secrets(
    monkeypatch, tmp_path, failure
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    requests = []

    def handler(request):
        requests.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("secret raw response", request=request)
        if failure == "http":
            return httpx.Response(401, text="secret raw response")
        if failure == "json":
            return httpx.Response(200, text="secret malformed json")
        return httpx.Response(200, text='{"data":{"usage":Infinity}}')

    mock_key_transport(monkeypatch, handler)
    result = costs.report_key_usage(tmp_path)
    assert result["status"] == "unavailable"
    assert len(requests) == 1
    assert "secret" not in json.dumps(result)
    assert "secret" not in (tmp_path / "spending.json").read_text()


def test_write_failure_does_not_hide_valid_observation_or_raise(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    mock_key_transport(monkeypatch, lambda _: httpx.Response(200, json={"data": {"usage": 0}}))

    def fail_write(*_):
        raise OSError("secret filesystem detail")

    monkeypatch.setattr(campaign, "atomic_json", fail_write)
    result = costs.report_key_usage(tmp_path)
    assert result["status"] == "available"
    assert result["key_total_usd"] == 0
    assert result["save_status"] == "unavailable"
    assert "secret" not in json.dumps(result)


def test_reporter_does_not_swallow_user_interrupt(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    def interrupted(_):
        raise KeyboardInterrupt

    mock_key_transport(monkeypatch, interrupted)
    with pytest.raises(KeyboardInterrupt):
        costs.report_key_usage(tmp_path)


def test_log_cost_helper_returns_and_saves_partial_report_without_printing_or_exit(
    monkeypatch, tmp_path, capsys
):
    source = tmp_path / "source.eval"
    source.touch()
    destination = tmp_path / "nested" / "costs.json"
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(
        costs,
        "read_eval_log",
        lambda _: SimpleNamespace(
            samples=[SimpleNamespace(events=[event("gen-good"), event("gen-bad")])]
        ),
    )

    def handler(request):
        assert request.headers["Authorization"] == "Bearer explicit-test-key"
        if request.url.params["id"] == "gen-good":
            return httpx.Response(200, json={"data": {"id": "gen-good", "total_cost": 0.02}})
        return httpx.Response(503)

    original_client = httpx.Client
    monkeypatch.setattr(
        costs.httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    report = costs.report_log_costs([source], destination, api_key="explicit-test-key")
    assert report == json.loads(destination.read_text())
    assert report["unique_generations"] == 2 and report["verified_generations"] == 1
    assert report["verified_total_usd"] == 0.02
    assert report["errors"] == [{"id": "gen-bad", "status": 503}]
    assert capsys.readouterr().out == ""
