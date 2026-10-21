"""Offline tests for optional key-wide spending observations."""

import json

import httpx
import pytest

from scheming_inspect import campaign, spending


def transport(monkeypatch, handler):
    original_client = httpx.Client
    options = []

    def client(**kwargs):
        options.append(kwargs)
        return original_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(spending.httpx, "Client", client)
    return options


def test_key_total_is_labeled_and_saved_after_one_bounded_get(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer secret-test-key"
        return httpx.Response(200, json={"data": {"usage": 12.5, "private": "omit"}})

    options = transport(monkeypatch, handler)
    result = spending.report_cost(tmp_path)
    assert result["status"] == "available"
    assert result["key_total_usd"] == 12.5
    assert result["scope"] == "openrouter_api_key_total"
    assert result["checked_at"]
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert str(requests[0].url) == spending.KEY_URL
    assert options == [{"timeout": 3, "follow_redirects": False}]
    saved = (tmp_path / "spending.json").read_text()
    assert json.loads(saved) == result
    assert "secret" not in saved and "private" not in saved


def test_missing_key_is_nonfatal_and_does_not_call_network(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(spending.httpx, "Client", lambda **_: pytest.fail("network"))
    result = spending.report_cost(tmp_path)
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
    transport(monkeypatch, lambda _: httpx.Response(200, json=body))
    result = spending.report_cost(tmp_path)
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

    transport(monkeypatch, handler)
    result = spending.report_cost(tmp_path)
    assert result["status"] == "unavailable"
    assert len(requests) == 1
    assert "secret" not in json.dumps(result)
    assert "secret" not in (tmp_path / "spending.json").read_text()


def test_write_failure_does_not_hide_valid_observation_or_raise(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    transport(monkeypatch, lambda _: httpx.Response(200, json={"data": {"usage": 0}}))

    def fail_write(*_):
        raise OSError("secret filesystem detail")

    monkeypatch.setattr(campaign, "atomic_json", fail_write)
    result = spending.report_cost(tmp_path)
    assert result["status"] == "available"
    assert result["key_total_usd"] == 0
    assert result["save_status"] == "unavailable"
    assert "secret" not in json.dumps(result)


def test_reporter_does_not_swallow_user_interrupt(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")

    def interrupted(_):
        raise KeyboardInterrupt

    transport(monkeypatch, interrupted)
    with pytest.raises(KeyboardInterrupt):
        spending.report_cost(tmp_path)
