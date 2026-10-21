import json
from types import SimpleNamespace

import pytest
from inspect_ai.model import GenerateConfig, get_model

from scheming_inspect import run


def catalog(pricing=None):
    return SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: {
            "data": [
                {
                    "id": "test/model",
                    "context_length": 1000,
                    "pricing": pricing or {"prompt": "0.000001", "completion": "0.000002"},
                }
            ]
        },
    )


def test_spending_bound_reserves_overshoot(monkeypatch):
    monkeypatch.setattr(run.httpx, "get", lambda *a, **kw: catalog())
    quote = run.openrouter_bound("openrouter/test/model", 3, 10000, 100)
    assert quote["conservative_max_usd"] >= 3 * (10000 + 1000 + 100) * 0.000002
    assert quote["provider"] == {
        "max_price": {"prompt": 1.0, "completion": 2.0, "request": 0},
        "require_parameters": True,
    }
    assert quote["cache_prompt"] is False
    assert quote["budget_policy"] == "provider-rate-caps-v1"


@pytest.mark.parametrize(
    "fee", ["request", "image", "web_search", "internal_reasoning", "future_fee"]
)
def test_extra_cost_components_fail_closed(monkeypatch, fee):
    monkeypatch.setattr(
        run.httpx,
        "get",
        lambda *a, **kw: catalog({"prompt": "0", "completion": "0", fee: "1"}),
    )
    with pytest.raises(ValueError, match="pricing components"):
        run.openrouter_bound("openrouter/test/model", 1, 100, 100)


@pytest.mark.parametrize("field", ["prompt", "completion"])
@pytest.mark.parametrize("bad_rate", ["-0.001", "NaN", "Infinity", "-Infinity", None, True])
def test_each_token_rate_must_be_finite_and_nonnegative(monkeypatch, field, bad_rate):
    pricing = {"prompt": "0.000001", "completion": "0.000002", field: bad_rate}
    monkeypatch.setattr(run.httpx, "get", lambda *a, **kw: catalog(pricing))
    with pytest.raises(ValueError, match="pricing/context"):
        run.openrouter_bound("openrouter/test/model", 1, 100, 100)


def test_free_rates_still_enforce_zero_caps(monkeypatch):
    monkeypatch.setattr(
        run.httpx, "get", lambda *a, **kw: catalog({"prompt": "0", "completion": "0"})
    )
    quote = run.openrouter_bound("openrouter/test/model", 1, 100, 100)
    assert quote["conservative_max_usd"] == 0
    assert quote["provider"]["max_price"] == {"prompt": 0, "completion": 0, "request": 0}


def test_missing_token_rate_fails_closed(monkeypatch):
    monkeypatch.setattr(run.httpx, "get", lambda *a, **kw: catalog({"completion": "0.000002"}))
    with pytest.raises(ValueError, match="pricing/context"):
        run.openrouter_bound("openrouter/test/model", 1, 100, 100)


def test_inspect_openrouter_adapter_forwards_cap_without_generation(monkeypatch):
    monkeypatch.setattr(run.httpx, "get", lambda *a, **kw: catalog())
    provider = run.openrouter_bound("openrouter/test/model", 1, 100, 100)["provider"]
    # Construct the installed adapter and inspect its outgoing parameters only.
    # The dummy key is not usable; no model.generate or HTTP request is made.
    model = get_model(
        "openrouter/test/model",
        api_key="test-only-not-a-real-key",
        provider=provider,
        memoize=False,
    )
    params = model.api.completion_params(
        GenerateConfig(max_tokens=100, cache_prompt=False), tools=True
    )
    assert params["extra_body"]["provider"] == provider


def test_execution_forwards_and_records_provider_caps(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setattr(run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    for name in ("uv.lock", "Dockerfile", "compose.yaml"):
        (tmp_path / name).write_text("offline fixture", encoding="utf-8")
    monkeypatch.setattr(run.httpx, "get", lambda *a, **kw: catalog())
    monkeypatch.setattr(run, "scheming", lambda **kwargs: kwargs)
    captured = {}

    def fake_eval(tasks, **kwargs):
        captured.update(kwargs)
        assert len(tasks) == 8
        return [SimpleNamespace(status="success", samples=[])]

    monkeypatch.setattr(run, "eval", fake_eval)
    monkeypatch.setattr("sys.argv", ["run", "--execute", "--model", "openrouter/test/model"])
    run.main()
    expected = {
        "max_price": {"prompt": 1.0, "completion": 2.0, "request": 0},
        "require_parameters": True,
    }
    assert captured["model_args"] == {"provider": expected}
    assert captured["cache_prompt"] is False
    assert captured["max_retries"] == captured["retry_on_error"] == 0
    saved = json.loads((tmp_path / "logs" / "smoke" / "run-plan.json").read_text())
    assert saved["reservation"]["provider"] == expected
    assert saved["reservation"]["budget_policy"] == "provider-rate-caps-v1"
    assert captured["metadata"]["spending_reservation"]["provider"] == expected


def test_dry_plan_cannot_call_provider(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["run", "--profile", "smoke"])
    monkeypatch.setattr(run.httpx, "get", lambda *a, **kw: pytest.fail("Network used for dry plan"))
    monkeypatch.setattr(run, "eval", lambda *a, **kw: pytest.fail("Models run for dry plan"))
    run.main()
    assert "Plan only" in capsys.readouterr().out


def test_budget_rejection_precedes_model_calls(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setattr(
        "sys.argv",
        ["run", "--execute", "--model", "openrouter/test/model", "--budget-usd", ".00001"],
    )
    monkeypatch.setattr(run.httpx, "get", lambda *a, **kw: catalog())
    monkeypatch.setattr(run, "eval", lambda *a, **kw: pytest.fail("Over-budget run executed"))
    with pytest.raises(SystemExit):
        run.main()
