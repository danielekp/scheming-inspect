"""Standalone runner wiring; all model and reporting calls are mocked."""

import json
from types import SimpleNamespace

import httpx
import pytest

from scheming_inspect import run


@pytest.fixture
def execution(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setattr(run, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(
        httpx, "get", lambda *_a, **_k: pytest.fail("Unexpected pricing/network call")
    )
    monkeypatch.setattr(run, "scheming", lambda **kwargs: kwargs)
    captured = {"reports": []}

    def fake_eval(tasks, **kwargs):
        captured.update(kwargs, tasks=tasks)
        return [SimpleNamespace(status="success", samples=[])]

    def fake_report(directory):
        captured["reports"].append(directory)
        return {"status": "unavailable", "key_total_usd": None}

    monkeypatch.setattr(run, "eval", fake_eval)
    monkeypatch.setattr(run, "report_key_usage", fake_report)
    return captured


def test_execution_preserves_experiment_settings_without_financial_gates(
    monkeypatch, tmp_path, execution
):
    monkeypatch.setattr("sys.argv", ["run", "--execute", "--model", "openrouter/test/model"])
    run.main()
    assert len(execution["tasks"]) == 8
    assert execution["model"] == "openrouter/test/model"
    assert execution["max_samples"] == execution["max_tasks"] == 1
    assert execution["cache_prompt"] is False
    assert execution["max_retries"] == execution["retry_on_error"] == 0
    assert execution["log_model_api"] is True
    assert "model_args" not in execution
    assert execution["metadata"] == {"reconstruction": True}
    saved = json.loads((tmp_path / "logs" / "smoke" / "run-plan.json").read_text())
    assert set(saved) == {"arguments", "cells"}
    assert "budget_usd" not in saved["arguments"]
    assert execution["reports"] == [tmp_path / "logs" / "smoke"]


def test_filter_repetitions_seed_and_limits_reach_tasks(monkeypatch, execution, tmp_path):
    log_dir = tmp_path / "custom-logs"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run",
            "--execute",
            "--profile",
            "smoke",
            "--experiment",
            "alignment",
            "--n",
            "3",
            "--seed",
            "17",
            "--max-tokens",
            "123",
            "--max-steps",
            "7",
            "--token-limit",
            "4567",
            "--log-dir",
            str(log_dir),
        ],
    )
    run.main()
    assert len(execution["tasks"]) == 2
    assert {task["condition"] for task in execution["tasks"]} == {"testing", "deployment"}
    for task in execution["tasks"]:
        assert task["n"] == 3
        assert task["seed"] == 17
        assert task["max_tokens"] == 123
        assert task["max_steps"] == 7
        assert task["token_limit"] == 4567
    assert execution["log_dir"] == str(log_dir)


def test_dry_plan_never_calls_models_or_cost_endpoint(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["run", "--profile", "smoke"])
    monkeypatch.setattr(run, "eval", lambda *_a, **_k: pytest.fail("Model call"))
    monkeypatch.setattr(run, "report_key_usage", lambda *_a: pytest.fail("Cost lookup"))
    run.main()
    assert "Plan only" in capsys.readouterr().out


@pytest.mark.parametrize(
    "option,value",
    [
        ("--n", "0"),
        ("--max-tokens", "0"),
        ("--max-steps", "-1"),
        ("--token-limit", "0"),
        ("--experiment", "absent"),
        ("--budget-usd", "5"),
    ],
)
def test_invalid_options_stop_before_dispatch(monkeypatch, option, value):
    monkeypatch.setattr("sys.argv", ["run", "--execute", option, value])
    monkeypatch.setattr(run, "eval", lambda *_a, **_k: pytest.fail("Model call"))
    with pytest.raises(SystemExit) as exc:
        run.main()
    assert exc.value.code == 2


def test_openrouter_still_requires_credentials(monkeypatch, execution):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["run", "--execute"])
    monkeypatch.setattr(run, "eval", lambda *_a, **_k: pytest.fail("Model call"))
    with pytest.raises(SystemExit):
        run.main()


def test_other_inspect_providers_do_not_require_openrouter_key(monkeypatch, execution):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["run", "--execute", "--model", "mockllm/model"])
    run.main()
    assert execution["model"] == "mockllm/model"
    assert len(execution["reports"]) == 1


def test_completed_eval_errors_remain_visible_with_optional_report(monkeypatch, execution):
    monkeypatch.setattr("sys.argv", ["run", "--execute"])
    monkeypatch.setattr(
        run,
        "eval",
        lambda *_a, **_k: [
            SimpleNamespace(
                status="success",
                samples=[SimpleNamespace(error="infrastructure error")],
            )
        ],
    )
    with pytest.raises(SystemExit, match="Run has errors"):
        run.main()
    assert len(execution["reports"]) == 1
