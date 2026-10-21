"""Offline accounting invariants; no test invokes a model or external service."""

import json
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest
from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, write_eval_log
from inspect_ai.model import GenerateConfig, ModelOutput

from scheming_inspect import budget
from scheming_inspect.budget import (
    BatchAlreadyExists,
    BillingError,
    BudgetError,
    BudgetExceeded,
    CampaignLedger,
)


def event(response_id="gen-paid", **kwargs):
    return SimpleNamespace(event="model", output=SimpleNamespace(response_id=response_id), **kwargs)


@pytest.fixture
def logs(monkeypatch, tmp_path):
    """JSON-tagged fake logs exercise accounting independently of Inspect parsing."""
    records = {}

    def make(events=None, *, status="success", error=None, usage=None, retries=None):
        name = f"log-{len(records)}"
        path = tmp_path / f"{name}.eval"
        path.write_text(json.dumps({"record": name}))
        records[name] = SimpleNamespace(
            status=status,
            samples=[
                SimpleNamespace(
                    events=events if events is not None else [event()],
                    error=error,
                    invalidation=None,
                    model_usage=usage or {},
                    error_retries=retries,
                )
            ],
        )
        return path

    monkeypatch.setattr(
        budget, "read_eval_log", lambda stream, **_: records[json.load(stream)["record"]]
    )
    return make


def lookup_cost(cost=0.25, seen=None):
    def lookup(response_id):
        if seen is not None:
            seen.append(response_id)
        return {"id": response_id, "total_cost": cost}

    return lookup


def test_reserve_survives_restart_and_cannot_run_twice(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = CampaignLedger(path, starting_spend_usd="0.013229117")
    ledger.reserve("base-1", 3, metadata={"sample_ids": [1, 2]})
    resumed = CampaignLedger(path, starting_spend_usd="0.013229117")
    assert resumed.pending()[0]["status"] == "reserved"
    assert resumed.snapshot()["committed_usd"] == pytest.approx(3.013229117)
    with pytest.raises(BatchAlreadyExists):
        resumed.reserve("base-1", 3)
    with pytest.raises(BudgetError, match="unresolved"):
        resumed.reserve("base-2", 1)


def test_settlement_releases_only_verified_difference_and_obeys_lifetime_cap(tmp_path, logs):
    ledger = CampaignLedger(tmp_path / "ledger.json", limit_usd=5, starting_spend_usd=1)
    ledger.reserve("one", 4)
    settled = ledger.settle("one", [logs()], complete=True, lookup=lookup_cost(2))
    assert settled["status"] == "settled"
    assert settled["actual_usd"] == 2
    assert ledger.snapshot()["remaining_usd"] == 2
    with pytest.raises(BudgetExceeded, match="lifetime"):
        ledger.reserve("too-large", "2.000000001")
    ledger.reserve("two", 2)
    assert ledger.snapshot()["remaining_usd"] == 0


def test_stage_limit_and_shared_global_limit(tmp_path, logs):
    ledger = CampaignLedger(tmp_path / "ledger.json", limit_usd=5, stage_limits={"base": 3})
    with pytest.raises(BudgetExceeded, match="stage"):
        ledger.reserve("base-too-large", 4)
    ledger.reserve("base", 3)
    ledger.settle("base", [logs()], complete=True, lookup=lookup_cost(2))
    with pytest.raises(BudgetExceeded, match="stage"):
        ledger.reserve("base-more", 2)
    ledger.reserve("audit", 3, stage="audit")
    assert ledger.snapshot()["committed_usd"] == 5
    assert ledger.snapshot()["stages"]["base"]["remaining_usd"] == 1


@pytest.mark.parametrize("amount", [float("nan"), float("inf"), -1, True, None, "oops", 0])
def test_invalid_reservation_is_never_written(tmp_path, amount):
    ledger = CampaignLedger(tmp_path / "ledger.json")
    with pytest.raises(BudgetError):
        ledger.reserve("invalid", amount)
    assert ledger.snapshot()["batches"] == {}


def test_existing_config_cannot_be_widened_or_reset(tmp_path):
    path = tmp_path / "ledger.json"
    CampaignLedger(path, limit_usd=5, starting_spend_usd=1, stage_limits={"base": 3})
    with pytest.raises(BudgetError, match="configuration"):
        CampaignLedger(path, limit_usd=50, starting_spend_usd=1, stage_limits={"base": 3})
    with pytest.raises(BudgetError, match="configuration"):
        CampaignLedger(path, limit_usd=5, starting_spend_usd=0, stage_limits={"base": 3})


@pytest.mark.parametrize("damage", ["missing", "bad-json", "bad-amount", "missing-batches"])
def test_corrupt_or_removed_ledger_never_silently_resets(tmp_path, damage):
    path = tmp_path / "ledger.json"
    CampaignLedger(path).reserve("paid-or-inflight", 5)
    if damage == "missing":
        path.unlink()
    elif damage == "bad-json":
        path.write_text("{")
    else:
        state = json.loads(path.read_text())
        if damage == "bad-amount":
            state["batches"]["paid-or-inflight"]["upper_usd"] = "NaN"
        else:
            del state["batches"]
        path.write_text(json.dumps(state))
    with pytest.raises(BudgetError):
        CampaignLedger(path)


def test_atomic_reservation_failure_preserves_original_ledger(monkeypatch, tmp_path):
    ledger = CampaignLedger(tmp_path / "ledger.json")
    original = ledger.path.read_bytes()

    def fail_replace(*_):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(budget.os, "replace", fail_replace)
    with pytest.raises(OSError):
        ledger.reserve("never-launched", 10)
    assert ledger.path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_concurrent_process_style_instances_cannot_double_reserve(tmp_path):
    path = tmp_path / "ledger.json"
    a, b = CampaignLedger(path, limit_usd=5), CampaignLedger(path, limit_usd=5)

    def reserve(args):
        ledger, name = args
        try:
            ledger.reserve(name, 4)
            return "reserved"
        except BudgetError:
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(reserve, [(a, "a"), (b, "b")]))
    assert sorted(results) == ["blocked", "reserved"]
    assert a.snapshot()["committed_usd"] == 4
    assert len(a.snapshot()["batches"]) == 1


def test_incomplete_batch_keeps_full_reserve_even_when_visible_ids_are_billed(tmp_path, logs):
    ledger = CampaignLedger(tmp_path / "ledger.json")
    ledger.reserve("crashed", 5)
    path = logs()
    seen = []
    partial = ledger.settle("crashed", [path], lookup=lookup_cost(0.1, seen))
    assert partial["status"] == "blocked"
    assert partial["actual_usd"] is None
    assert partial["verified_usd"] == 0.1
    assert partial["committed_usd"] == 5
    assert seen == ["gen-paid"]
    resumed = CampaignLedger(ledger.path)
    finished = resumed.settle("crashed", [path], complete=True, lookup=lookup_cost(99, seen))
    assert finished["status"] == "settled"
    assert finished["actual_usd"] == 0.1
    assert seen == ["gen-paid"]  # Billing IDs are durable across interruption.


@pytest.mark.parametrize(
    "kind",
    [
        "missing-id",
        "error",
        "pending",
        "retry",
        "sample-retry",
        "cancelled",
        "started",
        "usage-only",
        "unreadable",
        "empty",
    ],
)
def test_unaccounted_requests_keep_full_reservation(tmp_path, logs, kind):
    ledger = CampaignLedger(tmp_path / "ledger.json")
    ledger.reserve("uncertain", 5)
    if kind == "missing-id":
        paths = [logs([event(None)])]
    elif kind in {"error", "pending", "retry"}:
        attrs = (
            {"error": "provider failed"}
            if kind == "error"
            else {"pending": True}
            if kind == "pending"
            else {"retries": 1}
        )
        paths = [logs([event(**attrs)])]
    elif kind == "sample-retry":
        paths = [logs(retries=["previous failed attempt"])]
    elif kind in {"cancelled", "started"}:
        paths = [logs(status=kind)]
    elif kind == "usage-only":
        paths = [logs([], usage={"model": {"input_tokens": 10}})]
    elif kind == "unreadable":
        paths = [tmp_path / "missing.eval"]
    else:
        paths = []
    result = ledger.settle("uncertain", paths, complete=True, lookup=lookup_cost())
    assert result["status"] == "blocked"
    assert result["errors"]
    assert result["committed_usd"] == 5
    with pytest.raises(BudgetError, match="unresolved"):
        ledger.reserve("later", 1)


def test_sample_or_scorer_error_with_accounted_model_call_can_settle(tmp_path, logs):
    ledger = CampaignLedger(tmp_path / "ledger.json")
    ledger.reserve("scorer-failed", 5)
    path = logs(status="error", error="Scorer rejected a truncated but billed response")
    result = ledger.settle("scorer-failed", [path], complete=True, lookup=lookup_cost())
    assert result["status"] == "settled"
    assert result["actual_usd"] == 0.25
    assert result["warnings"]


def test_deduplicates_replayed_events_and_cache_reads_across_batches(tmp_path, logs):
    ledger = CampaignLedger(tmp_path / "ledger.json")
    seen = []
    ledger.reserve("base", 5)
    source = logs([event(), event(), event("gen-cached", cache="read")])
    first = ledger.settle("base", [source, source], complete=True, lookup=lookup_cost(0.2, seen))
    assert first["actual_usd"] == 0.2
    assert first["cached_events_excluded"] == 1
    ledger.reserve("audit", 5, stage="audit")
    audit = logs([event(), event("gen-judge"), event("gen-paid", cache="read")])
    second = ledger.settle("audit", [audit], complete=True, lookup=lookup_cost(0.3, seen))
    assert second["actual_usd"] == 0.3
    assert ledger.snapshot()["verified_usd"] == 0.5
    assert seen == ["gen-paid", "gen-judge"]


def test_starting_spend_ids_prevent_double_counting_existing_logs(tmp_path, logs):
    ledger = CampaignLedger(
        tmp_path / "ledger.json",
        starting_spend_usd=0.5,
        starting_generations={"gen-existing": 0.5},
    )
    ledger.reserve("audit", 1, stage="audit")
    seen = []
    settled = ledger.settle(
        "audit",
        [logs([event("gen-existing"), event("gen-new")])],
        complete=True,
        lookup=lookup_cost(0.1, seen),
    )
    assert settled["actual_usd"] == 0.1
    assert ledger.snapshot()["verified_usd"] == 0.6
    assert seen == ["gen-new"]


@pytest.mark.parametrize(
    "data",
    [
        None,
        {},
        {"id": "gen-other", "total_cost": 1},
        {"id": "gen-paid", "total_cost": None},
        {"id": "gen-paid", "total_cost": "0.1"},
        {"id": "gen-paid", "total_cost": True},
        {"id": "gen-paid", "total_cost": -1},
        {"id": "gen-paid", "total_cost": float("nan")},
        {"id": "gen-paid", "total_cost": float("inf")},
    ],
)
def test_malformed_billing_never_releases_reservation(tmp_path, logs, data):
    ledger = CampaignLedger(tmp_path / "ledger.json")
    ledger.reserve("bill", 5)
    result = ledger.settle("bill", [logs()], complete=True, lookup=lambda _: data)
    assert result["status"] == "blocked"
    assert result["verified_usd"] == 0
    assert result["committed_usd"] == 5


def test_partial_billing_survives_unknown_cost_then_recovers_without_double_count(tmp_path, logs):
    ledger = CampaignLedger(tmp_path / "ledger.json")
    ledger.reserve("bill", 5)
    path = logs([event("gen-a-known"), event("gen-b-later")])

    def incomplete(response_id):
        if response_id == "gen-b-later":
            raise RuntimeError("SECRET-DO-NOT-PERSIST")
        return {"id": response_id, "total_cost": 0.2}

    result = ledger.settle("bill", [path], complete=True, lookup=incomplete)
    assert result["status"] == "blocked"
    assert result["verified_usd"] == 0.2
    assert "SECRET" not in ledger.path.read_text()
    seen = []
    final = CampaignLedger(ledger.path).settle(
        "bill", [path], complete=True, lookup=lookup_cost(0.3, seen)
    )
    assert final["actual_usd"] == 0.5
    assert seen == ["gen-b-later"]


def test_disappearing_ids_do_not_free_unknown_spend(tmp_path, logs):
    ledger = CampaignLedger(tmp_path / "ledger.json")
    ledger.reserve("bill", 5)
    ledger.settle("bill", [logs([event("gen-a"), event("gen-b")])], lookup=lookup_cost())
    result = ledger.settle("bill", [logs([event("gen-a")])], complete=True, lookup=lookup_cost())
    assert result["status"] == "blocked"
    assert any("disappeared" in error for error in result["errors"])


def test_actual_charge_above_bound_blocks_and_counts_larger_amount(tmp_path, logs):
    ledger = CampaignLedger(tmp_path / "ledger.json", limit_usd=5)
    ledger.reserve("bound-failed", 3)
    result = ledger.settle("bound-failed", [logs()], complete=True, lookup=lookup_cost(6))
    assert result["status"] == "blocked"
    assert result["committed_usd"] == 6
    assert ledger.snapshot()["remaining_usd"] == 0
    assert any("upper bound" in error for error in result["errors"])


@pytest.mark.parametrize("file_format", ["json", "eval"])
def test_real_inspect_log_is_read_without_model_calls(tmp_path, file_format):
    output = ModelOutput(response_id="gen-real-schema")
    model_event = ModelEvent(
        model="openrouter/test/model",
        input=[],
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
        output=output,
    )
    log = EvalLog(
        status="success",
        eval=EvalSpec(
            created="2026-10-21T00:00:00Z",
            task="billing-test",
            dataset=EvalDataset(),
            model="openrouter/test/model",
            config=EvalConfig(),
        ),
        samples=[EvalSample(id=1, epoch=1, input="test", target="", events=[model_event])],
    )
    path = tmp_path / f"real.{file_format}"
    write_eval_log(log, path, format=file_format)
    result = budget.collect_generation_ids([path])
    assert result["ids"] == ["gen-real-schema"]
    assert result["errors"] == []


def mock_http(monkeypatch, handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(budget.httpx, "Client", lambda **_: client)


def test_billing_get_retries_eventual_metadata_but_never_model_endpoint(monkeypatch):
    requests, sleeps = [], []

    def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(404)
        if len(requests) == 2:
            return httpx.Response(200, json={"data": {"id": "gen-x", "total_cost": None}})
        return httpx.Response(
            200, json={"data": {"id": "gen-x", "total_cost": 0.05, "private": "omit"}}
        )

    mock_http(monkeypatch, handler)
    result = budget.fetch_generation("gen-x", api_key="fake-key", sleep=sleeps.append)
    assert result == {"id": "gen-x", "total_cost": 0.05}
    assert sleeps == [1, 2]
    assert all(
        request.method == "GET" and request.url.path == "/api/v1/generation" for request in requests
    )


def test_billing_retries_are_bounded_and_error_does_not_expose_secrets(monkeypatch):
    requests, sleeps = [], []

    def handler(request):
        requests.append(request)
        raise httpx.ReadTimeout("SECRET-DO-NOT-EXPOSE", request=request)

    mock_http(monkeypatch, handler)
    with pytest.raises(BillingError) as exc:
        budget.fetch_generation("gen-x", api_key="SECRET-KEY", sleep=sleeps.append)
    assert len(requests) == 4
    assert sleeps == [1, 2, 4]
    assert "SECRET" not in str(exc.value)


def test_billing_auth_error_does_not_retry(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(401)

    mock_http(monkeypatch, handler)
    with pytest.raises(BillingError, match="HTTP 401"):
        budget.fetch_generation("gen-x", api_key="fake", sleep=lambda _: pytest.fail("retry"))
    assert len(seen) == 1


def test_small_decimal_reservations_do_not_round_around_cap(tmp_path, logs):
    ledger = CampaignLedger(tmp_path / "ledger.json", limit_usd="0.3")
    ledger.reserve("one", "0.1")
    ledger.settle("one", [logs()], complete=True, lookup=lookup_cost(Decimal("0.1")))
    ledger.reserve("two", "0.2")
    assert ledger.snapshot()["remaining_usd"] == 0
