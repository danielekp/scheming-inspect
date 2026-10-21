"""Offline campaign, resume and real-ledger billing integration."""

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.event import ModelEvent
from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalLog,
    EvalPlan,
    EvalSample,
    EvalSpec,
    write_eval_log,
)
from inspect_ai.model import GenerateConfig, ModelOutput, get_model

from scheming_inspect import budget, campaign
from scheming_inspect.budget import CampaignLedger
from scheming_inspect.protocol import plan

PROVIDER = {
    "max_price": {"prompt": 0.1, "completion": 0.2, "request": 0},
    "require_parameters": True,
}


def task_factory(experiment="oversight", n=5, seed=42, **kwargs):
    return SimpleNamespace(
        config=GenerateConfig(parallel_tool_calls=False),
        dataset=MemoryDataset(
            [
                Sample(id=f"{experiment}-{i + 1:04d}", input=f"seed={seed}; original_index={i}")
                for i in range(n)
            ]
        ),
    )


def config():
    return dict(
        model=campaign.MODEL,
        max_steps=20,
        max_tokens=8000,
        token_limit=100000,
        max_samples=8,
        batch_size=2,
        budget_usd=50,
        stage_budget_usd=35,
    )


def create_log(task, kwargs, *, errors=(), missing=(), status="success"):
    samples = []
    for sample in task.dataset:
        if sample.id in missing:
            continue
        event = ModelEvent(
            model=campaign.MODEL,
            input=[],
            tools=[],
            tool_choice="none",
            config=GenerateConfig(max_retries=0),
            output=ModelOutput(response_id=f"gen-{sample.id}"),
        )
        error = (
            {"message": "scorer failed", "traceback": "test", "traceback_ansi": "test"}
            if sample.id in errors
            else None
        )
        samples.append(
            EvalSample(
                id=sample.id,
                epoch=1,
                input=sample.input,
                target="",
                events=[event],
                error=error,
                metadata={"behavior": {"completed": 1, "schema_valid": 1}},
            )
        )
    log = EvalLog(
        status=status,
        eval=EvalSpec(
            created="2026-10-21T00:00:00Z",
            task="campaign-test",
            dataset=EvalDataset(samples=len(samples)),
            model=campaign.MODEL,
            config=EvalConfig(epochs=1),
            metadata=kwargs["metadata"],
        ),
        samples=samples,
    )
    write_eval_log(log, Path(kwargs["log_dir"]) / "result.eval")
    return log


@pytest.fixture
def billing(monkeypatch):
    calls = []
    original_client = httpx.Client

    def handler(request):
        calls.append(request)
        return httpx.Response(
            200, json={"data": {"id": request.url.params["id"], "total_cost": 0.01}}
        )

    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-secret")
    monkeypatch.setattr(
        budget.httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs),
    )
    return calls


def execute(tmp_path, ledger, evaluator, **overrides):
    return campaign.execute_batch(
        task_factory(n=1),
        batch_id="batch-one",
        log_dir=tmp_path / "batch",
        ledger=ledger,
        upper_usd=1,
        stage="base",
        model=campaign.MODEL,
        provider=PROVIDER,
        fingerprint="frozen",
        eval_fn=evaluator,
        **overrides,
    )


def test_full_cell_slices_keep_original_ids_and_preflight_is_one_sample():
    cells = [dict(experiment="oversight", n=6, seed=19), dict(experiment="alignment", n=3, seed=19)]
    batches = campaign.make_schedule(cells, 2, campaign_fingerprint="x", task_factory=task_factory)
    assert [len(batch["sample_ids"]) for batch in batches] == [1, 2, 2, 1, 2, 1]
    for index, cell in enumerate(cells):
        selected = [batch for batch in batches if batch["cell_index"] == index]
        samples = []
        for batch in selected:
            task = campaign.sliced_task(cell, batch, limits={}, task_factory=task_factory)
            assert task.config.parallel_tool_calls is None
            samples.extend(task.dataset)
        original = list(task_factory(**cell).dataset)
        assert [(s.id, s.input) for s in samples] == [(s.id, s.input) for s in original]


def test_full_schedule_covers_65_cells_and_9094_samples_exactly():
    cells = plan("full")
    batches = campaign.make_schedule(
        cells, 32, campaign_fingerprint="full", task_factory=task_factory
    )
    assert len(cells) == 65
    assert sum(len(batch["sample_ids"]) for batch in batches) == 9094
    assert len(batches[0]["sample_ids"]) == 1
    assert all(1 <= len(batch["sample_ids"]) <= 32 for batch in batches)
    for index, cell in enumerate(cells):
        indices = [
            j
            for batch in batches
            if batch["cell_index"] == index
            for j in range(batch["start"], batch["stop"])
        ]
        assert indices == list(range(cell["n"]))


def test_dispatch_durable_reservation_caps_no_retries_and_exact_billing(tmp_path, billing):
    ledger = CampaignLedger(
        tmp_path / "ledger.json", starting_spend_usd=0.013229117, stage_limits={"base": 35}
    )
    calls = []

    def evaluator(task, **kwargs):
        calls.append(kwargs)
        assert ledger.pending()[0]["status"] == "reserved"
        assert campaign.read_json(tmp_path / "batch" / "batch.json")["status"] == "started"
        assert task.config.parallel_tool_calls is None
        return [create_log(task, kwargs)]

    result = execute(tmp_path, ledger, evaluator)
    assert result["status"] == "settled"
    assert ledger.snapshot()["verified_usd"] == pytest.approx(0.023229117)
    assert result["sample_status_counts"] == {"success": 1}
    call = calls[0]
    assert call["model_args"] == {"provider": PROVIDER}
    assert call["max_samples"] == 8 and call["max_tasks"] == call["epochs"] == 1
    assert call["max_retries"] == call["retry_on_error"] == 0
    assert call["cache_prompt"] is call["ctl_server"] is False
    assert len(billing) == 1 and billing[0].method == "GET"
    assert billing[0].url.path == "/api/v1/generation"
    assert billing[0].headers["authorization"] == "Bearer offline-test-secret"
    assert "offline-test-secret" not in ledger.path.read_text()


def test_settled_resume_never_dispatches_or_bills_twice(tmp_path, billing):
    ledger = CampaignLedger(tmp_path / "ledger.json")
    first = execute(tmp_path, ledger, lambda task, **kwargs: [create_log(task, kwargs)])
    again = execute(tmp_path, ledger, lambda *_a, **_k: pytest.fail("redispatch"), resume=True)
    assert again == first and len(billing) == 1
    with pytest.raises(campaign.CampaignError, match="--resume"):
        execute(tmp_path, ledger, lambda *_a, **_k: pytest.fail("redispatch"))


def test_billing_only_resume_releases_reservation_without_regenerating(tmp_path, monkeypatch):
    ledger = CampaignLedger(tmp_path / "ledger.json")
    lookups = []

    def delayed_bill(response_id, **kwargs):
        lookups.append(response_id)
        if len(lookups) == 1:
            raise budget.BillingError("Temporary billing lookup failure")
        return {"id": response_id, "total_cost": 0.01}

    monkeypatch.setattr(budget, "fetch_generation", delayed_bill)
    blocked = execute(tmp_path, ledger, lambda task, **kwargs: [create_log(task, kwargs)])
    assert blocked["status"] == "blocked"
    assert blocked["complete"] is True
    assert ledger.snapshot()["committed_usd"] == 1
    assert blocked["settlement"]["errors"] == [
        "Billing unavailable for gen-oversight-0001 (BillingError)."
    ]

    settled = execute(
        tmp_path,
        ledger,
        lambda *_a, **_k: pytest.fail("A billing retry regenerated a completed sample"),
        resume=True,
    )
    assert settled["status"] == "settled"
    assert ledger.snapshot()["committed_usd"] == pytest.approx(0.01)
    assert lookups == ["gen-oversight-0001", "gen-oversight-0001"]


def test_partial_batch_blocks_and_resume_does_not_pay_for_missing_samples(tmp_path, billing):
    ledger = CampaignLedger(tmp_path / "ledger.json")
    args = dict(
        batch_id="partial",
        log_dir=tmp_path / "partial",
        ledger=ledger,
        upper_usd=2,
        stage="base",
        model=campaign.MODEL,
        provider=PROVIDER,
        fingerprint="partial",
    )
    first = campaign.execute_batch(
        task_factory(n=2),
        **args,
        eval_fn=lambda task, **kwargs: [create_log(task, kwargs, missing=("oversight-0002",))],
    )
    assert first["status"] == "blocked" and ledger.snapshot()["committed_usd"] == 2
    again = campaign.execute_batch(
        None,
        **args,
        expected_sample_ids=["oversight-0001", "oversight-0002"],
        resume=True,
        eval_fn=lambda *_a, **_k: pytest.fail("redispatch"),
    )
    assert again["status"] == "blocked" and len(billing) == 1


def test_accounted_sample_error_is_visible_and_never_retried(tmp_path, billing):
    ledger = CampaignLedger(tmp_path / "ledger.json")
    result = execute(
        tmp_path,
        ledger,
        lambda task, **kwargs: [
            create_log(task, kwargs, errors=("oversight-0001",), status="error")
        ],
    )
    assert result["status"] == "settled"
    assert result["sample_status_counts"] == {"error": 1}
    assert ledger.snapshot()["verified_usd"] == 0.01


def test_started_crash_retains_reservation_and_does_not_redispatch(tmp_path, billing):
    ledger = CampaignLedger(tmp_path / "ledger.json")

    def crashed(*_a, **_k):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        execute(tmp_path, ledger, crashed)
    assert campaign.read_json(tmp_path / "batch" / "batch.json")["status"] == "interrupted"
    assert ledger.snapshot()["committed_usd"] == 1
    resumed = execute(tmp_path, ledger, lambda *_a, **_k: pytest.fail("redispatch"), resume=True)
    assert resumed["status"] == "blocked" and not billing


def test_settlement_journal_crash_window_recovers_without_resettling(
    tmp_path, billing, monkeypatch
):
    ledger = CampaignLedger(tmp_path / "ledger.json")
    execute(tmp_path, ledger, lambda task, **kwargs: [create_log(task, kwargs)])
    path = tmp_path / "batch" / "batch.json"
    interrupted = campaign.read_json(path)
    interrupted["status"] = "returned"
    campaign.atomic_json(path, interrupted)
    monkeypatch.setattr(ledger, "settle", lambda *_a, **_k: pytest.fail("already settled"))
    result = execute(tmp_path, ledger, lambda *_a, **_k: pytest.fail("redispatch"), resume=True)
    assert result["status"] == "settled" and len(billing) == 1


def test_budget_denial_leaves_no_started_journal_and_can_shrink(tmp_path, billing):
    ledger = CampaignLedger(tmp_path / "ledger.json", limit_usd=0.5)
    denied = execute(tmp_path, ledger, lambda *_a, **_k: pytest.fail("over cap"))
    assert denied["status"] == "pending_budget"
    assert not (tmp_path / "batch" / "batch.json").exists()
    assert not ledger.pending() and not billing
    allowed = campaign.execute_batch(
        task_factory(n=1),
        batch_id="batch-one",
        log_dir=tmp_path / "batch",
        ledger=ledger,
        upper_usd=0.25,
        stage="base",
        model=campaign.MODEL,
        provider=PROVIDER,
        fingerprint="frozen",
        eval_fn=lambda task, **kwargs: [create_log(task, kwargs)],
    )
    assert allowed["status"] == "settled"


def test_frozen_manifest_rejects_source_or_config_change(tmp_path):
    cells = [dict(experiment="oversight", n=2, seed=42)]
    first = campaign.prepare_manifest(
        tmp_path,
        config=config(),
        cells=cells,
        hashes={"code": "a"},
        resume=False,
        task_factory=task_factory,
    )
    assert (
        campaign.prepare_manifest(
            tmp_path,
            config=config(),
            cells=cells,
            hashes={"code": "a"},
            resume=True,
            task_factory=task_factory,
        )
        == first
    )
    for cfg, hashes in (
        ({**config(), "max_tokens": 4096}, {"code": "a"}),
        (config(), {"code": "b"}),
    ):
        with pytest.raises(campaign.CampaignError, match="differ"):
            campaign.prepare_manifest(
                tmp_path,
                config=cfg,
                cells=cells,
                hashes=hashes,
                resume=True,
                task_factory=task_factory,
            )


def test_rewritten_settled_log_is_not_silently_accepted(tmp_path, billing):
    ledger = CampaignLedger(tmp_path / "ledger.json")
    result = execute(tmp_path, ledger, lambda task, **kwargs: [create_log(task, kwargs)])
    Path(result["log_paths"][0]).write_bytes(b"changed")
    with pytest.raises(campaign.CampaignError, match="changed"):
        execute(tmp_path, ledger, lambda *_a, **_k: pytest.fail("redispatch"), resume=True)


def test_base_runner_persists_cell_mapping_and_compact_status(tmp_path, billing):
    cells = [dict(experiment="oversight", n=3, seed=42)]
    manifest = campaign.prepare_manifest(
        tmp_path,
        config=config(),
        cells=cells,
        hashes={"code": "frozen"},
        resume=False,
        task_factory=task_factory,
    )
    ledger = CampaignLedger(tmp_path / "budget.json", stage_limits={"base": 35})
    dispatches = []

    def evaluator(task, **kwargs):
        dispatches.append([sample.id for sample in task.dataset])
        return [create_log(task, kwargs)]

    def bound(*args):
        return dict(
            conservative_max_usd=1,
            provider=PROVIDER,
            context_length=1000,
            pricing={"prompt": "0.0000001", "completion": "0.0000002"},
        )

    result = campaign.run_base(
        tmp_path,
        manifest,
        ledger,
        resume=False,
        bound_fn=bound,
        task_factory=task_factory,
        eval_fn=evaluator,
        source_hash_fn=lambda: {"code": "frozen"},
    )
    assert result["status"] == "base_complete"
    assert dispatches == [["oversight-0001"], ["oversight-0002", "oversight-0003"]]
    assert [r["cell"] for r in result["completed_base_log_records"]] == cells * 2
    status = campaign.campaign_status(tmp_path)
    assert status["base_samples_settled"] == status["base_samples_total"] == 3
    assert status["budget"]["verified_usd"] == 0.03
    assert "generations" not in status["budget"] and "batches" not in status["budget"]


def bounded_log(task, kwargs, *, missing_bill=False):
    log = create_log(task, kwargs)
    log.eval.config.retry_on_error = 0
    log.eval.config.token_limit = 100000
    log.eval.model_args = kwargs["model_args"]
    log.plan = EvalPlan(config=GenerateConfig(max_retries=0, max_tokens=8000, cache_prompt=False))
    for sample in log.samples:
        sample.events[0].config = log.plan.config
    if missing_bill:
        log.samples[0].events[0].output.response_id = None
    write_eval_log(log, Path(kwargs["log_dir"]) / "result.eval")
    return [log]


def test_unknown_bill_keeps_reservation_and_base_progress_without_rerunning(tmp_path, billing):
    manifest = campaign.prepare_manifest(
        tmp_path,
        config=config(),
        cells=[dict(experiment="oversight", n=3, seed=42)],
        hashes={"code": "frozen"},
        resume=False,
        task_factory=task_factory,
    )
    ledger = CampaignLedger(tmp_path / "budget.json", limit_usd=2, allow_unverified_billing=True)
    dispatches = []

    def evaluator(task, **kwargs):
        dispatches.append([sample.id for sample in task.dataset])
        return bounded_log(task, kwargs, missing_bill=len(dispatches) == 1)

    result = campaign.run_base(
        tmp_path,
        manifest,
        ledger,
        resume=False,
        bound_fn=lambda *_: {"conservative_max_usd": 1, "provider": PROVIDER},
        task_factory=task_factory,
        eval_fn=evaluator,
        source_hash_fn=lambda: {"code": "frozen"},
    )
    assert result["status"] == "base_complete"
    assert [batch["status"] for batch in result["batches"]] == ["complete_reserved", "settled"]
    assert len(dispatches) == 2
    assert campaign.campaign_status(tmp_path)["base_samples_settled"] == 3
    assert sum(len(row["expected_sample_ids"]) for row in result["completed_base_log_records"]) == 3
    assert ledger.snapshot()["committed_usd"] == pytest.approx(1.02)
    assert ledger.snapshot()["verified_usd"] == pytest.approx(0.02)
    assert not ledger.pending()
    campaign.run_base(
        tmp_path,
        result,
        ledger,
        resume=True,
        eval_fn=lambda *_a, **_k: pytest.fail("reran recorded samples"),
        bound_fn=lambda *_: pytest.fail("repriced recorded samples"),
        source_hash_fn=lambda: {"code": "frozen"},
    )
    with pytest.raises(budget.BudgetExceeded):
        ledger.reserve("over_remaining_cap", 0.99)


def test_resume_after_cost_retention_before_journal_write_does_not_dispatch(
    tmp_path, billing, monkeypatch
):
    ledger = CampaignLedger(tmp_path / "ledger.json", allow_unverified_billing=True)
    atomic = campaign.atomic_json

    def interrupted(path, value):
        if Path(path).name == "batch.json" and value.get("status") == "complete_reserved":
            raise OSError("after retaining cost, before saving completed journal")
        atomic(path, value)

    monkeypatch.setattr(campaign, "atomic_json", interrupted)
    with pytest.raises(OSError):
        execute(tmp_path, ledger, lambda task, **kw: bounded_log(task, kw, missing_bill=True))
    assert campaign._ledger_batch(ledger, "batch-one")["status"] == "closed_reserved"
    monkeypatch.setattr(campaign, "atomic_json", atomic)
    result = execute(
        tmp_path, ledger, lambda *_a, **_k: pytest.fail("replayed completed work"), resume=True
    )
    assert result["status"] == "complete_reserved"
    assert result["settlement"]["actual_usd"] is None
    assert ledger.snapshot()["committed_usd"] == 1


def test_billing_opt_in_is_not_an_experimental_config_change(tmp_path, capsys):
    campaign.main(["--campaign-dir", str(tmp_path)])
    strict = json.loads(capsys.readouterr().out)
    campaign.main(["--campaign-dir", str(tmp_path), "--allow-unverified-billing"])
    tolerant = json.loads(capsys.readouterr().out)
    assert strict["config"] == tolerant["config"]


def test_plan_has_no_network_side_effects(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(campaign, "openrouter_bound", lambda *_a: pytest.fail("network"))
    monkeypatch.setattr(campaign, "inspect_eval", lambda *_a, **_k: pytest.fail("model"))
    campaign.main(["--campaign-dir", str(tmp_path)])
    output = json.loads(capsys.readouterr().out)
    assert output["event"] == "plan_only" and output["samples"] == 9094
    assert not list(tmp_path.iterdir())


def test_hashes_ignore_tests_and_timestamps_but_include_code(tmp_path):
    (tmp_path / "scheming_inspect").mkdir()
    source = tmp_path / "scheming_inspect" / "task.py"
    source.write_text("pass\n")
    (tmp_path / "tests").mkdir()
    test = tmp_path / "tests" / "test_task.py"
    test.write_text("pass\n")
    first = campaign.source_hashes(tmp_path)
    source.touch()
    test.write_text("changed\n")
    assert campaign.source_hashes(tmp_path) == first
    source.write_text("changed\n")
    assert campaign.source_hashes(tmp_path) != first


def test_roles_receive_provider_caps_and_prompt_cache_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(
        campaign, "get_model", lambda name, **kwargs: calls.append((name, kwargs)) or "role"
    )
    assert campaign._role_models({"grader": campaign.MODEL}, campaign.MODEL, PROVIDER) == {
        "grader": "role"
    }
    assert calls[0][1]["provider"] == PROVIDER
    assert calls[0][1]["config"].max_retries == 0
    assert calls[0][1]["config"].cache_prompt is False


def test_cleared_parallel_tool_calls_is_absent_from_openrouter_request():
    model = get_model(campaign.MODEL, api_key="offline-key", provider=PROVIDER)
    cell = dict(experiment="oversight", n=1, seed=42)
    batch = {"start": 0, "stop": 1, "sample_ids": ["oversight-0001"]}
    task = campaign.sliced_task(cell, batch, limits={}, task_factory=task_factory)
    params = model.api.completion_params(task.config, tools=True)
    assert "parallel_tool_calls" not in params
    explicit = model.api.completion_params(GenerateConfig(parallel_tool_calls=False), tools=True)
    assert explicit["parallel_tool_calls"] is False


def test_status_includes_poststage_progress_without_provider_calls(tmp_path, monkeypatch, capsys):
    from scheming_inspect import poststages

    campaign.prepare_manifest(
        tmp_path,
        config=config(),
        cells=[dict(experiment="oversight", n=1)],
        hashes={"code": "frozen"},
        resume=False,
        task_factory=task_factory,
    )
    campaign.atomic_json(tmp_path / "poststages.json", {"test_state": True})
    report = {
        "status": "running",
        "stages": {"fixed": {"planned": 160, "settled": 32, "pending": 128}},
        "blocked_batch": None,
        "exclusions": {},
    }
    monkeypatch.setattr(poststages, "progress_report", lambda state: report)
    monkeypatch.setattr(campaign, "openrouter_bound", lambda *_a: pytest.fail("network"))
    campaign.main(["--campaign-dir", str(tmp_path), "--status"])
    output = capsys.readouterr().out
    assert "Post stages: running" in output and "fixed: 32/160 settled; 128 pending" in output
