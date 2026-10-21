"""Offline recovery of real Inspect logs with permanent retained-cost accounting."""

import copy
from pathlib import Path

import httpx
import pytest
from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, write_eval_log
from inspect_ai.model import GenerateConfig, ModelOutput

from scheming_inspect import budget, campaign, recovery
from scheming_inspect.budget import BudgetExceeded, CampaignLedger

PROVIDER = {
    "max_price": {"prompt": 0.1, "completion": 0.2, "request": 0},
    "require_parameters": True,
}
OLD_HASHES = {"scheming_inspect/engine.py": "old", "scheming_inspect/data.py": "unchanged"}
NEW_HASHES = {**OLD_HASHES, "scheming_inspect/engine.py": "repaired"}
EVIDENCE = {"process_stopped": True, "worker": "offline-fixture"}


def write_log(task, kwargs, *, original=False):
    samples = []
    for index, sample in enumerate(task.dataset):
        if original and index == 3:
            continue  # A request/sample not present in the terminal log must be retried explicitly.
        metadata = copy.deepcopy(sample.metadata)
        metadata["behavior"] = {"completed": 1, "schema_valid": 1}
        error, limit = None, None
        if original and index == 0:
            metadata["behavior"]["schema_valid"] = 0  # retain bad output, do not cherry-pick
        if original and index == 1:
            error = {
                "message": "infrastructure failure",
                "traceback": "fixture",
                "traceback_ansi": "fixture",
            }
        if original and index == 2:
            metadata["behavior"] = {"completed": 0, "truncated": 1}
            limit = {"type": "token", "limit": 1}
        response_id = f"gen-{'old' if original else 'retry'}-{sample.id}"
        event = ModelEvent(
            model=campaign.MODEL,
            input=[],
            tools=[],
            tool_choice="none",
            config=GenerateConfig(max_retries=0),
            output=ModelOutput(response_id=response_id),
        )
        samples.append(
            EvalSample(
                id=sample.id,
                epoch=1,
                input=sample.input,
                target="",
                metadata=metadata,
                error=error,
                limit=limit,
                events=[event],
            )
        )
    log = EvalLog(
        status="error" if original else "success",
        samples=samples,
        eval=EvalSpec(
            created="2026-10-21T00:00:00Z",
            task="recovery-fixture",
            dataset=EvalDataset(samples=len(task.dataset)),
            model=campaign.MODEL,
            config=EvalConfig(epochs=1),
            metadata=kwargs["metadata"],
        ),
    )
    write_eval_log(log, Path(kwargs["log_dir"]) / "result.eval")
    return [log]


@pytest.fixture
def stopped(tmp_path, monkeypatch):
    requests = []
    client = httpx.Client

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200, json={"data": {"id": request.url.params["id"], "total_cost": 0.01}}
        )

    monkeypatch.setattr(
        budget.httpx,
        "Client",
        lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs),
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-only")
    cfg = dict(
        model=campaign.MODEL,
        max_steps=20,
        max_tokens=8000,
        token_limit=100000,
        max_samples=8,
        batch_size=32,
        budget_usd=5,
        stage_budget_usd=5,
        starting_spend_usd=0,
        retries=0,
        epochs=1,
        cache_prompt=False,
    )
    cell = dict(experiment="oversight", n=4, seed=42)
    manifest = campaign.prepare_manifest(
        tmp_path, config=cfg, cells=[cell], hashes=OLD_HASHES, resume=False
    )
    task = campaign.scheming(**cell)
    # One stopped four-sample batch is the small analogue of 13 retained / 19 retry.
    batch = manifest["batches"][0]
    batch.update(start=0, stop=4, sample_ids=[str(s.id) for s in task.dataset])
    manifest["batches"] = [batch]
    ledger = CampaignLedger(tmp_path / "budget.json", limit_usd=5, stage_limits={"base": 5})
    result = campaign.execute_batch(
        task,
        batch_id=batch["batch_id"],
        log_dir=tmp_path / "base" / batch["batch_id"],
        ledger=ledger,
        upper_usd=2,
        stage="base",
        model=campaign.MODEL,
        provider=PROVIDER,
        fingerprint=manifest["fingerprint"],
        eval_fn=lambda task, **kwargs: write_log(task, kwargs, original=True),
    )
    assert result["status"] == "blocked"
    batch.update(status="blocked", log_paths=result["log_paths"])
    manifest.update(status="blocked", blocker=result)
    campaign.atomic_json(tmp_path / "manifest.json", manifest)
    return tmp_path, manifest, ledger, requests


def prepare(directory):
    return recovery.prepare_recovery(
        directory,
        reason="Fix missing-key aggregation",
        process_evidence=EVIDENCE,
        hashes=NEW_HASHES,
    )


def bound(_model, n, _token_limit, _max_tokens):
    assert n == 2
    return {"conservative_max_usd": 1, "provider": PROVIDER}


def test_prepare_run_resume_preserves_outcomes_exact_ids_and_full_old_reservation(stopped):
    directory, old, ledger, requests = stopped
    original_log = Path(old["batches"][0]["log_paths"][0])
    original_bytes = original_log.read_bytes()
    manifest = prepare(directory)
    assert prepare(directory) == manifest  # idempotent, no second closure or billing
    assert len(requests) == 3
    batch = manifest["batches"][0]
    spec = batch["recovery"]
    assert spec["retained_sample_ids"] == ["oversight-0001", "oversight-0003"]
    assert spec["retry_sample_ids"] == ["oversight-0002", "oversight-0004"]
    assert spec["retained_status_counts"] == {"invalid_schema": 1, "limited": 1}
    assert [item["reason"] for item in spec["excluded_original_attempts"]] == [
        "sample_error",
        "unrecorded",
    ]
    assert manifest["fingerprint"] != old["fingerprint"]
    assert manifest["source_revisions"][0]["old_source_hashes"] == OLD_HASHES
    assert ledger.snapshot()["committed_usd"] == 2
    assert not ledger.pending()
    assert campaign._ledger_batch(ledger, batch["batch_id"])["status"] == "closed_reserved"
    attempts = []

    def evaluator(task, **kwargs):
        attempts.append([str(sample.id) for sample in task.dataset])
        assert ledger.snapshot()["committed_usd"] == 3  # old ceiling + new reservation
        return write_log(task, kwargs)

    result = recovery.run_recovery_batch(
        directory, manifest, batch, ledger, resume=True, bound_fn=bound, eval_fn=evaluator
    )
    assert attempts == [["oversight-0002", "oversight-0004"]]
    assert result["sample_status_counts"] == {"invalid_schema": 1, "limited": 1, "success": 2}
    assert [r["expected_sample_ids"] for r in result["canonical_log_records"]] == [
        ["oversight-0001", "oversight-0003"],
        ["oversight-0002", "oversight-0004"],
    ]
    assert ledger.snapshot()["committed_usd"] == pytest.approx(2.02)
    assert original_log.read_bytes() == original_bytes
    assert len(requests) == 5
    finished = campaign.run_base(
        directory,
        manifest,
        ledger,
        resume=True,
        bound_fn=lambda *_: pytest.fail("Recomputed a settled retry"),
        eval_fn=lambda *_a, **_k: pytest.fail("Dispatched a settled retry"),
        source_hash_fn=lambda: NEW_HASHES,
    )
    assert finished["status"] == "base_complete"
    assert len(requests) == 5
    assert campaign.campaign_status(directory)["base_samples_settled"] == 4
    ids = [
        sid
        for record in finished["completed_base_log_records"]
        for sid in record["expected_sample_ids"]
    ]
    assert sorted(ids) == sorted(batch["sample_ids"]) and len(set(ids)) == 4
    with pytest.raises(BudgetExceeded):
        ledger.reserve("would_exceed_global_cap", 2.99, stage="base")


def test_data_hash_migration_rejected_before_ledger_or_manifest_changes(stopped):
    directory, _, ledger, _ = stopped
    before = (directory / "manifest.json").read_bytes()
    ledger_before = ledger.path.read_bytes()
    with pytest.raises(campaign.CampaignError, match="repairs"):
        recovery.prepare_recovery(
            directory,
            reason="not a scorer-only repair",
            process_evidence=EVIDENCE,
            hashes={**NEW_HASHES, "scheming_inspect/data.py": "changed input facts"},
        )
    assert (directory / "manifest.json").read_bytes() == before
    assert ledger.path.read_bytes() == ledger_before
    assert not (directory / "recoveries").exists()


def test_restart_after_ledger_closure_finishes_without_second_closure(stopped, monkeypatch):
    directory, old, ledger, _ = stopped
    original_atomic = campaign.atomic_json

    def interrupted_write(path, value):
        if Path(path) == directory / "manifest.json" and value.get("status") == "recovery_ready":
            raise OSError("simulated interruption after ledger closure")
        original_atomic(path, value)

    monkeypatch.setattr(campaign, "atomic_json", interrupted_write)
    with pytest.raises(OSError, match="simulated interruption"):
        prepare(directory)
    assert (
        campaign._ledger_batch(ledger, old["batches"][0]["batch_id"])["status"] == "closed_reserved"
    )
    monkeypatch.setattr(campaign, "atomic_json", original_atomic)
    monkeypatch.setattr(
        CampaignLedger,
        "close_with_reservation",
        lambda *_a, **_k: pytest.fail("Closed an attempt twice"),
    )
    recovered = prepare(directory)
    assert recovered["status"] == "recovery_ready"
    assert len(recovered["source_revisions"]) == 1
    assert ledger.snapshot()["committed_usd"] == 2


def test_retry_reservation_must_fit_alongside_original_retained_cost(stopped):
    directory, _, ledger, requests = stopped
    manifest = prepare(directory)
    result = recovery.run_recovery_batch(
        directory,
        manifest,
        manifest["batches"][0],
        ledger,
        resume=True,
        bound_fn=lambda *_: {"conservative_max_usd": 3.01, "provider": PROVIDER},
        eval_fn=lambda *_a, **_k: pytest.fail("Exceeded combined budget"),
    )
    assert result["status"] == "pending_budget"
    assert ledger.snapshot()["committed_usd"] == 2
    assert not ledger.pending()
    assert len(requests) == 3
