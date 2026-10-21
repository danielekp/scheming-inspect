"""Offline recovery preserves original attempts without financial gates."""

import copy
from pathlib import Path

import httpx
import pytest
from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, write_eval_log
from inspect_ai.model import GenerateConfig, ModelOutput

from scheming_inspect import campaign, recovery

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
    monkeypatch.setattr(httpx, "Client", lambda **_kwargs: pytest.fail("Unexpected HTTP client"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = dict(
        model=campaign.MODEL,
        max_steps=20,
        max_tokens=8000,
        token_limit=100000,
        max_samples=8,
        batch_size=32,
        retries=0,
        epochs=1,
        cache_prompt=False,
    )
    cell = dict(experiment="oversight", n=4, seed=42)
    manifest = campaign.prepare_manifest(
        tmp_path, config=cfg, cells=[cell], hashes=OLD_HASHES, resume=False
    )
    task = campaign.scheming(**cell)
    batch = manifest["batches"][0]
    batch.update(start=0, stop=4, sample_ids=[str(s.id) for s in task.dataset])
    manifest["batches"] = [batch]
    (tmp_path / "budget.json").write_bytes(b"Historical budget: deliberately not parseable JSON.")
    result = campaign.execute_batch(
        task,
        batch_id=batch["batch_id"],
        log_dir=tmp_path / "base" / batch["batch_id"],
        stage="base",
        model=campaign.MODEL,
        fingerprint=manifest["fingerprint"],
        eval_fn=lambda task, **kwargs: write_log(task, kwargs, original=True),
    )
    assert result["status"] == "blocked"
    batch.update(status="blocked", log_paths=result["log_paths"])
    manifest.update(status="blocked", blocker=result)
    campaign.atomic_json(tmp_path / "manifest.json", manifest)
    return tmp_path, manifest


def prepare(directory, hashes=NEW_HASHES):
    return recovery.prepare_recovery(
        directory, reason="Recover infrastructure failure", process_evidence=EVIDENCE, hashes=hashes
    )


@pytest.mark.parametrize("finished_status", ["complete", "settled", "complete_reserved"])
def test_prepare_run_resume_preserves_outcomes_exact_ids_and_legacy_budget(
    stopped, finished_status
):
    directory, old = stopped
    original_log = Path(old["batches"][0]["log_paths"][0])
    original_bytes = original_log.read_bytes()
    budget_bytes = (directory / "budget.json").read_bytes()
    manifest = prepare(directory)
    assert prepare(directory) == manifest
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
    assert (
        campaign.read_json(directory / "recoveries" / spec["attempt_id"] / "manifest-before.json")
        == old
    )
    attempts = []

    def evaluator(task, **kwargs):
        attempts.append([str(sample.id) for sample in task.dataset])
        assert kwargs["model_args"] == {"provider": {}}
        assert "spending_reservation_usd" not in kwargs["metadata"]
        return write_log(task, kwargs)

    result = recovery.run_recovery_batch(directory, manifest, batch, resume=True, eval_fn=evaluator)
    assert attempts == [["oversight-0002", "oversight-0004"]]
    assert result["status"] == "complete"
    assert result["sample_status_counts"] == {"invalid_schema": 1, "limited": 1, "success": 2}
    assert [r["expected_sample_ids"] for r in result["canonical_log_records"]] == [
        ["oversight-0001", "oversight-0003"],
        ["oversight-0002", "oversight-0004"],
    ]
    # Compatibility with completed pre-migration recovery journals.
    journal_path = directory / "recovery-attempts" / spec["attempt_id"] / "batch.json"
    journal = campaign.read_json(journal_path)
    journal["status"] = finished_status
    if finished_status != "complete":
        journal["settlement"] = {"log_fingerprints": journal["log_fingerprints"]}
    campaign.atomic_json(journal_path, journal)
    finished = campaign.run_base(
        directory,
        manifest,
        resume=True,
        eval_fn=lambda *_a, **_k: pytest.fail("Redispatched an existing recovery"),
        source_hash_fn=lambda: NEW_HASHES,
    )
    assert finished["status"] == "base_complete"
    ids = [
        sid
        for record in finished["completed_base_log_records"]
        for sid in record["expected_sample_ids"]
    ]
    assert sorted(ids) == sorted(batch["sample_ids"]) and len(set(ids)) == 4
    assert original_log.read_bytes() == original_bytes
    assert (directory / "budget.json").read_bytes() == budget_bytes


def test_data_hash_migration_rejected_before_manifest_or_historical_records_change(stopped):
    directory, _ = stopped
    before = (directory / "manifest.json").read_bytes()
    budget_before = (directory / "budget.json").read_bytes()
    with pytest.raises(campaign.CampaignError, match="repairs"):
        prepare(directory, {**NEW_HASHES, "scheming_inspect/data.py": "changed input facts"})
    assert (directory / "manifest.json").read_bytes() == before
    assert (directory / "budget.json").read_bytes() == budget_before
    assert not (directory / "recoveries").exists()


def test_restart_after_preparation_snapshot_finishes_without_replacing_evidence(
    stopped, monkeypatch
):
    directory, _ = stopped
    original_atomic = campaign.atomic_json

    def interrupted_write(path, value):
        if Path(path) == directory / "manifest.json" and value.get("status") == "recovery_ready":
            raise OSError("simulated interruption after preparation")
        original_atomic(path, value)

    monkeypatch.setattr(campaign, "atomic_json", interrupted_write)
    with pytest.raises(OSError, match="simulated interruption"):
        prepare(directory)
    snapshot = next((directory / "recoveries").glob("*/manifest-before.json"))
    original_bytes = snapshot.read_bytes()
    monkeypatch.setattr(campaign, "atomic_json", original_atomic)
    recovered = prepare(directory)
    assert recovered["status"] == "recovery_ready"
    assert len(recovered["source_revisions"]) == 1
    assert snapshot.read_bytes() == original_bytes


@pytest.mark.parametrize(
    "changed", ["original_log", "original_journal", "prepared_plan", "partition"]
)
def test_changed_recovery_evidence_cannot_dispatch(stopped, changed):
    directory, _ = stopped
    manifest = prepare(directory)
    batch = manifest["batches"][0]
    spec = batch["recovery"]
    if changed == "original_log":
        with Path(spec["original_log"]).open("ab") as stream:
            stream.write(b"changed")
    elif changed == "original_journal":
        path = directory / "base" / batch["batch_id"] / "batch.json"
        value = campaign.read_json(path)
        value["changed"] = True
        campaign.atomic_json(path, value)
    elif changed == "prepared_plan":
        path = directory / "recoveries" / spec["attempt_id"] / "plan.json"
        value = campaign.read_json(path)
        value["recovery"]["retry_sample_ids"].reverse()
        campaign.atomic_json(path, value)
    else:
        spec["retry_sample_ids"].append(spec["retained_sample_ids"][0])
    with pytest.raises(campaign.CampaignError):
        recovery.run_recovery_batch(
            directory,
            manifest,
            batch,
            resume=True,
            eval_fn=lambda *_a, **_k: pytest.fail("Dispatched changed recovery evidence"),
        )


def test_unchanged_source_can_recover_infrastructure_without_financial_configuration(stopped):
    directory, _ = stopped
    (directory / "budget.json").unlink()
    manifest = prepare(directory, OLD_HASHES)
    assert manifest["batches"][0]["recovery"]["retry_sample_ids"] == [
        "oversight-0002",
        "oversight-0004",
    ]
    assert prepare(directory, OLD_HASHES) == manifest
    assert not (directory / "budget.json").exists()


@pytest.mark.parametrize("evidence", [{}, {"process_stopped": "true"}, {"process_stopped": False}])
def test_stopped_process_proof_still_required(stopped, evidence):
    directory, _ = stopped
    before = (directory / "manifest.json").read_bytes()
    with pytest.raises(campaign.CampaignError, match="stopped-process"):
        recovery.prepare_recovery(
            directory, reason="retry", process_evidence=evidence, hashes=NEW_HASHES
        )
    assert (directory / "manifest.json").read_bytes() == before


@pytest.mark.parametrize("journal_status", [None, "planned", "started"])
@pytest.mark.parametrize("batch_status", ["complete", "settled", "complete_reserved"])
def test_completed_recovery_never_dispatches_without_terminal_attempt_journal(
    stopped, journal_status, batch_status
):
    directory, _ = stopped
    manifest = prepare(directory)
    batch = manifest["batches"][0]
    batch["status"] = batch_status
    if journal_status:
        campaign.atomic_json(
            directory / "recovery-attempts" / batch["recovery"]["attempt_id"] / "batch.json",
            {"status": journal_status},
        )
    with pytest.raises(campaign.CampaignError, match="terminal attempt journal"):
        recovery.run_recovery_batch(
            directory,
            manifest,
            batch,
            resume=True,
            eval_fn=lambda *_a, **_k: pytest.fail("Redispatched completed recovery"),
        )
