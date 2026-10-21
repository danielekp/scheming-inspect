"""Offline uncapped execution, exact coverage, and resume regression tests."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.event import ModelEvent
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, write_eval_log
from inspect_ai.model import GenerateConfig, ModelOutput

from scheming_inspect import campaign
from scheming_inspect.protocol import plan


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


def execute(tmp_path, evaluator, **overrides):
    kwargs = dict(
        batch_id="one",
        log_dir=tmp_path / "one",
        stage="base",
        model=campaign.MODEL,
        fingerprint="frozen",
        eval_fn=evaluator,
    )
    kwargs.update(overrides)
    return campaign.execute_batch(task_factory(n=2), **kwargs)


def test_runs_without_financial_api_and_records_errors(tmp_path):
    calls = []

    def evaluator(task, **kwargs):
        calls.append(kwargs)
        return create_log(task, kwargs, errors=("oversight-0002",))

    result = execute(tmp_path, evaluator)
    assert result["status"] == "complete"
    assert result["sample_status_counts"] == {"success": 1, "error": 1}
    assert calls[0]["model_args"] == {"provider": {}}
    assert "spending_reservation_usd" not in calls[0]["metadata"]
    assert not (tmp_path / "budget.json").exists()
    assert result["log_fingerprints"]


def test_resume_does_not_regenerate_or_require_billing(tmp_path):
    first = execute(tmp_path, lambda task, **kw: create_log(task, kw))
    result = execute(tmp_path, lambda *_a, **_k: pytest.fail("Repeated generation"), resume=True)
    assert result == first
    with pytest.raises(campaign.CampaignError, match="already exists"):
        execute(tmp_path, lambda task, **kw: create_log(task, kw))
    Path(first["log_paths"][0]).write_bytes(b"changed")
    with pytest.raises(campaign.CampaignError, match="changed"):
        execute(tmp_path, lambda task, **kw: create_log(task, kw), resume=True)


@pytest.mark.parametrize("old_status", ["settled", "complete_reserved"])
def test_legacy_completed_results_are_preserved(tmp_path, old_status):
    result = execute(tmp_path, lambda task, **kw: create_log(task, kw))
    result.update(
        status=old_status,
        settlement={
            "status": "closed_reserved",
            "actual_usd": None,
            "log_fingerprints": result.pop("log_fingerprints"),
        },
    )
    campaign.atomic_json(tmp_path / "one" / "batch.json", result)
    assert execute(tmp_path, lambda *_a, **_k: pytest.fail("Rerun"), resume=True) == result


def test_partial_batch_is_not_replayed(tmp_path):
    result = execute(tmp_path, lambda task, **kw: create_log(task, kw, missing=("oversight-0002",)))
    assert result["status"] == "blocked"
    result = execute(tmp_path, lambda *_a, **_k: pytest.fail("Rerun"), resume=True)
    assert result["status"] == "blocked"
    assert len(result["samples"]) == 1


def test_interrupt_after_generation_reconciles_saved_log(tmp_path):
    def evaluator(task, **kwargs):
        create_log(task, kwargs)
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        execute(tmp_path, evaluator)
    assert (
        execute(tmp_path, lambda *_a, **_k: pytest.fail("Rerun"), resume=True)["status"]
        == "complete"
    )


def test_changed_identity_refused(tmp_path):
    execute(tmp_path, lambda task, **kw: create_log(task, kw))
    with pytest.raises(campaign.CampaignError, match="identity"):
        execute(
            tmp_path, lambda task, **kw: create_log(task, kw), resume=True, fingerprint="changed"
        )


def test_full_schedule_keeps_every_case_once():
    cells = plan("full")
    batches = campaign.make_schedule(cells, 32, campaign_fingerprint="test")
    assert sum(len(b["sample_ids"]) for b in batches) == 9094
    assert len(cells) == 65
    for index, cell in enumerate(cells):
        ids = [sid for b in batches if b["cell_index"] == index for sid in b["sample_ids"]]
        assert len(ids) == len(set(ids)) == cell["n"]


def test_base_loop_ignores_historical_caps_and_reports_each_new_batch(tmp_path, monkeypatch):
    cfg = config() | {"budget_usd": 0, "stage_budget_usd": 0, "starting_spend_usd": 99999}
    cells = [{"experiment": "oversight", "n": 5}]
    m = campaign.prepare_manifest(
        tmp_path, config=cfg, cells=cells, hashes={}, resume=False, task_factory=task_factory
    )
    attempts, reports = [], []
    monkeypatch.setattr(
        campaign, "report_cost", lambda root: reports.append(root) or {"status": "unavailable"}
    )

    def evaluator(task, **kwargs):
        attempts.extend(str(s.id) for s in task.dataset)
        return create_log(task, kwargs)

    out = campaign.run_base(
        tmp_path,
        m,
        resume=False,
        task_factory=task_factory,
        eval_fn=evaluator,
        source_hash_fn=lambda: {},
    )
    assert out["status"] == "base_complete"
    assert attempts == [f"oversight-{i:04d}" for i in range(1, 6)]
    assert len(reports) == 3
    campaign.run_base(
        tmp_path,
        out,
        resume=True,
        task_factory=task_factory,
        eval_fn=lambda *_a, **_k: pytest.fail("Rerun"),
        source_hash_fn=lambda: {},
    )
    assert len(reports) == 3
    assert campaign.campaign_status(tmp_path)["base_samples_settled"] == 5
    assert not (tmp_path / "budget.json").exists()


def test_resume_ignores_only_historical_financial_config(tmp_path):
    cfg = config()
    cells = [{"experiment": "oversight", "n": 2}]
    old = campaign.prepare_manifest(
        tmp_path,
        config=cfg | {"budget_usd": 1},
        cells=cells,
        hashes={},
        resume=False,
        task_factory=task_factory,
    )
    assert (
        campaign.prepare_manifest(
            tmp_path, config=cfg, cells=cells, hashes={}, resume=True, task_factory=task_factory
        )
        == old
    )
    with pytest.raises(campaign.CampaignError, match="configuration"):
        campaign.prepare_manifest(
            tmp_path,
            config=cfg | {"max_tokens": 1},
            cells=cells,
            hashes={},
            resume=True,
            task_factory=task_factory,
        )


def test_main_dry_run_has_no_budget(capsys, tmp_path):
    campaign.main(["--campaign-dir", str(tmp_path), "--profile", "full"])
    data = json.loads(capsys.readouterr().out)
    assert data["samples"] == 9094 and data["spending_limit"] is None
    assert "budget_usd" not in data["config"]


def test_missing_completed_journal_never_regenerates(tmp_path, monkeypatch):
    cfg = config()
    m = campaign.prepare_manifest(
        tmp_path,
        config=cfg,
        cells=[{"experiment": "oversight", "n": 2}],
        hashes={},
        resume=False,
        task_factory=task_factory,
    )
    monkeypatch.setattr(campaign, "report_cost", lambda _: {"status": "unavailable"})
    campaign.run_base(
        tmp_path,
        m,
        resume=False,
        task_factory=task_factory,
        eval_fn=lambda task, **kw: create_log(task, kw),
        source_hash_fn=lambda: {},
    )
    (tmp_path / "base" / m["batches"][0]["batch_id"] / "batch.json").unlink()
    with pytest.raises(campaign.CampaignError, match="missing.*journal"):
        campaign.run_base(
            tmp_path,
            m,
            resume=True,
            task_factory=lambda *_a, **_k: pytest.fail("Reconstructed completed task"),
            eval_fn=lambda *_a, **_k: pytest.fail("Rerun"),
            source_hash_fn=lambda: {},
        )
