"""Fresh campaign execution and whole-batch restart, using local logs only."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, write_eval_log
from inspect_ai.model import GenerateConfig

from scheming_inspect import campaign as c
from scheming_inspect.protocol import plan

MODEL = "mockllm/model"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(c, "report_key_usage", lambda _: {"status": "unavailable"})


def task_factory(experiment="oversight", n=2, **kwargs):
    return SimpleNamespace(
        dataset=[
            Sample(id=f"{experiment}-{i:04d}", input=f"original fixture {i}")
            for i in range(1, n + 1)
        ],
        config=GenerateConfig(),
    )


def config(**overrides):
    return dict(
        model=MODEL,
        max_steps=20,
        max_tokens=8000,
        token_limit=100000,
        max_samples=8,
        batch_size=2,
        poststages="none",
        **overrides,
    )


def batch(n=2):
    return {
        "batch_id": "base-000-00000",
        "sample_ids": [f"oversight-{i:04d}" for i in range(1, n + 1)],
        "status": "planned",
    }


def write_log(
    task, kwargs=None, *, missing=(), errors=(), status="success", behaviors=None, **eval_kwargs
):
    kwargs = kwargs or eval_kwargs
    samples = []
    for sample in task.dataset:
        if sample.id in missing:
            continue
        samples.append(
            EvalSample(
                id=sample.id,
                epoch=1,
                input=sample.input,
                target="",
                metadata={
                    "behavior": (behaviors or {}).get(
                        sample.id, {"completed": 1, "choice_recorded": 1}
                    )
                },
                error={
                    "message": "fixture error",
                    "traceback": "fixture",
                    "traceback_ansi": "fixture",
                }
                if sample.id in errors
                else None,
            )
        )
    log = EvalLog(
        status=status,
        samples=samples,
        eval=EvalSpec(
            created="2026-10-28T00:00:00Z",
            task="campaign-test",
            dataset=EvalDataset(samples=len(task.dataset)),
            model=kwargs["model"],
            config=EvalConfig(epochs=1),
            metadata=kwargs["metadata"],
        ),
    )
    write_eval_log(log, Path(kwargs["log_dir"]) / "result.eval")
    return [log]


def execute(directory, evaluator=write_log, *, builder=None, item=None, **kwargs):
    return c.execute_batch(
        builder or (lambda: task_factory()),
        batch=item or batch(),
        log_dir=directory / "one",
        model=MODEL,
        eval_fn=evaluator,
        **kwargs,
    )


def test_executes_without_hashes_or_financial_configuration(tmp_path):
    calls = []

    def evaluate(task, **kwargs):
        calls.append(kwargs)
        return write_log(task, kwargs, errors=("oversight-0002",))

    result = execute(tmp_path, evaluate)
    assert result["status"] == "complete"
    assert result["sample_status_counts"] == {"success": 1, "error": 1}
    assert calls[0]["max_retries"] == calls[0]["retry_on_error"] == 0
    assert calls[0]["max_samples"] == 8
    assert "model_args" not in calls[0]
    assert calls[0]["metadata"] == {"campaign_batch_id": batch()["batch_id"]}
    assert not any("hash" in key or "budget" in key for key in result)


def test_completed_batch_is_preserved_without_building_or_generating(tmp_path):
    first = execute(tmp_path)
    result = execute(
        tmp_path,
        builder=lambda: pytest.fail("Rebuilt completed task"),
        evaluator=lambda *_a, **_k: pytest.fail("Regenerated completed task"),
    )
    assert result == first


@pytest.mark.parametrize("status", ["success", "error", "cancelled"])
def test_stale_journal_with_full_terminal_coverage_retains_adverse_outcomes(tmp_path, status):
    result = execute(
        tmp_path,
        lambda task, **kw: write_log(
            task,
            kw,
            status=status,
            errors=("oversight-0002",),
            behaviors={
                "oversight-0001": {"completed": 1, "choice_recorded": 0, "choice_rejected": 1}
            },
        ),
    )
    result["status"] = "started"
    c.atomic_json(tmp_path / "one" / "batch.json", result)
    out = execute(tmp_path, builder=lambda: pytest.fail("Rebuilt complete log"))
    assert out["status"] == "complete"
    assert out["sample_status_counts"] == {"blocked_choice": 1, "error": 1}
    assert not (tmp_path / "_interrupted").exists()


def test_partial_attempt_is_archived_and_entire_batch_rebuilt(tmp_path):
    first = execute(tmp_path, lambda task, **kw: write_log(task, kw, missing=("oversight-0002",)))
    assert first["status"] == "interrupted"
    original = (tmp_path / "one" / "result.eval").read_bytes()
    calls = []

    def evaluate(task, **kwargs):
        calls.append([sample.id for sample in task.dataset])
        return write_log(task, kwargs)

    result = execute(tmp_path, evaluate)
    assert calls == [batch()["sample_ids"]]
    assert result["status"] == "complete"
    assert len(result["archived_attempts"]) == 1
    archived = Path(result["archived_attempts"][0])
    assert archived.parent == tmp_path / "_interrupted"
    assert (archived / "result.eval").read_bytes() == original


def test_nonterminal_log_with_all_ids_is_archived(tmp_path):
    first = execute(tmp_path, lambda task, **kw: write_log(task, kw, status="started"))
    assert first["status"] == "interrupted"
    assert len(execute(tmp_path)["archived_attempts"]) == 1


@pytest.mark.parametrize("exception", [KeyboardInterrupt, RuntimeError])
def test_exception_is_recorded_and_complete_saved_log_is_not_replayed(tmp_path, exception):
    def evaluate(task, **kwargs):
        write_log(task, kwargs)
        raise exception("fixture")

    with pytest.raises(exception):
        execute(tmp_path, evaluate)
    assert c.read_json(tmp_path / "one" / "batch.json")["error_type"] == exception.__name__
    assert (
        execute(tmp_path, builder=lambda: pytest.fail("Rebuilt saved results"))["status"]
        == "complete"
    )


def test_no_saved_log_after_interrupt_restarts_once(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        execute(tmp_path, lambda *_a, **_k: (_ for _ in ()).throw(KeyboardInterrupt()))
    assert execute(tmp_path)["status"] == "complete"


def test_completed_missing_log_and_changed_batch_identity_never_regenerate(tmp_path):
    first = execute(tmp_path)
    changed = batch()
    changed["sample_ids"] = ["another-id"]
    with pytest.raises(c.CampaignError, match="identity"):
        execute(tmp_path, item=changed)
    Path(first["log_paths"][0]).unlink()
    with pytest.raises(c.CampaignError, match="completed batch"):
        execute(tmp_path, builder=lambda: pytest.fail("Rebuilt lost completed log"))


@pytest.mark.parametrize("change", ["batch", "model", "duplicate", "unexpected", "epoch"])
def test_wrong_log_identity_is_not_archived_or_replayed(tmp_path, change):
    first = execute(tmp_path, lambda task, **kw: write_log(task, kw, missing=("oversight-0002",)))
    path = Path(first["log_paths"][0])
    log = c.read_log(path)
    if change == "batch":
        log.eval.metadata["campaign_batch_id"] = "other"
    elif change == "model":
        log.eval.model = "mockllm/other"
    elif change == "duplicate":
        log.samples.append(log.samples[0])
    elif change == "unexpected":
        log.samples[0].id = "other"
    else:
        log.samples[0].epoch = 2
    write_eval_log(log, path)
    with pytest.raises(c.CampaignError):
        # Inspect's ZIP writer coalesces duplicate IDs; inject that malformed parsed case.
        execute(
            tmp_path,
            builder=lambda: pytest.fail("Rebuilt wrong log"),
            read_log_fn=(lambda _: log) if change == "duplicate" else None,
        )
    assert path.exists() and not (tmp_path / "_interrupted").exists()


def test_wrong_generated_ids_stop_before_model_call(tmp_path):
    with pytest.raises(c.CampaignError, match="original sample IDs"):
        execute(
            tmp_path,
            builder=lambda: task_factory(n=1),
            evaluator=lambda *_a, **_k: pytest.fail("Model call"),
        )
    assert c.read_json(tmp_path / "one" / "batch.json")["status"] == "interrupted"


def test_batch_lock_prevents_two_campaign_processes(tmp_path):
    with c.exclusive_lock(tmp_path / ".campaign.lock"):
        with pytest.raises(c.CampaignError, match="Another process"):
            with c.exclusive_lock(tmp_path / ".campaign.lock"):
                pytest.fail("Second lock acquired")


def test_full_matrix_schedule_preserves_all_original_ids():
    cells = plan("full")
    batches = c.make_schedule(cells, 32, task_factory=task_factory)
    assert len(cells) == 65 and sum(len(b["sample_ids"]) for b in batches) == 9094
    assert batches[0]["batch_id"] == "base-000-00000" and len(batches[0]["sample_ids"]) == 32
    for index, cell in enumerate(cells):
        ids = [
            sample_id for b in batches if b["cell_index"] == index for sample_id in b["sample_ids"]
        ]
        assert ids == [sample.id for sample in task_factory(**cell).dataset]


def test_run_base_slices_original_cells_and_emits_only_complete_records(tmp_path):
    manifest = c.prepare_manifest(
        tmp_path,
        config=config(),
        cells=[{"experiment": "oversight", "n": 5}],
        task_factory=task_factory,
    )
    observed = []

    def evaluate(task, **kwargs):
        observed.extend((sample.id, sample.input) for sample in task.dataset)
        return write_log(task, kwargs)

    result = c.run_base(tmp_path, manifest, task_factory=task_factory, eval_fn=evaluate)
    assert result["status"] == "base_complete"
    assert observed == [(f"oversight-{i:04d}", f"original fixture {i}") for i in range(1, 6)]
    records = c.base_log_records(result)
    assert [row["expected_sample_ids"] for row in records] == [
        b["sample_ids"] for b in result["batches"]
    ]
    assert set(records[0]) == {"path", "cell", "cell_index", "batch_id", "expected_sample_ids"}
    assert "completed_base_log_records" not in result
    c.run_base(tmp_path, result, task_factory=lambda **_kw: pytest.fail("Rebuilt complete base"))
    assert c.campaign_status(tmp_path)["completed_samples"] == 5


def test_interrupted_base_resumes_with_whole_batch_and_excludes_archived_logs(tmp_path):
    manifest = c.prepare_manifest(
        tmp_path,
        config=config(),
        cells=[{"experiment": "oversight", "n": 2}],
        task_factory=task_factory,
    )
    result = c.run_base(
        tmp_path,
        manifest,
        task_factory=task_factory,
        eval_fn=lambda task, **kw: write_log(task, kw, missing=("oversight-0002",)),
    )
    assert result["status"] == "interrupted" and c.base_log_records(result) == []
    result = c.run_base(
        tmp_path, c.read_manifest(tmp_path), task_factory=task_factory, eval_fn=write_log
    )
    assert result["status"] == "base_complete" and len(c.base_log_records(result)) == 1
    assert "_interrupted" not in Path(c.base_log_records(result)[0]["path"]).parts


def test_old_format_rejected_and_existing_fresh_directory_requires_resume(tmp_path):
    c.atomic_json(tmp_path / "manifest.json", {"version": 1, "status": "complete"})
    with pytest.raises(c.CampaignError, match="new directory"):
        c.read_manifest(tmp_path)
    (tmp_path / "manifest.json").unlink()
    c.prepare_manifest(
        tmp_path,
        config=config(),
        cells=[{"experiment": "oversight", "n": 1}],
        task_factory=task_factory,
    )
    with pytest.raises(c.CampaignError, match="--resume"):
        c.prepare_manifest(tmp_path, config=config(), cells=[], task_factory=task_factory)


def test_cli_resume_uses_saved_settings_and_cells_not_defaults(tmp_path, monkeypatch):
    cfg = config() | {"max_tokens": 123, "seed": 19, "profile": "smoke", "batch_size": 1}
    manifest = c.prepare_manifest(
        tmp_path, config=cfg, cells=[{"experiment": "oversight", "n": 1}], task_factory=task_factory
    )
    called = []

    def run(directory, saved):
        called.append(saved)
        return saved | {"status": "base_complete"}

    monkeypatch.setattr(c, "run_base", run)
    monkeypatch.setattr(c, "load_dotenv", lambda *_a, **_k: None)
    c.main(["--campaign-dir", str(tmp_path), "--execute", "--resume"])
    assert called[0]["config"] == cfg and called[0]["cells"] == manifest["cells"]


def test_dry_plan_does_not_create_campaign_or_call_models(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(c, "inspect_eval", lambda *_a, **_k: pytest.fail("Model call"))
    c.main(["--campaign-dir", str(tmp_path), "--profile", "full"])
    result = json.loads(capsys.readouterr().out)
    assert result["samples"] == 9094
    assert not (tmp_path / "manifest.json").exists()


def test_corrupt_partial_log_is_archived_but_corrupt_completed_log_is_preserved(tmp_path):
    partial = execute(tmp_path, lambda task, **kw: write_log(task, kw, missing=("oversight-0002",)))
    path = Path(partial["log_paths"][0])
    path.write_bytes(b"interrupted ZIP write")
    resumed = execute(tmp_path)
    assert resumed["status"] == "complete"
    assert (
        Path(resumed["archived_attempts"][0]) / path.name
    ).read_bytes() == b"interrupted ZIP write"
    path.write_bytes(b"missing completed evidence")
    with pytest.raises(c.CampaignError, match="completed batch"):
        execute(tmp_path, builder=lambda: pytest.fail("Regenerated completed batch"))
    assert path.read_bytes() == b"missing completed evidence"


@pytest.mark.parametrize("resume", [False, True])
def test_poststage_batch_size_rejected_before_base_calls(tmp_path, monkeypatch, resume):
    monkeypatch.setattr(c, "run_base", lambda *_a, **_kw: pytest.fail("Base calls"))
    if resume:
        cfg = config() | {"batch_size": 1, "poststages": "all"}
        c.prepare_manifest(
            tmp_path,
            config=cfg,
            cells=[{"experiment": "oversight", "n": 1}],
            task_factory=task_factory,
        )
    args = [
        "--campaign-dir",
        str(tmp_path),
        "--execute",
        "--poststages",
        "all",
        "--batch-size",
        "1",
    ]
    with pytest.raises(SystemExit) as exc:
        c.main(args + (["--resume"] if resume else []))
    assert exc.value.code == 2


def test_status_reports_interrupted_poststage_even_with_complete_base(tmp_path, monkeypatch):
    from scheming_inspect import poststages

    c.prepare_manifest(
        tmp_path,
        config=config(),
        cells=[{"experiment": "oversight", "n": 1}],
        task_factory=task_factory,
    )
    c.atomic_json(tmp_path / "poststages.json", {"status": "interrupted"})
    report = {"status": "interrupted", "pending_samples": 4}
    monkeypatch.setattr(poststages, "progress_report", lambda _: report)
    status = c.campaign_status(tmp_path)
    assert status["status"] == "interrupted" and status["poststages"] == report


@pytest.mark.parametrize(
    "behavior,error,limit,expected",
    [
        ({"completed": 1, "choice_recorded": 0, "choice_rejected": 1}, "failure", None, "error"),
        ({"completed": 1, "choice_recorded": 0, "choice_rejected": 1}, None, "token", "limited"),
        ({"completed": 0, "choice_recorded": 0, "choice_rejected": 1}, None, None, "incomplete"),
        ({"completed": 1, "schema_valid": 0}, None, None, "invalid_schema"),
        ({"completed": 1, "choice_recorded": 0, "choice_rejected": 0}, None, None, "no_choice"),
    ],
)
def test_sample_outcome_keeps_failure_priority_and_distinguishes_missing_choice(
    behavior, error, limit, expected
):
    sample = SimpleNamespace(
        metadata={"behavior": behavior}, error=error, limit=limit, invalidation=None
    )
    assert c.sample_status(sample) == expected
