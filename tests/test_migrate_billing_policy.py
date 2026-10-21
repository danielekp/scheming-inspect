"""Offline accounting revisions preserve execution records and fail closed."""

import copy
import json
from types import SimpleNamespace

import pytest

from scheming_inspect import campaign as c
from scripts import migrate_billing_policy as migration


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def stopped(tmp_path, monkeypatch):
    directory = tmp_path / "campaign"
    batch_id = "old-base-003-00128"
    log_dir = directory / "base" / batch_id
    log_dir.mkdir(parents=True)
    log_path = log_dir / "result.eval"
    log_path.write_bytes(b"immutable offline evaluation")
    hashes = {key: "old" for key in migration.ALLOWED_FILES}
    hashes.update(
        {
            "scheming_inspect/engine.py": "engine",
            "scheming_inspect/tasks.py": "tasks",
            "data/scenarios.json": "data",
        }
    )
    current = dict(
        hashes,
        **{
            "scheming_inspect/budget.py": "new-budget",
            "scheming_inspect/campaign.py": "new-campaign",
        },
    )
    config = dict(
        model=c.MODEL,
        max_steps=20,
        max_tokens=8000,
        token_limit=100000,
        max_samples=8,
        batch_size=32,
        budget_usd=50,
        stage_budget_usd=35,
        starting_spend_usd=0.01,
        retries=0,
        epochs=1,
        cache_prompt=False,
    )
    provider = {
        "max_price": {"prompt": 0.1, "completion": 0.2, "request": 0},
        "require_parameters": True,
    }
    ids = ["email-0129", "email-0130"]
    batch = dict(
        batch_id=batch_id,
        cell_index=0,
        sample_ids=ids,
        status="blocked",
        reservation={"provider": provider},
    )
    manifest = dict(
        config=config,
        cells=[dict(experiment="email", n=300)],
        source_hashes=hashes,
        batches=[batch],
        status="blocked",
        completed_base_logs=[],
    )
    manifest["fingerprint"] = c.fingerprint(migration.identity(manifest))
    metadata = dict(
        batch_id=batch_id,
        stage="base",
        model=c.MODEL,
        fingerprint=manifest["fingerprint"],
        sample_ids=ids,
        max_samples=8,
        model_roles={},
        log_dir=str(log_dir),
    )
    journal = metadata | dict(
        status="blocked",
        complete=True,
        errors=[],
        upper_usd=1,
        provider=provider,
        log_paths=[str(log_path)],
    )
    record = dict(
        status="blocked",
        upper_usd="1",
        metadata=metadata,
        log_fingerprints={str(log_path): migration.sha(log_path.read_bytes())},
    )
    ledger = dict(
        limit_usd="50",
        starting_spend_usd="0.01",
        stage_limits={"base": "35"},
        batches={batch_id: record},
        generations={},
    )
    paths = dict(
        manifest=directory / "manifest.json",
        budget=directory / "budget.json",
        journal=log_dir / "batch.json",
        log=log_path,
    )
    dump(paths["manifest"], manifest)
    dump(paths["budget"], ledger)
    dump(paths["journal"], journal)
    log = SimpleNamespace(
        status="success",
        eval=SimpleNamespace(
            model=c.MODEL,
            metadata={
                "campaign_batch_id": batch_id,
                "campaign_fingerprint": manifest["fingerprint"],
            },
        ),
        samples=[
            SimpleNamespace(
                id=key,
                epoch=1,
                error=None,
                limit=None,
                metadata={"behavior": {"completed": 1, "schema_valid": 1}},
            )
            for key in ids
        ],
    )
    monkeypatch.setattr(c, "source_hashes", lambda: copy.deepcopy(current))

    def forbidden(*args, **kwargs):
        pytest.fail("Migration must not dispatch or contact billing")

    monkeypatch.setattr(c, "inspect_eval", forbidden)
    monkeypatch.setattr(c, "get_model", forbidden)
    return SimpleNamespace(
        directory=directory,
        paths=paths,
        manifest=manifest,
        ledger=ledger,
        journal=journal,
        current=current,
        log=log,
        read=lambda path: log,
        before={key: path.read_bytes() for key, path in paths.items()},
    )


def apply(stopped):
    return migration.migrate(
        stopped.directory,
        reason="Preserve uncertain bills without rerunning complete work",
        read_log_fn=stopped.read,
    )


def test_revision_archives_exact_bytes_and_only_changes_manifest_metadata(stopped):
    result = apply(stopped)
    assert result["status"] == "applied"
    updated = c.read_json(stopped.paths["manifest"])
    assert updated["fingerprint"] == c.fingerprint(migration.identity(updated))
    revision = updated["source_revisions"][0]
    assert revision["accounting_only"] is True
    assert revision["old_source_hashes"] == stopped.manifest["source_hashes"]
    assert revision["new_source_hashes"] == stopped.current
    archive = stopped.directory / "accounting-revisions"
    plans = list(archive.glob("*/plan.json"))
    assert len(plans) == 1
    for kind in ("manifest", "budget", "journal"):
        assert (plans[0].parent / f"{kind}-before.json").read_bytes() == stopped.before[kind]
    for key, value in stopped.manifest.items():
        if key not in {"source_hashes", "fingerprint", "updated_at", "source_revisions"}:
            assert updated[key] == value
    for kind in ("budget", "journal", "log"):
        assert stopped.paths[kind].read_bytes() == stopped.before[kind]
    original = stopped.paths["manifest"].read_bytes()
    assert apply(stopped)["status"] == "already_applied"
    assert stopped.paths["manifest"].read_bytes() == original
    assert len(list(archive.iterdir())) == 1


@pytest.mark.parametrize(
    "path",
    [
        "scheming_inspect/engine.py",
        "scheming_inspect/tasks.py",
        "data/scenarios.json",
        "pyproject.toml",
    ],
)
def test_rejects_non_accounting_changes_and_new_inventory(stopped, path):
    stopped.current[path] = "changed"
    with pytest.raises(c.CampaignError, match="Only existing"):
        apply(stopped)
    assert stopped.paths["manifest"].read_bytes() == stopped.before["manifest"]
    assert not (stopped.directory / "accounting-revisions").exists()


def test_rejects_config_change_even_with_recomputed_fingerprint(stopped):
    stopped.manifest["config"]["budget_usd"] = 51
    stopped.manifest["fingerprint"] = c.fingerprint(migration.identity(stopped.manifest))
    dump(stopped.paths["manifest"], stopped.manifest)
    with pytest.raises(c.CampaignError, match="budget caps differ"):
        apply(stopped)


def test_rejects_model_change_even_with_recomputed_fingerprint(stopped):
    stopped.manifest["config"]["model"] = "openrouter/other/model"
    stopped.manifest["fingerprint"] = c.fingerprint(migration.identity(stopped.manifest))
    dump(stopped.paths["manifest"], stopped.manifest)
    with pytest.raises(c.CampaignError, match="identity or frozen bounds"):
        apply(stopped)


@pytest.mark.parametrize(
    "damage",
    ["missing_sample", "wrong_log_fingerprint", "changed_file", "unfinished", "provider_cap"],
)
def test_rejects_incomplete_or_changed_execution_evidence(stopped, damage):
    if damage == "missing_sample":
        stopped.log.samples.pop()
    elif damage == "wrong_log_fingerprint":
        stopped.log.eval.metadata["campaign_fingerprint"] = "unknown"
    elif damage == "changed_file":
        stopped.paths["log"].write_bytes(b"changed")
    elif damage == "unfinished":
        stopped.journal["complete"] = False
        dump(stopped.paths["journal"], stopped.journal)
    elif damage == "provider_cap":
        stopped.journal["provider"] = {"max_price": {"prompt": 100}}
        dump(stopped.paths["journal"], stopped.journal)
    with pytest.raises(c.CampaignError):
        apply(stopped)
    assert stopped.paths["manifest"].read_bytes() == stopped.before["manifest"]


@pytest.mark.parametrize("lock", [".campaign.lock", ".billing-retry.lock"])
def test_requires_both_nonblocking_worker_locks(stopped, lock):
    with c.exclusive_lock(stopped.directory / lock):
        with pytest.raises(c.CampaignError, match="Another process"):
            apply(stopped)
    assert stopped.paths["manifest"].read_bytes() == stopped.before["manifest"]


@pytest.mark.parametrize("after_replace", [False, True])
def test_interrupted_commit_resumes_without_duplicate_revision(stopped, monkeypatch, after_replace):
    atomic_json = c.atomic_json

    def interrupt(path, value):
        if path == stopped.paths["manifest"]:
            if after_replace:
                atomic_json(path, value)
            raise OSError("simulated process interruption")
        atomic_json(path, value)

    monkeypatch.setattr(c, "atomic_json", interrupt)
    with pytest.raises(OSError, match="interruption"):
        apply(stopped)
    monkeypatch.setattr(c, "atomic_json", atomic_json)
    result = apply(stopped)
    assert result["status"] == ("already_applied" if after_replace else "applied")
    assert len(c.read_json(stopped.paths["manifest"])["source_revisions"]) == 1
    assert len(list((stopped.directory / "accounting-revisions").iterdir())) == 1
    for kind in ("budget", "journal", "log"):
        assert stopped.paths[kind].read_bytes() == stopped.before[kind]


def test_original_journal_fingerprint_from_recorded_history_is_accepted(stopped):
    old = stopped.manifest["fingerprint"]
    old_hashes = copy.deepcopy(stopped.manifest["source_hashes"])
    stopped.manifest["source_hashes"]["scheming_inspect/recovery.py"] = "intermediate"
    new = c.fingerprint(migration.identity(stopped.manifest))
    stopped.manifest["fingerprint"] = new
    original_revision = {
        "old_fingerprint": old,
        "new_fingerprint": new,
        "old_source_hashes": old_hashes,
        "new_source_hashes": copy.deepcopy(stopped.manifest["source_hashes"]),
    }
    stopped.manifest["source_revisions"] = [original_revision]
    dump(stopped.paths["manifest"], stopped.manifest)
    assert apply(stopped)["status"] == "applied"
    revisions = c.read_json(stopped.paths["manifest"])["source_revisions"]
    assert len(revisions) == 2
    assert revisions[0] == original_revision


def test_applied_archive_corruption_is_rejected_on_repeat(stopped):
    apply(stopped)
    plan_path = next((stopped.directory / "accounting-revisions").glob("*/plan.json"))
    plan = c.read_json(plan_path)
    plan["reason"] = "tampered"
    dump(plan_path, plan)
    with pytest.raises(c.CampaignError, match="checksum"):
        apply(stopped)


def test_historical_fingerprint_does_not_authorize_changed_generation_caps(stopped):
    old = stopped.manifest["fingerprint"]
    stopped.manifest["config"]["max_tokens"] = 16000
    new = c.fingerprint(migration.identity(stopped.manifest))
    stopped.manifest["fingerprint"] = new
    stopped.manifest["source_revisions"] = [
        {
            "old_fingerprint": old,
            "new_fingerprint": new,
            "old_source_hashes": copy.deepcopy(stopped.manifest["source_hashes"]),
            "new_source_hashes": copy.deepcopy(stopped.manifest["source_hashes"]),
        }
    ]
    dump(stopped.paths["manifest"], stopped.manifest)
    with pytest.raises(c.CampaignError, match="Historical fingerprint"):
        apply(stopped)
