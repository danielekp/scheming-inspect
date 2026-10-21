"""One-way runner revision preserves results and historical financial evidence."""

import hashlib

import pytest

from scheming_inspect import campaign as c
from scripts.use_uncapped_runner import POLICY, migrate


@pytest.fixture
def stopped(tmp_path):
    log = tmp_path / "base" / "done" / "result.eval"
    log.parent.mkdir(parents=True)
    log.write_bytes(b"original saved results")
    cells = [{"experiment": "oversight", "n": 2}]
    cfg = {"model": c.MODEL, "budget_usd": 50, "stage_budget_usd": 35}
    hashes = {"scheming_inspect/campaign.py": "old", "data/scenarios.json": "unchanged"}
    record = dict(
        path=str(log),
        cell_index=0,
        cell=cells[0],
        batch_id="done",
        expected_sample_ids=["one"],
        recovery_retained=False,
        log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
    )
    m = dict(
        config=cfg,
        cells=cells,
        source_hashes=hashes,
        status="pending_budget",
        total_samples=2,
        completed_base_log_records=[record],
        completed_base_logs=[str(log)],
        batches=[
            dict(
                batch_id="done",
                cell_index=0,
                sample_ids=["one"],
                status="complete_reserved",
                log_paths=[str(log)],
            ),
            dict(batch_id="next", cell_index=0, sample_ids=["two"], status="pending_budget"),
        ],
    )
    m["fingerprint"] = c.fingerprint({k: m[k] for k in ("config", "cells", "source_hashes")})
    c.atomic_json(tmp_path / "manifest.json", m)
    c.atomic_json(
        tmp_path / "budget.json",
        {"batches": {"done": {"status": "closed_reserved"}}, "limit_usd": 50},
    )
    return tmp_path, m, {**hashes, "scheming_inspect/campaign.py": "new"}, log


def test_switch_preserves_results_budget_and_case_inventory(stopped):
    directory, before, hashes, log = stopped
    original = log.read_bytes()
    budget = (directory / "budget.json").read_bytes()
    out = migrate(directory, hashes=hashes)
    assert out["execution_policy"] == POLICY
    assert out["config"] == before["config"]  # historical settings, now ignored
    assert out["completed_base_log_records"] == before["completed_base_log_records"]
    assert out["batches"][0] == before["batches"][0]
    assert out["batches"][1]["status"] == "planned"
    assert log.read_bytes() == original
    assert (directory / "budget.json").read_bytes() == budget
    assert out["status"] == "planned"
    assert migrate(directory, hashes=hashes) == out
    assert len(out["source_revisions"]) == 1


@pytest.mark.parametrize("issue", ["data_change", "active", "missing_log", "journal"])
def test_switch_refuses_unrelated_or_ambiguous_changes(stopped, issue):
    directory, manifest, hashes, log = stopped
    if issue == "data_change":
        hashes["data/scenarios.json"] = "different"
    if issue == "active":
        manifest["status"] = "running"
        c.atomic_json(directory / "manifest.json", manifest)
    if issue == "missing_log":
        log.unlink()
    if issue == "journal":
        c.atomic_json(directory / "base" / "next" / "batch.json", {"status": "started"})
    before = (directory / "manifest.json").read_bytes()
    with pytest.raises((c.CampaignError, FileNotFoundError)):
        migrate(directory, hashes=hashes)
    assert (directory / "manifest.json").read_bytes() == before
