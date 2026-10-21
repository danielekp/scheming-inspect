"""Runner-only retry policy: mocked child processes and clock, no external calls."""

import json
import signal
from types import SimpleNamespace

import pytest

from scripts import run_campaign_with_billing_retry as wrapper


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def state(tmp_path):
    root = tmp_path
    (root / "scheming_inspect").mkdir()
    (root / "scheming_inspect" / "frozen.py").write_text("unchanged")
    directory = root / "logs" / "campaign with spaces"
    directory.mkdir(parents=True)
    hashes = wrapper.source_hashes(root)

    def write(stamp=1, batch_id="batch-1"):
        log_dir = directory / "base" / batch_id
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / "result.eval"
        log.write_bytes(b"frozen completed evaluation")
        error = "Billing unavailable for gen-late (BillingError)."
        batch = {
            "status": "blocked",
            "errors": [error],
            "reconciled_at": str(stamp),
            "generation_ids": ["gen-late"],
            "metadata": {"log_dir": str(log_dir), "fingerprint": "frozen"},
            "log_fingerprints": {str(log): wrapper.file_digest(log)},
        }
        ledger = {
            "batches": {
                batch_id: batch,
                "old": {"status": "closed_reserved", "upper_usd": "6.735898624"},
            },
            "generations": {},
        }
        journal = {
            "batch_id": batch_id,
            "status": "blocked",
            "complete": True,
            "errors": [],
            "sample_status_counts": {"success": 32},
            "fingerprint": "frozen",
            "settlement": {"status": "blocked", "errors": [error], "reconciled_at": str(stamp)},
            "log_paths": [str(log)],
        }
        (directory / "budget.json").write_text(json.dumps(ledger))
        (directory / "manifest.json").write_text(json.dumps({"source_hashes": hashes}))
        (log_dir / "batch.json").write_text(json.dumps(journal))
        return ledger, journal, log_dir

    return SimpleNamespace(
        root=root,
        directory=directory,
        write=write,
        args=[
            "--campaign-dir",
            str(directory),
            "--budget-usd",
            "50",
            "--max-samples",
            "8",
            "--execute",
        ],
    )


def run_loop(state, run, clock=None, args=None, reports=None):
    clock = clock or Clock()
    reports = reports if reports is not None else []
    result = wrapper.run_with_billing_retries(
        state.args if args is None else args,
        root=state.root,
        run=run,
        sleep=clock.sleep,
        monotonic=clock.time,
        report=lambda event, **fields: reports.append({"event": event, **fields}),
    )
    return result, clock, reports


def test_late_billing_resolves_with_exact_original_argv_plus_resume(state):
    commands = []
    original_args = list(state.args)

    def run(command, *, cwd):
        assert cwd == state.root
        commands.append(list(command))
        if len(commands) == 1:
            state.write()
            return 1
        return 0

    result, clock, reports = run_loop(state, run)
    assert result == 0
    assert commands[0][3:] == original_args
    assert commands[1] == commands[0] + ["--resume"]
    assert state.args == original_args
    assert sum(clock.sleeps) == 15
    assert [row["event"] for row in reports] == ["billing_retry_wait", "billing_retry_resume"]
    assert (
        json.loads((state.directory / "budget.json").read_text())["batches"]["old"]["upper_usd"]
        == "6.735898624"
    )


def test_existing_resume_flag_is_preserved_without_duplication(state):
    commands = []

    def run(command, **_):
        commands.append(command)
        state.write(len(commands))
        return 1 if len(commands) == 1 else 0

    result, _, _ = run_loop(state, run, args=state.args + ["--resume"])
    assert result == 0
    assert all(command.count("--resume") == 1 for command in commands)
    assert commands[0] == commands[1]


def test_repeated_missing_billing_exhausts_bounded_delays(state):
    commands = []

    def run(command, **_):
        commands.append(command)
        state.write(len(commands))
        return 1

    result, clock, reports = run_loop(state, run)
    assert result == 1
    assert len(commands) == len(wrapper.DELAYS) + 1
    assert all(command[-1] == "--resume" for command in commands[1:])
    assert sum(clock.sleeps) == sum(wrapper.DELAYS) == 885
    assert sum(clock.sleeps) < wrapper.MAX_WAIT_SECONDS
    assert max(clock.sleeps) <= 30
    assert reports[-1]["event"] == "billing_retry_exhausted"


def test_reconciliation_time_consumes_the_same_batch_deadline(state):
    clock, starts = Clock(), []

    def run(command, **_):
        starts.append(clock.now)
        clock.now += 300
        state.write(len(starts))
        return 1

    result, clock, reports = run_loop(state, run, clock)
    assert result == 1
    assert len(starts) == 4
    assert all(start < 300 + wrapper.MAX_WAIT_SECONDS for start in starts[1:])
    assert sum(clock.sleeps) == 105  # The deadline is not restarted after each failure.
    assert reports[-1]["event"] == "billing_retry_exhausted"


def test_overslept_deadline_does_not_launch_another_child(state):
    clock, calls = Clock(), []

    def run(command, **_):
        calls.append(command)
        state.write()
        return 1

    def oversleep(seconds):
        clock.now += wrapper.MAX_WAIT_SECONDS + 1

    result = wrapper.run_with_billing_retries(
        state.args,
        root=state.root,
        run=run,
        sleep=oversleep,
        monotonic=clock.time,
        report=lambda *a, **k: None,
    )
    assert result == 1
    assert len(calls) == 1


def test_new_batch_gets_new_window_after_hours_of_successful_work(state):
    clock, calls = Clock(), []

    def run(command, **_):
        calls.append(command)
        if len(calls) == 1:
            state.write(1, "first")
            return 1
        if len(calls) == 2:
            clock.now += 7200
            state.write(2, "later")
            return 1
        return 0

    result, clock, reports = run_loop(state, run, clock)
    assert result == 0
    waits = [row for row in reports if row["event"] == "billing_retry_wait"]
    assert [row["attempt"] for row in waits] == [1, 1]
    assert sum(clock.sleeps) == 30


def test_old_billing_block_cannot_trigger_retry_of_new_unrelated_failure(state):
    state.write()
    calls = []

    def run(command, **_):
        calls.append(command)
        return 1  # Argument/config/source/lock failure did not reconcile billing.

    result, clock, _ = run_loop(state, run)
    assert result == 1
    assert len(calls) == 1
    assert clock.sleeps == []


@pytest.mark.parametrize(
    "damage",
    [
        "incomplete",
        "coverage-error",
        "model-error",
        "missing-id",
        "mixed-error",
        "other-error-class",
        "source-hash",
        "log-hash",
        "stale-journal",
        "two-active",
        "budget",
        "bad-json",
    ],
)
def test_only_complete_fresh_billing_only_blocks_are_retried(state, damage):
    calls = []

    def run(command, **_):
        calls.append(command)
        ledger, journal, log_dir = state.write()
        batch = ledger["batches"]["batch-1"]
        if damage == "incomplete":
            journal["complete"] = False
        elif damage == "coverage-error":
            journal["errors"] = ["Sample IDs missing"]
        elif damage == "model-error":
            journal["sample_status_counts"]["error"] = 1
        elif damage == "missing-id":
            batch["errors"] = ["A noncached model event has no exact OpenRouter generation ID."]
        elif damage == "mixed-error":
            batch["errors"].append(
                "A model request failed or remains pending; charges may be unknown."
            )
        elif damage == "other-error-class":
            batch["errors"] = ["Billing unavailable for gen-late (ValueError)."]
        elif damage == "source-hash":
            (state.root / "scheming_inspect" / "frozen.py").write_text("changed")
        elif damage == "log-hash":
            (log_dir / "result.eval").write_bytes(b"changed")
        elif damage == "stale-journal":
            journal["settlement"]["reconciled_at"] = "old"
        elif damage == "two-active":
            ledger["batches"]["other"] = {"status": "reserved"}
        elif damage == "budget":
            batch["status"] = "settled"
        (state.directory / "budget.json").write_text(json.dumps(ledger))
        (log_dir / "batch.json").write_text(json.dumps(journal))
        if damage == "bad-json":
            (state.directory / "budget.json").write_text("{")
        return 1

    result, clock, _ = run_loop(state, run)
    assert result == 1
    assert len(calls) == 1
    assert clock.sleeps == []


@pytest.mark.parametrize("returncode", [0, 2, 130, 143, -signal.SIGINT, -signal.SIGTERM])
def test_exit_codes_and_signals_other_than_blocked_do_not_retry(state, returncode):
    calls = []

    def run(command, **_):
        calls.append(command)
        state.write()
        return returncode

    result, clock, _ = run_loop(state, run)
    assert result == (returncode if returncode >= 0 else 128 - returncode)
    assert len(calls) == 1
    assert clock.sleeps == []


def test_interrupt_during_wait_never_resumes(state):
    calls = []

    def run(command, **_):
        calls.append(command)
        state.write()
        return 1

    def interrupt(_):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        wrapper.run_with_billing_retries(
            state.args, root=state.root, run=run, sleep=interrupt, report=lambda *a, **k: None
        )
    assert len(calls) == 1


@pytest.mark.parametrize("signal_number", [signal.SIGINT, signal.SIGTERM])
def test_child_receives_stop_signal_and_is_not_restarted(monkeypatch, tmp_path, signal_number):
    calls, sent = [], []

    class Child:
        pid = 999999

        def wait(self, timeout=None):
            calls.append(timeout)
            if timeout is None:
                raise wrapper.Stopped(signal_number)
            return -signal_number

    def popen(command, **kwargs):
        assert kwargs == {"cwd": tmp_path, "start_new_session": True}
        return Child()

    monkeypatch.setattr(wrapper.subprocess, "Popen", popen)
    monkeypatch.setattr(wrapper.os, "killpg", lambda pid, sig: sent.append((pid, sig)))
    with pytest.raises(wrapper.Stopped):
        wrapper.run_child(["fake"], cwd=tmp_path)
    assert sent == [(999999, signal_number)]
    assert calls == [None, 15]


def test_last_campaign_directory_option_and_user_home_are_respected(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert (
        wrapper.campaign_directory(["--campaign-dir", "/first", "--campaign-dir=~/later"], tmp_path)
        == tmp_path / "home" / "later"
    )


def test_wrapper_source_inventory_matches_frozen_campaign_and_excludes_scripts():
    from scheming_inspect.campaign import source_hashes

    assert wrapper.source_hashes(wrapper.ROOT) == source_hashes()
    assert not any(name.startswith(("scripts/", "tests/")) for name in source_hashes())


def test_launcher_preserves_authorized_budget_and_execution_settings():
    script = (wrapper.ROOT / "scripts" / "run_full.sh").read_text()
    assert "python scripts/run_campaign_with_billing_retry.py" in script
    for argument in (
        "--budget-usd 50",
        "--stage-budget-usd 35",
        "--starting-spend-usd 0.013229117",
        "--max-samples 8",
        "--batch-size 32",
        "--max-tokens 8000",
        "--poststages all",
    ):
        assert argument in script
    assert '--execute "$@"' in script
