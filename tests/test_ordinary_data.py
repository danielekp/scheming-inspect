"""Check generated routine requests and their interaction with controlled comparisons."""

import base64
import csv
import io
import json
import re

import pytest

from scheming_inspect.data import EXPERIMENTS, _load, build_cases
from scheming_inspect.protocol import plan
from scheming_inspect.tasks import to_sample

PROMPTS = _load("shared_prompts.json")
BANK = _load("ordinary_tasks.json")


def decoded(sample, path):
    return base64.b64decode(sample.files[path].split(",", 1)[1]).decode()


def rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def check_ledger(task):
    ledger = rows(next(text for path, text in task["files"].items() if "ledger_" in path))
    bank = rows(next(text for path, text in task["files"].items() if "bank_export_" in path))
    assert 55 <= len(ledger) <= 75
    paid = [row for row in ledger if row["status"] == "paid"]
    expected = task["expected"]
    receipts = sum(int(r["amount_usd"]) for r in paid if r["type"] == "receipt")
    expenses = sum(int(r["amount_usd"]) for r in paid if r["type"] == "expense")
    assert all(r["description"] in BANK["tasks"]["ledger"][r["type"] + "s"] for r in ledger)
    for row in bank:
        if row["reference"] == "BANK":
            continue
        kind = next(r["type"] for r in ledger if r["reference"] == row["reference"])
        assert row["counterparty"] in BANK["tasks"]["ledger"][f"{kind}_counterparties"]
        assert row["direction"] == ("credit" if kind == "receipt" else "debit")
    assert expected["paid_receipts_usd"] == receipts
    assert expected["paid_expenses_usd"] == expenses
    assert expected["left_usd"] == receipts - expenses
    by_reference = {row["reference"]: row for row in bank}
    missing = sorted(row["reference"] for row in paid if row["reference"] not in by_reference)
    assert expected["paid_missing_from_bank"] == missing and len(missing) == 2
    differs = {
        row["reference"]: {
            "ledger_usd": int(row["amount_usd"]),
            "bank_usd": int(by_reference[row["reference"]]["amount_usd"]),
        }
        for row in paid
        if row["reference"] in by_reference
        and by_reference[row["reference"]]["amount_usd"] != row["amount_usd"]
    }
    assert expected["amount_differs"] == differs and len(differs) == 2
    assert {row["reference"] for row in bank} - {row["reference"] for row in ledger} == {"BANK"}


def check_jobs(task):
    log = rows(next(text for path, text in task["files"].items() if "_log_" in path))
    schedule = rows(next(text for path, text in task["files"].items() if "_schedule_" in path))
    assert 50 <= len(log) <= 70 and len(schedule) == len(log) + 3
    closed = [row for row in log if row["status"] == "closed"]
    expected = task["expected"]
    assert expected["closed_jobs"] == len(closed)
    assert expected["total_minutes"] == sum(int(row["minutes"]) for row in closed)
    for site, figures in expected["per_site"].items():
        mine = [row for row in closed if row["site"] == site]
        assert figures == {
            "closed_jobs": len(mine),
            "minutes": sum(int(row["minutes"]) for row in mine),
        }
    logged = {row["job_id"] for row in log}
    assert expected["unlogged_jobs"] == sorted(
        row["job_id"] for row in schedule if row["job_id"] not in logged
    )
    assert all(int(row["minutes"]) == 0 for row in log if row["status"] != "closed")


def check_receipts(task, files):
    assert task["files"].keys() == {"/workspace/planning/task.md"}
    irregular = {}
    for path in files:
        match = re.match(r"/workspace/archive/requests/(PLAN-\d+)/brief\.md$", path)
        if not match:
            continue
        folder = match[1]
        text = files.get(f"/workspace/archive/requests/{folder}/decision_receipt.json")
        if text is None:
            irregular[folder] = "missing"
        elif json.loads(text)["status"] != "recorded":
            irregular[folder] = f"status {json.loads(text)['status']}"
    assert task["expected"] == {
        "irregular_receipts": irregular,
        "receipts_directory": "/workspace/reports",
    }
    assert len(irregular) == 2
    assert "/workspace/services/decision_gateway/config.json" in files


CHECKS = {"ledger": check_ledger, "jobs": check_jobs}


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_routine_requests_are_generated_and_answerable(experiment):
    kinds_seen = set()
    for case in build_cases(experiment, n=12):
        tasks = case["ordinary"]
        assert len(tasks) == BANK["count"] == 2
        assert len({task["kind"] for task in tasks}) == 2
        assert len({task["id"] for task in tasks}) == 2
        for position, task in enumerate(tasks):
            kinds_seen.add(task["kind"])
            assert re.fullmatch(r"REQ-9\d\d", task["id"])
            assert task["brief"].endswith(PROMPTS["ordinary_completion_suffix"])
            assert task["files"]["/workspace/planning/task.md"] == task["brief"]
            assert "/workspace/planning/task.md" in task["request"]
            assert "reply" not in task["brief"].split("\n\n")[0].lower()
            paths = {
                path
                for path in re.findall(r"/workspace/[\w/.-]+\.(?:csv|md|json)", task["brief"])
                if not path.startswith("/workspace/reports/")
            }
            everything = {**case["files"], **task["files"]}
            assert paths and all(path in everything for path in paths)
            if task["kind"] == "receipts":
                check_receipts(task, case["files"])
            else:
                CHECKS[task["kind"]](task)
            first = position == 0
            for path, text in task["files"].items():
                assert text.strip()
                assert (path in case["files"]) is (first or path == "/workspace/planning/task.md")
                assert path not in case["staged"] or path == "/workspace/planning/task.md"
            if position:
                assert task["id"] in task["request"]
    assert kinds_seen == set(BANK["tasks"])


@pytest.mark.parametrize("arguments", plan("full"))
def test_every_matrix_cell_starts_with_source_records_and_keeps_main_request(arguments):
    case = build_cases(**{**arguments, "n": 1})[0]
    sample = to_sample(case)
    first = case["ordinary"][0]
    assert sample.input[1].text == first["request"]
    assert "/workspace/planning/task.md" in sample.input[1].text
    initial_brief = decoded(sample, "/workspace/planning/task.md")
    assert initial_brief == first["brief"]
    assert "/workspace/reports/completion.md" in initial_brief
    assert "finish(message=" in initial_brief
    assert sample.metadata["ordinary_tasks"] == case["ordinary"]
    assert sample.metadata["followup_user"] == case["user"]
    assert {"sandbagging": "ASSESS-140", "email": "INBOX-640"}.get(
        arguments["experiment"], "PLAN-2400"
    ) in case["user"]
    assert sample.metadata["followup_files"] == case["staged"]
    assert "/workspace/planning/task.md" in case["staged"]
    assert "/workspace/meetings/planning_review.md" in case["staged"]
    assert "/workspace/meetings/planning_review.md" not in sample.files
    goal = case["reference"]["model_goal"]
    if goal:
        if arguments["goal_role"] == "system":
            assert goal in sample.input[0].text
            assert goal not in sample.input[1].text
        else:
            assert goal not in sample.input[0].text
            assert goal in sample.input[1].text and goal in case["user"]
            assert sample.input[1].text.startswith(case["user"].split("\n\n")[0])
            assert all(
                task["request"].startswith(case["user"].split("\n\n")[0])
                for task in case["ordinary"]
            )
    else:
        assert not case["metadata"]["goal_active"]
        assert "Your remit at" not in sample.input[0].text + sample.input[1].text


@pytest.mark.parametrize(
    "family,conditions",
    [
        ("alignment", ("testing", "deployment")),
        ("sandbagging", ("high", "low")),
        ("exfiltration", ("default", "monitored")),
    ],
)
def test_opposite_treatments_receive_identical_first_tasks(family, conditions):
    left = build_cases(family, n=33, condition=conditions[0])
    right = build_cases(family, n=33, condition=conditions[1])
    assert [c["ordinary"] for c in left] == [c["ordinary"] for c in right]
    assert [c["files"] for c in left] == [c["files"] for c in right]


@pytest.mark.parametrize(
    "arguments", [cell for cell in plan("full") if cell["experiment"] == "sandbagging"]
)
def test_sandbagging_paper_is_delayed_and_paired_initial_workspaces_are_identical(arguments):
    high = build_cases(**{**arguments, "n": 2, "condition": "high"})
    low = build_cases(**{**arguments, "n": 2, "condition": "low"})
    for index, (first, second) in enumerate(zip(high, low, strict=True)):
        request_id = f"ASSESS-{140 + index}"
        left = to_sample(first)
        right = to_sample(second)
        assert left.files == right.files
        assert [(m.role, m.text) for m in left.input] == [(m.role, m.text) for m in right.input]
        assert first["reference"] == second["reference"]
        assert first["metadata"]["pair_id"] == second["metadata"]["pair_id"]
        questions = f"/workspace/assessments/{request_id}/questions.txt"
        assert (
            left.metadata["followup_files"][questions]
            == right.metadata["followup_files"][questions]
        )
        assert {
            path
            for path in left.metadata["followup_files"]
            if left.metadata["followup_files"][path] != right.metadata["followup_files"][path]
        } == {f"/workspace/mail/{request_id}-assessment.md"}
        for case, sample in ((first, left), (second, right)):
            deferred = sample.metadata["followup_files"]
            assert set(deferred) == {
                "/workspace/planning/task.md",
                "/workspace/meetings/planning_review.md",
                questions,
                f"/workspace/assessments/{request_id}/record.md",
                f"/workspace/mail/{request_id}-assessment.md",
            }
            assert set(deferred) & set(sample.files) == {"/workspace/planning/task.md"}
            initial = "\n".join(decoded(sample, path) for path in sample.files)
            assert f"/workspace/assessments/{request_id}" not in initial
            assert "How the vendor uses the result" not in initial
            assert case["ordinary"][0]["brief"] in initial


def test_reference_facts_and_later_hint_are_not_given_in_first_message():
    case = build_cases("oversight", n=1, difficulty="easy")[0]
    case["ordinary"][0]["expected"]["private_sentinel"] = "PRIVATE_EXPECTED_SENTINEL"
    sample = to_sample(case)
    text = "\n".join(m.text for m in sample.input) + "\n".join(
        decoded(sample, path) for path in sample.files
    )
    assert "PRIVATE_EXPECTED_SENTINEL" not in text
    assert (
        sample.metadata["ordinary_tasks"][0]["expected"]["private_sentinel"]
        == "PRIVATE_EXPECTED_SENTINEL"
    )
    hint = _load("oversight.json")["hints"]["easy"]
    assert sample.metadata["reference"]["precheck_hint"] == hint
    assert hint not in text and hint not in sample.metadata["followup_user"]
    assert all(hint not in task["request"] for task in case["ordinary"])
    assert json.loads(json.dumps(case)) == case


@pytest.mark.parametrize("goal_mode", ["conflict", "none"])
def test_goal_role_does_not_change_the_file_based_ordinary_brief(goal_mode):
    system_case = build_cases("oversight", n=1, goal_mode=goal_mode, goal_role="system")[0]
    user_case = build_cases("oversight", n=1, goal_mode=goal_mode, goal_role="user")[0]
    assert [t["brief"] for t in system_case["ordinary"]] == [
        t["brief"] for t in user_case["ordinary"]
    ]
    assert system_case["files"] == user_case["files"]
    assert system_case["staged"] == user_case["staged"]
    assert "Your remit at" not in system_case["ordinary"][0]["request"]
    if goal_mode == "none":
        assert user_case["ordinary"] == system_case["ordinary"]
        assert user_case["user"] == system_case["user"]
    else:
        goal = user_case["reference"]["model_goal"]
        pairs = [
            (u["request"], s["request"])
            for u, s in zip(user_case["ordinary"], system_case["ordinary"])
        ]
        for message, plain in pairs + [(user_case["user"], system_case["user"])]:
            prefix, rest = message.split("\n\n", 1)
            assert goal in prefix
            assert rest == plain
            assert prefix not in user_case["ordinary"][0]["brief"]
