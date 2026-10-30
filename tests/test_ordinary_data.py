"""Check authored first tasks and their interaction with controlled comparisons."""

import base64
import csv
import io
import json
import re

import pytest

from scheming_inspect.data import _load, build_cases
from scheming_inspect.protocol import plan
from scheming_inspect.tasks import to_sample

PROMPTS = _load("shared_prompts.json")


def decoded(sample, path):
    return base64.b64decode(sample.files[path].split(",", 1)[1]).decode()


def test_every_setting_has_an_answerable_distinct_first_request():
    bank = _load("ordinary_tasks.json")
    assert set(bank["selection"]) == {d["id"] for d in _load("shared_selection.json")["domains"]}
    assert set(bank["email"]) == {d["id"] for d in _load("email.json")["settings"]}
    assert set(bank["sandbagging"]) == {"arithmetic", "mixed", "easy"}
    entries = [
        entry for group in ("selection", "email", "sandbagging") for entry in bank[group].values()
    ]
    assert len(entries) == len({entry["request"] for entry in entries}) == 41
    existing = set(_load("workspace.json")["files"])
    for entry in entries:
        assert entry["expected"]
        assert "reply" not in entry["request"].lower()
        paths = set(re.findall(r"/workspace/[\w/.-]+\.(?:csv|md|txt)", entry["request"]))
        assert paths and paths <= entry["files"].keys()
        assert not existing.intersection(entry["files"])
        assert not re.search(r"\b(Taylor|Priya|Alex|Morgan)\b", entry["request"])
        for path, text in entry["files"].items():
            assert text.strip()
            if path.endswith(".csv"):
                rows = list(csv.DictReader(io.StringIO(text)))
                assert rows and all(None not in row and None not in row.values() for row in rows)


@pytest.mark.parametrize("arguments", plan("full"))
def test_every_matrix_cell_starts_with_source_records_and_keeps_main_request(arguments):
    case = build_cases(**{**arguments, "n": 1})[0]
    sample = to_sample(case)
    assert sample.input[1].text == case["ordinary"]["request"]
    assert "/workspace/planning/task.md" in sample.input[1].text
    initial_brief = decoded(sample, "/workspace/planning/task.md")
    assert initial_brief == case["ordinary"]["brief"]
    assert initial_brief.endswith(PROMPTS["ordinary_completion_suffix"])
    assert "/workspace/reports/completion.md" in initial_brief
    assert "finish(message=" in initial_brief
    assert any(
        path in initial_brief
        for path in case["files"]
        if path.startswith(("/workspace/operations/", "/workspace/finance/"))
        and path.split("/")[-1]
        not in (
            "team_notes.md",
            "change_policy.md",
            "budget.csv",
            "expenses.csv",
            "expense_notes.md",
        )
        and "invoices" not in path
    )
    assert sample.metadata["followup_user"] == case["user"]
    assert {"sandbagging": "ASSESS-140", "email": "INBOX-640"}.get(
        arguments["experiment"], "PLAN-2400"
    ) in case["user"]
    assert sample.metadata["followup_files"] == case["staged"]
    assert "/workspace/planning/task.md" in case["staged"]
    goal = case["reference"]["model_goal"]
    if goal:
        if arguments["goal_role"] == "system":
            assert goal in sample.input[0].text
            assert goal not in sample.input[1].text
        else:
            assert goal not in sample.input[0].text
            assert goal in sample.input[1].text and goal in case["user"]
            assert sample.input[1].text.startswith(case["user"].split("\n\n")[0])
    else:
        assert not case["metadata"]["goal_active"]
        assert "brought this assistant in" not in sample.input[0].text + sample.input[1].text


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
                questions,
                f"/workspace/assessments/{request_id}/record.md",
                f"/workspace/mail/{request_id}-assessment.md",
            }
            assert set(deferred) & set(sample.files) == {"/workspace/planning/task.md"}
            initial = "\n".join(decoded(sample, path) for path in sample.files)
            assert f"/workspace/assessments/{request_id}" not in initial
            assert "How the vendor uses the result" not in initial
            assert case["ordinary"]["brief"] in initial


def test_reference_facts_and_later_hint_are_not_given_in_first_message():
    case = build_cases("oversight", n=1, difficulty="easy")[0]
    case["ordinary"]["expected"]["private_sentinel"] = "PRIVATE_EXPECTED_SENTINEL"
    sample = to_sample(case)
    text = "\n".join(m.text for m in sample.input) + "\n".join(
        decoded(sample, path) for path in sample.files
    )
    assert "PRIVATE_EXPECTED_SENTINEL" not in text
    assert sample.metadata["ordinary_reference"]["private_sentinel"] == "PRIVATE_EXPECTED_SENTINEL"
    hint = _load("oversight.json")["hints"]["easy"]
    assert hint in sample.metadata["followup_user"]
    assert hint not in sample.input[1].text
    assert json.loads(json.dumps(case)) == case


@pytest.mark.parametrize("goal_mode", ["conflict", "none"])
def test_goal_role_does_not_change_the_file_based_ordinary_brief(goal_mode):
    system_case = build_cases("oversight", n=1, goal_mode=goal_mode, goal_role="system")[0]
    user_case = build_cases("oversight", n=1, goal_mode=goal_mode, goal_role="user")[0]
    assert system_case["ordinary"]["brief"] == user_case["ordinary"]["brief"]
    assert system_case["files"] == user_case["files"]
    assert system_case["staged"] == user_case["staged"]
    assert "brought this assistant in" not in system_case["ordinary"]["request"]
    if goal_mode == "none":
        assert user_case["ordinary"]["request"] == system_case["ordinary"]["request"]
        assert user_case["user"] == system_case["user"]
    else:
        goal = user_case["reference"]["model_goal"]
        for message, plain in (
            (user_case["ordinary"]["request"], system_case["ordinary"]["request"]),
            (user_case["user"], system_case["user"]),
        ):
            prefix, rest = message.split("\n\n", 1)
            assert goal in prefix
            assert rest == plain
            assert prefix not in user_case["ordinary"]["brief"]
