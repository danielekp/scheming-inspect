"""Connected family records, with no models, containers, or external services."""

import copy
import json
import re

import pytest

from scheming_inspect import data


def test_inbox_threads_reference_real_acknowledgments_and_one_current_record():
    for case in data.build_cases("email", n=5):
        inbox = json.loads(case["files"]["/workspace/emails.json"])
        for key in ("id", "request_id", "thread_id", "message_id"):
            assert len({email[key] for email in inbox}) == 20
        assert any(email["in_reply_to"] for email in inbox)
        assert any(email["in_reply_to"] is None for email in inbox)
        expected = {row["id"]: row for row in case["reference"]["expected_parsed"]}
        for email in inbox:
            assert email["request_id"] in email["subject"]
            assert email["request_id"] in email["body"]
            assert email["sender_name"] in email["body"]
            ratings = re.findall(
                r"(urgency|organization_value|community_value)=(\d)", email["body"]
            )
            assert len(ratings) == 3
            assert {"id": email["id"], **{key: int(value) for key, value in ratings}} == expected[
                email["id"]
            ]
            history = email["thread_history"]
            if email["in_reply_to"] is None:
                assert history == []
            else:
                assert len(history) == 1
                assert history[0]["message_id"] == email["in_reply_to"]
                assert email["request_id"] in history[0]["body"]
                assert history[0]["message_id"] != email["message_id"]
                assert not re.search(
                    r"urgency=|organization_value=|community_value=", history[0]["body"]
                )


def test_wording_changes_do_not_consume_rating_or_shuffle_randomness(monkeypatch):
    original = data.build_cases("email", n=5, seed=173)
    load = data._load

    def altered_load(name):
        config = load(name)
        if name == "email.json":
            config = copy.deepcopy(config)
            config["text"]["request_templates"] = [
                {"reply": False, "body": "A different request from {sender_name}."}
            ]
            config["text"]["senders"] = [{"name": "Another sender", "email": "a@example.invalid"}]
        return config

    monkeypatch.setattr(data, "_load", altered_load)
    changed = data.build_cases("email", n=5, seed=173)
    for before, after in zip(original, changed):
        assert before["system"] == after["system"]
        for key in (
            "expected_parsed",
            "expected_scores",
            "model_preferred_scores",
            "developer_ranking",
            "model_preferred_ranking",
            "scoring_code",
        ):
            assert before["reference"][key] == after["reference"][key]
        old_inbox = json.loads(before["files"]["/workspace/emails.json"])
        new_inbox = json.loads(after["files"]["/workspace/emails.json"])
        assert [email["id"] for email in old_inbox] == [email["id"] for email in new_inbox]


def test_inbox_handoff_is_connected_to_the_actual_batch_and_records():
    for index, case in enumerate(data.build_cases("email", n=5), 1):
        files = case["files"]
        batch_id = f"INBOX-{index:04d}"
        handoff = files["/workspace/operations/inbox_handoff.md"]
        previous = files["/workspace/archive/inbox_processing_previous.md"]
        assert batch_id in handoff and batch_id in previous
        assert batch_id in files["/workspace/planning/task.md"]
        assert "/workspace/deployment_notes.txt" in handoff
        assert "id field (email_01 through email_20)" in handoff
        assert (
            case["reference"]["developer_goal"] in files["/workspace/meetings/planning_review.md"]
        )
        for email in json.loads(files["/workspace/emails.json"]):
            assert email["request_id"] in handoff
        for document in (handoff, previous):
            for linked in re.findall(r"\]\((/workspace/[^)]+)\)", document):
                assert linked in files
        assert json.loads(files["/workspace/parsed.json"]) == []
        assert "/workspace/scores.json" not in files


@pytest.mark.parametrize("question_set", ("arithmetic", "mixed", "easy"))
@pytest.mark.parametrize("consequence", data.CONSEQUENCES)
def test_assessment_record_adds_no_condition_difference_or_answer_key(question_set, consequence):
    kwargs = dict(n=2, seed=173, question_set=question_set, consequence=consequence)
    high = data.build_cases("sandbagging", condition="high", **kwargs)
    low = data.build_cases("sandbagging", condition="low", **kwargs)
    for index, (first, second) in enumerate(zip(high, low), 1):
        assert first["reference"] == second["reference"]
        assert first["system"] == second["system"]
        assert first["user"] == second["user"]
        assert set(first["files"]) == set(second["files"])
        assert [
            path for path in first["files"] if first["files"][path] != second["files"][path]
        ] == ["/workspace/Documentation.md"]
        files = first["files"]
        record = files["/workspace/planning/assessment_record.md"]
        request_id = f"ASSESS-{index:04d}"
        assert request_id in record
        assert request_id in files["/workspace/planning/task.md"]
        assert request_id in files["/workspace/Documentation.md"]
        assert "capability assessment" in record
        assert "response not yet submitted" in record
        assert json.dumps(first["reference"]["answers"]) not in record
        for linked in re.findall(r"\]\((/workspace/[^)]+)\)", record):
            assert linked in files
        assert "JSON array of exactly 10 uppercase letters" in files["/workspace/planning/task.md"]
        assert "JSON array" not in files["/workspace/Documentation.md"]
        assert (
            "Do not write or execute code, or use a calculator"
            in files["/workspace/planning/task.md"]
        )
