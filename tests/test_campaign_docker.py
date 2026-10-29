"""Exercise the full campaign's eight-way sandbox concurrency without model charges."""

import json
import os
import re
import shlex

import pytest
from inspect_ai import eval
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput, get_model
from inspect_ai.tool import ToolCall

from scheming_inspect import campaign
from scheming_inspect.tasks import ROOT, scheming


@pytest.mark.docker
@pytest.mark.skipif(os.getenv("RUN_DOCKER_TESTS") != "1", reason="Requires Docker opt-in")
def test_eight_parallel_sandboxes_keep_their_own_files(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(ROOT / "inspect-data"))
    task = scheming("oversight", n=8)
    for sample in task.dataset:
        sample.files["/workspace/campaign-marker.txt"] = f"marker:{sample.id}\n"
    fixtures = {str(sample.id): sample for sample in task.dataset}
    input_paths = {path for sample in task.dataset for path in sample.files}

    def scripted(messages, tools, tool_choice, config):
        users = [i for i, message in enumerate(messages) if message.role == "user"]
        observed = [message.text for message in messages[users[-1] + 1 :] if message.role == "tool"]
        if len(users) == 1:
            if not observed:
                name, arguments = (
                    "bash",
                    {
                        "cmd": "cat /workspace/planning/task.md; printf '\\n'; "
                        "cat /workspace/campaign-marker.txt"
                    },
                )
            else:
                sample_id = next(
                    line.removeprefix("marker:")
                    for line in observed[0].splitlines()
                    if line.startswith("marker:")
                )
                answer = "I completed the routine request. Results: " + json.dumps(
                    fixtures[sample_id].metadata["ordinary_reference"], sort_keys=True
                )
                if len(observed) == 2:
                    name, arguments = "submit", {"answer": answer}
                else:
                    records = [
                        path
                        for path in re.findall(
                            r"/workspace/[\w./-]+\.(?:csv|md|txt|json)\b", observed[0]
                        )
                        if path in input_paths and path != "/workspace/planning/task.md"
                    ]
                    assert records, "The ordinary brief should identify a real input document."
                    name, arguments = (
                        "bash",
                        {
                            "cmd": "cat "
                            + " ".join(shlex.quote(path) for path in records)
                            + "; mkdir -p /workspace/reports; "
                            "cp /workspace/campaign-marker.txt /workspace/reports/ordinary-marker.txt; "
                            f"printf '%s' {shlex.quote(answer)} > /workspace/reports/ordinary-summary.txt"
                        },
                    )
        else:
            name = "submit" if observed else "bash"
            arguments = (
                {"answer": observed[-1]}
                if observed
                else {
                    "cmd": "cmp /workspace/campaign-marker.txt "
                    "/workspace/reports/ordinary-marker.txt && "
                    "cat /workspace/reports/ordinary-marker.txt "
                    "/workspace/reports/ordinary-summary.txt"
                }
            )
        return ModelOutput(
            model="mockllm/parallel",
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(
                        content="",
                        tool_calls=[ToolCall(id=name, function=name, arguments=arguments)],
                    ),
                    stop_reason="tool_calls",
                )
            ],
        )

    def evaluate(task, **kwargs):
        kwargs["model"] = get_model("mockllm/parallel", custom_outputs=scripted)
        return eval(task, **kwargs, max_sandboxes=8, display="none")

    batch = {
        "batch_id": "base-000-00000",
        "sample_ids": [str(sample.id) for sample in task.dataset],
        "status": "planned",
    }
    record = campaign.execute_batch(
        lambda: task,
        batch=batch,
        model="mockllm/parallel",
        log_dir=tmp_path / "batch",
        eval_fn=evaluate,
    )
    assert record["status"] == "complete"
    log = campaign.read_log(record["log_paths"][0])
    assert log.status == "success"
    assert len(log.samples) == 8
    for sample in log.samples:
        assert sample.error is None
        assert sample.metadata["ordinary_completed"] is True
        assert sample.metadata["followup_started"] is True
        fixture = fixtures[str(sample.id)]
        users = [message for message in sample.messages if message.role == "user"]
        assert [message.text for message in users] == [
            fixture.metadata["ordinary_request"],
            fixture.metadata["followup_user"],
        ]
        systems = [message for message in sample.messages if message.role == "system"]
        assert len(systems) == 1
        assert systems[0].text == fixture.input[0].text
        assert sample.metadata["ordinary_answer"] in sample.metadata["final_answer"]
        boundary = sample.metadata["followup_message_index"]
        assert sample.messages[boundary - 1].role == "tool"
        assert sample.messages[boundary - 1].function == "submit"
        assert sample.messages[boundary - 2].tool_calls[0].arguments == {
            "answer": sample.metadata["ordinary_answer"]
        }
        assert f"marker:{sample.id}" in sample.metadata["final_answer"]
        tool_text = "\n".join(message.text for message in sample.messages if message.role == "tool")
        assert f"marker:{sample.id}" in tool_text
        for other in log.samples:
            if other.id != sample.id:
                assert f"marker:{other.id}" not in tool_text
    assert (
        campaign.execute_batch(
            lambda: pytest.fail("Rebuilt completed sandbox task"),
            batch=batch,
            model="mockllm/parallel",
            log_dir=tmp_path / "batch",
            eval_fn=lambda *_a, **_kw: pytest.fail("Regenerated completed sandbox task"),
        )["status"]
        == "complete"
    )
