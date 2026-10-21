"""Exercise the full campaign's eight-way sandbox concurrency without model charges."""

import os

import pytest
from inspect_ai import eval
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelOutput, get_model
from inspect_ai.tool import ToolCall

from scheming_inspect.tasks import ROOT, scheming


@pytest.mark.docker
@pytest.mark.skipif(os.getenv("RUN_DOCKER_TESTS") != "1", reason="Requires Docker opt-in")
def test_eight_parallel_sandboxes_keep_their_own_files(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(ROOT / "inspect-data"))
    task = scheming("oversight", n=8)
    for sample in task.dataset:
        sample.files["/workspace/campaign-marker.txt"] = f"marker:{sample.id}\n"

    def scripted(messages, tools, tool_choice, config):
        observed = [message.text for message in messages if message.role == "tool"]
        name = "submit" if observed else "bash"
        arguments = (
            {"answer": observed[-1]} if observed else {"cmd": "cat /workspace/campaign-marker.txt"}
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

    logs = eval(
        task,
        model=get_model("mockllm/parallel", custom_outputs=scripted),
        log_dir=str(tmp_path),
        max_samples=8,
        max_sandboxes=8,
        display="none",
        ctl_server=False,
    )
    assert logs[0].status == "success"
    assert len(logs[0].samples) == 8
    for sample in logs[0].samples:
        assert sample.error is None
        tool_text = "\n".join(message.text for message in sample.messages if message.role == "tool")
        assert f"marker:{sample.id}" in tool_text
        for other in logs[0].samples:
            if other.id != sample.id:
                assert f"marker:{other.id}" not in tool_text
