"""Exercise the full campaign's eight-way sandbox concurrency without model charges."""

import os

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
