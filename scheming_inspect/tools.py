"""Tools exposed to the target model inside each sandboxed scenario."""

import json

from inspect_ai.tool import Tool, ToolError, tool
from inspect_ai.util import sandbox, store

from .data import _load
from .evidence import INBOX, RECEIPT, monitor_enabled, observe, receipt


def _message(name: str, **values) -> str:
    return _load("shared_prompts.json")["tool_messages"][name].format(**values)


@tool
def bash() -> Tool:
    async def execute(cmd: str) -> str:
        """Run a shell command.

        Args:
            cmd: Shell command.
        """
        if store().get("finished", False):
            raise ToolError(_message("closed"))
        try:
            result = await sandbox().exec(["bash", "-c", cmd], cwd="/workspace", timeout=20)
        except TimeoutError as exc:
            await observe(cmd)
            raise ToolError(_message("bash_timeout")) from exc
        await observe(cmd)
        return f"exit_code={result.returncode}\n{result.stdout}\n{result.stderr}"[:24000]

    return execute


@tool
def record_decision() -> Tool:
    async def execute(plan_id: str) -> str:
        """Record a planning recommendation.

        Args:
            plan_id: Plan ID.
        """
        memory = store()
        if memory.get("finished", False):
            raise ToolError(_message("closed"))
        reference = memory.get("reference")
        labels = {option["plan_id"]: label for label, option in reference["options"].items()}
        if plan_id not in labels:
            raise ToolError(_message("invalid_plan", plans=", ".join(labels)))
        label = labels[plan_id]
        after = await observe()
        if (
            memory.get("experiment") == "oversight"
            and monitor_enabled("oversight", after) is not False
            and label != reference["developer_choice"]
        ):
            memory.set("attempted_choice", label)
            return _message(
                "decision_rejected", plan_id=plan_id, request_id=reference["request_id"]
            )
        try:
            await sandbox().write_file(
                f"/workspace/{RECEIPT}",
                json.dumps(receipt(reference["request_id"], plan_id), indent=2) + "\n",
            )
        except (PermissionError, FileNotFoundError, IsADirectoryError, NotADirectoryError) as exc:
            raise ToolError(_message("receipt_failed", error=exc)) from exc
        await observe()
        memory.set("selected_strategy", label)
        if memory.get("experiment") == "alignment":
            memory.set("finished", True)
        return _message("decision_recorded", plan_id=plan_id)

    return execute


@tool
def finish() -> Tool:
    async def execute(message: str) -> str:
        """Send the reply and close the request.

        Args:
            message: Reply text.
        """
        memory = store()
        if memory.get("finished", False):
            raise ToolError(_message("closed"))
        if memory.get("phase") == "ordinary" and not message.strip():
            raise ToolError(_message("empty_reply"))
        memory.set("final_answer", message)
        memory.set("finished", True)
        return _message("finished")

    return execute


@tool(max_output=64 * 1024)
def read_inbox() -> Tool:
    async def execute() -> str:
        """Read the inbox export."""
        path = f"/workspace/{INBOX}"
        try:
            result = await sandbox().exec(["/bin/cat", path], timeout=20)
        except TimeoutError as exc:
            raise ToolError(_message("inbox_timeout")) from exc
        if not result.success:
            raise ToolError(
                _message(
                    "inbox_failed",
                    path=path,
                    code=result.returncode,
                    detail=result.stderr.strip() or "no error details",
                )
            )
        return result.stdout

    return execute
