"""Inbox reads preserve sandbox output and distinguish file errors from infrastructure."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from inspect_ai.tool import ToolDef, ToolError
from inspect_ai.util import ExecResult

from scheming_inspect import tools
from scheming_inspect.data import build_cases
from scheming_inspect.evidence import INBOX

PATH = f"/workspace/{INBOX}"


def fake_sandbox(result=None, error=None):
    return SimpleNamespace(
        exec=AsyncMock(return_value=result, side_effect=error),
        read_file=AsyncMock(side_effect=AssertionError("Inbox helper used read_file")),
    )


def test_full_inboxes_fit_the_tool_output_limit():
    limit = ToolDef(tools.read_inbox()).max_output
    assert limit is not None and limit > 0
    # Cover all five organizations and six wording rotations.
    for case in build_cases("email", n=30):
        assert len(case["staged"][PATH].encode()) < limit


@pytest.mark.parametrize("raw", ['[\n  {"subject": "Hello"}\n]\n', "not JSON\n", ""])
async def test_inbox_returns_raw_stdout_from_fixed_sandbox_path(raw):
    sandbox = fake_sandbox(ExecResult(success=True, returncode=0, stdout=raw, stderr=""))
    helper = tools.read_inbox()
    with (
        patch.object(tools, "sandbox", return_value=sandbox),
        patch("builtins.open", side_effect=AssertionError("Inbox helper opened a host file")),
        patch.object(Path, "open", side_effect=AssertionError("Inbox helper opened a host path")),
    ):
        assert await helper() == raw
    sandbox.exec.assert_awaited_once_with(["/bin/cat", PATH], timeout=20)
    sandbox.read_file.assert_not_called()


@pytest.mark.parametrize(
    "returncode,stderr",
    [
        (1, f"/bin/cat: {PATH}: No such file or directory\n"),
        (1, f"/bin/cat: {PATH}: Permission denied\n"),
        (2, f"/bin/cat: {PATH}: Input/output error\n"),
    ],
)
async def test_inbox_command_failure_preserves_actual_diagnostic(returncode, stderr):
    sandbox = fake_sandbox(
        ExecResult(success=False, returncode=returncode, stdout="", stderr=stderr)
    )
    with patch.object(tools, "sandbox", return_value=sandbox), pytest.raises(ToolError) as error:
        await tools.read_inbox()()
    assert stderr.strip() in str(error.value)
    assert str(returncode) in str(error.value)
    sandbox.exec.assert_awaited_once_with(["/bin/cat", PATH], timeout=20)


async def test_inbox_timeout_is_a_recoverable_tool_error():
    sandbox = fake_sandbox(error=TimeoutError("sandbox command exceeded its deadline"))
    with patch.object(tools, "sandbox", return_value=sandbox), pytest.raises(ToolError) as error:
        await tools.read_inbox()()
    assert "20" in str(error.value)
    assert "time" in str(error.value).lower()


@pytest.mark.parametrize(
    "failure",
    [FileNotFoundError("container runtime disappeared"), RuntimeError("sandbox transport failed")],
)
async def test_inbox_infrastructure_errors_propagate(failure):
    sandbox = fake_sandbox(error=failure)
    with (
        patch.object(tools, "sandbox", return_value=sandbox),
        pytest.raises(type(failure)) as error,
    ):
        await tools.read_inbox()()
    assert error.value is failure
    sandbox.read_file.assert_not_called()
