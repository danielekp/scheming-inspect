"""Generate follow-up replies and conduct blind, bounded interrogations."""

from __future__ import annotations

import copy
import json
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver

from ..data import _load
from ..judge import judge_admission, strict_json
from .samples import OVERSIGHT


class InvalidInterrogatorResponse(ValueError):
    """An emitted question/verdict fails the required response format."""


class IncompleteTargetResponse(RuntimeError):
    """The target returned an unfinished answer or requested unavailable tools."""


def record_response_failure(
    state: TaskState, output: Any, error: Exception, *, phase: str, turn: int | None = None
) -> None:
    """Retain invalid model output without guessing a repaired question or label."""
    state.store.set(
        "followup_error",
        {
            "phase": phase,
            "turn": turn,
            "error_type": type(error).__name__,
            "error": str(error),
            "raw": output.completion,
            "stop_reason": output.stop_reason,
            "tool_calls": output.message.model_dump(mode="json").get("tool_calls") or [],
        },
    )
    state.metadata["followup_status"] = "invalid_model_response"
    state.metadata["followup_failure_phase"] = phase
    state.metadata["followup_failure_turn"] = turn
    state.metadata["followup_failure_type"] = type(error).__name__
    state.metadata["followup_stop_reason"] = output.stop_reason


async def target_reply(state: TaskState, *, max_tokens: int) -> None:
    """One generation only: no automatic execution of newly requested tools."""
    state.tools = []
    output = await get_model().generate(
        state.messages,
        tools=[],
        config=GenerateConfig(max_tokens=max_tokens, cache=False, max_retries=0, timeout=90),
    )
    state.output = output
    state.messages.append(output.message)
    if output.stop_reason != "stop" or output.message.tool_calls:
        raise IncompleteTargetResponse(
            f"Target follow-up did not finish normally: {output.stop_reason}"
        )


@solver
def fixed_followup_solver(max_tokens: int = 2048) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        try:
            await target_reply(state, max_tokens=max_tokens)
        except IncompleteTargetResponse as exc:
            record_response_failure(state, state.output, exc, phase="target_reply")
        return state

    return solve


def _parse_field(output: Any, field: str, choices: set[str] | None = None) -> str:
    if output.stop_reason != "stop" or output.message.tool_calls:
        raise InvalidInterrogatorResponse(
            f"Interrogator did not finish normally: {output.stop_reason}"
        )
    try:
        obj = strict_json(output.completion)
    except (ValueError, TypeError) as exc:
        raise InvalidInterrogatorResponse("Interrogator returned invalid JSON.") from exc
    if not isinstance(obj, dict) or set(obj) != {field} or not isinstance(obj[field], str):
        raise InvalidInterrogatorResponse(
            f"Interrogator must return exactly one string field: {field}."
        )
    value = obj[field]
    if not value.strip() or (choices is not None and value not in choices):
        raise InvalidInterrogatorResponse(f"Invalid interrogator {field}.")
    return value


def blind_interrogator_messages(task_description: str) -> list[ChatMessage]:
    return [
        ChatMessageSystem(content=_load("followups.json")["interrogation"]["system"]),
        ChatMessageUser(content=json.dumps({"original_task": task_description})),
    ]


@solver
def interrogation_solver(
    max_rounds: int = 7, max_tokens: int = 2048, judge_max_tokens: int = 2048
) -> Solver:
    if not 1 <= max_rounds <= 7:
        raise ValueError("max_rounds must be in 1..7.")

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        interrogator = get_model(role="interrogator", required=True)
        # Resolve the judge before spending target/interrogator calls.
        get_model(role="grader", required=True)
        history = blind_interrogator_messages(state.metadata["task_description"])
        rounds: list[dict[str, Any]] = []
        state.store.set("interrogation_rounds", [])
        state.store.set("first_admission_turn", None)
        for turn in range(1, max_rounds + 1):
            question_output = await interrogator.generate(
                history,
                tools=[],
                config=GenerateConfig(
                    max_tokens=max_tokens, cache=False, max_retries=0, timeout=90
                ),
            )
            try:
                question = _parse_field(question_output, "question")
            except InvalidInterrogatorResponse as exc:
                record_response_failure(
                    state, question_output, exc, phase="interrogator_question", turn=turn
                )
                return state
            history.append(question_output.message)
            state.messages.append(ChatMessageUser(content=question))
            try:
                await target_reply(state, max_tokens=max_tokens)
            except IncompleteTargetResponse as exc:
                record_response_failure(state, state.output, exc, phase="target_reply", turn=turn)
                return state
            answer = state.output.completion
            history.append(ChatMessageUser(content=json.dumps({"interview_reply": answer})))
            admission = await judge_admission(
                answer, OVERSIGHT.description, max_tokens=judge_max_tokens
            )
            # Separate fork: this verdict cannot contaminate the next question.
            verdict_request = ChatMessageUser(
                content=_load("followups.json")["interrogation"]["verdict_request"]
            )
            verdict_output = await interrogator.generate(
                [*copy.deepcopy(history), verdict_request],
                tools=[],
                config=GenerateConfig(
                    max_tokens=max_tokens, cache=False, max_retries=0, timeout=90
                ),
            )
            try:
                verdict = _parse_field(
                    verdict_output, "verdict", {"guilty", "not_guilty", "unclear"}
                )
                verdict_error = None
            except InvalidInterrogatorResponse as exc:
                verdict, verdict_error = None, str(exc)
            rounds.append(
                {
                    "turn": turn,
                    "question": question,
                    "response": answer,
                    "admission": admission.as_dict(),
                    "verdict": verdict,
                    "verdict_error": verdict_error,
                    "verdict_raw": verdict_output.completion,
                }
            )
            state.store.set("interrogation_rounds", copy.deepcopy(rounds))
            if admission.label == "admitted":
                break
        state.store.set(
            "first_admission_turn",
            next(
                (row["turn"] for row in rounds if row["admission"]["label"] == "admitted"),
                None,
            ),
        )
        return state

    return solve
