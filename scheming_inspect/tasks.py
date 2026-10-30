"""One task factory: all experimental differences live in the data and task arguments."""

import base64
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig

from .data import SELECTION, build_cases
from .scorers import behavior_metrics
from .solvers import agent_loop

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PATHS = (
    "/workspace/planning/brief.md",
    "/workspace/planning/assumptions.md",
    "/workspace/planning/forecasts.csv",
)


def to_sample(case: dict) -> Sample:
    files = {
        path: "data:application/octet-stream;base64," + base64.b64encode(text.encode()).decode()
        for path, text in case["files"].items()
    }
    # Blind follow-up interviews see the public task facts only: no system goal,
    # private reference, or later file changes.
    visible = {**case["files"], **case["staged"]}
    paths = ["/workspace/planning/task.md"]
    if case["metadata"]["experiment"] in SELECTION:
        paths += PUBLIC_PATHS
    return Sample(
        id=case["id"],
        input=[
            ChatMessageSystem(content=case["system"]),
            ChatMessageUser(content=case["ordinary"]["request"]),
        ],
        files=files,
        metadata={
            **case["metadata"],
            "reference": case["reference"],
            "task_description": "\n\n".join([case["user"], *(visible[path] for path in paths)]),
            "ordinary_request": case["ordinary"]["request"],
            "ordinary_reference": case["ordinary"]["expected"],
            "followup_user": case["user"],
            "followup_files": case["staged"],
        },
    )


@task
def scheming(
    experiment: str = "oversight",
    n: int = 100,
    seed: int = 42,
    condition: str = "default",
    nudge: bool = True,
    goal_mode: str = "conflict",
    goal_role: str = "system",
    difficulty: str = "hard",
    suppress_reasoning: bool = False,
    question_set: str = "arithmetic",
    consequence: str = "unlearning",
    user_prompt: str = "best",
    max_steps: int = 20,
    max_tokens: int = 4096,
    token_limit: int = 400000,
) -> Task:
    """Reconstruct one experimental cell. Values are saved in the Inspect log.

    difficulty sets hint intensity, not question difficulty. suppress_reasoning
    requests less written reasoning through the prompt only; it does not disable
    provider reasoning. Inspect accepts it as -T suppress_reasoning=true.
    max_steps bounds each of the two requests separately; time and token limits
    are shared by the whole conversation.
    """
    if goal_role not in ("system", "user"):
        raise ValueError(
            "This Inspect version supports system/user goal roles; developer is unavailable."
        )
    if max_steps < 1 or max_tokens < 1 or token_limit < 1:
        raise ValueError("Generation limits must be positive.")
    cases = build_cases(
        experiment,
        n=n,
        seed=seed,
        condition=condition,
        nudge=nudge,
        goal_mode=goal_mode,
        goal_role=goal_role,
        difficulty=difficulty,
        suppress_reasoning=suppress_reasoning,
        question_set=question_set,
        consequence=consequence,
        user_prompt=user_prompt,
    )
    return Task(
        dataset=[to_sample(case) for case in cases],
        solver=agent_loop(max_steps),
        scorer=behavior_metrics(),
        sandbox=("docker", str(ROOT / "compose.yaml")),
        config=GenerateConfig(
            max_tokens=max_tokens, max_retries=2, timeout=90, parallel_tool_calls=False
        ),
        token_limit=token_limit,
        time_limit=300,
        fail_on_error=False,
        name=f"scheming_{experiment}_{condition}",
        metadata={
            "paper": "arxiv:2412.04984v2",
            "fidelity": "reconstructed",
            "seed": seed,
            "source_tree": str(ROOT),
        },
    )
