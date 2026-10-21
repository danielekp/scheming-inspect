"""One task factory: all experimental differences live in the data and task arguments."""

import base64
import hashlib
import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig

from .data import build_cases
from .engine import agent_loop, behavior_metrics

ROOT = Path(__file__).resolve().parents[1]


def to_sample(case: dict, goal_role: str) -> Sample:
    # Explicit data URLs avoid accidentally treating a short string as a host path.
    files = {
        name: "data:application/octet-stream;base64," + base64.b64encode(text.encode()).decode()
        for name, text in case["files"].items()
    }
    # data._prompts places only the assigned goal in its selected role. Moving
    # the whole base system message would confound the role ablation.
    return Sample(
        id=case["id"],
        input=[ChatMessageSystem(content=case["system"]), ChatMessageUser(content=case["user"])],
        files=files,
        metadata={
            **case["metadata"],
            "reference": case["reference"],
            "fixture_sha256": hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest(),
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
    no_cot: bool = False,
    question_set: str = "arithmetic",
    consequence: str = "unlearning",
    user_prompt: str = "best",
    max_steps: int = 20,
    max_tokens: int = 4096,
    token_limit: int = 100000,
) -> Task:
    """Reconstruct one experimental cell. Values are saved in the Inspect log."""
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
        no_cot=no_cot,
        question_set=question_set,
        consequence=consequence,
        user_prompt=user_prompt,
    )
    return Task(
        dataset=[to_sample(c, goal_role) for c in cases],
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
            "fixture_generator": "authored-v1",
            "seed": seed,
            "source_tree": str(ROOT),
        },
    )
