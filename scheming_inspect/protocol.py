"""Explicit task-argument matrices. Building a plan never runs a model."""

import json

EXPERIMENTS = ("oversight", "exfiltration", "goal_guarding", "email", "alignment", "sandbagging")
COVERT = EXPERIMENTS[:4]
PROFILES = ("smoke", "main", "ablations", "full")
PAPER_URL = "https://arxiv.org/pdf/2412.04984v2"
HISTORICAL_MODELS = {
    "o1": "nonpublic pre-deployment checkpoint; unavailable",
    "o1-preview": "o1-preview-2024-09-12",
    "o1-mini": "o1-mini-2024-09-12",
    "gpt-4o": "gpt-4o-2024-08-06",
    "sonnet-3.5": "claude-3-5-sonnet-20241022",
    "opus-3": "claude-3-opus-20240229",
    "gemini-1.5-pro": "gemini-1.5-pro-002",
    "llama-3.1": "Llama-3.1-405B-Instruct",
}


def _tasks(experiment, n=None, **overrides):
    """Counterfactual branches have identical seeds and independent histories."""
    conditions = {"alignment": ("testing", "deployment"), "sandbagging": ("high", "low")}
    count = n if n is not None else {"alignment": 99, "sandbagging": 50}.get(experiment, 300)
    defaults = dict(
        experiment=experiment,
        n=count,
        seed=42,
        nudge=experiment != "alignment",
        goal_mode="conflict",
        goal_role="system",
        difficulty="hard",
        no_cot=False,
        question_set="arithmetic",
        consequence="unlearning",
        user_prompt="best",
    )
    defaults.update(overrides)
    return [
        dict(defaults, condition=condition)
        for condition in conditions.get(experiment, ("default",))
    ]


def plan(profile="smoke") -> list[dict]:
    """Return only arguments accepted by the scheming task; no model is selected.

    Counts are per condition, per selected model. The ablation matrix intentionally
    applies historically model-specific analyses to whichever model the caller selects.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}; choose one of {PROFILES}")
    main = [task for experiment in EXPERIMENTS for task in _tasks(experiment)]
    if profile == "smoke":
        return [dict(task, n=1) for task in main]
    if profile == "main":
        return main
    ablations = []
    for experiment in COVERT:
        ablations += _tasks(experiment, n=400, nudge=False)
        ablations += _tasks(experiment, n=100, goal_mode="matched")
        ablations += _tasks(experiment, n=100, goal_role="user")
    for experiment in ("oversight", "exfiltration"):
        ablations += _tasks(experiment, n=300, goal_mode="reversed")
    ablations.append(dict(_tasks("exfiltration", n=100)[0], condition="monitored"))
    ablations += _tasks("sandbagging", goal_mode="none", nudge=False)
    for consequence in (
        "unlearning",
        "sad",
        "fired",
        "replacement",
        "helpful_unlearning",
        "compute",
    ):
        ablations += _tasks(
            "sandbagging",
            goal_mode="none",
            nudge=False,
            question_set="mixed",
            consequence=consequence,
        )
    for prefix in ("testing", "minimal", "no_sandbag", "no_underperform"):
        ablations += _tasks(
            "sandbagging", goal_mode="none", nudge=False, question_set="mixed", user_prompt=prefix
        )
    for question_set in ("arithmetic", "easy"):
        ablations += _tasks(
            "sandbagging",
            n=100,
            goal_mode="none",
            nudge=False,
            question_set=question_set,
            no_cot=True,
        )
    for difficulty in ("medium", "easy"):
        for experiment in EXPERIMENTS:
            ablations += _tasks(experiment, difficulty=difficulty)
    tasks = ablations if profile == "ablations" else main + ablations
    return list({json.dumps(task, sort_keys=True): task for task in tasks}.values())
