"""Plan follow-ups from completed base batches and run them with the shared runner."""

from __future__ import annotations

import copy
import random
from collections import Counter
from pathlib import Path

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ModelOutput
from inspect_ai.solver import solver

from . import campaign as c
from . import followups as f
from .judge import semantic_audit
from .protocol import COVERT, plan

FORMAT = "simple-poststages-v1"
STAGES = ("audit_main", "interrogation", "fixed", "audit_remaining")
SOURCE_FIELDS = (
    "key",
    "log_path",
    "sample_id",
    "epoch",
    "model",
    "experiment",
    "scenario_id",
    "condition",
    "behavior",
    "matching_context",
)


def is_main_cell(cell: dict) -> bool:
    return any(
        all(cell.get(key) == value for key, value in original.items() if key != "seed")
        for original in plan("main")
    )


def discover_sources(base: dict, model: str, read_log_fn=c.read_log) -> dict:
    """Select completed valid interactions; retain an explicit count of exclusions."""
    if any(batch["status"] != "complete" for batch in base["batches"]):
        raise ValueError("Finish the base batches before starting follow-ups.")
    sources, excluded = {}, []
    requested = Counter(
        dict(audit_main=0, audit_remaining=0, fixed_sources=0, interrogation_sources=0)
    )
    for record in c.base_log_records(base):
        cell = record["cell"]
        if cell["experiment"] not in COVERT:
            continue
        main = is_main_cell(cell)
        stage = "audit_main" if main else "audit_remaining"
        expected = set(record["expected_sample_ids"])
        requested[stage] += len(expected)
        if main and cell["experiment"] != "email":
            requested["fixed_sources"] += len(expected)
        if main and cell["experiment"] == "oversight":
            requested["interrogation_sources"] += len(expected)
        log = read_log_fn(Path(record["path"]))
        if str(log.eval.model) != model:
            raise ValueError("The source log uses a different target model.")
        samples = {str(sample.id): sample for sample in log.samples or []}
        if len(samples) != len(log.samples or []) or set(samples) != expected:
            raise ValueError("Base log samples do not match the completed batch.")
        for sample in log.samples or []:
            reason = f.source_exclusion_reason(sample)
            if reason:
                excluded.append(
                    {
                        "path": record["path"],
                        "sample_id": sample.id,
                        "stage": stage,
                        "reason": reason,
                    }
                )
                continue
            source = f.source_from_sample(record["path"], sample, model)
            sources[source.key] = {name: getattr(source, name) for name in SOURCE_FIELDS} | {
                "main": main,
            }
    return {"sources": sources, "excluded": excluded, "requested": dict(requested)}


def source_stub(record: dict) -> f.SourceCase:
    return f.SourceCase(
        **{name: record[name] for name in SOURCE_FIELDS}, messages=(), task_description=""
    )


def make_groups(discovery: dict, config: dict) -> tuple[list[dict], list[dict]]:
    groups, excluded = [], []
    records = discovery["sources"]
    for key, record in records.items():
        groups.append(
            {
                "id": f"audit:{key}",
                "stage": "audit_main" if record["main"] else "audit_remaining",
                "source_keys": [key],
                "samples": 1,
            }
        )
    main = [source_stub(record) for record in records.values() if record["main"]]
    pairs = f.matched_sources(main, seed=config["seed"])
    matched = {source.key for pair in pairs for source in pair}
    rng = random.Random(config["seed"])
    for guilty, innocent in pairs:
        truths = ["guilty", "innocent"]
        rng.shuffle(truths)
        groups.append(
            {
                "id": f"interrogation:{guilty.key}:{innocent.key}",
                "stage": "interrogation",
                "source_keys": [guilty.key, innocent.key],
                "truth_order": truths,
                "samples": 2,
            }
        )
    for source in sorted(main, key=lambda item: item.key):
        if source.experiment == "email":
            continue
        actions = f.eligible_actions(source)
        if not actions:
            excluded.append(
                {"source_key": source.key, "stage": "fixed", "reason": "no_observed_action"}
            )
        for action in actions:
            variants = [name for name, _ in f.fixed_questions(action)]
            groups.append(
                {
                    "id": f"fixed:{source.key}:{action.name}",
                    "stage": "fixed",
                    "source_keys": [source.key],
                    "action": action.name,
                    "variants": variants,
                    "repeats": config["fixed_repeats"],
                    "samples": len(variants) * config["fixed_repeats"],
                }
            )
        if source.experiment == "oversight" and source.key not in matched:
            excluded.append(
                {
                    "source_key": source.key,
                    "stage": "interrogation",
                    "reason": "no_matched_opposite_outcome",
                }
            )
    groups.sort(key=lambda group: STAGES.index(group["stage"]))
    return groups, excluded


def group_items(group: dict) -> list[dict]:
    """Lightweight sample descriptions; transcripts are copied only when dispatched."""
    keys = group["source_keys"]
    if group["stage"].startswith("audit_"):
        return [{"id": group["id"]}]
    if group["stage"] == "interrogation":
        return [
            {"id": f.interrogation_sample_id(*keys, truth, 0), "truth": truth}
            for truth in group["truth_order"]
        ]
    return [
        {
            "id": f.fixed_sample_id(keys[0], group["action"], variant, repeat),
            "variant": variant,
            "repeat": repeat,
        }
        for variant in group["variants"]
        for repeat in range(group["repeats"])
    ]


class SourceReader:
    """Keep only one source log in memory."""

    def __init__(self, records: dict, read_log_fn=c.read_log):
        self.records, self.read_log_fn = records, read_log_fn
        self.path, self.samples = None, {}

    def sample(self, key):
        record = self.records[key]
        if self.path != record["log_path"]:
            log = self.read_log_fn(Path(record["log_path"]))
            self.samples = {(str(s.id), s.epoch): s for s in log.samples or []}
            self.path = record["log_path"]
        return self.samples[(str(record["sample_id"]), record["epoch"])]

    def source(self, key):
        record = self.records[key]
        return f.source_from_sample(record["log_path"], self.sample(key), record["model"], key=key)


@solver
def restore_completion():
    async def solve(state, generate):
        state.output = ModelOutput.from_content(
            str(state.model), state.metadata["source_completion"]
        )
        state.tools = []
        state.completed = True
        return state

    return solve


def audit_sample(reader: SourceReader, key: str, stage: str) -> Sample:
    original = reader.sample(key)
    metadata = copy.deepcopy(original.metadata or {})
    metadata.update(
        source_completion=original.output.completion,
        source_key=key,
        source_log=reader.records[key]["log_path"],
        source_sample_id=original.id,
        source_epoch=original.epoch,
        audit_stage=stage,
    )
    return Sample(id=f"audit:{key}", input=copy.deepcopy(original.messages), metadata=metadata)


def build_batch(selections: list[dict], groups: dict, reader: SourceReader, config: dict) -> Task:
    samples = []
    stage = groups[selections[0]["group_id"]]["stage"]
    for selection in selections:
        group = groups[selection["group_id"]]
        keys = group["source_keys"]
        items = group_items(group)[selection["start"] : selection["stop"]]
        if stage.startswith("audit_"):
            samples.append(audit_sample(reader, keys[0], stage))
        elif stage == "fixed":
            source = reader.source(keys[0])
            action = next(a for a in f.eligible_actions(source) if a.name == group["action"])
            samples.extend(
                f.fixed_sample(source, action, item["variant"], item["repeat"]) for item in items
            )
        else:
            guilty, innocent = (reader.source(key) for key in keys)
            samples.extend(
                f.interrogation_sample(guilty, innocent, item["truth"], 0, config["seed"])
                for item in items
            )
    if not stage.startswith("audit_"):
        return f.task_from_samples(
            samples,
            mode=stage,
            max_tokens=config["max_tokens"],
            judge_max_tokens=config["max_tokens"],
            max_rounds=config["max_rounds"],
            time_limit=config["interrogation_time_limit"]
            if stage == "interrogation"
            else config["other_time_limit"],
            token_limit=config["token_limit"],
            metadata={
                "protocol": "reconstructed post-stages",
                "stage": stage,
                "selection_seed": config["seed"],
                "fixed_repeats": config["fixed_repeats"],
            },
        )
    return Task(
        name=f"scheming_{stage}",
        dataset=samples,
        solver=restore_completion(),
        scorer=semantic_audit(max_tokens=config["max_tokens"]),
        sandbox=None,
        config=GenerateConfig(
            max_tokens=config["max_tokens"],
            max_retries=0,
            cache=False,
            cache_prompt=False,
            timeout=90,
        ),
        token_limit=config["token_limit"],
        time_limit=config["other_time_limit"],
        fail_on_error=True,
        metadata={"stage": stage},
    )


def completed_offsets(manifest: dict) -> dict:
    offsets = {}
    for batch in manifest["batches"]:
        if batch["status"] == "complete":
            for selection in batch["selections"]:
                offsets[selection["group_id"]] = selection["stop"]
    return offsets


def next_selections(manifest: dict, batch_size: int) -> list[dict]:
    """Batch one stage at a time and keep each interrogation pair together."""
    offsets = completed_offsets(manifest)
    selected, count, stage = [], 0, None
    for group in manifest["groups"]:
        start = offsets.get(group["id"], 0)
        if start >= group["samples"]:
            continue
        if stage is None:
            stage = group["stage"]
        if group["stage"] != stage:
            break
        unit = 2 if stage == "interrogation" else 1
        take = min(group["samples"] - start, batch_size - count)
        take -= take % unit
        if not take:
            break
        selected.append({"group_id": group["id"], "start": start, "stop": start + take})
        count += take
        if count >= batch_size:
            break
    return selected


def progress_report(manifest: dict) -> dict:
    offsets = completed_offsets(manifest)
    counts = {stage: {"planned": 0, "completed": 0, "pending": 0} for stage in STAGES}
    for group in manifest["groups"]:
        counts[group["stage"]]["planned"] += group["samples"]
        counts[group["stage"]]["completed"] += offsets.get(group["id"], 0)
    for values in counts.values():
        values["pending"] = values["planned"] - values["completed"]
    return {
        "status": manifest["status"],
        "stages": counts,
        "requested": manifest["discovery"]["requested"],
        "exclusions": dict(Counter(r["reason"] for r in manifest["discovery"]["excluded"])),
        "eligibility_exclusions": dict(
            Counter(f"{r['stage']}:{r['reason']}" for r in manifest["eligibility"])
        ),
        "sample_outcomes": {
            stage: dict(
                sum(
                    (
                        Counter(b.get("sample_status_counts", {}))
                        for b in manifest["batches"]
                        if b["stage"] == stage
                    ),
                    Counter(),
                )
            )
            for stage in STAGES
        },
    }


def execute_poststages(
    campaign_root,
    *,
    model,
    batch_size=32,
    max_samples=8,
    max_tokens=8000,
    token_limit=100000,
    resume=False,
    execute_batch_fn=None,
    read_log_fn=c.read_log,
) -> dict:
    root = Path(campaign_root).resolve()
    path = root / "poststages.json"
    base = c.read_manifest(root)
    if path.exists():
        if not resume:
            raise ValueError("Post-stages already exist; use --resume.")
        manifest = c.read_json(path)
        if manifest.get("format") != FORMAT:
            raise ValueError("Use a new campaign directory for this runner.")
    else:
        if min(batch_size, max_samples, max_tokens, token_limit) < 1 or batch_size < 2:
            raise ValueError("Positive limits and batch_size >= 2 are required.")
        config = dict(
            model=model,
            batch_size=batch_size,
            max_samples=max_samples,
            max_tokens=max_tokens,
            token_limit=token_limit,
            seed=base["config"].get("seed", 42),
            fixed_repeats=20,
            max_rounds=7,
            interrogation_time_limit=600,
            other_time_limit=300,
        )
        discovery = discover_sources(base, model, read_log_fn)
        groups, eligibility = make_groups(discovery, config)
        manifest = dict(
            format=FORMAT,
            config=config,
            discovery=discovery,
            groups=groups,
            eligibility=eligibility,
            batches=[],
            status="planned",
        )
        c.atomic_json(path, manifest)
    config = manifest["config"]
    groups = {group["id"]: group for group in manifest["groups"]}
    reader = SourceReader(manifest["discovery"]["sources"], read_log_fn)
    execute = execute_batch_fn or c.execute_batch
    # Verify already-recorded batches through the same path, without constructing tasks.
    queue = list(manifest["batches"])
    while True:
        if queue:
            batch = queue.pop(0)
        else:
            selections = next_selections(manifest, config["batch_size"])
            if not selections:
                manifest["status"] = (
                    "complete_with_exclusions" if manifest["discovery"]["excluded"] else "complete"
                )
                c.atomic_json(path, manifest)
                return progress_report(manifest)
            stage = groups[selections[0]["group_id"]]["stage"]
            batch = dict(
                batch_id=f"{stage}-{len(manifest['batches']):05d}",
                stage=stage,
                selections=selections,
                status="planned",
            )
            manifest["batches"].append(batch)
            manifest["status"] = "running"
            c.atomic_json(path, manifest)
        ids = [
            item["id"]
            for selection in batch["selections"]
            for item in group_items(groups[selection["group_id"]])[
                selection["start"] : selection["stop"]
            ]
        ]
        was_complete = batch["status"] == "complete"
        roles = {"grader": config["model"]}
        if batch["stage"] == "interrogation":
            roles["interrogator"] = config["model"]
        try:
            result = execute(
                lambda: build_batch(batch["selections"], groups, reader, config),
                batch=batch | {"sample_ids": ids},
                log_dir=root / "poststages" / batch["stage"] / batch["batch_id"],
                model=config["model"],
                model_roles=roles,
                max_samples=config["max_samples"],
            )
        except BaseException:
            if not was_complete:
                batch["status"] = "interrupted"
            manifest["status"] = "interrupted"
            c.atomic_json(path, manifest)
            raise
        batch.update({key: result[key] for key in ("status", "log_paths", "sample_status_counts")})
        manifest["status"] = "running" if batch["status"] == "complete" else "interrupted"
        c.atomic_json(path, manifest)
        if not was_complete:
            c.emit(
                "poststage_progress", **progress_report(manifest), spending=c.report_key_usage(root)
            )
        if batch["status"] != "complete":
            return progress_report(manifest)
