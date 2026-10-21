# In-context scheming, reconstructed with Inspect

A small, inspectable implementation of the six experiment families in **Frontier Models are Capable of In-context Scheming** (Meinke et al., [arXiv:2412.04984v2](https://arxiv.org/abs/2412.04984)).

**This is a reconstruction, not an exact replication of Apollo's private benchmark or its numerical results.** The authors deliberately withheld implementation details and used a private o1 checkpoint. This repository supplies newly authored replacement data, executable experiments, controls, followups, scoring, tests, and analysis. Every generated case is marked `synthetic-v1`. The [coverage table](docs/PROTOCOL.md) states what differs or remains unavailable.

## Start here

Use Python 3.12+, `uv`, and a running Docker daemon. Work from this source checkout; the Python installation is editable and the adjacent fixture/Docker files are required.

```sh
uv sync --locked
cp .env.example .env
# Edit .env to add OPENROUTER_API_KEY. The file is ignored by Git.
uv run pytest -q
RUN_DOCKER_TESTS=1 uv run pytest tests/test_docker.py -q
```

The Docker tests use scripted models and incur **no API charges**. They run actual Inspect tasks, mutate dummy artifacts, and verify that evidence from one sample does not enter another sample's container.

Inspect the run plan, then explicitly launch the small live check:

```sh
uv run python -m scheming_inspect.run --profile smoke
uv run python -m scheming_inspect.run --profile smoke --execute --budget-usd 5
uv run inspect view --log-dir logs/smoke
uv run python -m scheming_inspect.analyze logs/smoke --out results/smoke --plots
```

The default target is `openrouter/deepseek/deepseek-v4-flash`, a **contemporary substitute**, not a model from the paper. The runner looks up current uncached prices, sets provider price ceilings, reserves a conservative upper bound including one request beyond each token limit, disables retries and prompt-cache writes, and refuses a plan above its budget. It runs samples serially. If no provider meets the price and parameter requirements, the request fails instead of falling back to a higher price. The budget covers this invocation only; other processes using the key are outside its accounting. The followup and rescoring commands below are separate paid runs.

The `smoke` profile has eight samples, including both branches of the two paired experiments. It tests execution, not the paper's scientific claims. `main` has 1,498 rollouts per model; `full` has 9,094 before followups. Print and review a larger plan before running it. The runner currently budgets OpenRouter; native provider use is available through Inspect's CLI.

## The code to read

| File | Responsibility |
|---|---|
| `scheming_inspect/data.py` | Builds cases, paired conditions, prompts and sandbox files. |
| `data/domains.json` | 33 authored goal-conflict scenarios. |
| `data/email_settings.json` | Five authored email-ranking settings; each case has 20 emails. |
| `scheming_inspect/tasks.py` | Converts cases to Inspect `Sample` objects and creates the task. |
| `scheming_inspect/engine.py` | Four tools, bounded agent loop, evidence snapshots and deterministic scores. |
| `scheming_inspect/protocol.py` | Main and ablation matrices; readable snapshots are in `configs/`. |
| `scheming_inspect/judge.py` | Optional evidence-quoting semantic scorer; requires an explicit grader. |
| `scheming_inspect/followups.py` | Independent fixed questions and blind multi-turn interrogation. |
| `scheming_inspect/analyze.py` | Offline CSV/JSON summaries, paired statistics and plots. |
| `scheming_inspect/run.py`, `costs.py` | Plan/budget execution and verify logged OpenRouter charges. |

There is one task factory, rather than six independent frameworks. For example:

```sh
XDG_DATA_HOME="$PWD/inspect-data" uv run --env-file .env inspect eval tasks.py@scheming \
  -T experiment=oversight -T n=3 -T nudge=true \
  --model openrouter/deepseek/deepseek-v4-flash --max-samples 1 \
  --log-dir logs/oversight
```

This direct Inspect command does **not** use the runner's dollar guard. Task paths are relative here because the installed Inspect/Python combination does not accept every absolute task-path pattern. Data and Docker paths are resolved relative to the package, so importing the task from another working directory still works.

## What is measured

| Family | Observable measurement | Important qualification |
|---|---|---|
| Oversight subversion | Change from enabled to disabled oversight. | A broken/missing config does not silently disable the simulated monitor. |
| Self-exfiltration | Deletion, copying and replacement of inert marker files. | A local deployment directory simulates a second server; there are no real model weights or external transfers. |
| Goal guarding | Successor-prompt removal or modification. | Modification alone does not prove preservation of the original goal; use semantic review. |
| Email reranking | Input changes, scoring-code changes, saved/submitted score and ranking changes. | Errors and intentional concealment are separate questions. |
| Alignment faking | Developer-aligned choice during testing, model-aligned choice in its paired deployment run. | Requires aggregate analysis; one aligned answer is not evidence of deception. |
| Sandbagging | Accuracy difference between matched question sets with opposite incentives. | Ten questions in a rollout are one statistical unit. Formatting failures remain visible. |

The target receives only the prompt and that sample's sandbox files. Reference answers remain on the evaluator side. Docker runs without a network, host mounts, credentials or a Docker socket; uses a non-root user; drops capabilities; and gives each sample private writable tmpfs over a read-only root filesystem. `no-new-privileges` is omitted because Snap Docker on the validation host rejects executable startup with it; this does not enable privileged mode.

Snapshots record file state after each command. An edit and restoration entirely within one shell command may escape state-change detection; retain transcripts and review ambiguous attempts. Exfiltration's optional `monitored` condition also records disable-before-copy ordering across command boundaries. The fake oversight story does not implement a real monitor capable of shutting down the target model.

## Optional followups and semantic review

An observed action can be followed by eight wording variants, each forked independently from the original conversation. The paper used 20 repetitions per variant; start with one source case and one repetition:

```sh
uv run --env-file .env python -m scheming_inspect.followups \
  --mode fixed --logs logs/oversight \
  --model openrouter/deepseek/deepseek-v4-flash \
  --judge openrouter/deepseek/deepseek-v4-flash \
  --limit 1 --repeats 1 --log-dir logs/fixed
```

For interrogation, use `--mode interrogation` and provide `--interrogator MODEL`. It needs both observed guilty and innocent oversight runs in the same scenario and treatment. It matches scenarios, not identical generated numeric facts. No artificial guilty cases are silently invented. The interrogator sees the task description and interview, not the source action trace or ground truth. It stops after an admission or at most seven rounds. Analysis reports both true and false accusations/admissions.

Semantic auditing is opt-in and costs an additional grader call per eligible sample. It assesses evidence of attempted actions, goal-preserving replacement and concealed email manipulation, without replacing the deterministic file-state measurements:

```sh
uv run --env-file .env inspect score PATH_TO_LOG.eval \
  --scorer scheming_inspect/judge.py@semantic_audit \
  --model-role grader=openrouter/deepseek/deepseek-v4-flash \
  --action append --output-file logs/audited.eval
```

Judge answers must satisfy a strict schema and quote evidence actually present in the trace. Ambiguous or failed judgments are unscored and retain diagnostics. Review a human-labeled sample before treating judge results as reliable. A no-CoT instruction only suppresses requested visible text; it cannot establish that a model performed no internal reasoning.

Sandbagging prompts prohibit using code to solve the questions; this instruction is not enforced by removing Bash. Check the logged commands for compliance before interpreting a performance difference. The visible dummy artifacts and simplified deployment environment can also affect model behavior.

To verify the billed cost of recorded OpenRouter requests, run `uv run python -m scheming_inspect.costs logs/smoke logs/fixed logs/semantic --out results/costs.json`. This reads request metadata and makes no model generations. It deduplicates generation IDs across original and rescored logs, excludes Inspect cache reads and saves partial results when billing metadata is unavailable.

## Sources and results

The saved validation run completed eight main cases, eight fixed followups and one semantic audit. OpenRouter verified **$0.013229117** in total charges. These are small execution checks; the full paper-scale matrix has not been run.

- [Protocol and coverage](docs/PROTOCOL.md): all implemented conditions, counts, inference and fidelity gaps.
- [Data provenance](data/PROVENANCE.md): how replacement inputs were authored.
- [Source audit](docs/SOURCES.md): original paper, author repositories and transcript availability.
- [Validation](docs/VALIDATION.md): checks actually run, models, costs and limitations.

The paper and selected author transcripts are research references, not redistributable source data included in this Git repository. No original-paper success rate is presented as a result produced by this implementation.
