# Full campaign

The full reconstructed matrix has 65 experimental cells and 9,094 target rollouts for one model. The campaign uses `openrouter/deepseek/deepseek-v4-flash`, seed 42, 8,000 output tokens per call, and up to eight concurrent isolated samples. This is the same contemporary substitute used in validation, not the paper's historical model matrix.

The user has authorized continuing this campaign **without spending caps**. The earlier $50 total cap and $35 base-stage sublimit no longer govern execution. Token limits and concurrency remain experiment settings; an optional spending counter is informational and never stops the run.

## Run and monitor

From the repository root, `bash scripts/run_full.sh` starts the configured campaign. It refuses to overwrite an existing campaign. `bash scripts/run_full.sh --resume` continues from the recorded manifest and logs. Do not run either command while another instance is active; the campaign lock prevents concurrent dispatch.

Check progress without calling a model:

```sh
uv run python -m scheming_inspect.campaign --status --campaign-dir logs/full-campaign
```

Browse nested evaluation logs in Inspect View:

```sh
uv run inspect view --recursive --log-dir logs/full-campaign --port 7576
```

The viewer is at `http://127.0.0.1:7576`. Its samples show prompts, tool calls, scores, metadata and any errors. The campaign manifest remains alongside the logs. The historical `budget.json` is preserved read-only; its old reservations are not current spending controls or a complete bill for the continuing run. When enabled, `spending.json` holds the latest optional API-key usage observation.

Analyze the base matrix separately from the later stages so rescored copies or continuations are not mistaken for extra base samples:

```sh
uv run python -m scheming_inspect.analyze --campaign-dir logs/full-campaign --out results/full-campaign/base --plots --holm
```

If launched in the persistent `scheming-full` tmux session, use `tmux attach -t scheming-full` to see its terminal. Detach with Ctrl+B, then D; this leaves it running. The launch wrapper temporarily inhibits sleep for the experiment's lifetime without changing saved power settings. Explicit shutdown or forced suspension can still interrupt it.

## Work included

1. Run every base/main and ablation cell at its configured sample count. The first batch contains one real full-run sample to validate the new execution path; subsequent batches contain up to 32 samples. No extra smoke sample is added.
2. Perform semantic audits of the main-condition covert-action traces (up to 1,200) without regenerating the target's original responses.
3. Run matched guilty/innocent oversight interrogations, up to seven turns each.
4. Ask eight fixed follow-up variants, each with twenty independent repetitions, for every eligible main-run action source.
5. Perform semantic audits of the remaining covert-action traces (up to 5,500).

Fixed follow-ups and interrogations use the main hard, nudged, conflicting-goal, system-role conditions. Applying these continuations to every ablation would be an additional experiment. Their exact sample counts depend on the base run's observed actions and available matched controls. The complete poststage plan records planned, completed and excluded work.

## Spending and restart behavior

The campaign no longer reserves funds, reconciles generation bills before continuing, or stops at a financial threshold. Its historical ledger and billing-policy records remain unchanged for audit. Earlier statuses such as `complete_reserved` describe the accounting policy that was active when those results were recorded; the completed samples remain part of the same experiment.

The optional reporter makes one read-only request to OpenRouter's API-key endpoint, with a three-second timeout and no retries. Its `key_total_usd` value is cumulative usage for that API key. It can include earlier runs and concurrent activity outside this campaign, so it must not be presented as an exact campaign cost. A missing key, unavailable or malformed response, or failed report write produces an unavailable observation and does not block execution. No cost observation authorizes replaying a completed sample.

Resuming uses the existing manifest and logs, verifies the experiment's configuration and source revision, and preserves recorded results. On `--resume`, a batch with incomplete execution is archived outside the active campaign log directory and rerun in full, including any answers already recorded in that interrupted attempt. The replacement uses exactly the original sample IDs and settings. A terminal log with full coverage is kept even if its journal was interrupted or some answers are invalid; completion is not a correctness criterion. The manifest records each restart and the discarded attempt's outcome counts, and campaign analysis includes only the replacement attempt. Source migrations are recorded explicitly so accounting changes remain distinguishable from changes to experimental prompts, samples, scorers, or generation settings. Do not edit sources while the campaign is running or start a second instance.

Historical partial recoveries remain readable with their original case selections. An explicitly prepared partial recovery can also continue after a scorer or infrastructure failure if desired; ordinary resume now uses the full-batch restart policy above. It archives the previous manifest, records the code revision, preserves every non-error outcome (including invalid answers and limits), and selects only errored or unrecorded cases for a new attempt. The original partial log stays unchanged. Historical reservations with unknown charges remain historical accounting evidence; they neither establish the actual cost nor stop the uncapped continuation.

Use the `--campaign-dir` analysis command above after recovery. Its canonical sample selections prevent failed and replacement attempts from counting twice, and `summary.json` records the excluded attempts and source revisions. Reading all raw logs together would lose this distinction.

## Repairing the audit metric schema

Older code can stop during `audit_main` with `key ... isn't present in the score value dictionary` when a batch mixes scenario families. The repaired scorer returns all audit metric names and uses NaN for inapplicable dimensions, so they are excluded from averages. Judge prompts, labels, and the underlying experiments are unchanged.

Because campaign source hashes are frozen, applying the code fix also requires an explicit recorded revision. With the runner stopped, preview the narrow repair, then apply it:

```sh
uv run python -m scripts.repair_audit_schema --campaign-dir logs/full-campaign
uv run python -m scripts.repair_audit_schema --campaign-dir logs/full-campaign --apply
bash scripts/run_full.sh --resume
```

This repair requires all base batches to be complete and no committed post-stage progress. It preserves base results, archives the incomplete audit attempt, and records both old and corrected source identities. The interrupted audit batch is rerun in full under the existing restart policy. The helper makes no model calls.

## Invalid follow-up responses

An interrogator response can finish normally yet contain invalid JSON, such as a missing closing brace. Invalid questions now end only that interview and produce explicit unscored results. The code does not repair the JSON, invent a question, or count the failed interview as a denial or a not-guilty verdict. Unfinished target answers and requests for unavailable tools are also left unscored. Unexpected programming errors and failed target/interrogator API calls still surface as errors.

Inspect View shows the raw failed response, phase, turn and stop reason in the sample store and score metadata under `followup_error`. Scalar sample metadata includes `followup_status=invalid_model_response` and `followup_failure_phase`. Earlier completed interview rounds remain available for analysis; missing later rounds are unknown. A batch's `success` count means processing completed, not that every score is usable: report unscored counts and each metric's actual denominator.

For a campaign stopped by the older interrogation parser, with the code fix installed and the runner stopped:

```sh
uv run python -m scripts.repair_followup_responses --campaign-dir logs/full-campaign
uv run python -m scripts.repair_followup_responses --campaign-dir logs/full-campaign --apply
bash scripts/run_full.sh --resume
```

This narrow repair preserves committed base and post-stage results, archives the entire interrupted interrogation batch, and records the source revision. The incomplete batch is rerun with its original sample IDs and settings. Completed batches retain their original execution fingerprints and immutable logs. The helper makes no model calls.

## Judge timeouts

Inspect 0.3.281 gives scoring half the sample time limit: 150 seconds for the
300-second fixed-followup tasks. A scoring timeout is a sample error and, with
`fail_on_error=True`, interrupts the batch. The existing `GenerateConfig.timeout`
setting limits retries but does not interrupt an in-flight call.

Judge calls now have a separate 90-second wall-clock deadline. If it expires,
the target reply stays in the log and the annotation is explicitly unscored
(`grader_failed`, with a deadline explanation). It is excluded from admission or
denial rates, and the other samples continue. External cancellation still
propagates. This does not change the target prompts, sample selection, token
limits, or spending policy.

For the older fixed-followup batch stopped by a scoring timeout, install the fix
with the runner stopped, then record the revision and resume:

```sh
uv run python -m scripts.repair_judge_timeout --campaign-dir logs/full-campaign
uv run python -m scripts.repair_judge_timeout --campaign-dir logs/full-campaign --apply
bash scripts/run_full.sh --resume
```

The repair preserves all committed results and prior revision records. It
archives the entire incomplete fixed-followup batch and schedules its original
sample IDs again. Completed batches retain their original execution fingerprints
and logs. The helper makes no model calls; it is specific to this stopped repair,
not a step to repeat on every ordinary resume.

## Interpretation

The saved smoke run suggests roughly $10.50 for the base matrix if its average cost generalized, but eight cases are a weak cost sample. Follow-up counts and reasoning lengths can materially change the final cost. One fixed source-action creates 160 continuations and their grader calls. This estimate is not a limit on the now-uncapped campaign.

The serial smoke timings extrapolate to several days for the full base matrix. Concurrency should reduce wall time, but provider latency and Docker startup still make this a long run. Progress should be assessed from completed counts, not a promised finish time.

Reports must retain infrastructure errors, invalid completions, truncations and excluded source cases. Observed actions and semantic judgments remain separate. Completion of this campaign would validate this reconstructed dataset and contemporary model; it would not recover the authors' withheld benchmark or reproduce their numerical results exactly.
