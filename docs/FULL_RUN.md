# Full campaign

The full reconstructed matrix has 65 experimental cells and 9,094 target rollouts for one model. The campaign uses `openrouter/deepseek/deepseek-v4-flash`, seed 42, 8,000 output tokens per call, and up to eight concurrent isolated samples. This is the same contemporary substitute used in validation, not the paper's historical model matrix.

The total authorized spending ceiling is **$50**, including the earlier **$0.013229117** validation cost. The base stage has a $35 sublimit to preserve room for additional measurements. Unused base allocation remains available under the global cap. An allocation does not guarantee that every planned stage will fit.

## Run and monitor

From the repository root, `bash scripts/run_full.sh` starts the configured campaign. It refuses to overwrite an existing campaign. `bash scripts/run_full.sh --resume` reconciles an existing campaign before attempting any new batch. Do not run either command while another instance is active; the campaign lock prevents concurrent dispatch.

Check progress without calling a model:

```sh
uv run python -m scheming_inspect.campaign --status --campaign-dir logs/full-campaign
```

Browse nested evaluation logs in Inspect View:

```sh
uv run inspect view --recursive --log-dir logs/full-campaign --port 7576
```

The viewer is at `http://127.0.0.1:7576`. Its samples show prompts, tool calls, scores, metadata and any errors. The campaign manifest and `budget.json` remain alongside the logs. An in-flight reservation is a cost ceiling, not a charge already billed.

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

The original simple runner reserved the worst-case cost of the entire matrix upfront. At the checked rates, its full-matrix reservation was about $1,914; this was not an expected bill. The campaign instead reserves the maximum cost of a batch before making requests, then reconciles exact OpenRouter generation charges before releasing unused reservation. Provider price ceilings, output/token limits and disabled client retries bound each batch.

The ledger is written atomically and protected by a file lock. Request IDs are deduplicated, including repeated source events in rescored logs. Inspect cache reads are not new paid requests. Read-only billing lookups may be retried; model generations are not automatically retried.

The full-run launcher now passes `--allow-unverified-billing`, as authorized on 21 October 2026. When every sample is recorded and only billing availability is uncertain, the runner verifies the original execution bounds and permanently retains the **entire original batch reservation**. The execution journal reports `complete_reserved`; the financial ledger reports `closed_reserved`, with `actual_usd: null`. These samples enter analysis exactly once, and the next batch can run if its reservation fits under the unchanged $50 total and $35 base-stage caps. Missing bills are never treated as zero cost. Retained allowances can exhaust the cap even when verified spending is lower.

This option does not bypass missing or changed logs, incomplete coverage, sample errors, retries that undermine the bound, or charges above the reservation. Without the option, billing remains strict; the older wrapper retries temporary billing-record lookup failures for up to 15 minutes. Neither mode regenerates a completed batch. The runtime choice is recorded in `billing-policy.json` and does not change prompts, samples, scorers, or generation settings.

Resuming uses the existing manifest and logs. It verifies the configuration and source fingerprint and does not silently replay an already-started batch. Do not delete the ledger, change the source while the campaign runs, or start a second campaign to bypass a stop. A stopped or budget-limited campaign remains explicitly incomplete.

For the stopped campaign that predates this accounting change, `scripts/migrate_billing_policy.py` records an explicit accounting-only source revision and archives the prior manifest, ledger, and blocked journal. It does not alter results, financial limits, or experimental settings. The subsequent ordinary resume reconciles the existing log before starting new samples.

An explicitly prepared recovery can continue after a scorer or infrastructure failure. It archives the previous manifest, records the code revision, preserves every non-error outcome (including invalid answers and limits), and selects only errored or unrecorded cases for a new attempt. The original partial log stays unchanged. If interrupted requests have unknown charges, their full original reservation remains permanently committed under `closed_reserved`; it is not reported as verified spending. Recovery attempts require an additional reservation under the same $50 cap.

Use the `--campaign-dir` analysis command above after recovery. Its canonical sample selections prevent failed and replacement attempts from counting twice, and `summary.json` records the excluded attempts and source revisions. Reading all raw logs together would lose this distinction.

## Interpretation

The saved smoke run suggests roughly $10.50 for the base matrix if its average cost generalized, but eight cases are a weak cost sample. Follow-up counts and reasoning lengths can materially change the final cost. One fixed source-action creates 160 continuations and their grader calls. The $50 cap takes precedence over completing all stages.

The serial smoke timings extrapolate to several days for the full base matrix. Concurrency should reduce wall time, but provider latency, Docker startup and billing reconciliation still make this a long run. Progress should be assessed from completed counts, not a promised finish time.

Reports must retain infrastructure errors, invalid completions, truncations and excluded source cases. Observed actions and semantic judgments remain separate. Completion of this campaign would validate this reconstructed dataset and contemporary model; it would not recover the authors' withheld benchmark or reproduce their numerical results exactly.
