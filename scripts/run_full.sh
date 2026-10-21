#!/usr/bin/env bash
# The complete authorized campaign. Add --resume to reconcile an existing run.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export XDG_DATA_HOME="$PWD/inspect-data"
exec uv run --locked python scripts/run_campaign_with_billing_retry.py \
  --campaign-dir "$PWD/logs/full-campaign" \
  --profile full \
  --model openrouter/deepseek/deepseek-v4-flash \
  --budget-usd 50 \
  --stage-budget-usd 35 \
  --starting-spend-usd 0.013229117 \
  --max-tokens 8000 \
  --max-steps 20 \
  --token-limit 100000 \
  --max-samples 8 \
  --batch-size 32 \
  --seed 42 \
  --poststages all \
  --allow-unverified-billing \
  --execute "$@"
