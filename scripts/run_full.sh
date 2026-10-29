#!/usr/bin/env bash
# Full reconstructed campaign. Add --resume to use saved settings and continue.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export XDG_DATA_HOME="$PWD/inspect-data"
exec uv run --locked python -m scheming_inspect.campaign \
  --campaign-dir "$PWD/logs/campaign" \
  --profile full \
  --model openrouter/deepseek/deepseek-v4-flash \
  --max-tokens 8000 \
  --max-steps 20 \
  --token-limit 200000 \
  --max-samples 8 \
  --batch-size 32 \
  --seed 42 \
  --poststages all \
  --execute "$@"
