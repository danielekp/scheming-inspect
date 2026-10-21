#!/usr/bin/env bash
# Full campaign without spending limits. Add --resume to continue saved work.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export XDG_DATA_HOME="$PWD/inspect-data"
exec uv run --locked python -m scheming_inspect.campaign \
  --campaign-dir "$PWD/logs/full-campaign" \
  --profile full \
  --model openrouter/deepseek/deepseek-v4-flash \
  --max-tokens 8000 \
  --max-steps 20 \
  --token-limit 100000 \
  --max-samples 8 \
  --batch-size 32 \
  --seed 42 \
  --poststages all \
  --execute "$@"
