#!/usr/bin/env bash
set -Eeuo pipefail

# Real GPU/API acceptance: run only on the configured server, never as a local test.
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
load_config
[[ -n "${AIGC_API_KEY:-}" ]] || fail "AIGC_API_KEY is required"
activate_environment

API_URL="${AIGC_ACCEPTANCE_API_URL:-http://127.0.0.1:${AIGC_APP_PORT:-8000}}"
INPUT_PATH="${AIGC_ACCEPTANCE_INPUT:-$ROOT/assets/warmup-portrait.png}"
export AIGC_API_KEY
python -m aigc_photomaker.smoke --api-url "$API_URL" --input "$INPUT_PATH" "$@"
