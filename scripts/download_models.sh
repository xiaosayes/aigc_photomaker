#!/usr/bin/env bash
set -Eeuo pipefail

# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
load_config
activate_environment

arguments=(
    download
    --manifest "$ROOT/config/models.lock.json"
    --models-root "$COMFY_DIR/models"
)
if [[ "${AIGC_INCLUDE_OPTIONAL_MODELS:-0}" == "1" ]]; then
    arguments+=(--include-optional)
fi
if [[ "$AIGC_OFFLINE" == "1" ]]; then
    arguments+=(--offline-dir "$ROOT/offline/models")
fi
python -m aigc_photomaker.model_manager "${arguments[@]}"
python "$ROOT/scripts/extract_antelopev2.py" \
    --archive "$COMFY_DIR/models/instantid/antelopev2.zip" \
    --destination "$COMFY_DIR/models/insightface/models/antelopev2"
