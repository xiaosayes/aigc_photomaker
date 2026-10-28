#!/usr/bin/env bash
set -Eeuo pipefail

# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
load_config

stop_process api
stop_process worker
stop_process comfy
log "API, Worker, and ComfyUI are stopped"
