#!/usr/bin/env bash
set -Eeuo pipefail

# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
load_config
activate_environment

export CUDA_VISIBLE_DEVICES="${AIGC_GPU_ID:-1}"
export HF_HUB_OFFLINE="${AIGC_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${AIGC_OFFLINE:-0}"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
export no_proxy="$NO_PROXY"

started=()
cleanup_done=0
cleanup_started() {
    local name
    if (( cleanup_done )); then
        return 0
    fi
    cleanup_done=1
    trap - ERR
    trap '' INT TERM
    set +e
    for name in api worker comfy; do
        if [[ " ${started[*]} " == *" $name "* ]]; then
            stop_process "$name"
        fi
    done
}
on_error() {
    local line="$1"
    trap - ERR
    printf '[aigc] ERROR: startup failed at line %s; cleaning up this attempt\n' "$line" >&2
    exit 1
}
on_signal() {
    local signal_name="$1"
    trap - ERR
    printf '[aigc] Startup interrupted by %s; cleaning up this attempt\n' "$signal_name" >&2
    case "$signal_name" in
        INT) exit 130 ;;
        TERM) exit 143 ;;
    esac
}
trap cleanup_started EXIT
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap 'on_error $LINENO' ERR

start_owned_process() {
    local name="$1"
    # Keep launch and ownership registration atomic with respect to INT/TERM.
    trap '' INT TERM
    start_process "$@"
    started+=("$name")
    trap 'on_signal INT' INT
    trap 'on_signal TERM' TERM
}

for service_name in comfy worker api; do
    if pid_is_valid "$service_name"; then
        fail "$service_name is already running; use status.sh or stop.sh first"
    fi
    rm -f "$(pid_file "$service_name")"
done

verify_arguments=(
    verify
    --manifest "$ROOT/config/models.lock.json"
    --models-root "$COMFY_DIR/models"
)
if [[ "${AIGC_INCLUDE_OPTIONAL_MODELS:-0}" == "1" ]]; then
    verify_arguments+=(--include-optional)
fi
python -m aigc_photomaker.model_manager "${verify_arguments[@]}"
check_gpu
# Never mistake an unrelated existing HTTP service for the process we just launched.
ensure_port_available 127.0.0.1 "${AIGC_COMFY_PORT:-8188}" comfy
ensure_port_available "${AIGC_APP_HOST:-0.0.0.0}" "${AIGC_APP_PORT:-8000}" api

mkdir -p "$COMFY_DIR/input"
cp "$ROOT/assets/warmup-portrait.png" "$COMFY_DIR/input/warmup-portrait.png"

comfy_extra_args=()
read -r -a comfy_extra_args <<< "${AIGC_COMFY_EXTRA_ARGS:-}" || true
start_owned_process comfy "$APP_RUNTIME/logs/comfy.log" \
    python "$COMFY_DIR/main.py" --listen 127.0.0.1 --port "${AIGC_COMFY_PORT:-8188}" \
    --preview-method none "${comfy_extra_args[@]}"
wait_http "http://127.0.0.1:${AIGC_COMFY_PORT:-8188}/system_stats" "${AIGC_COMFY_START_TIMEOUT:-300}"

python -m aigc_photomaker.warmup

start_owned_process worker "$APP_RUNTIME/logs/worker.log" python -m aigc_photomaker.worker

start_owned_process api "$APP_RUNTIME/logs/api.log" \
    python -m uvicorn --factory aigc_photomaker.app:create_app \
    --host "${AIGC_APP_HOST:-0.0.0.0}" --port "${AIGC_APP_PORT:-8000}"
wait_http "http://127.0.0.1:${AIGC_APP_PORT:-8000}/api/v1/health/ready" \
    "${AIGC_API_START_TIMEOUT:-120}"

trap - EXIT ERR INT TERM
log "ComfyUI, both-mode warmup, Worker, and API are ready"
log "ComfyUI: http://127.0.0.1:${AIGC_COMFY_PORT:-8188} (loopback only)"
log "API/Web listen: http://${AIGC_APP_HOST:-0.0.0.0}:${AIGC_APP_PORT:-8000}"
log "Browser: http://SERVER_IP:${AIGC_APP_PORT:-8000} (replace SERVER_IP)"
