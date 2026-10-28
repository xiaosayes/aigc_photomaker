#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$ROOT/.runtime"
APP_RUNTIME="$RUNTIME_DIR/app"
COMFY_DIR="${AIGC_COMFY_DIR:-$RUNTIME_DIR/ComfyUI}"
mkdir -p "$APP_RUNTIME/logs" "$APP_RUNTIME/pids" "$ROOT/data/inputs" "$ROOT/data/outputs"

# Every pip invocation goes through pip_run so offline mode cannot silently use an index.
PIP_OFFLINE_ARGS=(--no-index --find-links="$ROOT/offline/wheelhouse")

log() {
    printf '[aigc] %s\n' "$*"
}

fail() {
    printf '[aigc] ERROR: %s\n' "$*" >&2
    return 1
}

load_config() {
    # Snapshot explicit operational values without evaluating their contents. The version
    # lock and .env supply defaults, but cannot override the caller's exported choices.
    local name
    local -A operational_overrides=()
    while IFS= read -r name; do
        [[ -n "$name" ]] || continue
        operational_overrides["$name"]="${!name}"
    done < <(compgen -v AIGC_ || true)
    for name in HF_ENDPOINT PIP_INDEX_URL PIP_EXTRA_INDEX_URL PYTORCH_INDEX_URL GITHUB_PROXY; do
        if [[ -v "$name" ]]; then
            operational_overrides["$name"]="${!name}"
        fi
    done

    local locked_comfyui_repo locked_comfyui_version locked_comfyui_commit
    local locked_photomaker_commit locked_instantid_commit locked_torch_version locked_torchvision_version locked_pytorch_cuda
    # shellcheck source=/dev/null
    source "$ROOT/config/versions.lock.env"
    locked_comfyui_repo="$COMFYUI_REPO"
    locked_comfyui_version="$COMFYUI_VERSION"
    locked_comfyui_commit="$COMFYUI_COMMIT"
    locked_photomaker_commit="$PHOTOMAKER_PLUS_COMMIT"
    locked_instantid_commit="$INSTANTID_COMMIT"
    locked_torch_version="$TORCH_VERSION"
    locked_torchvision_version="$TORCHVISION_VERSION"
    locked_pytorch_cuda="$PYTORCH_CUDA"

    if [[ -f "$ROOT/.env" ]]; then
        set -a
        # shellcheck source=/dev/null
        source "$ROOT/.env"
        set +a
    fi

    # Source identity and immutable versions always come from versions.lock.env.
    COMFYUI_REPO="$locked_comfyui_repo"
    COMFYUI_VERSION="$locked_comfyui_version"
    COMFYUI_COMMIT="$locked_comfyui_commit"
    PHOTOMAKER_PLUS_COMMIT="$locked_photomaker_commit"
    INSTANTID_COMMIT="$locked_instantid_commit"
    TORCH_VERSION="$locked_torch_version"
    TORCHVISION_VERSION="$locked_torchvision_version"
    PYTORCH_CUDA="$locked_pytorch_cuda"
    for name in "${!operational_overrides[@]}"; do
        printf -v "$name" '%s' "${operational_overrides[$name]}"
        export "$name"
    done

    AIGC_OFFLINE="${AIGC_OFFLINE:-0}"
    if [[ "$AIGC_OFFLINE" != "0" && "$AIGC_OFFLINE" != "1" ]]; then
        fail "AIGC_OFFLINE must be 0 or 1"
    fi
    AIGC_GPU_ID="${AIGC_GPU_ID:-1}"
    AIGC_CONDA_ENV="${AIGC_CONDA_ENV:-aigc-photomaker}"
    export AIGC_CONDA_ENV
    export CUDA_VISIBLE_DEVICES="${AIGC_GPU_ID:-1}"
    COMFY_DIR="${AIGC_COMFY_DIR:-$RUNTIME_DIR/ComfyUI}"
    [[ "$COMFY_DIR" == /* ]] || COMFY_DIR="$ROOT/$COMFY_DIR"
    APP_RUNTIME="$RUNTIME_DIR/app"
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
    export PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
    export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost,::1"
    export no_proxy="$NO_PROXY"
    if [[ "$AIGC_OFFLINE" == "1" ]]; then
        export PIP_NO_INDEX=1
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
    else
        unset PIP_NO_INDEX
        export HF_HUB_OFFLINE=0
        export TRANSFORMERS_OFFLINE=0
    fi
}

activate_environment() {
    command -v conda >/dev/null 2>&1 || fail "conda is required"
    # shellcheck source=/dev/null
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${AIGC_CONDA_ENV:-aigc-photomaker}"
}

pip_run() {
    local offline_args=()
    if [[ "${AIGC_OFFLINE:-0}" == "1" ]]; then
        offline_args=("${PIP_OFFLINE_ARGS[@]}")
    fi
    python -m pip "${offline_args[@]}" "$@"
}

require_ubuntu_x86_64() {
    [[ "$(uname -m)" == "x86_64" ]] || fail "Ubuntu x86_64 is required"
    [[ -r /etc/os-release ]] || fail "Cannot identify this operating system"
    # shellcheck source=/dev/null
    source /etc/os-release
    [[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "22.04" ]] || \
        fail "Ubuntu 22.04 x86_64 is required"
}

check_gpu() {
    local gpu_id="${AIGC_GPU_ID:-1}"
    local total_mib used_mib free_mib min_free_mib compute_pids
    command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi is required"
    nvidia-smi -i "$gpu_id" --query-gpu=name --format=csv,noheader >/dev/null || \
        fail "Physical GPU $gpu_id is unavailable"
    total_mib="$(nvidia-smi -i "$gpu_id" --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)"
    used_mib="$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1)"
    [[ "$total_mib" =~ ^[0-9]+$ && "$used_mib" =~ ^[0-9]+$ ]] || \
        fail "Cannot read GPU $gpu_id memory data"
    min_free_mib="${AIGC_GPU_MIN_FREE_MIB:-20480}"
    [[ "$min_free_mib" =~ ^[0-9]+$ ]] || fail "AIGC_GPU_MIN_FREE_MIB must be an integer"
    free_mib=$((10#$total_mib - 10#$used_mib))
    (( free_mib >= 10#$min_free_mib )) || \
        fail "GPU $gpu_id has ${free_mib} MiB free (minimum ${min_free_mib} MiB required)"
    compute_pids="$(nvidia-smi -i "$gpu_id" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    [[ -z "${compute_pids//[[:space:]]/}" ]] || \
        log "WARNING: GPU $gpu_id has compute processes: ${compute_pids//$'\n'/, }"
}

ensure_port_available() {
    local host="$1" port="$2" service_name="$3"
    python - "$host" "$port" "$service_name" <<'PY'
import socket
import sys

host, raw_port, service_name = sys.argv[1:]
try:
    port = int(raw_port)
    if not 1 <= port <= 65535:
        raise ValueError
except ValueError:
    print(f"[aigc] ERROR: invalid TCP port for {service_name}: {raw_port}", file=sys.stderr)
    raise SystemExit(1)

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind((host, port))
except OSError:
    print(
        f"[aigc] ERROR: {service_name} cannot bind {host}:{port}; "
        f"another process is using the address. Inspect it with: "
        f"ss -ltnp | grep ':{port}'",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
}

rotate_log() {
    local path="$1" index previous
    [[ -f "$path" ]] || return 0
    if (( $(stat -c %s "$path") >= 50 * 1024 * 1024 )); then
        for (( index=5; index>=1; index-- )); do
            previous=$((index - 1))
            if (( index == 1 )); then
                [[ ! -f "$path" ]] || mv -f "$path" "$path.1"
            elif [[ -f "$path.$previous" ]]; then
                mv -f "$path.$previous" "$path.$index"
            fi
        done
    fi
}

pid_file() {
    printf '%s/pids/%s.pid' "$APP_RUNTIME" "$1"
}

pid_is_valid() {
    local name="$1" file pid marker cwd
    file="$(pid_file "$name")"
    [[ -f "$file" ]] || return 1
    read -r pid < "$file" || return 1
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    (( pid > 1 )) || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    case "$name" in
        comfy) marker="$COMFY_DIR/main.py" ;;
        worker) marker="aigc_photomaker.worker" ;;
        api) marker="aigc_photomaker.app:create_app" ;;
        *) return 1 ;;
    esac
    tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq -- "$marker" || return 1
    cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
    [[ "$cwd" == "$ROOT" ]]
}

show_pid() {
    local name="$1" file pid
    file="$(pid_file "$name")"
    if pid_is_valid "$name"; then
        read -r pid < "$file"
        printf '%s: running (pid %s)\n' "$name" "$pid"
        return 0
    fi
    printf '%s: stopped\n' "$name"
    return 1
}

start_process() {
    local name="$1" log_path="$2" pid
    shift 2
    if pid_is_valid "$name"; then
        fail "$name is already running"
    fi
    rm -f "$(pid_file "$name")"
    rotate_log "$log_path"
    (
        cd "$ROOT"
        setsid "$@" >> "$log_path" 2>&1 &
        pid=$!
        printf '%s\n' "$pid" > "$(pid_file "$name")"
    )
}

stop_process() {
    local name="$1" file pid attempt
    file="$(pid_file "$name")"
    if ! pid_is_valid "$name"; then
        rm -f "$file"
        return 0
    fi
    read -r pid < "$file"
    kill -TERM -- "-$pid" 2>/dev/null || true
    for (( attempt=0; attempt<30; attempt++ )); do
        if ! kill -0 "$pid" 2>/dev/null; then
            rm -f "$file"
            return 0
        fi
        sleep 1
    done
    log "$name did not stop after SIGTERM; sending SIGKILL"
    kill -KILL -- "-$pid" 2>/dev/null || true
    for (( attempt=0; attempt<5; attempt++ )); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
        fail "Unable to stop $name process group $pid"
    fi
    rm -f "$file"
}

wait_http() {
    local url="$1" timeout_seconds="${2:-60}" elapsed=0
    while (( elapsed < timeout_seconds )); do
        if curl --noproxy '*' --fail --silent --show-error --max-time 5 \
            --output /dev/null "$url"; then
            return 0
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    fail "Timed out waiting for $url"
}
