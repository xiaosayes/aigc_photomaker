#!/usr/bin/env bash
set -Eeuo pipefail

# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
load_config

status=0
for service_name in api worker comfy; do
    show_pid "$service_name" || status=1
done

printf '\nListening TCP ports (ComfyUI %s and API %s):\n' \
    "${AIGC_COMFY_PORT:-8188}" "${AIGC_APP_PORT:-8000}"
if command -v ss >/dev/null 2>&1; then
    if listeners="$(ss -ltnp)"; then
        printf '%s\n' "$listeners"
        for port in "${AIGC_COMFY_PORT:-8188}" "${AIGC_APP_PORT:-8000}"; do
            if ! grep -Eq ":${port}([[:space:]]|$)" <<<"$listeners"; then
                printf 'No listener on TCP port %s\n' "$port"
                status=1
            fi
        done
    else
        printf 'ss failed while inspecting TCP listeners\n'
        status=1
    fi
else
    printf 'ss is unavailable; cannot verify TCP listeners\n'
    status=1
fi

printf '\nPhysical GPU %s:\n' "${AIGC_GPU_ID:-1}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -i "${AIGC_GPU_ID:-1}" \
        --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
        --format=csv,noheader || status=1
else
    printf 'nvidia-smi is unavailable\n'
    status=1
fi

printf '\nDisk usage:\n'
df -h "$ROOT" || status=1
printf '\nHealth:\n'
if curl --noproxy '*' --silent --show-error --fail-with-body --max-time 5 \
    "http://127.0.0.1:${AIGC_APP_PORT:-8000}/api/v1/health/ready"; then
    printf '\n'
else
    printf 'API health is unavailable\n'
    status=1
fi
exit "$status"
