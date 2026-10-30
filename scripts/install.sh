#!/usr/bin/env bash
set -Eeuo pipefail

# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
load_config

AIGC_SKIP_PYTHON_INSTALL="${AIGC_SKIP_PYTHON_INSTALL:-0}"
if [[ "$AIGC_SKIP_PYTHON_INSTALL" != "0" && "$AIGC_SKIP_PYTHON_INSTALL" != "1" ]]; then
    fail "AIGC_SKIP_PYTHON_INSTALL must be 0 or 1"
fi
if [[ "$AIGC_OFFLINE" == "1" && "$AIGC_SKIP_PYTHON_INSTALL" == "1" ]]; then
    fail "AIGC_OFFLINE=1 cannot be combined with AIGC_SKIP_PYTHON_INSTALL=1; offline prefix extraction would replace manually installed packages"
fi

require_ubuntu_x86_64
command -v conda >/dev/null 2>&1 || fail "conda is required"
if ! command -v gcc >/dev/null 2>&1 || ! command -v g++ >/dev/null 2>&1; then
    fail "C/C++ compilers are required; run: sudo apt-get install -y build-essential"
fi
command -v git >/dev/null 2>&1 || fail "git is required"
command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v tar >/dev/null 2>&1 || fail "tar is required"
if [[ "$AIGC_OFFLINE" == "1" ]]; then
    command -v sha256sum >/dev/null 2>&1 || fail "sha256sum is required for offline verification"
    [[ -f "$ROOT/offline/SHA256SUMS" ]] || fail "Offline SHA256SUMS is missing"
    (cd "$ROOT/offline" && sha256sum -c SHA256SUMS)
fi
check_gpu

configure_comfy_urls() {
    COMFY_CLONE_URL="$COMFYUI_REPO"
    COMFY_ARCHIVE_URL="${COMFYUI_REPO%.git}/archive/$COMFYUI_COMMIT.tar.gz"
    if [[ -n "${GITHUB_PROXY:-}" ]]; then
        COMFY_CLONE_URL="${GITHUB_PROXY%/}/$COMFYUI_REPO"
        COMFY_ARCHIVE_URL="${GITHUB_PROXY%/}/$COMFY_ARCHIVE_URL"
    fi
}

install_comfy_online() {
    local work
    configure_comfy_urls
    work="$(mktemp -d "$RUNTIME_DIR/comfy-install.XXXXXX")"
    if git clone "$COMFY_CLONE_URL" "$work/ComfyUI" && \
        git -C "$work/ComfyUI" checkout --detach "$COMFYUI_COMMIT"; then
        :
    else
        log "Git clone/checkout failed; trying the pinned source tarball"
        rm -rf "$work/ComfyUI"
        curl --fail --location --retry 3 --output "$work/ComfyUI.tar.gz" "$COMFY_ARCHIVE_URL"
        mkdir -p "$work/ComfyUI"
        tar -xzf "$work/ComfyUI.tar.gz" -C "$work/ComfyUI" --strip-components=1
    fi
    printf '%s\n' "$COMFYUI_COMMIT" > "$work/ComfyUI/.aigc-commit"
    rm -rf "$COMFY_DIR"
    mv "$work/ComfyUI" "$COMFY_DIR"
    rm -rf "$work"
}

install_comfy_offline() {
    local archive="$ROOT/offline/comfyui/ComfyUI.tar.gz" work actual_commit
    [[ -f "$archive" ]] || fail "Offline ComfyUI archive is missing: $archive"
    work="$(mktemp -d "$RUNTIME_DIR/comfy-install.XXXXXX")"
    mkdir -p "$work/ComfyUI"
    tar -xzf "$archive" -C "$work/ComfyUI"
    [[ -f "$work/ComfyUI/.aigc-commit" ]] || fail "Offline ComfyUI archive has no .aigc-commit"
    read -r actual_commit < "$work/ComfyUI/.aigc-commit"
    [[ "$actual_commit" == "$COMFYUI_COMMIT" ]] || fail "Offline ComfyUI commit does not match versions.lock.env"
    rm -rf "$COMFY_DIR"
    mv "$work/ComfyUI" "$COMFY_DIR"
    rm -rf "$work"
}

install_comfy() {
    local current_commit="" instantid_expected instantid_actual
    mkdir -p "$RUNTIME_DIR"
    if [[ "$AIGC_OFFLINE" == "1" ]]; then
        install_comfy_offline
    elif [[ -d "$COMFY_DIR/.git" ]]; then
        current_commit="$(git -C "$COMFY_DIR" rev-parse HEAD)"
        if [[ "$current_commit" == "$COMFYUI_COMMIT" ]]; then
            :
        else
            configure_comfy_urls
            if git -C "$COMFY_DIR" fetch "$COMFY_CLONE_URL" "$COMFYUI_COMMIT" && \
                git -C "$COMFY_DIR" checkout --detach --force "$COMFYUI_COMMIT"; then
                :
            else
                log "Existing checkout fetch failed; using clone or pinned tarball fallback"
                install_comfy_online
            fi
        fi
        printf '%s\n' "$COMFYUI_COMMIT" > "$COMFY_DIR/.aigc-commit"
    elif [[ -e "$COMFY_DIR" && -f "$COMFY_DIR/.aigc-commit" ]] && \
        [[ "$(<"$COMFY_DIR/.aigc-commit")" == "$COMFYUI_COMMIT" ]]; then
        :
    elif [[ -e "$COMFY_DIR" ]]; then
        install_comfy_online
    else
        install_comfy_online
    fi
    [[ "$(<"$COMFY_DIR/.aigc-commit")" == "$COMFYUI_COMMIT" ]] || \
        fail "ComfyUI source commit verification failed"
    if [[ -d "$COMFY_DIR/.git" ]]; then
        [[ "$(git -C "$COMFY_DIR" rev-parse HEAD)" == "$COMFYUI_COMMIT" ]] || \
            fail "ComfyUI checkout does not match the locked commit"
    fi

    # Only these custom-node directories are owned and replaced by this installer.
    rm -rf "$COMFY_DIR/custom_nodes/ComfyUI-PhotoMaker-Plus"
    rm -rf "$COMFY_DIR/custom_nodes/ComfyUI_InstantID"
    mkdir -p "$COMFY_DIR/custom_nodes"
    cp -a "$ROOT/vendor/ComfyUI-PhotoMaker-Plus" \
        "$COMFY_DIR/custom_nodes/ComfyUI-PhotoMaker-Plus"
    instantid_expected="$(tr -d '[:space:]' <<< "$INSTANTID_COMMIT")"
    instantid_actual="$(tr -d '[:space:]' < "$ROOT/vendor/ComfyUI_InstantID/.aigc-commit")"
    [[ "$instantid_actual" == "$instantid_expected" ]] || \
        fail "Vendored InstantID commit does not match versions.lock.env"
    cp -a "$ROOT/vendor/ComfyUI_InstantID" \
        "$COMFY_DIR/custom_nodes/ComfyUI_InstantID"
    instantid_actual="$(tr -d '[:space:]' < "$COMFY_DIR/custom_nodes/ComfyUI_InstantID/.aigc-commit")"
    [[ "$instantid_actual" == "$instantid_expected" ]] || \
        fail "Installed InstantID commit verification failed"
}

ensure_api_key() {
    umask 077
    if [[ -f "$ROOT/.env" ]]; then
        chmod 600 "$ROOT/.env"
    else
        : > "$ROOT/.env"
    fi
    python - "$ROOT/.env" <<'PY'
import re
import secrets
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = path.read_text(encoding="utf-8").splitlines()
pattern = re.compile(r"^\s*(?:export\s+)?AIGC_API_KEY\s*=\s*(.*?)\s*$")
valid = []
for index, line in enumerate(lines):
    match = pattern.match(line)
    if match is None:
        continue
    value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    if value:
        valid.append((index, line))

if valid:
    selected = valid[0][0]
    replacement = valid[0][1]
else:
    selected = next((i for i, line in enumerate(lines) if pattern.match(line)), len(lines))
    replacement = f"AIGC_API_KEY={secrets.token_urlsafe(48)}"

result = []
for index, line in enumerate(lines):
    if not pattern.match(line):
        result.append(line)
    elif index == selected:
        result.append(replacement)
if selected == len(lines):
    result.append(replacement)
path.write_text("\n".join(result) + ("\n" if result else ""), encoding="utf-8")
PY
    log "Ensured one non-empty private API key in .env (generated keys use 48 random bytes)"
    chmod 600 "$ROOT/.env"
}

install_conda_online() {
    local conda_env="${AIGC_CONDA_ENV:-aigc-photomaker}"
    if conda run -n "$conda_env" python -c \
        'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))' >/dev/null 2>&1; then
        :
    elif [[ -d "$(conda info --base)/envs/$conda_env" ]]; then
        fail "Existing conda environment $conda_env is not a working Python 3.11 environment"
    else
        conda create -y -n "$conda_env" python=3.11 pip
    fi
}

install_conda_offline() {
    local archive="$ROOT/offline/conda/aigc-photomaker.tar.gz"
    local conda_base prefix conda_env
    [[ -f "$archive" ]] || fail "Offline conda-pack archive is missing: $archive"
    conda_base="$(conda info --base)"
    conda_env="${AIGC_CONDA_ENV:-aigc-photomaker}"
    prefix="$conda_base/envs/$conda_env"
    if [[ "${CONDA_DEFAULT_ENV:-}" == "$conda_env" ]]; then
        # shellcheck source=/dev/null
        source "$conda_base/etc/profile.d/conda.sh"
        conda deactivate
    fi
    rm -rf "$prefix"
    mkdir -p "$prefix"
    tar -xzf "$archive" -C "$prefix"
    [[ -x "$prefix/bin/conda-unpack" ]] || fail "Offline Conda archive has no conda-unpack"
    "$prefix/bin/conda-unpack"
    "$prefix/bin/python" -c \
        'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))'
    "$prefix/bin/python" -m pip --version >/dev/null
}

install_comfy

if [[ "$AIGC_OFFLINE" == "1" ]]; then
    install_conda_offline
else
    install_conda_online
fi
# Activation happens only after an offline prefix has completed conda-unpack.
# shellcheck source=/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${AIGC_CONDA_ENV:-aigc-photomaker}"

if [[ "$AIGC_SKIP_PYTHON_INSTALL" == "0" ]]; then
    pip_run install --upgrade pip setuptools wheel
    pip_run install torch==2.9.1 torchvision==0.24.1 \
        --index-url "$PYTORCH_INDEX_URL"
    # Build prerequisites must be installed before InsightFace is built without isolation.
    pip_run install numpy==1.26.4 Cython==3.0.12 build
    pip_run install --no-build-isolation -r "$ROOT/requirements/app.txt" -r "$ROOT/requirements/ml.txt"
    pip_run install -r "$COMFY_DIR/requirements.txt"
    pip_run install -e "$ROOT" --no-deps
else
    log "Skipping Python package installation (AIGC_SKIP_PYTHON_INSTALL=1)"
fi
# Always install the local project itself; this does not install its dependencies.
# It is required for `python -m aigc_photomaker.model_manager` and all other module entry points.
pip_run install -e "$ROOT" --no-deps
pip_run check

ensure_api_key
bash "$ROOT/scripts/download_models.sh"
python -m pytest -q "$ROOT/tests"
log "Installation complete"
