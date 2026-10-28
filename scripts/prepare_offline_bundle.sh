#!/usr/bin/env bash
set -Eeuo pipefail

# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
load_config

[[ "$AIGC_OFFLINE" == "0" ]] || fail "Bundle preparation requires a networked machine"
require_ubuntu_x86_64
[[ "$(uname -m)" == "x86_64" ]] || fail "An x86_64 bundle host is required"
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail "Python 3.11 is required"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Python 3.11 is required"'
command -v conda >/dev/null 2>&1 || fail "conda is required on the bundle builder"
command -v git >/dev/null 2>&1 || fail "git is required"
command -v curl >/dev/null 2>&1 || fail "curl is required"
command -v tar >/dev/null 2>&1 || fail "tar is required"
curl --fail --silent --show-error --location --max-time 20 --output /dev/null "$PIP_INDEX_URL" || \
    fail "The configured package index is unreachable"

work="$(mktemp -d "$RUNTIME_DIR/offline-bundle.XXXXXX")"
cleanup() {
    rm -rf "$work"
}
trap cleanup EXIT

clone_url="$COMFYUI_REPO"
archive_url="${COMFYUI_REPO%.git}/archive/$COMFYUI_COMMIT.tar.gz"
if [[ -n "${GITHUB_PROXY:-}" ]]; then
    clone_url="${GITHUB_PROXY%/}/$COMFYUI_REPO"
    archive_url="${GITHUB_PROXY%/}/$archive_url"
fi
if git clone "$clone_url" "$work/ComfyUI" && \
    git -C "$work/ComfyUI" checkout --detach "$COMFYUI_COMMIT"; then
    :
else
    log "Git clone/checkout failed; trying the pinned source tarball"
    rm -rf "$work/ComfyUI"
    curl --fail --location --retry 3 --output "$work/comfy.tar.gz" "$archive_url"
    mkdir -p "$work/ComfyUI"
    tar -xzf "$work/comfy.tar.gz" -C "$work/ComfyUI" --strip-components=1
fi
[[ "$(git -C "$work/ComfyUI" rev-parse HEAD 2>/dev/null || printf '%s' "$COMFYUI_COMMIT")" == "$COMFYUI_COMMIT" ]] || \
    fail "ComfyUI source does not match the locked commit"
printf '%s\n' "$COMFYUI_COMMIT" > "$work/ComfyUI/.aigc-commit"

mkdir -p "$ROOT/offline/comfyui" "$ROOT/offline/conda" "$ROOT/offline/wheelhouse" "$ROOT/offline/models"
tar --exclude=.git -czf "$ROOT/offline/comfyui/ComfyUI.tar.gz" -C "$work" ComfyUI
cat > "$ROOT/offline/SOURCE_COMMITS" <<EOF
COMFYUI_REPO=$COMFYUI_REPO
COMFYUI_COMMIT=$COMFYUI_COMMIT
PHOTOMAKER_PLUS_COMMIT=$PHOTOMAKER_PLUS_COMMIT
TORCH_VERSION=$TORCH_VERSION
TORCHVISION_VERSION=$TORCHVISION_VERSION
PYTORCH_CUDA=$PYTORCH_CUDA
EOF

builder_env="aigc-photomaker-builder"
builder_prefix="$(conda info --base)/envs/$builder_env"
if conda run -n "$builder_env" python -c \
    'import sys; raise SystemExit(sys.version_info[:2] != (3, 11))' >/dev/null 2>&1 && \
    conda run -n "$builder_env" python -m pip --version >/dev/null 2>&1; then
    :
elif [[ -d "$builder_prefix" ]]; then
    fail "Existing $builder_env is not a working Python 3.11+pip environment"
else
    conda create -y -n "$builder_env" python=3.11 pip
fi
conda run -n "$builder_env" python -m pip install --upgrade conda-pack
conda run -n "$builder_env" conda-pack -n "$builder_env" \
    -o "$ROOT/offline/conda/aigc-photomaker.tar.gz" --force

"$PYTHON_BIN" -m venv "$work/venv"
# shellcheck source=/dev/null
source "$work/venv/bin/activate"
pip_run install --upgrade pip setuptools wheel
pip_run install numpy==1.26.4 Cython==3.0.12 build httpx==0.28.1

pip_run download --dest "$ROOT/offline/wheelhouse" pip setuptools wheel \
    numpy==1.26.4 Cython==3.0.12 build
pip_run download --dest "$ROOT/offline/wheelhouse" \
    torch==2.9.1 torchvision==0.24.1 --index-url "$PYTORCH_INDEX_URL"
pip_run download --no-build-isolation --dest "$ROOT/offline/wheelhouse" \
    -r "$ROOT/requirements/app.txt" -r "$ROOT/requirements/ml.txt"
pip_run download --dest "$ROOT/offline/wheelhouse" -r "$work/ComfyUI/requirements.txt"

PYTHONPATH="$ROOT/src" python -m aigc_photomaker.model_manager download \
    --manifest "$ROOT/config/models.lock.json" \
    --models-root "$ROOT/offline/models" \
    --include-optional

(
    cd "$ROOT/offline"
    find comfyui conda models wheelhouse -type f -print0 | sort -z | xargs -0 sha256sum
    sha256sum SOURCE_COMMITS
) > "$ROOT/offline/SHA256SUMS"
log "Offline bundle prepared with conda-pack, locked sources, wheels, models, and SHA256SUMS"
