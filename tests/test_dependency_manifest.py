import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
INSTALLER = ROOT / "scripts" / "install.sh"


def normalize_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_requirements(path=REQUIREMENTS):
    requirements = {}
    indexes = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("--index-url ") or line.startswith("--extra-index-url "):
            indexes.append(line)
            continue
        assert not line.startswith("-"), f"unsupported option on line {line_number}: {line}"
        match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(.*)$", line)
        assert match is not None, f"invalid requirement on line {line_number}: {line}"
        name, specifier = match.groups()
        normalized = normalize_name(name)
        assert normalized not in requirements, f"duplicate package: {name}"
        requirements[normalized] = specifier.strip()
    return requirements, indexes


def test_requirements_have_unique_normalized_names_and_supported_indexes():
    requirements, indexes = parse_requirements()
    assert requirements
    assert indexes == [
        "--index-url https://pypi.org/simple",
        "--extra-index-url https://download.pytorch.org/whl/cu126",
    ]
    assert "--no-build-isolation" not in REQUIREMENTS.read_text(encoding="utf-8")


def test_requirements_keep_locked_torch_comfyui_and_ml_pins():
    requirements, _ = parse_requirements()
    expected = {
        "torch": "==2.9.1",
        "torchvision": "==0.24.1",
        "comfyui-frontend-package": "==1.55.18",
        "comfyui-workflow-templates": "==0.11.89",
        "comfyui-embedded-docs": "==0.5.14",
        "comfy-kitchen": "==0.2.39",
        "comfy-aimdo": "==0.5.6",
        "numpy": "==1.26.4",
        "cython": "==3.0.12",
        "onnx": "==1.17.0",
        "insightface": "==0.7.3",
        "onnxruntime-gpu": "==1.20.2",
    }
    assert {name: requirements.get(name) for name in expected} == expected
    assert "onnxruntime" not in requirements
    lock = (ROOT / "config" / "versions.lock.env").read_text(encoding="utf-8")
    assert "COMFYUI_VERSION=v0.42.1" in lock
    assert "PYTORCH_CUDA=cu126" in lock


def test_requirements_document_non_isolated_build_prerequisites():
    text = REQUIREMENTS.read_text(encoding="utf-8")
    assert (
        "python -m pip install --upgrade pip setuptools wheel "
        "numpy==1.26.4 Cython==3.0.12 build"
    ) in text
    assert "PIP_NO_BUILD_ISOLATION=1 python -m pip install -r requirements.txt" in text


def test_skip_install_preserves_default_and_non_pip_install_steps():
    installer = INSTALLER.read_text(encoding="utf-8")
    assert 'AIGC_SKIP_PYTHON_INSTALL="${AIGC_SKIP_PYTHON_INSTALL:-0}"' in installer
    assert 'if [[ "$AIGC_SKIP_PYTHON_INSTALL" == "0" ]]; then' in installer
    install_block = installer.split(
        'if [[ "$AIGC_SKIP_PYTHON_INSTALL" == "0" ]]; then', 1
    )[1].split("pip_run check", 1)[0]
    assert 'pip_run install --no-build-isolation -r "$ROOT/requirements/app.txt"' in install_block
    assert 'pip_run install -r "$COMFY_DIR/requirements.txt"' in install_block
    assert 'fi\n# Always install the local project itself' in installer
    assert 'pip_run install -e "$ROOT" --no-deps\npip_run check' in installer
    post_install = installer.split("pip_run check", 1)[1]
    assert "ensure_api_key" in post_install
    assert '"$ROOT/scripts/download_models.sh"' in post_install
    assert 'python -m pytest -q "$ROOT/tests"' in post_install


def test_conda_environment_is_configurable_and_local_package_is_always_installed():
    installer = INSTALLER.read_text(encoding="utf-8")
    library = (ROOT / "scripts" / "lib.sh").read_text(encoding="utf-8")
    assert 'AIGC_CONDA_ENV="${AIGC_CONDA_ENV:-aigc-photomaker}"' in library
    assert 'conda activate "${AIGC_CONDA_ENV:-aigc-photomaker}"' in library
    assert 'conda create -y -n "$conda_env" python=3.11 pip' in installer
    assert 'prefix="$conda_base/envs/$conda_env"' in installer
    assert 'conda run -n aigc-photomaker' not in installer
    assert 'conda activate aigc-photomaker' not in installer
    assert 'AIGC_CONDA_ENV=photomaker' in (ROOT / ".env.example").read_text(encoding="utf-8")
    assert 'pip_run install -e "$ROOT" --no-deps\npip_run check' in installer


def test_skip_install_rejects_offline_before_comfyui_or_prefix_changes():
    installer = INSTALLER.read_text(encoding="utf-8")
    guard = (
        'if [[ "$AIGC_OFFLINE" == "1" && "$AIGC_SKIP_PYTHON_INSTALL" == "1" ]]; then\n'
        '    fail "AIGC_OFFLINE=1 cannot be combined with AIGC_SKIP_PYTHON_INSTALL=1; '
        'offline prefix extraction would replace manually installed packages"\n'
        "fi"
    )
    assert guard in installer
    assert installer.index(guard) < installer.index("\ninstall_comfy\n")
    assert 'fail "AIGC_SKIP_PYTHON_INSTALL must be 0 or 1"' in installer
