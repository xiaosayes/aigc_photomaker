import os
from pathlib import Path
import shutil
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_gpu_preflight(tmp_path, *, total="24576", used="1632", pids="", extra_env=None):
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "scripts" / "lib.sh", project / "scripts" / "lib.sh")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    nvidia_smi = bin_dir / "nvidia-smi"
    nvidia_smi.write_text(
        """#!/usr/bin/env bash
set -eu
query=
for arg in "$@"; do
    case "$arg" in
        --query-gpu=*) query="${arg#--query-gpu=}" ;;
        --query-compute-apps=*) query=compute-pids ;;
    esac
done
case "$query" in
    name) printf 'NVIDIA GeForce RTX 3090\\n' ;;
    memory.total) printf '%s\\n' "$FAKE_GPU_TOTAL" ;;
    memory.used) printf '%s\\n' "$FAKE_GPU_USED" ;;
    compute-pids) [[ -z "$FAKE_GPU_PIDS" ]] || printf '%s\\n' "$FAKE_GPU_PIDS" ;;
    *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
            "FAKE_GPU_TOTAL": total,
            "FAKE_GPU_USED": used,
            "FAKE_GPU_PIDS": pids,
            "AIGC_GPU_ID": "1",
        }
    )
    env.pop("AIGC_GPU_MAX_USED_MIB", None)
    env.pop("AIGC_GPU_MIN_FREE_MIB", None)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", "-c", 'source scripts/lib.sh; check_gpu'],
        cwd=project,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_available_memory_passes_and_compute_pid_only_warns(tmp_path):
    result = run_gpu_preflight(tmp_path, used="1632", pids="4242")

    assert result.returncode == 0, result.stderr
    assert "warning" in result.stdout.lower()
    assert "4242" in result.stdout


def test_insufficient_free_memory_fails(tmp_path):
    result = run_gpu_preflight(tmp_path, used="5000")

    assert result.returncode != 0
    assert "19576" in result.stderr
    assert "20480" in result.stderr


def test_legacy_max_used_limit_is_ignored(tmp_path):
    result = run_gpu_preflight(
        tmp_path,
        used="1632",
        extra_env={"AIGC_GPU_MAX_USED_MIB": "1024"},
    )

    assert result.returncode == 0, result.stderr


def test_custom_minimum_free_memory_is_enforced(tmp_path):
    result = run_gpu_preflight(
        tmp_path,
        used="6000",
        extra_env={"AIGC_GPU_MIN_FREE_MIB": "19000"},
    )

    assert result.returncode != 0
    assert "18576" in result.stderr
    assert "19000" in result.stderr


def test_malformed_memory_data_fails(tmp_path):
    for field in ("total", "used"):
        kwargs = {field: "not-a-number"}
        result = run_gpu_preflight(tmp_path / field, **kwargs)

        assert result.returncode != 0
        assert "memory" in result.stderr.lower()
