# Server-pending: run these real loopback-socket deployment tests on Ubuntu.
import shutil
import socket
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_port_preflight(tmp_path, port):
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "scripts" / "lib.sh", project / "scripts" / "lib.sh")
    return subprocess.run(
        [
            "bash",
            "-c",
            f"source scripts/lib.sh; ensure_port_available 127.0.0.1 {port} comfy",
        ],
        cwd=project,
        text=True,
        capture_output=True,
        check=False,
    )


def test_occupied_loopback_port_fails_before_startup(tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        result = run_port_preflight(tmp_path, port)

    assert result.returncode != 0
    assert "comfy" in result.stderr
    assert f"127.0.0.1:{port}" in result.stderr
    assert "ss -ltnp" in result.stderr


def test_free_loopback_port_passes(tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    result = run_port_preflight(tmp_path, port)

    assert result.returncode == 0, result.stderr


def test_start_and_status_use_the_same_configurable_comfy_port():
    start = (REPO_ROOT / "scripts" / "start.sh").read_text(encoding="utf-8")
    status = (REPO_ROOT / "scripts" / "status.sh").read_text(encoding="utf-8")

    assert 'ensure_port_available 127.0.0.1 "${AIGC_COMFY_PORT:-8188}" comfy' in start
    assert '--port "${AIGC_COMFY_PORT:-8188}"' in start
    assert 'http://127.0.0.1:${AIGC_COMFY_PORT:-8188}/system_stats' in start
    assert 'for port in "${AIGC_COMFY_PORT:-8188}" "${AIGC_APP_PORT:-8000}"' in status
    assert "AIGC_COMFY_PORT=8188" in (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
