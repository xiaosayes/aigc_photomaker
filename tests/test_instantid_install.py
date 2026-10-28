# Server-pending: InstantID source-lock deployment checks.
from pathlib import Path


def test_instantid_commit_lock_is_exact_and_lf_normalized():
    lock = Path("config/versions.lock.env").read_text(encoding="utf-8")
    commit = "72495e806bc2ab9c41581e15ccaa1bcf83c477e8"
    assert f"INSTANTID_COMMIT={commit}" in lock
    raw = Path("vendor/ComfyUI_InstantID/.aigc-commit").read_bytes()
    assert raw == f"{commit}\n".encode("ascii")
    assert b"\r" not in raw


def test_installer_ignores_crlf_when_comparing_instantid_commits():
    installer = Path("scripts/install.sh").read_text(encoding="utf-8")
    assert installer.count("tr -d '[:space:]'") >= 3
    assert 'instantid_actual="$(' in installer
    assert 'instantid_expected="$(' in installer
