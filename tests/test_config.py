# Server-pending: run these dependency/behavior tests on the GPU server.
import os

import pytest
from pydantic import ValidationError

from aigc_photomaker.config import Settings


def test_api_key_is_required(monkeypatch):
    monkeypatch.delenv("AIGC_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_defaults_match_runtime_contract(monkeypatch):
    # _env_file=None disables dotenv, not variables exported by install.sh.
    for name in tuple(os.environ):
        if name.startswith("AIGC_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AIGC_API_KEY", "a" * 32)
    settings = Settings(_env_file=None)
    assert settings.app_port == 8000
    assert settings.comfy_port == 8188
    assert settings.max_upload_bytes == 10 * 1024 * 1024
    assert settings.worker_lease_seconds == 60
    assert settings.standard_steps == 25
    assert settings.turbo_steps == 8
    assert settings.standard_cfg == 4.0
    assert settings.checkpoint_filename == "sd_xl_base_1.0.safetensors"
    assert settings.instantid_weight == 0.8


def test_custom_port_environment_is_still_honored(monkeypatch):
    monkeypatch.setenv("AIGC_API_KEY", "a" * 32)
    monkeypatch.setenv("AIGC_APP_PORT", "8186")
    monkeypatch.setenv("AIGC_COMFY_PORT", "8185")

    settings = Settings(_env_file=None)

    assert settings.app_port == 8186
    assert settings.comfy_port == 8185
    assert settings.comfy_url == "http://127.0.0.1:8185"
