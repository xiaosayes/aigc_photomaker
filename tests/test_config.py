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
    assert settings.standard_steps == 20
    assert settings.turbo_steps == 4
    assert settings.standard_cfg == 4.0
    assert settings.standard_sampler == "euler"
    assert settings.turbo_cfg == 1.0
    assert settings.flux_base_unet_filename == "flux-2-klein-base-4b-fp8.safetensors"
    assert settings.flux_turbo_unet_filename == "flux-2-klein-4b-fp8.safetensors"
    assert settings.flux_text_encoder_filename == "qwen_3_4b.safetensors"
    assert settings.flux_vae_filename == "flux2-vae.safetensors"


def test_custom_port_environment_is_still_honored(monkeypatch):
    monkeypatch.setenv("AIGC_API_KEY", "a" * 32)
    monkeypatch.setenv("AIGC_APP_PORT", "8186")
    monkeypatch.setenv("AIGC_COMFY_PORT", "8185")

    settings = Settings(_env_file=None)

    assert settings.app_port == 8186
    assert settings.comfy_port == 8185
    assert settings.comfy_url == "http://127.0.0.1:8185"
