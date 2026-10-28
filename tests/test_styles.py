# Server-pending: run these behavior tests on the GPU server.
import json
from pathlib import Path

import pytest

from aigc_photomaker.styles import StylePreset, load_styles


STYLES_PATH = Path("config/styles.json")


def test_styles_load_all_presets_with_required_copy():
    styles = load_styles(STYLES_PATH)

    assert set(styles) == {"cyberpunk", "anime", "ink_wash", "oil_painting"}
    for key, style in styles.items():
        assert style.key == key
        assert style.label
        assert style.description
        assert "person img" not in style.positive
        assert "person" in style.positive
        for term in ("solo", "one person", "single subject"):
            assert term in style.positive
        for term in (
            "low quality", "face asymmetry", "deformed eyes", "bad anatomy",
            "extra fingers", "watermark", "text", "logo", "multiple people",
            "two people", "twins", "duplicate person", "extra face",
        ):
            assert term in style.negative

    expected_loras = {
        "cyberpunk": ("lah-cyberpunk-sdxl.safetensors", 1.0, "cyberpunk style"),
        "anime": ("anime-sdxl.safetensors", 2.0, "anime"),
        "ink_wash": ("chinese-ink-sdxl.safetensors", 0.8, "Chinese Ink"),
        "oil_painting": ("oil-painting-sdxl.safetensors", 2.0, "oil painting"),
    }
    for key, (filename, strength, trigger) in expected_loras.items():
        assert styles[key].lora_filename == filename
        assert styles[key].lora_strength == strength
        assert trigger in styles[key].positive


def test_style_preset_requires_an_explicit_lora_contract():
    with pytest.raises(TypeError):
        StylePreset(
            key="invalid",
            label="Invalid",
            description="Missing LoRA configuration",
            positive="portrait of a person",
            negative="low quality",
        )


def test_styles_file_is_valid_json():
    assert isinstance(json.loads(STYLES_PATH.read_text(encoding="utf-8")), dict)


def test_missing_style_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_styles(tmp_path / "missing.json")
