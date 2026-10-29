# Server-pending: run these behavior tests on the GPU server.
import json
from pathlib import Path

import pytest

from aigc_photomaker.styles import load_styles


STYLES_PATH = Path("config/styles.json")


def test_styles_load_all_presets_with_required_copy_and_no_sdxl_lora():
    styles = load_styles(STYLES_PATH)

    assert set(styles) == {"cyberpunk", "anime", "ink_wash", "oil_painting"}
    for key, style in styles.items():
        assert style.key == key
        assert style.label
        assert style.description
        assert "person img" not in style.positive
        assert "person" in style.positive
        assert not hasattr(style, "lora_filename")
        assert not hasattr(style, "lora_strength")
        for term in ("solo", "one person", "single subject"):
            assert term in style.positive
        for term in (
            "low quality", "face asymmetry", "deformed eyes", "bad anatomy",
            "extra fingers", "watermark", "text", "logo", "multiple people",
            "two people", "twins", "duplicate person", "extra face",
        ):
            assert term in style.negative

    triggers = {
        "cyberpunk": "cyberpunk style",
        "anime": "anime",
        "ink_wash": "Chinese Ink",
        "oil_painting": "oil painting",
    }
    for key, trigger in triggers.items():
        assert trigger in styles[key].positive


def test_styles_file_is_valid_json():
    assert isinstance(json.loads(STYLES_PATH.read_text(encoding="utf-8")), dict)


def test_missing_style_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_styles(tmp_path / "missing.json")
