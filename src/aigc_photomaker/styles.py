import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StylePreset:
    key: str
    label: str
    description: str
    positive: str
    negative: str
    lora_filename: str
    lora_strength: float


def load_styles(path: Path) -> dict[str, StylePreset]:
    data = json.loads(path.read_text(encoding="utf-8"))
    styles = {}
    for key, values in data.items():
        preset = StylePreset(key=key, **values)
        if (
            not preset.lora_filename.endswith(".safetensors")
            or "/" in preset.lora_filename
            or "\\" in preset.lora_filename
            or not 0 < preset.lora_strength <= 3
        ):
            raise ValueError(f"STYLE_LORA_CONFIG:{key}")
        styles[key] = preset
    return styles
