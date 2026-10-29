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


def load_styles(path: Path) -> dict[str, StylePreset]:
    data = json.loads(path.read_text(encoding="utf-8"))
    styles = {}
    for key, values in data.items():
        preset = StylePreset(key=key, **values)
        if "person img" in preset.positive:
            raise ValueError(f"STYLE_LEGACY_TRIGGER:{key}")
        styles[key] = preset
    return styles
