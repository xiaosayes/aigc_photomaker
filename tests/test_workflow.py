# Server-pending: run these SDXL + InstantID workflow tests on the GPU server.
from pathlib import Path
import subprocess
import sys

import pytest

from aigc_photomaker.config import Settings
from aigc_photomaker.styles import load_styles
from aigc_photomaker.workflow import ComfyImageRef, build_workflow, prepare_workflow


def settings():
    return Settings(api_key="a" * 32, _env_file=None)


def test_standard_contains_sdxl_instantid_style_and_output_guard_chain():
    graph = build_workflow("standard")

    assert graph["1"]["class_type"] == "CheckpointLoaderSimple"
    assert graph["1"]["inputs"]["ckpt_name"] == "sd_xl_base_1.0.safetensors"
    assert graph["2"]["class_type"] == "InstantIDModelLoader"
    assert graph["2"]["inputs"]["instantid_file"] == "ip-adapter.bin"
    assert graph["3"]["class_type"] == "InstantIDFaceAnalysis"
    assert graph["4"]["class_type"] == "ControlNetLoader"
    assert graph["4"]["inputs"]["control_net_name"] == (
        "instantid/diffusion_pytorch_model.safetensors"
    )
    assert graph["16"]["class_type"] == "InstantIDSingleFaceInputGuard"
    assert graph["16"]["inputs"] == {"images": ["5", 0], "faceanalysis": ["3", 0]}
    assert graph["13"]["class_type"] == "LoraLoaderModelOnly"
    assert graph["15"]["class_type"] == "ApplyInstantID"
    assert graph["15"]["inputs"]["model"] == ["13", 0]
    assert graph["15"]["inputs"]["image"] == ["16", 0]
    assert graph["9"]["inputs"]["model"] == ["15", 0]
    assert graph["9"]["inputs"]["positive"] == ["15", 1]
    assert graph["9"]["inputs"]["negative"] == ["15", 2]
    assert graph["9"]["inputs"]["steps"] == 25
    assert graph["8"]["inputs"] == {"width": 1024, "height": 1024, "batch_size": 1}
    assert graph["14"]["class_type"] == "InstantIDSingleFaceOutputGuard"
    assert graph["14"]["inputs"] == {"images": ["11", 0], "faceanalysis": ["3", 0]}
    assert graph["12"]["inputs"]["images"] == ["14", 0]


def test_turbo_places_lcm_between_style_lora_and_instantid():
    graph = build_workflow("turbo")

    assert graph["10"]["class_type"] == "LoraLoaderModelOnly"
    assert graph["10"]["inputs"]["model"] == ["13", 0]
    assert graph["15"]["inputs"]["model"] == ["10", 0]
    assert graph["9"]["inputs"]["model"] == ["15", 0]
    assert graph["9"]["inputs"]["sampler_name"] == "lcm"
    assert graph["9"]["inputs"]["steps"] == 8
    assert graph["9"]["inputs"]["cfg"] == 1.8
    assert graph["8"]["inputs"]["width"] == 1024
    assert graph["8"]["inputs"]["height"] == 1024


def test_parameter_injection_uses_instantid_without_photomaker_trigger():
    styles = load_styles(Path("config/styles.json"))
    graph = prepare_workflow(
        "standard", ComfyImageRef("face.png"), styles["cyberpunk"],
        42, "aigc/task", settings(),
    )

    assert graph["5"]["inputs"]["image"] == "face.png"
    assert graph["9"]["inputs"]["seed"] == 42
    assert graph["12"]["inputs"]["filename_prefix"] == "aigc/task"
    assert graph["6"]["inputs"]["text"] == styles["cyberpunk"].positive
    assert "person img" not in graph["6"]["inputs"]["text"]
    assert graph["7"]["inputs"]["text"] == styles["cyberpunk"].negative
    assert graph["1"]["inputs"]["ckpt_name"] == "sd_xl_base_1.0.safetensors"
    assert graph["3"]["inputs"]["provider"] == settings().face_provider
    assert graph["13"]["inputs"]["lora_name"] == "lah-cyberpunk-sdxl.safetensors"
    assert graph["13"]["inputs"]["strength_model"] == 1.0


def test_prepare_uses_mode_settings_subfolder_and_instantid_weight():
    styles = load_styles(Path("config/styles.json"))
    configured = settings().model_copy(
        update={
            "standard_steps": 31,
            "standard_cfg": 6.5,
            "standard_sampler": "euler",
            "standard_scheduler": "normal",
            "turbo_steps": 10,
            "turbo_cfg": 2.0,
            "instantid_weight": 0.65,
        }
    )
    original = build_workflow("turbo")

    graph = prepare_workflow(
        "turbo", ComfyImageRef("face.png", subfolder="uploads"), styles["anime"],
        7, "aigc/turbo", configured,
    )

    assert graph["5"]["inputs"]["image"] == "uploads/face.png"
    assert graph["9"]["inputs"]["steps"] == 10
    assert graph["9"]["inputs"]["cfg"] == 2.0
    assert graph["9"]["inputs"]["sampler_name"] == "lcm"
    assert graph["9"]["inputs"]["scheduler"] == "normal"
    assert graph["13"]["inputs"]["lora_name"] == "anime-sdxl.safetensors"
    assert graph["13"]["inputs"]["strength_model"] == 2.0
    assert graph["15"]["inputs"]["weight"] == 0.65
    assert original["9"]["inputs"]["steps"] == 8


@pytest.mark.parametrize(
    ("key", "filename", "strength"),
    [
        ("cyberpunk", "lah-cyberpunk-sdxl.safetensors", 1.0),
        ("anime", "anime-sdxl.safetensors", 2.0),
        ("ink_wash", "chinese-ink-sdxl.safetensors", 0.8),
        ("oil_painting", "oil-painting-sdxl.safetensors", 2.0),
    ],
)
def test_each_style_selects_exactly_one_style_lora(key, filename, strength):
    styles = load_styles(Path("config/styles.json"))
    graph = prepare_workflow(
        "standard", ComfyImageRef("face.png"), styles[key], 42, "aigc/task", settings()
    )
    style_loaders = [
        node for node in graph.values()
        if node["class_type"] == "LoraLoaderModelOnly"
        and node["inputs"]["lora_name"] != "lcm-lora-sdxl.safetensors"
    ]
    assert len(style_loaders) == 1
    assert style_loaders[0]["inputs"]["lora_name"] == filename
    assert style_loaders[0]["inputs"]["strength_model"] == strength


def test_prepare_standard_uses_custom_sampler_and_scheduler():
    styles = load_styles(Path("config/styles.json"))
    configured = settings().model_copy(
        update={"standard_sampler": "euler", "standard_scheduler": "normal"}
    )
    graph = prepare_workflow(
        "standard", ComfyImageRef("face.png"), styles["anime"], 7, "aigc/test", configured
    )
    assert graph["9"]["inputs"]["sampler_name"] == "euler"
    assert graph["9"]["inputs"]["scheduler"] == "normal"


def test_committed_workflow_artifacts_are_reproducible():
    completed = subprocess.run(
        [sys.executable, "scripts/build_workflows.py", "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_prepare_rejects_non_input_images_and_invalid_modes():
    styles = load_styles(Path("config/styles.json"))
    with pytest.raises(ValueError, match="INVALID_IMAGE_TYPE"):
        prepare_workflow(
            "standard", ComfyImageRef("face.png", type="output"), styles["anime"],
            7, "aigc/test", settings(),
        )
    with pytest.raises(ValueError, match="INVALID_MODE"):
        build_workflow("other")
