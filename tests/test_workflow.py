# Server-pending: FLUX.2 Klein workflow tests for the GPU server.
from pathlib import Path
import subprocess
import sys

import pytest

from aigc_photomaker.config import Settings
from aigc_photomaker.styles import load_styles
from aigc_photomaker.workflow import ComfyImageRef, build_workflow, prepare_workflow


def settings():
    return Settings(api_key="a" * 32, _env_file=None)


def assert_common_flux_chain(graph):
    assert graph["2"]["class_type"] == "CLIPLoader"
    assert graph["2"]["inputs"]["clip_name"] == "qwen_3_4b.safetensors"
    assert graph["2"]["inputs"]["type"] == "flux2"
    assert graph["3"]["class_type"] == "VAELoader"
    assert graph["3"]["inputs"]["vae_name"] == "flux2-vae.safetensors"
    assert graph["4"]["class_type"] == "InstantIDFaceAnalysis"
    assert graph["18"]["class_type"] == "ImageScale"
    assert graph["18"]["inputs"]["width"] == 1024
    assert graph["18"]["inputs"]["height"] == 1024
    assert graph["18"]["inputs"]["crop"] == "center"
    assert graph["16"]["class_type"] == "FluxSingleFaceInputGuard"
    assert graph["16"]["inputs"] == {"images": ["18", 0], "faceanalysis": ["4", 0]}
    assert graph["19"]["class_type"] == "VAEEncode"
    assert graph["19"]["inputs"] == {"pixels": ["16", 0], "vae": ["3", 0]}
    assert graph["20"]["class_type"] == "ReferenceLatent"
    assert graph["20"]["inputs"] == {"conditioning": ["6", 0], "latent": ["19", 0]}
    assert graph["21"]["class_type"] == "ReferenceLatent"
    assert graph["21"]["inputs"] == {"conditioning": ["7", 0], "latent": ["19", 0]}
    assert graph["15"]["class_type"] == "CFGGuider"
    assert graph["15"]["inputs"]["model"] == ["1", 0]
    assert graph["15"]["inputs"]["positive"] == ["20", 0]
    assert graph["15"]["inputs"]["negative"] == ["21", 0]
    assert graph["13"]["class_type"] == "Flux2Scheduler"
    assert graph["13"]["inputs"]["width"] == 1024
    assert graph["13"]["inputs"]["height"] == 1024
    assert graph["17"]["class_type"] == "RandomNoise"
    assert graph["10"]["class_type"] == "KSamplerSelect"
    assert graph["9"]["class_type"] == "SamplerCustomAdvanced"
    assert graph["9"]["inputs"] == {
        "noise": ["17", 0], "guider": ["15", 0], "sampler": ["10", 0],
        "sigmas": ["13", 0], "latent_image": ["8", 0],
    }
    assert graph["8"]["class_type"] == "EmptyFlux2LatentImage"
    assert graph["8"]["inputs"] == {"width": 1024, "height": 1024, "batch_size": 1}
    assert graph["14"]["class_type"] == "FluxSingleFaceOutputGuard"
    assert graph["14"]["inputs"] == {"images": ["11", 0], "faceanalysis": ["4", 0]}
    assert graph["12"]["inputs"]["images"] == ["14", 0]
    assert all(node["class_type"] != "LoraLoaderModelOnly" for node in graph.values())


def test_standard_uses_distilled_fp8_with_eight_steps_and_advanced_sampling():
    graph = build_workflow("standard")
    assert graph["1"]["class_type"] == "UNETLoader"
    assert graph["1"]["inputs"]["unet_name"] == "flux-2-klein-4b-fp8.safetensors"
    assert graph["13"]["inputs"]["steps"] == 8
    assert graph["15"]["inputs"]["cfg"] == 1.0
    assert graph["10"]["inputs"]["sampler_name"] == "euler"
    assert_common_flux_chain(graph)


def test_turbo_uses_distilled_fp8_and_four_steps():
    graph = build_workflow("turbo")
    assert graph["1"]["inputs"]["unet_name"] == "flux-2-klein-4b-fp8.safetensors"
    assert graph["13"]["inputs"]["steps"] == 4
    assert graph["15"]["inputs"]["cfg"] == 1.0
    assert_common_flux_chain(graph)


def test_parameter_injection_uses_reference_image_and_no_legacy_lora():
    styles = load_styles(Path("config/styles.json"))
    graph = prepare_workflow(
        "standard", ComfyImageRef("face.png"), styles["cyberpunk"],
        42, "aigc/task", settings(),
    )
    assert graph["5"]["inputs"]["image"] == "face.png"
    assert graph["17"]["inputs"]["noise_seed"] == 42
    assert graph["12"]["inputs"]["filename_prefix"] == "aigc/task"
    assert graph["6"]["inputs"]["text"] == styles["cyberpunk"].positive
    assert "person img" not in graph["6"]["inputs"]["text"]
    assert graph["7"]["inputs"]["text"] == styles["cyberpunk"].negative
    assert graph["4"]["inputs"]["provider"] == "CPU"


def test_prepare_uses_mode_settings_and_subfolder():
    styles = load_styles(Path("config/styles.json"))
    configured = settings().model_copy(
        update={
            "standard_steps": 31,
            "standard_cfg": 6.5,
            "standard_sampler": "euler_ancestral",
            "turbo_steps": 6,
            "turbo_cfg": 1.5,
        }
    )
    standard = prepare_workflow(
        "standard", ComfyImageRef("face.png", subfolder="uploads"), styles["anime"],
        7, "aigc/standard", configured,
    )
    turbo = prepare_workflow(
        "turbo", ComfyImageRef("face.png"), styles["anime"],
        7, "aigc/turbo", configured,
    )
    assert standard["5"]["inputs"]["image"] == "uploads/face.png"
    assert standard["13"]["inputs"]["steps"] == 31
    assert standard["15"]["inputs"]["cfg"] == 6.5
    assert standard["10"]["inputs"]["sampler_name"] == "euler_ancestral"
    assert turbo["13"]["inputs"]["steps"] == 6
    assert turbo["15"]["inputs"]["cfg"] == 1.5
    assert turbo["10"]["inputs"]["sampler_name"] == "euler"


def test_committed_workflow_artifacts_are_reproducible():
    completed = subprocess.run(
        [sys.executable, "scripts/build_workflows.py", "--check"],
        capture_output=True, text=True, check=False,
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
