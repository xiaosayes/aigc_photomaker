from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .styles import StylePreset

if TYPE_CHECKING:
    from .config import Settings


@dataclass(frozen=True)
class ComfyImageRef:
    name: str
    subfolder: str = ""
    type: str = "input"


def node(class_type, **inputs):
    return {"class_type": class_type, "inputs": inputs, "_meta": {"title": class_type}}


def build_workflow(mode: str) -> dict:
    if mode not in {"standard", "turbo"}:
        raise ValueError("INVALID_MODE")
    standard = mode == "standard"
    graph = {
        "1": node(
            "UNETLoader",
            unet_name=(
                "flux-2-klein-4b-fp8.safetensors"
            ),
            weight_dtype="default",
        ),
        "2": node("CLIPLoader", clip_name="qwen_3_4b.safetensors", type="flux2", device="default"),
        "3": node("VAELoader", vae_name="flux2-vae.safetensors"),
        "4": node("InstantIDFaceAnalysis", provider="CPU"),
        "5": node("LoadImage", image="warmup-portrait.png"),
        "6": node("CLIPTextEncode", clip=["2", 0], text="solo, one person, portrait of a person"),
        "7": node("CLIPTextEncode", clip=["2", 0], text="low quality, multiple people, extra face"),
        "8": node("EmptyFlux2LatentImage", width=1024, height=1024, batch_size=1),
        "9": node(
            "SamplerCustomAdvanced",
            noise=["17", 0],
            guider=["15", 0],
            sampler=["10", 0],
            sigmas=["13", 0],
            latent_image=["8", 0],
        ),
        "10": node("KSamplerSelect", sampler_name="euler"),
        "11": node("VAEDecode", samples=["9", 0], vae=["3", 0]),
        "12": node("SaveImage", images=["14", 0], filename_prefix="aigc-photomaker/output"),
        "13": node("Flux2Scheduler", steps=8 if standard else 4, width=1024, height=1024),
        "14": node("FluxSingleFaceOutputGuard", images=["11", 0], faceanalysis=["4", 0]),
        "15": node(
            "CFGGuider",
            model=["1", 0],
            positive=["20", 0],
            negative=["21", 0],
            cfg=1.0,
        ),
        "16": node("FluxSingleFaceInputGuard", images=["18", 0], faceanalysis=["4", 0]),
        "17": node("RandomNoise", noise_seed=0),
        "18": node("ImageScale", image=["5", 0], upscale_method="lanczos", width=1024, height=1024, crop="center"),
        "19": node("VAEEncode", pixels=["16", 0], vae=["3", 0]),
        "20": node("ReferenceLatent", conditioning=["6", 0], latent=["19", 0]),
        "21": node("ReferenceLatent", conditioning=["7", 0], latent=["19", 0]),
    }
    return graph


def prepare_workflow(
    mode: str,
    image: ComfyImageRef,
    style: StylePreset,
    seed: int,
    output_prefix: str,
    settings: "Settings",
) -> dict:
    if image.type != "input":
        raise ValueError("INVALID_IMAGE_TYPE")
    graph = deepcopy(build_workflow(mode))
    graph["1"]["inputs"]["unet_name"] = (
        settings.flux_standard_unet_filename if mode == "standard" else settings.flux_turbo_unet_filename
    )
    graph["2"]["inputs"]["clip_name"] = settings.flux_text_encoder_filename
    graph["3"]["inputs"]["vae_name"] = settings.flux_vae_filename
    graph["4"]["inputs"]["provider"] = settings.face_provider
    graph["5"]["inputs"]["image"] = (
        f"{image.subfolder}/{image.name}" if image.subfolder else image.name
    )
    graph["6"]["inputs"]["text"] = style.positive
    graph["7"]["inputs"]["text"] = style.negative
    graph["17"]["inputs"]["noise_seed"] = seed
    if mode == "standard":
        graph["13"]["inputs"]["steps"] = settings.standard_steps
        graph["15"]["inputs"]["cfg"] = settings.standard_cfg
        graph["10"]["inputs"]["sampler_name"] = settings.standard_sampler
    else:
        graph["13"]["inputs"]["steps"] = settings.turbo_steps
        graph["15"]["inputs"]["cfg"] = settings.turbo_cfg
        graph["10"]["inputs"]["sampler_name"] = "euler"
    graph["12"]["inputs"]["filename_prefix"] = output_prefix
    return graph
