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
    graph = {
        "1": node("CheckpointLoaderSimple", ckpt_name="sd_xl_base_1.0.safetensors"),
        "2": node("InstantIDModelLoader", instantid_file="ip-adapter.bin"),
        "3": node("InstantIDFaceAnalysis", provider="CPU"),
        "4": node(
            "ControlNetLoader",
            control_net_name="instantid/diffusion_pytorch_model.safetensors",
        ),
        "5": node("LoadImage", image="warmup-portrait.png"),
        "6": node(
            "CLIPTextEncode",
            clip=["1", 1],
            text="solo, one person, single subject, portrait of a person",
        ),
        "7": node(
            "CLIPTextEncode",
            clip=["1", 1],
            text="low quality, deformed face, multiple people, extra fingers, bad anatomy",
        ),
        "8": node("EmptyLatentImage", width=1024, height=1024, batch_size=1),
        "9": node(
            "KSampler",
            model=["15", 0],
            positive=["15", 1],
            negative=["15", 2],
            latent_image=["8", 0],
            seed=0,
            steps=25,
            cfg=4.0,
            sampler_name="dpmpp_2m",
            scheduler="karras",
            denoise=1.0,
        ),
        "11": node("VAEDecode", samples=["9", 0], vae=["1", 2]),
        "12": node("SaveImage", images=["14", 0], filename_prefix="aigc-photomaker/output"),
        "13": node(
            "LoraLoaderModelOnly",
            model=["1", 0],
            lora_name="lah-cyberpunk-sdxl.safetensors",
            strength_model=1.0,
        ),
        "14": node(
            "InstantIDSingleFaceOutputGuard",
            images=["11", 0],
            faceanalysis=["3", 0],
        ),
        "15": node(
            "ApplyInstantID",
            instantid=["2", 0],
            insightface=["3", 0],
            control_net=["4", 0],
            image=["16", 0],
            model=["13", 0],
            positive=["6", 0],
            negative=["7", 0],
            weight=0.8,
            start_at=0.0,
            end_at=1.0,
        ),
        "16": node(
            "InstantIDSingleFaceInputGuard",
            images=["5", 0],
            faceanalysis=["3", 0],
        ),
    }
    if mode == "turbo":
        graph["10"] = node(
            "LoraLoaderModelOnly",
            model=["13", 0],
            lora_name="lcm-lora-sdxl.safetensors",
            strength_model=1.0,
        )
        graph["15"]["inputs"]["model"] = ["10", 0]
        graph["9"]["inputs"].update(
            steps=8, cfg=1.8, sampler_name="lcm", scheduler="normal"
        )
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
    graph["1"]["inputs"]["ckpt_name"] = settings.checkpoint_filename
    graph["3"]["inputs"]["provider"] = settings.face_provider
    graph["5"]["inputs"]["image"] = (
        f"{image.subfolder}/{image.name}" if image.subfolder else image.name
    )
    graph["6"]["inputs"]["text"] = style.positive
    graph["7"]["inputs"]["text"] = style.negative
    graph["13"]["inputs"].update(
        lora_name=style.lora_filename,
        strength_model=style.lora_strength,
    )
    graph["15"]["inputs"]["weight"] = settings.instantid_weight
    graph["9"]["inputs"]["seed"] = seed
    if mode == "standard":
        graph["9"]["inputs"].update(
            steps=settings.standard_steps,
            cfg=settings.standard_cfg,
            sampler_name=settings.standard_sampler,
            scheduler=settings.standard_scheduler,
        )
    else:
        graph["9"]["inputs"].update(steps=settings.turbo_steps, cfg=settings.turbo_cfg)
    graph["12"]["inputs"]["filename_prefix"] = output_prefix
    return graph
