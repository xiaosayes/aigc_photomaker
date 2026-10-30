import json
from pathlib import Path


def load_models():
    return json.loads(Path("config/models.lock.json").read_text(encoding="utf-8"))["models"]


def test_manifest_has_verified_flux_defaults():
    defaults = {model["key"]: model for model in load_models() if model["required"]}
    assert set(defaults) == {
        "flux2_klein_turbo_fp8",
        "flux2_qwen_text_encoder",
        "flux2_vae",
        "antelopev2_archive",
    }
    assert all(len(model["sha256"]) == 64 and model["size"] > 0 for model in defaults.values())

    expected = {
        "flux2_klein_turbo_fp8": (
            "black-forest-labs/FLUX.2-klein-4b-fp8",
            "5b4408e59397a4a37ccb46afe426d8ed86379441",
            "flux-2-klein-4b-fp8.safetensors",
            4070624520,
            "97ed34fe0567e436200f2faee3939b88f2b5d99f8af2a4dc16532c4245c0ccb6",
            "diffusion_models/",
        ),
        "flux2_qwen_text_encoder": (
            "Comfy-Org/Flux-2-Klein-4B",
            "5b3abca5c0dc1ecd4eaf3340868d09063fc1c965",
            "split_files/text_encoders/qwen_3_4b.safetensors",
            8044982048,
            "6c671498573ac2f7a5501502ccce8d2b08ea6ca2f661c458e708f36b36edfc5a",
            "text_encoders/",
        ),
        "flux2_vae": (
            "Comfy-Org/Flux-2-Klein-4B",
            "5b3abca5c0dc1ecd4eaf3340868d09063fc1c965",
            "split_files/vae/flux2-vae.safetensors",
            336211292,
            "868fe7b343cc8f3a19dbcfcafbc3d5f888802be3f89bd81b65b3621a066ce8f3",
            "vae/",
        ),
    }
    for key, (repo, revision, filename, size, sha256, prefix) in expected.items():
        model = defaults[key]
        assert model["repo"] == repo
        assert model["revision"] == revision
        assert model["filename"] == filename
        assert model["size"] == size
        assert model["sha256"] == sha256
        assert model["relative_path"].startswith(prefix)


def test_base_fp8_is_optional_with_preserved_metadata():
    models = {model["key"]: model for model in load_models()}
    base = models["flux2_klein_base_fp8"]
    assert base["required"] is False
    assert base["repo"] == "black-forest-labs/FLUX.2-klein-base-4b-fp8"
    assert base["revision"] == "103db268c10d4d3921101b46057671f9ac460da6"
    assert base["filename"] == "flux-2-klein-base-4b-fp8.safetensors"
    assert base["size"] == 4089498488
    assert base["sha256"] == "44bab3a86fe98b85d21dd2a4729ebdc3ae51fb8a39f76e457e18c724219e6840"
    assert base["relative_path"] == "diffusion_models/flux-2-klein-base-4b-fp8.safetensors"


def test_sdxl_instantid_and_style_loras_are_optional_for_rollback():
    models = {model["key"]: model for model in load_models()}
    optional = {
        "realvisxl_v5", "realvisxl_v4", "sdxl_base_1_0", "photomaker_v2",
        "flux2_klein_base_fp8",
        "face_detection", "face_recognition", "lcm_lora_sdxl",
        "instantid_ip_adapter", "instantid_controlnet", "cyberpunk_lora",
        "anime_lora", "ink_wash_lora", "oil_painting_lora",
    }
    assert all(models[key]["required"] is False for key in optional)
