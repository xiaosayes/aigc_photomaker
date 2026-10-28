import json
from pathlib import Path


def test_manifest_has_verified_instantid_defaults():
    models = json.loads(Path("config/models.lock.json").read_text(encoding="utf-8"))["models"]
    defaults = {model["key"]: model for model in models if model["required"]}
    assert set(defaults) == {
        "sdxl_base_1_0", "lcm_lora_sdxl", "cyberpunk_lora", "anime_lora",
        "ink_wash_lora", "oil_painting_lora", "instantid_ip_adapter",
        "instantid_controlnet", "antelopev2_archive",
    }
    assert all(len(model["sha256"]) == 64 and model["size"] > 0 for model in defaults.values())

    expected = {
        "sdxl_base_1_0": (
            "stabilityai/stable-diffusion-xl-base-1.0",
            "31e35c80fc4829d14f90153f4c74cd59c90b779f6afe05a74cd6120b893f7e5b",
            "checkpoints/",
        ),
        "instantid_ip_adapter": (
            "InstantX/InstantID",
            "02b3618e36d803784166660520098089a81388e61a93ef8002aa79a5b1c546e1",
            "instantid/",
        ),
        "instantid_controlnet": (
            "InstantX/InstantID",
            "c8127be9f174101ebdafee9964d856b49b634435cf6daa396d3f593cf0bbbb05",
            "controlnet/",
        ),
        "antelopev2_archive": (
            "MonsterMMORPG/tools",
            "8e182f14fc6e80b3bfa375b33eb6cff7ee05d8ef7633e738d1c89021dcf0c5c5",
            "instantid/",
        ),
        "cyberpunk_lora": (
            "Muapi/lah-cyberpunk-sdxl",
            "17c7200db4a88b879e0dc850c13c0281b8fe8f2fffb11df1a8c3d697cbb1b103",
            "loras/",
        ),
        "anime_lora": (
            "ntc-ai/SDXL-LoRA-slider.anime",
            "996f4ac936d328c0db1c8c43abe6e1b1ae92c61416fd5df5d6d3787c3e5ccf7b",
            "loras/",
        ),
        "ink_wash_lora": (
            "ming-yang/sdxl_chinese_ink_lora",
            "79d5a46e204b08179217622703e93342249298318d2bc9a80bc2ae02e9bee229",
            "loras/",
        ),
        "oil_painting_lora": (
            "ntc-ai/SDXL-LoRA-slider.oil-painting",
            "76939f71a95487755ef3eca62f82e7156bca8b682fc054d978d56d8f47bf4368",
            "loras/",
        ),
    }
    for key, (repo, sha256, prefix) in expected.items():
        assert defaults[key]["repo"] == repo
        assert defaults[key]["sha256"] == sha256
        assert defaults[key]["relative_path"].startswith(prefix)


def test_legacy_photomaker_models_are_optional_for_rollback():
    models = json.loads(Path("config/models.lock.json").read_text(encoding="utf-8"))["models"]
    legacy = {model["key"]: model for model in models}
    for key in ("realvisxl_v5", "photomaker_v2", "face_detection", "face_recognition"):
        assert legacy[key]["required"] is False
