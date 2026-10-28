# Server-pending: diagnostic helper tests; no real ComfyUI or GPU is required.
import importlib.util
from pathlib import Path


path = Path(__file__).parents[1] / "scripts" / "inspect_latest_task.py"
spec = importlib.util.spec_from_file_location("inspect_latest_task", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def graph(mode="standard"):
    result = {
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "cyberpunk style"}},
        "9": {"class_type": "KSampler", "inputs": {"model": ["15", 0]}},
        "15": {
            "class_type": "ApplyInstantID",
            "inputs": {
                "model": ["13", 0], "instantid": ["2", 0],
                "insightface": ["3", 0], "control_net": ["4", 0],
                "image": ["16", 0],
            },
        },
        "16": {
            "class_type": "InstantIDSingleFaceInputGuard",
            "inputs": {"images": ["5", 0], "faceanalysis": ["3", 0]},
        },
        "12": {"class_type": "SaveImage", "inputs": {"images": ["14", 0]}},
        "13": {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"lora_name": "lah-cyberpunk-sdxl.safetensors"},
        },
        "14": {"class_type": "InstantIDSingleFaceOutputGuard", "inputs": {}},
    }
    if mode == "turbo":
        result["15"]["inputs"]["model"] = ["10", 0]
        result["10"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["13", 0]}}
    return result


def test_prompt_graph_uses_real_comfyui_history_layout():
    expected = graph()
    entry = {"prompt": [1, "prompt-id", expected, {}, ["12"]]}
    assert module.prompt_graph(entry) is expected


def test_checks_confirm_standard_and_turbo_style_lora_paths():
    for mode in ("standard", "turbo"):
        checks = module.workflow_checks(graph(mode))
        assert checks["style_lora_present"] is True
        assert checks["style_lora_applies_to_sampler"] is True
        assert checks["output_guard_present"] is True
        assert checks["instantid_identity_chain_present"] is True


def test_checks_reject_missing_style_path_and_guard():
    broken = graph()
    broken["9"]["inputs"]["model"] = ["13", 0]
    broken["12"]["inputs"]["images"] = ["11", 0]
    broken["15"]["inputs"]["control_net"] = None
    checks = module.workflow_checks(broken)
    assert checks["style_lora_applies_to_sampler"] is False
    assert checks["output_guard_present"] is False
    assert checks["instantid_identity_chain_present"] is False
