# Server-pending: FLUX.2 diagnostic helper tests; no real ComfyUI or GPU is required.
import importlib.util
from pathlib import Path


path = Path(__file__).parents[1] / "scripts" / "inspect_latest_task.py"
spec = importlib.util.spec_from_file_location("inspect_latest_task", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def graph(mode="standard"):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": f"{mode}.safetensors"}},
        "9": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["17", 0], "guider": ["15", 0], "sampler": ["10", 0],
                "sigmas": ["13", 0], "latent_image": ["8", 0],
            },
        },
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "12": {"class_type": "SaveImage", "inputs": {"images": ["14", 0]}},
        "13": {"class_type": "Flux2Scheduler", "inputs": {"steps": 20 if mode == "standard" else 4}},
        "14": {
            "class_type": "FluxSingleFaceOutputGuard",
            "inputs": {"images": ["11", 0], "faceanalysis": ["4", 0]},
        },
        "15": {
            "class_type": "CFGGuider",
            "inputs": {"model": ["1", 0], "positive": ["20", 0], "negative": ["21", 0]},
        },
        "16": {
            "class_type": "FluxSingleFaceInputGuard",
            "inputs": {"images": ["18", 0], "faceanalysis": ["4", 0]},
        },
        "17": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}},
        "19": {"class_type": "VAEEncode", "inputs": {"pixels": ["16", 0], "vae": ["3", 0]}},
        "20": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["6", 0], "latent": ["19", 0]}},
        "21": {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["7", 0], "latent": ["19", 0]}},
    }


def test_prompt_graph_uses_real_comfyui_history_layout():
    expected = graph()
    entry = {"prompt": [1, "prompt-id", expected, {}, ["12"]]}
    assert module.prompt_graph(entry) is expected


def test_checks_confirm_standard_and_turbo_flux_chains():
    for mode in ("standard", "turbo"):
        checks = module.workflow_checks(graph(mode))
        assert checks["flux2_reference_chain_present"] is True
        assert checks["flux2_samplers_present"] is True
        assert checks["output_guard_present"] is True


def test_checks_reject_incomplete_flux_chain():
    broken = graph()
    broken["15"]["inputs"]["positive"] = ["6", 0]
    broken["9"]["inputs"]["guider"] = ["1", 0]
    broken["12"]["inputs"]["images"] = ["11", 0]
    checks = module.workflow_checks(broken)
    assert checks["flux2_reference_chain_present"] is False
    assert checks["flux2_samplers_present"] is False
    assert checks["output_guard_present"] is False


def test_checks_reject_missing_vae_sampler_and_guard_inputs():
    broken = graph()
    broken["19"]["inputs"]["vae"] = ["2", 0]
    broken["20"]["inputs"]["conditioning"] = ["5", 0]
    broken["16"]["inputs"]["images"] = ["5", 0]
    broken["9"]["inputs"]["sampler"] = ["1", 0]
    broken["9"]["inputs"]["latent_image"] = ["1", 0]
    broken["14"]["inputs"]["images"] = []
    checks = module.workflow_checks(broken)
    assert checks["flux2_reference_chain_present"] is False
    assert checks["flux2_samplers_present"] is False
    assert checks["output_guard_present"] is False
