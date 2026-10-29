#!/usr/bin/env python3
"""Build deterministic ComfyUI API and UI workflow artifacts without ComfyUI."""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aigc_photomaker.workflow import build_workflow  # noqa: E402


OUTPUT_TYPES = {
    "CheckpointLoaderSimple": ["MODEL", "CLIP", "VAE"],
    "UNETLoader": ["MODEL"],
    "CLIPLoader": ["CLIP"],
    "VAELoader": ["VAE"],
    "InstantIDFaceAnalysis": ["FACEANALYSIS"],
    "FluxSingleFaceInputGuard": ["IMAGE"],
    "FluxSingleFaceOutputGuard": ["IMAGE"],
    "EmptyFlux2LatentImage": ["LATENT"],
    "SamplerCustomAdvanced": ["LATENT", "LATENT"],
    "KSamplerSelect": ["SAMPLER"],
    "Flux2Scheduler": ["SIGMAS"],
    "CFGGuider": ["GUIDER"],
    "RandomNoise": ["NOISE"],
    "ImageScale": ["IMAGE"],
    "VAEEncode": ["LATENT"],
    "ReferenceLatent": ["CONDITIONING"],
    "LoadImage": ["IMAGE", "MASK"],
    "CLIPTextEncode": ["CONDITIONING"],
    "EmptyLatentImage": ["LATENT"],
    "KSampler": ["LATENT"],
    "LoraLoaderModelOnly": ["MODEL"],
    "VAEDecode": ["IMAGE"],
    "SaveImage": [],
}

POSITIONS = {
    1: [40, 40],
    2: [40, 250],
    3: [40, 450],
    4: [400, 40],
    5: [400, 280],
    6: [760, 250],
    7: [760, 40],
    8: [760, 520],
    9: [1160, 200],
    10: [1080, 720],
    11: [1540, 200],
    12: [1900, 200],
    13: [760, 720],
    14: [1540, 520],
    15: [1160, 500],
    16: [400, 520],
    17: [1160, 40],
    18: [400, 800],
    19: [800, 800],
    20: [800, 40],
    21: [800, 440],
}


def is_link(value):
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def widgets_for(class_type, inputs):
    if class_type == "KSampler":
        values = [
            inputs["seed"],
            "fixed",
            inputs["steps"],
            inputs["cfg"],
            inputs["sampler_name"],
            inputs["scheduler"],
            inputs["denoise"],
        ]
    elif class_type == "LoadImage":
        values = [inputs["image"], "image"]
    else:
        values = [value for value in inputs.values() if not is_link(value)]
    return values


def topological_order(graph):
    remaining = set(graph)
    order = []
    while remaining:
        ready = sorted(
            (
                node_id
                for node_id in remaining
                if all(
                    value[0] not in remaining
                    for value in graph[node_id]["inputs"].values()
                    if is_link(value)
                )
            ),
            key=int,
        )
        if not ready:
            raise ValueError("WORKFLOW_CYCLE")
        order.extend(ready)
        remaining.difference_update(ready)
    return order


def to_ui(graph):
    links = []
    incoming = {}
    outgoing = {node_id: {} for node_id in graph}
    for target_id, api_node in graph.items():
        target_slot = 0
        for input_name, value in api_node["inputs"].items():
            if not is_link(value):
                continue
            origin_id, origin_slot = value
            link_id = len(links) + 1
            origin_type = OUTPUT_TYPES[graph[origin_id]["class_type"]][origin_slot]
            links.append(
                [link_id, int(origin_id), origin_slot, int(target_id), target_slot, origin_type]
            )
            incoming.setdefault(target_id, []).append((input_name, origin_type, link_id))
            outgoing[origin_id].setdefault(origin_slot, []).append(link_id)
            target_slot += 1

    dependency_order = {
        node_id: index for index, node_id in enumerate(topological_order(graph))
    }
    nodes = []
    for node_id in sorted(graph, key=int):
        api_node = graph[node_id]
        class_type = api_node["class_type"]
        output_types = OUTPUT_TYPES[class_type]
        outputs = [
            {
                "name": output_type,
                "type": output_type,
                "links": outgoing[node_id].get(slot, []),
                "slot_index": slot,
            }
            for slot, output_type in enumerate(output_types)
        ]
        nodes.append(
            {
                "id": int(node_id),
                "type": class_type,
                "pos": POSITIONS[int(node_id)],
                "size": [320, 200],
                "flags": {},
                "order": dependency_order[node_id],
                "mode": 0,
                "inputs": [
                    {"name": name, "type": input_type, "link": link_id}
                    for name, input_type, link_id in incoming.get(node_id, [])
                ],
                "outputs": outputs,
                "properties": {"Node name for S&R": class_type},
                "widgets_values": widgets_for(class_type, api_node["inputs"]),
            }
        )

    return {
        "last_node_id": max(map(int, graph)),
        "last_link_id": len(links),
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }


def serialize(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def expected_files():
    files = {}
    for mode in ("standard", "turbo"):
        graph = build_workflow(mode)
        files[PROJECT_ROOT / "workflows" / f"{mode}_api.json"] = serialize(graph)
        files[PROJECT_ROOT / "workflows" / f"{mode}_ui.json"] = serialize(to_ui(graph))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="verify committed JSON without modifying it"
    )
    args = parser.parse_args()
    files = expected_files()
    if args.check:
        mismatches = [
            str(path.relative_to(PROJECT_ROOT))
            for path, content in files.items()
            if not path.is_file() or path.read_text(encoding="utf-8") != content
        ]
        if mismatches:
            print("Workflow JSON is out of date: " + ", ".join(mismatches), file=sys.stderr)
            return 1
        print("Workflow JSON is reproducible and up to date (4 files).")
        return 0
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode("utf-8"))
        print(f"Wrote {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
