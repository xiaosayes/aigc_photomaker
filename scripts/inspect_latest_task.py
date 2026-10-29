#!/usr/bin/env python3
"""Print the actual ComfyUI graph for the latest or selected task without secrets."""

import argparse
import json
import sqlite3
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aigc_photomaker.config import get_settings  # noqa: E402


def prompt_graph(entry):
    """Extract [number, prompt_id, graph, extra_data, outputs] safely."""
    prompt = entry.get("prompt")
    if not isinstance(prompt, (list, tuple)) or len(prompt) < 3 or not isinstance(prompt[2], dict):
        raise ValueError("ComfyUI history has an invalid prompt layout")
    return prompt[2]


def workflow_checks(graph):
    reference_chain_present = (
        graph.get("19", {}).get("class_type") == "VAEEncode"
        and graph.get("19", {}).get("inputs", {}).get("pixels") == ["16", 0]
        and graph.get("19", {}).get("inputs", {}).get("vae") == ["3", 0]
        and graph.get("20", {}).get("class_type") == "ReferenceLatent"
        and graph.get("20", {}).get("inputs", {}).get("conditioning") == ["6", 0]
        and graph.get("20", {}).get("inputs", {}).get("latent") == ["19", 0]
        and graph.get("21", {}).get("class_type") == "ReferenceLatent"
        and graph.get("21", {}).get("inputs", {}).get("conditioning") == ["7", 0]
        and graph.get("21", {}).get("inputs", {}).get("latent") == ["19", 0]
        and graph.get("15", {}).get("inputs", {}).get("positive") == ["20", 0]
        and graph.get("15", {}).get("inputs", {}).get("negative") == ["21", 0]
        and graph.get("16", {}).get("class_type") == "FluxSingleFaceInputGuard"
        and graph.get("16", {}).get("inputs", {}).get("images") == ["18", 0]
        and graph.get("16", {}).get("inputs", {}).get("faceanalysis") == ["4", 0]
    )
    samplers_present = (
        graph.get("9", {}).get("class_type") == "SamplerCustomAdvanced"
        and graph.get("9", {}).get("inputs", {}).get("guider") == ["15", 0]
        and graph.get("9", {}).get("inputs", {}).get("sampler") == ["10", 0]
        and graph.get("9", {}).get("inputs", {}).get("sigmas") == ["13", 0]
        and graph.get("9", {}).get("inputs", {}).get("noise") == ["17", 0]
        and graph.get("9", {}).get("inputs", {}).get("latent_image") == ["8", 0]
        and graph.get("10", {}).get("class_type") == "KSamplerSelect"
        and graph.get("13", {}).get("class_type") == "Flux2Scheduler"
        and graph.get("17", {}).get("class_type") == "RandomNoise"
    )
    return {
        "flux2_reference_chain_present": reference_chain_present,
        "flux2_samplers_present": samplers_present,
        "output_guard_present": (
            graph.get("14", {}).get("class_type") == "FluxSingleFaceOutputGuard"
            and graph.get("14", {}).get("inputs", {}).get("images") == ["11", 0]
            and graph.get("14", {}).get("inputs", {}).get("faceanalysis") == ["4", 0]
            and graph.get("12", {}).get("inputs", {}).get("images") == ["14", 0]
        ),
    }


def load_task(database, task_id):
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        if task_id:
            row = connection.execute(
                """SELECT id, style, mode, seed, status, prompt_id
                   FROM tasks WHERE id = ?""",
                (task_id,),
            ).fetchone()
        else:
            row = connection.execute(
                """SELECT id, style, mode, seed, status, prompt_id
                   FROM tasks ORDER BY created_at DESC LIMIT 1"""
            ).fetchone()
    finally:
        connection.close()
    return dict(row) if row is not None else None


def fetch_history(prompt_id):
    settings = get_settings()
    # Never route loopback diagnostics through HTTP proxy environment variables.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"{settings.comfy_url}/history/{prompt_id}", timeout=10) as response:
        history = json.load(response)
    if prompt_id not in history or not isinstance(history[prompt_id], dict):
        raise ValueError("ComfyUI history does not contain the requested prompt")
    return history[prompt_id]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", help="Inspect this task instead of the latest task")
    args = parser.parse_args(argv)

    database = PROJECT_ROOT / "data" / "tasks.db"
    if not database.is_file():
        parser.error(f"Task database is missing: {database}")
    task = load_task(database, args.task_id)
    if task is None:
        parser.error("Task not found")
    print("TASK:")
    print(json.dumps(task, ensure_ascii=False, indent=2))

    prompt_id = task.get("prompt_id")
    if not prompt_id:
        parser.error("The selected task has no prompt_id; it may have failed before submission")
    graph = prompt_graph(fetch_history(prompt_id))

    print("\nWORKFLOW CHECKS:")
    print(json.dumps(workflow_checks(graph), ensure_ascii=False, indent=2))
    for node_id in tuple(str(index) for index in range(1, 22)):
        if node_id in graph:
            print(f"\nNODE {node_id}:")
            print(json.dumps(graph[node_id], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Inspection failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
