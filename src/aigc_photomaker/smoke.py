"""Real Standard/Turbo GPU smoke checks and manual recovery drill guidance."""

from __future__ import annotations

import argparse
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .acceptance import (
    GpuMonitor,
    check_project_gpu_processes,
    contains_oom,
    detect_oom_since,
    error_details,
    expect_task_error,
    log_offsets,
    run_generation,
)
from .benchmark import collect_evidence
from .config import PROJECT_ROOT

SMOKE_BASE_SEED = 12_000

RECOVERY_INSTRUCTIONS = r"""
Manual recovery drills (run separately; they intentionally restart real services):
1. Start a generation, then record its task and prompt IDs without printing the API key:
   sqlite3 data/tasks.db "SELECT id AS task_id,prompt_id,status FROM tasks ORDER BY created_at DESC LIMIT 1;"

Worker recovery (full managed restart; do not manually kill or start individual PIDs):
   bash scripts/stop.sh && bash scripts/start.sh
   # Query the same task until terminal and repeat the SQL above. The prompt_id must
   # remain unchanged: an existing prompt_id must not be resubmitted to ComfyUI.

ComfyUI recovery (full managed restart of API, Worker, and ComfyUI):
   bash scripts/stop.sh && bash scripts/start.sh
   # Verify the same prompt_id is queried/recovered and no duplicate prompt is submitted.

Cancellation and failed-task query (replace TASK_ID only):
   curl --fail-with-body -X POST -H "X-API-Key: ${AIGC_API_KEY}" __API_URL__/api/v1/tasks/TASK_ID/cancel
   curl --fail-with-body -H "X-API-Key: ${AIGC_API_KEY}" __API_URL__/api/v1/tasks/TASK_ID
   # Confirm cancellation becomes terminal and a failed task remains queryable with
   # its sanitized structured error. Never paste the API key into a report.
""".strip()


@dataclass
class SmokeReport:
    passed: bool
    checks: list[dict[str, Any]]
    gpu: dict[str, Any]
    evidence: dict[str, Any]
    json_path: Path
    markdown_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "passed": self.passed,
            "checks": self.checks,
            "gpu": self.gpu,
            "evidence": self.evidence,
            "result_contract": {
                "width": 1024,
                "height": 1024,
                "format": "PNG",
                "mode": "RGB",
                "full_decode_required": True,
            },
            "recovery": {
                "manual_required": True,
                "covers": [
                    "worker restart with existing prompt_id",
                    "ComfyUI restart without duplicate submission",
                    "cancellation",
                    "failed task query",
                ],
            },
        }


def _png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _solid_png() -> bytes:
    return _png_bytes(Image.new("RGB", (256, 256), "navy"))


def _duplicate_face_png(input_path: str | Path) -> bytes:
    with Image.open(input_path) as source:
        source.load()
        face = source.convert("RGB").resize((512, 512))
    duplicate = Image.new("RGB", (1024, 1024))
    duplicate.paste(face, (0, 0))
    duplicate.paste(face, (512, 0))
    return _png_bytes(duplicate)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _write_report(report: SmokeReport, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = f"smoke-{_timestamp()}"
    report.json_path = report_dir / f"{stem}.json"
    report.markdown_path = report_dir / f"{stem}.md"
    payload = report.to_dict()
    report.json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# GPU smoke acceptance",
        "",
        f"**Result: {'PASS' if report.passed else 'FAIL'}**",
        "",
        "| Check | Passed | Detail |",
        "|---|---|---|",
    ]
    for check in report.checks:
        detail = json.dumps(
            {key: value for key, value in check.items() if key not in {"name", "passed"}},
            ensure_ascii=False,
            sort_keys=True,
        )
        lines.append(f"| {check['name']} | {check['passed']} | {detail} |")
    lines.extend(
        [
            "",
            "## GPU and version evidence",
            "",
            "```json",
            json.dumps({"gpu": report.gpu, "evidence": report.evidence}, indent=2),
            "```",
            "",
            "Worker/ComfyUI recovery, cancellation, and failed-task drills remain explicit manual server steps.",
            "No API key or authorization header is included in this report.",
            "",
        ]
    )
    report.markdown_path.write_text("\n".join(lines), encoding="utf-8")


def run_smoke(
    api_url: str,
    api_key: str,
    input_path: str | Path,
    *,
    style: str = "cyberpunk",
    report_dir: str | Path = PROJECT_ROOT / "reports",
) -> SmokeReport:
    checks: list[dict[str, Any]] = []
    offsets = log_offsets()
    with GpuMonitor() as monitor:
        for index, mode in enumerate(("standard", "turbo")):
            seed = SMOKE_BASE_SEED + index
            try:
                result = run_generation(
                    api_url, api_key, input_path, mode, seed, style
                )
                checks.append(
                    {
                        "name": f"{mode}_1024_rgb_png",
                        "passed": True,
                        "task_id": result.task_id,
                        "seed": seed,
                        "client_total_seconds": result.client_total_seconds,
                        "server_stages": result.server_stages,
                        "image": result.image,
                    }
                )
            except Exception as exc:
                detail = error_details(exc, api_key)
                checks.append(
                    {
                        "name": f"{mode}_1024_rgb_png",
                        "passed": False,
                        "seed": seed,
                        "error": detail,
                    }
                )
        negative_cases = (
            (
                "solid_color_no_face",
                _solid_png(),
                "solid-color.png",
                "NO_FACE_DETECTED",
                SMOKE_BASE_SEED + 2,
            ),
            (
                "side_by_side_duplicate_faces",
                _duplicate_face_png(input_path),
                "duplicate-faces.png",
                "MULTIPLE_FACES_DETECTED",
                SMOKE_BASE_SEED + 3,
            ),
        )
        for name, content, filename, expected_code, seed in negative_cases:
            try:
                detail = expect_task_error(
                    api_url,
                    api_key,
                    content,
                    filename,
                    style=style,
                    mode="standard",
                    seed=seed,
                    expected_code=expected_code,
                )
                checks.append({"name": name, "passed": True, **detail})
            except Exception as exc:
                checks.append(
                    {
                        "name": name,
                        "passed": False,
                        "expected_code": expected_code,
                        "error": error_details(exc, api_key),
                    }
                )
    process_check = check_project_gpu_processes()
    checks.append({"name": "project_processes_only_on_physical_gpu1", **process_check})
    oom = detect_oom_since(offsets) or any(
        contains_oom(check.get("error", {})) for check in checks
    )
    checks.append({"name": "no_oom", "passed": not oom, "oom_detected": oom})
    gpu = {
        "physical_gpu_id": 1,
        "peak_vram_mib": monitor.peak_vram_mib,
        "peak_utilization_percent": monitor.peak_utilization_percent,
        "sample_count": len(monitor.samples),
        "sampling_errors": monitor.errors,
        "process_isolation": process_check,
    }
    passed = all(check["passed"] for check in checks) and monitor.peak_vram_mib is not None
    report = SmokeReport(passed, checks, gpu, collect_evidence(), Path(), Path())
    _write_report(report, Path(report_dir))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "assets" / "warmup-portrait.png"))
    parser.add_argument("--style", default="cyberpunk")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api_key = os.environ.get("AIGC_API_KEY", "")
    if not api_key:
        raise SystemExit("AIGC_API_KEY is required (its value is never logged)")
    try:
        report = run_smoke(
            args.api_url,
            api_key,
            args.input,
            style=args.style,
            report_dir=args.report_dir,
        )
    except Exception as exc:
        print(f"Smoke checks could not produce a report: {type(exc).__name__}")
        return 2
    print(f"JSON report: {report.json_path}")
    print(f"Markdown report: {report.markdown_path}")
    print(RECOVERY_INSTRUCTIONS.replace("__API_URL__", args.api_url.rstrip("/")))
    print(f"GPU smoke: {'PASS' if report.passed else 'FAIL'}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
