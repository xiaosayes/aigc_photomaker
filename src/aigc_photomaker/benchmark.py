"""Real HTTP/GPU benchmark reports.  Run only on the provisioned GPU server."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .acceptance import (
    GpuMonitor,
    contains_oom,
    detect_oom_since,
    error_details,
    log_offsets,
    run_generation,
)
from .config import PROJECT_ROOT, get_settings
from .task_service import MAX_SEED

STANDARD_MINIMUM_SUCCESSES = 20
TURBO_MINIMUM_SUCCESSES = 10
P95_LIMIT_SECONDS = 30.0
DEFAULT_BASE_SEED = 11_000


def percentile(values: Iterable[float], percent: float) -> float | None:
    """Nearest-rank percentile, independently testable without API/GPU dependencies."""
    if not 0 < percent <= 100:
        raise ValueError("percent must be in (0, 100]")
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    rank = max(1, math.ceil(percent / 100 * len(ordered)))
    return ordered[rank - 1]


def _summary(
    durations: list[float],
    failures: int,
    peak_vram_mib: float | None,
    minimum_successes: int,
    *,
    oom: bool,
    enforce_p95: bool,
) -> dict[str, Any]:
    successes = len(durations)
    total = successes + failures
    p50 = percentile(durations, 50)
    p95 = percentile(durations, 95)
    reasons = []
    if successes < minimum_successes:
        reasons.append(f"requires_at_least_{minimum_successes}_successful_samples")
    if failures:
        reasons.append("failures_present")
    if enforce_p95 and (p95 is None or p95 > P95_LIMIT_SECONDS):
        reasons.append("p95_exceeds_30_seconds")
    if oom:
        reasons.append("oom_detected")
    if peak_vram_mib is None:
        reasons.append("physical_gpu1_vram_unavailable")
    return {
        "passed": not reasons,
        "successful_samples": successes,
        "required_successful_samples": minimum_successes,
        "failures": failures,
        "failure_rate": failures / total if total else 0.0,
        "p50_seconds": p50,
        "p95_seconds": p95,
        "min_seconds": min(durations) if durations else None,
        "max_seconds": max(durations) if durations else None,
        "p95_limit_seconds": P95_LIMIT_SECONDS,
        "p95_within_30_seconds": p95 is not None and p95 <= P95_LIMIT_SECONDS,
        "p95_gate_enforced": enforce_p95,
        "peak_vram_mib_physical_gpu1": peak_vram_mib,
        "oom_detected": oom,
        "failure_reasons": reasons,
    }


def evaluate_standard(
    durations: Iterable[float],
    failures: int,
    peak_vram_mib: float | None,
    oom: bool = False,
) -> dict[str, Any]:
    return _summary(
        list(durations),
        failures,
        peak_vram_mib,
        STANDARD_MINIMUM_SUCCESSES,
        oom=oom,
        enforce_p95=True,
    )


def evaluate_turbo(
    durations: Iterable[float],
    failures: int,
    peak_vram_mib: float | None,
    oom: bool = False,
) -> dict[str, Any]:
    # Turbo has the same reporting fields, but its acceptance contract is ten
    # successful samples, zero failures, no OOM, and real GPU1 VRAM evidence.
    return _summary(
        list(durations),
        failures,
        peak_vram_mib,
        TURBO_MINIMUM_SUCCESSES,
        oom=oom,
        enforce_p95=False,
    )


def evaluate_alternate(
    samples: list[dict[str, Any]], failures: int, oom: bool
) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for mode in ("standard", "turbo"):
        durations = [
            float(sample["client_total_seconds"])
            for sample in samples
            if sample.get("mode") == mode
        ]
        modes[mode] = {
            "successful_samples": len(durations),
            "p50_seconds": percentile(durations, 50),
            "p95_seconds": percentile(durations, 95),
            "min_seconds": min(durations) if durations else None,
            "max_seconds": max(durations) if durations else None,
        }
    passed = (
        failures == 0
        and not oom
        and all(modes[mode]["successful_samples"] > 0 for mode in modes)
    )
    return {
        "passed": passed,
        "acceptance_scope": "alternating-only",
        "single_mode_standard_gate_applied": False,
        "failures": failures,
        "failure_rate": failures / (len(samples) + failures) if samples or failures else 0.0,
        "oom_detected": oom,
        "modes": modes,
    }


def seed_sequence(base_seed: int, runs: int) -> list[int]:
    if not 0 <= base_seed <= MAX_SEED:
        raise ValueError(f"base_seed must be between 0 and {MAX_SEED}")
    return [(base_seed + index) % (MAX_SEED + 1) for index in range(runs)]


def warmup_seed(base_seed: int) -> int:
    if not 0 <= base_seed <= MAX_SEED:
        raise ValueError(f"base_seed must be between 0 and {MAX_SEED}")
    return (base_seed - 1) % (MAX_SEED + 1)


@dataclass
class BenchmarkReport:
    mode: str
    summary: dict[str, Any]
    samples: list[dict[str, Any]]
    failures_detail: list[dict[str, Any]]
    warmup: dict[str, Any]
    evidence: dict[str, Any]
    gpu: dict[str, Any]
    json_path: Path
    markdown_path: Path
    transitions: list[dict[str, Any]] | None = None

    @property
    def passed(self) -> bool:
        return bool(self.summary["passed"])

    def to_dict(self) -> dict[str, Any]:
        report = {
            "schema_version": 1,
            "benchmark": self.mode,
            "summary": self.summary,
            "warmup": self.warmup,
            "samples": self.samples,
            "errors": self.failures_detail,
            "gpu": self.gpu,
            "evidence": self.evidence,
            "measurement": {
                "clock": "time.perf_counter",
                "scope": "multipart creation through result download and full Pillow decode/display-readiness",
                "poll_interval_ms": 500,
                "gpu_sample_interval_ms": 100,
                "gpu_id": 1,
            },
        }
        if self.transitions is not None:
            report["mode_transitions"] = self.transitions
        return report


def collect_evidence() -> dict[str, Any]:
    settings = get_settings()
    versions = {}
    for line in (PROJECT_ROOT / "config" / "versions.lock.env").read_text(
        encoding="utf-8"
    ).splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            versions[key] = value
    manifest = json.loads(
        (PROJECT_ROOT / "config" / "models.lock.json").read_text(encoding="utf-8")
    )
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except Exception:
        commit = "unavailable"
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        package_version = tomllib.load(stream)["project"]["version"]
    return {
        "application_version": package_version,
        "version_commit": commit,
        "versions": versions,
        "models": [
            {
                "key": model["key"],
                "filename": model["filename"],
                "revision": model["revision"],
                "sha256": model["sha256"],
            }
            for model in manifest["models"]
            if model.get("required", False)
        ],
        "parameters": {
            "width": 1024,
            "height": 1024,
            "image_format": "PNG",
            "image_mode": "RGB",
            "modes": {
                "standard": {
                    "steps": settings.standard_steps,
                    "cfg": settings.standard_cfg,
                    "sampler": settings.standard_sampler,
                    "scheduler": settings.standard_scheduler,
                },
                "turbo": {
                    "steps": settings.turbo_steps,
                    "cfg": settings.turbo_cfg,
                    "sampler": "lcm",
                    "scheduler": "normal",
                },
            },
        },
    }


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"# Benchmark: {report['benchmark']}",
        "",
        f"**Result: {'PASS' if summary['passed'] else 'FAIL'}**",
        "",
        "## Acceptance summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Samples (successful generations only)",
            "",
            "| Mode | Run | Seed | Task | Client total (s) | Server stages (s) |",
            "|---|---:|---:|---|---:|---|",
        ]
    )
    for sample in report["samples"]:
        lines.append(
            f"| {sample['mode']} | {sample['run_index']} | {sample['seed']} | "
            f"{sample['task_id']} | {sample['client_total_seconds']:.6f} | "
            f"{json.dumps(sample['server_stages'], sort_keys=True)} |"
        )
    lines.extend(["", "## Errors", "", "```json", json.dumps(report["errors"], indent=2), "```"])
    if "mode_transitions" in report:
        lines.extend(
            [
                "",
                "## Alternating-mode transitions",
                "",
                "These observed transition request latencies are separate from single-mode Standard P95 acceptance.",
                "",
                "```json",
                json.dumps(report["mode_transitions"], indent=2),
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## Model, seed, parameter, and version evidence",
            "",
            f"Warmup seed: `{report['warmup'].get('seed')}`. Measured seeds are listed per sample.",
            "",
            "```json",
            json.dumps(report["evidence"], indent=2, sort_keys=True),
            "```",
            "",
            "No API key or authorization header is included in this report.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_report(report: BenchmarkReport, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = f"benchmark-{report.mode}-{_timestamp()}"
    report.json_path = report_dir / f"{stem}.json"
    report.markdown_path = report_dir / f"{stem}.md"
    payload = report.to_dict()
    report.json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report.markdown_path.write_text(_markdown(payload), encoding="utf-8")


def _gpu_report(monitor: GpuMonitor) -> dict[str, Any]:
    return {
        "physical_gpu_id": 1,
        "peak_vram_mib": monitor.peak_vram_mib,
        "peak_utilization_percent": monitor.peak_utilization_percent,
        "sample_count": len(monitor.samples),
        "sampling_errors": monitor.errors,
        "command": "nvidia-smi --id=1 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits",
    }


def run_benchmark(
    api_url: str,
    api_key: str,
    input_path: str | Path,
    mode: str,
    runs: int,
    *,
    base_seed: int = DEFAULT_BASE_SEED,
    style: str = "cyberpunk",
    report_dir: str | Path = PROJECT_ROOT / "reports",
) -> BenchmarkReport:
    if mode not in {"standard", "turbo"}:
        raise ValueError("mode must be standard or turbo")
    if runs < 1:
        raise ValueError("runs must be positive")
    offsets = log_offsets()
    samples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    warmup: dict[str, Any] = {"seed": warmup_seed(base_seed), "measured": False}
    with GpuMonitor() as monitor:
        try:
            warmup_result = run_generation(
                api_url, api_key, input_path, mode, warmup["seed"], style
            )
            warmup.update(
                task_id=warmup_result.task_id,
                status="succeeded",
                client_total_seconds=warmup_result.client_total_seconds,
            )
        except Exception as exc:
            detail = error_details(exc, api_key)
            warmup.update(status="failed", error=detail)
            errors.append({"run_index": "warmup", "mode": mode, "seed": warmup["seed"], "error": detail})
        if warmup.get("status") == "succeeded":
            for run_index, seed in enumerate(seed_sequence(base_seed, runs)):
                started = time.perf_counter()
                try:
                    result = run_generation(
                        api_url, api_key, input_path, mode, seed, style
                    )
                    sample = result.to_dict()
                    sample["run_index"] = run_index
                    samples.append(sample)
                except Exception as exc:
                    errors.append(
                        {
                            "run_index": run_index,
                            "mode": mode,
                            "seed": seed,
                            "client_total_seconds": time.perf_counter() - started,
                            "error": error_details(exc, api_key),
                        }
                    )
    oom = detect_oom_since(offsets) or any(contains_oom(item["error"]) for item in errors)
    durations = [sample["client_total_seconds"] for sample in samples]
    evaluate = evaluate_standard if mode == "standard" else evaluate_turbo
    summary = evaluate(durations, len(errors), monitor.peak_vram_mib, oom)
    report = BenchmarkReport(
        mode,
        summary,
        samples,
        errors,
        warmup,
        collect_evidence(),
        _gpu_report(monitor),
        Path(),
        Path(),
    )
    _write_report(report, Path(report_dir))
    return report


def run_alternating_benchmark(
    api_url: str,
    api_key: str,
    input_path: str | Path,
    runs: int,
    *,
    base_seed: int = DEFAULT_BASE_SEED,
    style: str = "cyberpunk",
    report_dir: str | Path = PROJECT_ROOT / "reports",
) -> BenchmarkReport:
    if runs < 1:
        raise ValueError("runs must be positive")
    offsets = log_offsets()
    samples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    warmup: dict[str, Any] = {"seed": warmup_seed(base_seed), "mode": "standard", "measured": False}
    with GpuMonitor() as monitor:
        try:
            warmup_result = run_generation(
                api_url, api_key, input_path, "standard", warmup["seed"], style
            )
            warmup.update(task_id=warmup_result.task_id, status="succeeded")
        except Exception as exc:
            detail = error_details(exc, api_key)
            warmup.update(status="failed", error=detail)
            errors.append({"run_index": "warmup", "mode": "standard", "error": detail})
        if warmup.get("status") == "succeeded":
            previous_mode = "standard"
            for run_index, seed in enumerate(seed_sequence(base_seed, runs * 2)):
                mode = "standard" if run_index % 2 == 0 else "turbo"
                started = time.perf_counter()
                try:
                    result = run_generation(
                        api_url, api_key, input_path, mode, seed, style
                    )
                    sample = result.to_dict()
                    sample["run_index"] = run_index
                    sample["mode_changed"] = mode != previous_mode
                    samples.append(sample)
                    if mode != previous_mode:
                        transitions.append(
                            {
                                "run_index": run_index,
                                "from_mode": previous_mode,
                                "to_mode": mode,
                                "seed": seed,
                                "observed_switch_request_latency_seconds": result.client_total_seconds,
                                "server_stages": result.server_stages,
                                "peak_vram_mib_physical_gpu1_so_far": monitor.peak_vram_mib,
                            }
                        )
                    previous_mode = mode
                except Exception as exc:
                    errors.append(
                        {
                            "run_index": run_index,
                            "mode": mode,
                            "seed": seed,
                            "client_total_seconds": time.perf_counter() - started,
                            "error": error_details(exc, api_key),
                        }
                    )
    oom = detect_oom_since(offsets) or any(contains_oom(item["error"]) for item in errors)
    summary = evaluate_alternate(samples, len(errors), oom)
    summary["peak_vram_mib_physical_gpu1"] = monitor.peak_vram_mib
    if monitor.peak_vram_mib is None:
        summary["passed"] = False
    report = BenchmarkReport(
        "alternate-modes",
        summary,
        samples,
        errors,
        warmup,
        collect_evidence(),
        _gpu_report(monitor),
        Path(),
        Path(),
        transitions,
    )
    _write_report(report, Path(report_dir))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "assets" / "warmup-portrait.png"))
    parser.add_argument("--mode", choices=("standard", "turbo"), default="standard")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--alternate-modes", action="store_true")
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--style", default="cyberpunk")
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api_key = os.environ.get("AIGC_API_KEY", "")
    if not api_key:
        raise SystemExit("AIGC_API_KEY is required (its value is never logged)")
    try:
        if args.alternate_modes:
            report = run_alternating_benchmark(
                args.api_url,
                api_key,
                args.input,
                args.runs,
                base_seed=args.base_seed,
                style=args.style,
                report_dir=args.report_dir,
            )
        else:
            report = run_benchmark(
                args.api_url,
                api_key,
                args.input,
                args.mode,
                args.runs,
                base_seed=args.base_seed,
                style=args.style,
                report_dir=args.report_dir,
            )
    except Exception as exc:
        # Never print exception args: a local handler or URL might contain sensitive data.
        print(f"Benchmark could not produce a report: {type(exc).__name__}")
        return 2
    print(f"JSON report: {report.json_path}")
    print(f"Markdown report: {report.markdown_path}")
    print(f"Benchmark {report.mode}: {'PASS' if report.passed else 'FAIL'}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
