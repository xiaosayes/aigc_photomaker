# Server-pending: statistical tests are offline-safe, but the complete suite runs on the server.
import io
import json

import pytest
from PIL import Image

from aigc_photomaker import acceptance, benchmark
from aigc_photomaker.task_service import MAX_SEED


def png_bytes(size=(1024, 1024), mode="RGB", image_format="PNG"):
    output = io.BytesIO()
    Image.new(mode, size, "navy" if mode == "RGB" else 0).save(
        output, format=image_format
    )
    return output.getvalue()


def test_percentile_and_standard_gate():
    assert benchmark.percentile([1, 2, 3, 4], 95) == 4
    report = benchmark.evaluate_standard([29.0] * 20, failures=0, peak_vram_mib=22000)
    assert report["passed"] is True
    assert benchmark.evaluate_standard([31.0] * 20, 0, 22000)["passed"] is False
    assert benchmark.evaluate_standard([20.0] * 19, 0, 22000)["passed"] is False


def test_standard_gate_rejects_failures_oom_and_missing_gpu_evidence():
    assert benchmark.evaluate_standard([20.0] * 20, 1, 22000)["passed"] is False
    assert benchmark.evaluate_standard(
        [20.0] * 20, 0, 22000, oom=True
    )["passed"] is False
    assert benchmark.evaluate_standard([20.0] * 20, 0, None)["passed"] is False


def test_turbo_requires_ten_successful_samples_without_using_standard_p95_gate():
    report = benchmark.evaluate_turbo([31.0] * 10, failures=0, peak_vram_mib=22000)
    assert report["passed"] is True
    assert report["p95_seconds"] == 31.0
    assert report["p95_within_30_seconds"] is False
    assert benchmark.evaluate_turbo([20.0] * 9, 0, 22000)["passed"] is False
    assert benchmark.evaluate_turbo([20.0] * 10, 1, 22000)["passed"] is False


def test_percentile_uses_nearest_rank_and_rejects_invalid_input():
    assert benchmark.percentile([4.0, 1.0, 3.0, 2.0], 50) == 2.0
    assert benchmark.percentile([], 95) is None
    with pytest.raises(ValueError):
        benchmark.percentile([1.0], 0)


def test_full_result_decode_requires_display_ready_1024_rgb_png():
    metadata = acceptance.decode_result_png(png_bytes())
    assert metadata == {
        "format": "PNG",
        "mode": "RGB",
        "width": 1024,
        "height": 1024,
        "bytes": len(png_bytes()),
    }
    for invalid in (
        png_bytes(size=(512, 512)),
        png_bytes(mode="L"),
        png_bytes(image_format="JPEG"),
    ):
        with pytest.raises(acceptance.AcceptanceError):
            acceptance.decode_result_png(invalid)


def test_benchmark_uses_distinct_warmup_and_measured_seeds(tmp_path, monkeypatch):
    calls = []

    def fake_generation(api_url, api_key, input_path, mode, seed, style):
        calls.append((mode, seed))
        return acceptance.GenerationResult(
            task_id=f"task-{seed}",
            mode=mode,
            seed=seed,
            client_total_seconds=1.0,
            server_stages={"total": 0.8},
            image={"format": "PNG", "mode": "RGB", "width": 1024, "height": 1024},
        )

    class FakeMonitor:
        peak_vram_mib = 1234.0
        peak_utilization_percent = 80.0
        samples = [{"memory_used_mib": 1234.0, "utilization_percent": 80.0}]
        errors = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(benchmark, "run_generation", fake_generation)
    monkeypatch.setattr(benchmark, "GpuMonitor", FakeMonitor)
    monkeypatch.setattr(benchmark, "collect_evidence", lambda: {"version": "test"})
    monkeypatch.setattr(benchmark, "detect_oom_since", lambda offsets: False)

    report = benchmark.run_benchmark(
        "http://127.0.0.1:8000",
        "not-logged",
        "face.png",
        "turbo",
        2,
        base_seed=100,
        report_dir=tmp_path,
    )

    assert calls == [("turbo", 99), ("turbo", 100), ("turbo", 101)]
    assert [sample["seed"] for sample in report.samples] == [100, 101]
    assert report.warmup["seed"] == 99
    assert "not-logged" not in json.dumps(report.to_dict())
    assert (tmp_path / report.json_path.name).is_file()
    assert (tmp_path / report.markdown_path.name).is_file()


def test_negative_smoke_accepts_asynchronous_face_policy_error(monkeypatch):
    monkeypatch.setattr(
        acceptance,
        "create_task",
        lambda *args, **kwargs: {"task_id": "negative-task"},
    )
    monkeypatch.setattr(
        acceptance,
        "wait_for_terminal_task",
        lambda *args, **kwargs: {
            "status": "failed",
            "error": {"code": "NO_FACE_DETECTED"},
        },
    )
    result = acceptance.expect_task_error(
        "http://127.0.0.1:8000",
        "secret",
        b"png",
        "solid.png",
        style="cyberpunk",
        mode="standard",
        seed=1,
        expected_code="NO_FACE_DETECTED",
    )
    assert result["passed"] is True
    assert result["task_id"] == "negative-task"


def test_seed_sequence_stays_within_api_limit():
    assert benchmark.seed_sequence(MAX_SEED, 3) == [MAX_SEED, 0, 1]
    assert benchmark.warmup_seed(0) == MAX_SEED


def test_alternating_summary_is_not_a_single_mode_acceptance_gate():
    samples = [
        {"mode": "standard", "client_total_seconds": 31.0},
        {"mode": "turbo", "client_total_seconds": 2.0},
    ] * 10
    report = benchmark.evaluate_alternate(samples, failures=0, oom=False)
    assert report["passed"] is True
    assert report["acceptance_scope"] == "alternating-only"
    assert report["modes"]["standard"]["p95_seconds"] == 31.0
    assert "p95_within_30_seconds" not in report["modes"]["standard"]


GPU_INDICES = {"GPU-zero": 0, "GPU-one": 1}


def classify_attributions(processes, attributions, expected=None):
    return acceptance.classify_gpu_processes(
        GPU_INDICES,
        processes,
        lambda pid: attributions.get(pid, "indeterminate"),
        expected or {"comfy": 20},
    )


def test_unattributed_other_service_does_not_fail_project_isolation():
    result = classify_attributions(
        [{"gpu_uuid": "GPU-one", "pid": 10}, {"gpu_uuid": "GPU-one", "pid": 20}],
        {10: "indeterminate", 20: "project"},
    )
    assert result["passed"] is True
    assert result["status"] == "passed"
    assert result["project_processes"] == [{"pid": 20, "gpu_index": 1, "role": "comfy"}]
    assert result["indeterminate_processes"][0]["pid"] == 10
    assert result["ignored_unrelated_processes"] == 0


@pytest.mark.parametrize(
    "command",
    [
        "/usr/bin/env python train.py",
        "uv run python train.py",
        "/usr/bin/python3.11 train.py",
        "/usr/bin/unknown-runtime train.py",
    ],
)
def test_direct_classifier_treats_wrappers_python_and_unknown_as_indeterminate(command):
    assert acceptance._classify_command(command, "/tmp") == "indeterminate"


def test_direct_classifier_allows_only_explicitly_safe_unrelated_executable():
    command = "/usr/bin/ffmpeg -i input.mp4 output.mp4"
    assert acceptance._classify_command(command, "/tmp") == "unrelated"


@pytest.mark.parametrize(
    ("command", "cwd"),
    [
        (
            "/usr/bin/env python -m aigc_photomaker.worker",
            str(acceptance.PROJECT_ROOT),
        ),
        (
            "uv run python /opt/custom/ComfyUI/main.py --listen 127.0.0.1",
            str(acceptance.PROJECT_ROOT),
        ),
    ],
)
def test_direct_classifier_recognizes_full_project_command_lines(command, cwd):
    assert acceptance._classify_command(command, cwd) == "project"


def test_readable_unrelated_process_on_gpu0_is_ignored():
    result = classify_attributions(
        [{"gpu_uuid": "GPU-zero", "pid": 10}, {"gpu_uuid": "GPU-one", "pid": 20}],
        {10: "unrelated", 20: "project"},
    )
    assert result["passed"] is True
    assert result["status"] == "passed"
    assert result["ignored_unrelated_processes"] == 1
    assert result["project_processes"] == [{"pid": 20, "gpu_index": 1, "role": "comfy"}]


def test_custom_path_project_comfy_on_gpu0_fails_closed(monkeypatch):
    monkeypatch.setattr(
        acceptance,
        "_read_process_details",
        lambda pid: ("/opt/custom/ComfyUI/main.py --listen 127.0.0.1", str(acceptance.PROJECT_ROOT)),
    )
    assert acceptance._attribute_process(10, {}) == "project"
    result = classify_attributions(
        [{"gpu_uuid": "GPU-zero", "pid": 10}, {"gpu_uuid": "GPU-one", "pid": 20}],
        {10: "project", 20: "project"},
    )
    assert result["passed"] is False
    assert result["status"] == "failed"


def test_expected_project_process_on_gpu1_passes_without_proc_read(monkeypatch):
    monkeypatch.setattr(
        acceptance,
        "_read_process_details",
        lambda pid: (_ for _ in ()).throw(PermissionError("unreadable")),
    )
    assert acceptance._attribute_process(20, {20: "comfy"}) == "project"
    result = classify_attributions(
        [{"gpu_uuid": "GPU-one", "pid": 20}],
        {20: "project"},
    )
    assert result["passed"] is True
    assert result["missing_expected_gpu_processes"] == []


def test_missing_expected_gpu_process_prevents_isolation_pass():
    result = classify_attributions(
        [{"gpu_uuid": "GPU-zero", "pid": 10}],
        {10: "unrelated"},
    )
    assert result["passed"] is False
    assert result["status"] == "failed"
    assert result["missing_expected_gpu_processes"] == ["comfy"]
