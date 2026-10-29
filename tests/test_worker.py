# Server-pending: run these fake-ComfyUI and real-SQLite behavior tests on the GPU server.
import io
import json
import signal
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from aigc_photomaker.comfy import ExecutionResult
from aigc_photomaker.config import Settings
from aigc_photomaker.errors import AppError
from aigc_photomaker.repository import TaskRepository
from aigc_photomaker.styles import StylePreset
from aigc_photomaker.warmup import run_warmup
from aigc_photomaker.worker import TaskProcessor, Worker
from aigc_photomaker.workflow import ComfyImageRef


STYLE = StylePreset(
    key="portrait",
    label="Portrait",
    description="Warmup and worker test style",
    positive="portrait of a person",
    negative="low quality",
)


def settings(tmp_path):
    return Settings(
        api_key="a" * 32,
        _env_file=None,
        data_dir=tmp_path / "data",
        runtime_dir=tmp_path / "runtime",
        comfy_dir=tmp_path / "comfy",
    )


def repository(tmp_path):
    repo = TaskRepository(tmp_path / "tasks.db")
    repo.create_schema()
    return repo


def claimed_task(repo, task_id="task-1"):
    repo.create(task_id, f"inputs/{task_id}.png", "portrait", "standard", 42)
    task = repo.claim("worker-1", 60)
    assert task is not None
    return task


def png_bytes(width=1024, height=1024):
    output = io.BytesIO()
    Image.new("RGB", (width, height), "blue").save(output, format="PNG")
    return output.getvalue()


class FakeComfyClient:
    def __init__(self, tmp_path, error=None):
        self.tmp_path = tmp_path
        self.error = error
        self.calls = []
        self.upload_calls = []
        self.submit_calls = []
        self.wait_calls = []
        self.cancel_calls = []
        self.download_calls = []
        self.graphs = []
        self.on_wait = None
        self.closed = False

    def upload_image(self, path):
        self.calls.append("upload_image")
        self.upload_calls.append(Path(path))
        return ComfyImageRef("uploaded.png", "inputs", "input")

    def submit(self, graph, worker_id):
        self.calls.append("submit")
        self.submit_calls.append((graph, worker_id))
        self.graphs.append(graph)
        return "prompt-1"

    def wait_for_result(
        self,
        prompt_id,
        timeout_seconds,
        heartbeat=None,
        is_cancel_requested=None,
        on_progress=None,
    ):
        # Match ComfyUIClient: warmup intentionally uses the optional no-op callbacks.
        heartbeat = heartbeat or (lambda: None)
        is_cancel_requested = is_cancel_requested or (lambda: False)
        on_progress = on_progress or (lambda value, maximum: None)
        self.calls.append("wait_for_result")
        self.wait_calls.append(prompt_id)
        heartbeat()
        on_progress(1, 25)
        if self.on_wait is not None:
            self.on_wait(heartbeat, is_cancel_requested)
        if self.error is not None:
            raise self.error
        return ExecutionResult(prompt_id, [ComfyImageRef("result.png")], {})

    def cancel(self, prompt_id):
        self.calls.append("cancel")
        self.cancel_calls.append(prompt_id)

    def download_image(self, ref, destination):
        self.calls.append("download_image")
        self.download_calls.append((ref, Path(destination)))
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(png_bytes())

    def close(self):
        self.closed = True


@pytest.fixture
def processor_setup(tmp_path):
    configured = settings(tmp_path)
    data_dir = configured.resolve(configured.data_dir)
    data_dir.mkdir(parents=True)
    repo = repository(tmp_path)
    fake = FakeComfyClient(tmp_path)
    processor = TaskProcessor(
        settings=configured,
        repository=repo,
        comfy=fake,
        styles={"portrait": STYLE},
        worker_id="worker-1",
    )
    return configured, data_dir, repo, fake, processor


def test_new_task_success_uploads_submits_downloads_and_records_timings(processor_setup):
    configured, data_dir, repo, fake, processor = processor_setup
    task = claimed_task(repo)
    source = data_dir / task.input_path
    source.parent.mkdir(parents=True)
    source.write_bytes(png_bytes())

    processor.process(task)

    assert fake.calls == ["upload_image", "submit", "wait_for_result", "download_image"]
    graph, client_id = fake.submit_calls[0]
    assert client_id == f"aigc-photomaker-{task.task_id}"
    assert repo.get(task.task_id).submission_client_id == client_id
    assert repo.get(task.task_id).output_prefix == f"aigc-photomaker/{task.task_id}"
    assert graph["5"]["inputs"]["image"] == "inputs/uploaded.png"
    assert graph["17"]["inputs"]["noise_seed"] == 42
    staged_output = fake.download_calls[0][1]
    assert staged_output.parent == (data_dir / task.output_path).parent
    assert task.claim_token in staged_output.name
    assert staged_output != data_dir / task.output_path
    assert not staged_output.exists()
    assert (data_dir / task.output_path).read_bytes() == png_bytes()
    succeeded = repo.get(task.task_id)
    assert succeeded.status == "succeeded"
    assert succeeded.prompt_id == "prompt-1"
    assert succeeded.progress_value == 1
    assert succeeded.progress_max == 25
    assert set(succeeded.timings) == {"upload", "submit", "execute", "download", "total"}
    assert all(value >= 0 for value in succeeded.timings.values())


def test_resume_does_not_upload_or_submit_twice(processor_setup):
    _, _, repo, fake, processor = processor_setup
    task = claimed_task(repo)
    assert repo.mark_submitted(task.task_id, task.claim_token, "prompt-existing")
    resumed = repo.get(task.task_id)

    processor.process(resumed)

    assert fake.upload_calls == []
    assert fake.submit_calls == []
    assert fake.wait_calls == [resumed.prompt_id]
    assert fake.calls == ["wait_for_result", "download_image"]
    assert repo.get(task.task_id).status == "succeeded"


def test_cancel_requested_before_processing_interrupts_and_cancels(processor_setup):
    _, _, repo, fake, processor = processor_setup
    task = claimed_task(repo)
    assert repo.mark_submitted(task.task_id, task.claim_token, "prompt-existing")
    repo.request_cancel(task.task_id)

    processor.process(repo.get(task.task_id))

    assert fake.cancel_calls == ["prompt-existing"]
    assert fake.wait_calls == []
    cancelled = repo.get(task.task_id)
    assert cancelled.status == "cancelled"
    assert cancelled.error is None


def test_cancellation_during_execution_is_not_failed(processor_setup):
    _, _, repo, fake, processor = processor_setup
    task = claimed_task(repo)

    def cancel_during_wait(heartbeat, is_cancel_requested):
        repo.request_cancel(task.task_id)
        assert is_cancel_requested()
        raise AppError(409, "CANCELLED", "Generation was cancelled", False)

    fake.on_wait = cancel_during_wait
    processor.process(task)

    assert repo.get(task.task_id).status == "cancelled"
    assert repo.get(task.task_id).error is None


def test_comfy_error_persists_structured_code_and_retryable(tmp_path):
    configured = settings(tmp_path)
    data_dir = configured.resolve(configured.data_dir)
    data_dir.mkdir(parents=True)
    repo = repository(tmp_path)
    fake = FakeComfyClient(
        tmp_path, AppError(500, "GPU_OOM", "ComfyUI ran out of GPU memory", False)
    )
    processor = TaskProcessor(configured, repo, fake, {"portrait": STYLE}, "worker-1")
    task = claimed_task(repo)

    processor.process(task)

    failed = repo.get(task.task_id)
    assert failed.status == "failed"
    assert failed.error == {
        "code": "GPU_OOM",
        "message": "ComfyUI ran out of GPU memory",
        "retryable": False,
    }


def test_missing_result_is_structured_error(processor_setup):
    _, _, repo, fake, processor = processor_setup
    task = claimed_task(repo)
    fake.wait_for_result = lambda *args, **kwargs: ExecutionResult("prompt-1", [], {})

    processor.process(task)

    assert repo.get(task.task_id).status == "failed"
    assert repo.get(task.task_id).error["code"] == "RESULT_MISSING"
    assert repo.get(task.task_id).error["retryable"] is True


def test_old_claim_token_cannot_complete_recovered_task(processor_setup):
    _, data_dir, repo, fake, processor = processor_setup
    old_task = claimed_task(repo)
    final_output = data_dir / old_task.output_path
    final_output.parent.mkdir(parents=True)
    final_output.write_bytes(b"original-output")

    def recover(heartbeat, is_cancel_requested):
        repo.force_lease_expiry(old_task.task_id)
        current = repo.claim("worker-2", 60)
        assert current is not None
        assert current.claim_token != old_task.claim_token
        with pytest.raises(AppError) as caught:
            heartbeat()
        assert caught.value.code == "WORKER_CLAIM_LOST"

    fake.on_wait = recover
    processor.process(old_task)

    current = repo.get(old_task.task_id)
    assert current.status == "running"
    assert current.worker_id == "worker-2"
    assert current.timings == {}
    assert current.error is None
    assert fake.download_calls == []
    assert final_output.read_bytes() == b"original-output"
    assert list(final_output.parent.glob(f"{final_output.name}.*")) == []


def test_claim_loss_during_download_cleans_staging_and_preserves_output(processor_setup):
    _, data_dir, repo, fake, processor = processor_setup
    old_task = claimed_task(repo)
    final_output = data_dir / old_task.output_path
    final_output.parent.mkdir(parents=True)
    final_output.write_bytes(b"original-output")
    original_download = fake.download_image

    def recover_during_download(ref, destination):
        original_download(ref, destination)
        repo.force_lease_expiry(old_task.task_id)
        assert repo.claim("worker-2", 60).claim_token != old_task.claim_token

    fake.download_image = recover_during_download
    processor.process(old_task)

    current = repo.get(old_task.task_id)
    assert current.status == "running"
    assert current.worker_id == "worker-2"
    assert final_output.read_bytes() == b"original-output"
    assert list(final_output.parent.glob(f"{final_output.name}.*")) == []


def test_generic_exception_after_cancel_request_becomes_cancelled(processor_setup):
    _, _, repo, fake, processor = processor_setup
    task = claimed_task(repo)

    def cancel_then_fail(path):
        repo.request_cancel(task.task_id)
        raise RuntimeError("upload failed")

    fake.upload_image = cancel_then_fail
    processor.process(task)

    cancelled = repo.get(task.task_id)
    assert cancelled.status == "cancelled"
    assert cancelled.error is None


def test_cancellation_after_download_does_not_promote_or_succeed(processor_setup):
    _, data_dir, repo, fake, processor = processor_setup
    task = claimed_task(repo)
    final_output = data_dir / task.output_path
    final_output.parent.mkdir(parents=True)
    final_output.write_bytes(b"previous-output")
    original_download = fake.download_image

    def cancel_after_download(ref, destination):
        original_download(ref, destination)
        repo.request_cancel(task.task_id)

    fake.download_image = cancel_after_download
    processor.process(task)

    cancelled = repo.get(task.task_id)
    assert cancelled.status == "cancelled"
    assert cancelled.timings == {}
    assert final_output.read_bytes() == b"previous-output"
    assert list(final_output.parent.glob(f"{final_output.name}.*")) == []


def test_repository_success_and_failure_are_cancel_guarded_and_terminal(processor_setup):
    _, _, repo, _, _ = processor_setup
    task = claimed_task(repo)
    repo.request_cancel(task.task_id)

    assert not repo.succeed(task.task_id, task.claim_token, {"total": 1.0})
    assert not repo.fail(
        task.task_id,
        task.claim_token,
        {"code": "INTERNAL_ERROR", "message": "late", "retryable": False},
    )
    assert repo.mark_cancelled(task.task_id, task.claim_token)
    cancelled = repo.get(task.task_id)
    assert cancelled.status == "cancelled"
    assert cancelled.timings == {}
    assert cancelled.error is None
    assert not repo.succeed(task.task_id, task.claim_token, {"total": 2.0})
    assert not repo.fail(task.task_id, task.claim_token, {"code": "LATE"})
    assert repo.get(task.task_id) == cancelled


def test_worker_writes_atomic_heartbeat_and_stops_claiming(tmp_path):
    configured = settings(tmp_path)
    repo = repository(tmp_path)
    fake = FakeComfyClient(tmp_path)
    processor = TaskProcessor(configured, repo, fake, {"portrait": STYLE}, "worker-1")
    worker = Worker(configured, repo, processor, worker_id="worker-1")

    worker._write_heartbeat()
    heartbeat_path = configured.resolve(configured.runtime_dir) / "worker-heartbeat.json"
    payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))

    assert payload["worker_id"] == "worker-1"
    assert isinstance(payload["heartbeat_at"], float)
    assert not list(heartbeat_path.parent.glob("worker-heartbeat.json.*.part"))
    worker.stop()
    assert worker._stop.is_set()


def test_blocking_client_work_does_not_stop_lease_or_readiness_heartbeat(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("aigc_photomaker.worker.HEARTBEAT_INTERVAL_SECONDS", 0.05)
    configured = settings(tmp_path)
    data_dir = configured.resolve(configured.data_dir)
    data_dir.mkdir(parents=True)

    class TrackingRepository(TaskRepository):
        def __init__(self, path):
            super().__init__(path)
            self.heartbeat_calls = []
            self.enough_heartbeats = threading.Event()
            self.succeeded = threading.Event()

        def heartbeat(self, task_id, claim_token, lease_seconds):
            result = super().heartbeat(task_id, claim_token, lease_seconds)
            self.heartbeat_calls.append(time.monotonic())
            if len(self.heartbeat_calls) >= 3:
                self.enough_heartbeats.set()
            return result

        def succeed(self, task_id, claim_token, timings, output_path=None):
            result = super().succeed(task_id, claim_token, timings, output_path)
            if result:
                self.succeeded.set()
            return result

    repo = TrackingRepository(tmp_path / "tasks.db")
    repo.create_schema()
    # The Worker must perform the only claim; pre-claiming hides it for 60 seconds.
    task = repo.create("task-1", "inputs/task-1.png", "portrait", "standard", 42)
    source = data_dir / task.input_path
    source.parent.mkdir(parents=True)
    source.write_bytes(png_bytes())
    fake = FakeComfyClient(tmp_path)
    upload_started = threading.Event()
    release_upload = threading.Event()
    downloaded = threading.Event()
    original_upload = fake.upload_image
    original_download = fake.download_image

    def blocking_upload(path):
        upload_started.set()
        assert release_upload.wait(5)
        return original_upload(path)

    def recording_download(ref, destination):
        original_download(ref, destination)
        downloaded.set()

    fake.upload_image = blocking_upload
    fake.download_image = recording_download
    processor = TaskProcessor(configured, repo, fake, {"portrait": STYLE}, "worker-1")
    worker = Worker(configured, repo, processor, worker_id="worker-1")
    file_heartbeats = []
    enough_file_heartbeats = threading.Event()
    original_write = worker._write_heartbeat

    def recording_write():
        original_write()
        file_heartbeats.append(time.monotonic())
        if len(file_heartbeats) >= 3:
            enough_file_heartbeats.set()

    worker._write_heartbeat = recording_write
    thread = threading.Thread(target=worker.run_forever)
    thread.start()
    try:
        assert upload_started.wait(2)
        assert repo.enough_heartbeats.wait(2)
        assert enough_file_heartbeats.wait(2)
        assert len(fake.submit_calls) == 0
        payload = json.loads(
            (configured.resolve(configured.runtime_dir) / "worker-heartbeat.json").read_text()
        )
        assert payload["task_id"] == task.task_id
        release_upload.set()
        assert downloaded.wait(2)
        assert repo.succeeded.wait(2)
    finally:
        release_upload.set()
        worker.stop()
        thread.join(5)

    assert not thread.is_alive()
    assert repo.get(task.task_id).status == "succeeded"
    assert len(fake.wait_calls) == 1
    assert len(repo.heartbeat_calls) >= 3
    assert len(file_heartbeats) >= 3


def test_idle_worker_polls_for_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr("aigc_photomaker.worker.POLL_INTERVAL_SECONDS", 0.01)
    configured = settings(tmp_path)

    class PollingRepository(TaskRepository):
        def __init__(self, path):
            super().__init__(path)
            self.claim_calls = 0
            self.polled = threading.Event()

        def claim(self, worker_id, lease_seconds):
            self.claim_calls += 1
            if self.claim_calls >= 3:
                self.polled.set()
            return super().claim(worker_id, lease_seconds)

    repo = PollingRepository(tmp_path / "tasks.db")
    repo.create_schema()
    fake = FakeComfyClient(tmp_path)
    processor = TaskProcessor(configured, repo, fake, {"portrait": STYLE}, "worker-1")
    worker = Worker(configured, repo, processor, worker_id="worker-1")
    thread = threading.Thread(target=worker.run_forever)
    thread.start()
    try:
        assert repo.polled.wait(2)
    finally:
        worker.stop()
        thread.join(5)

    assert not thread.is_alive()
    assert repo.claim_calls >= 3
    assert fake.calls == []


def test_sigterm_stops_new_claims_and_cancels_active_task(tmp_path):
    configured = settings(tmp_path)

    class ClaimRepository(TaskRepository):
        def __init__(self, path):
            super().__init__(path)
            self.claim_calls = 0

        def claim(self, worker_id, lease_seconds):
            self.claim_calls += 1
            return super().claim(worker_id, lease_seconds)

    repo = ClaimRepository(tmp_path / "tasks.db")
    repo.create_schema()
    # Leave the task queued so run_forever executes it before the first assertion.
    task = repo.create("task-1", "inputs/task-1.png", "portrait", "standard", 42)
    fake = FakeComfyClient(tmp_path)

    def terminate(heartbeat, is_cancel_requested):
        signal.raise_signal(signal.SIGTERM)
        assert is_cancel_requested()
        raise AppError(409, "CANCELLED", "Generation was cancelled", False)

    fake.on_wait = terminate
    processor = TaskProcessor(configured, repo, fake, {"portrait": STYLE}, "worker-1")
    worker = Worker(configured, repo, processor, worker_id="worker-1")

    worker.run_forever()

    assert worker._stop.is_set()
    assert repo.claim_calls == 1
    assert repo.get(task.task_id).status == "cancelled"
    assert fake.cancel_calls == ["prompt-1"]


def test_warmup_uses_production_steps_and_verifies_single_face_png(tmp_path, monkeypatch):
    configured = settings(tmp_path)
    fake = FakeComfyClient(tmp_path)
    monkeypatch.setattr("aigc_photomaker.warmup.ComfyUIClient", lambda url: fake)

    report = run_warmup(configured)

    assert fake.closed
    assert [graph["13"]["inputs"]["steps"] for graph in fake.graphs] == [20, 4]
    assert [graph["17"]["inputs"]["noise_seed"] for graph in fake.graphs] == [0, 0]
    assert all(graph["8"]["inputs"]["width"] == 1024 for graph in fake.graphs)
    assert all(graph["8"]["inputs"]["height"] == 1024 for graph in fake.graphs)
    assert fake.graphs[0]["11"]["class_type"] == "VAEDecode"
    assert fake.graphs[0]["1"]["inputs"]["unet_name"] == "flux-2-klein-base-4b-fp8.safetensors"
    assert fake.graphs[1]["1"]["inputs"]["unet_name"] == "flux-2-klein-4b-fp8.safetensors"
    assert fake.graphs[1]["10"]["class_type"] == "KSamplerSelect"
    assert all(graph["14"]["class_type"] == "FluxSingleFaceOutputGuard" for graph in fake.graphs)
    assert all("cyberpunk style" in graph["6"]["inputs"]["text"] for graph in fake.graphs)
    assert all("solo" in graph["6"]["inputs"]["text"] for graph in fake.graphs)
    uploaded = configured.resolve(configured.comfy_dir) / "input" / "warmup-portrait.png"
    assert uploaded.read_bytes() == (Path("assets") / "warmup-portrait.png").read_bytes()
    assert report["status"] == "succeeded"
    assert set(report["modes"]) == {"standard", "turbo"}
    assert {mode: values["steps"] for mode, values in report["modes"].items()} == {
        "standard": 20,
        "turbo": 4,
    }
    persisted = json.loads(
        (configured.resolve(configured.runtime_dir) / "warmup.json").read_text(encoding="utf-8")
    )
    assert persisted == report
    for mode in ("standard", "turbo"):
        output = configured.resolve(configured.runtime_dir) / f"warmup-{mode}.png"
        with Image.open(output) as image:
            assert image.format == "PNG"
            assert image.size == (1024, 1024)
            assert image.mode == "RGB"


def test_warmup_rejects_wrong_dimensions_and_does_not_publish_status(tmp_path, monkeypatch):
    configured = settings(tmp_path)
    fake = FakeComfyClient(tmp_path)

    def small_download(ref, destination):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(png_bytes(2, 3))

    fake.download_image = small_download
    monkeypatch.setattr("aigc_photomaker.warmup.ComfyUIClient", lambda url: fake)

    with pytest.raises(AppError) as caught:
        run_warmup(configured, modes=("standard",))

    assert caught.value.code == "RESULT_MISSING"
    assert not (configured.resolve(configured.runtime_dir) / "warmup.json").exists()


def test_warmup_rejects_non_png_and_does_not_publish_status(tmp_path, monkeypatch):
    configured = settings(tmp_path)
    fake = FakeComfyClient(tmp_path)

    def invalid_download(ref, destination):
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(b"not a png")

    fake.download_image = invalid_download
    monkeypatch.setattr("aigc_photomaker.warmup.ComfyUIClient", lambda url: fake)

    with pytest.raises(AppError) as caught:
        run_warmup(configured, modes=("standard",))

    assert caught.value.code == "RESULT_MISSING"
    assert not (configured.resolve(configured.runtime_dir) / "warmup.json").exists()


def test_failed_warmup_rerun_invalidates_previous_success(tmp_path, monkeypatch):
    configured = settings(tmp_path)
    runtime_dir = configured.resolve(configured.runtime_dir)
    runtime_dir.mkdir(parents=True)
    status_path = runtime_dir / "warmup.json"
    status_path.write_text(json.dumps({"status": "succeeded", "modes": {}}))
    fake = FakeComfyClient(
        tmp_path, AppError(503, "COMFYUI_UNAVAILABLE", "ComfyUI is unavailable", True)
    )
    monkeypatch.setattr("aigc_photomaker.warmup.ComfyUIClient", lambda url: fake)

    with pytest.raises(AppError) as caught:
        run_warmup(configured, modes=("standard",))

    assert caught.value.code == "COMFYUI_UNAVAILABLE"
    assert not status_path.exists()
    assert fake.closed
