# Server-pending: coordinated final-review regressions; do not run locally.
import asyncio
import errno
import io
import json
import sqlite3
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from aigc_photomaker.app import RawBodyLimitMiddleware, create_app
from aigc_photomaker.comfy import ExecutionResult
from aigc_photomaker.config import Settings
from aigc_photomaker.db import connect
from aigc_photomaker.errors import AppError, disk_space_error
from aigc_photomaker.images import normalize_image
from aigc_photomaker.repository import TaskRepository
from aigc_photomaker.retention import RetentionCleaner
from aigc_photomaker.worker import TaskProcessor, Worker
from aigc_photomaker.workflow import ComfyImageRef

NOW = 2_000_000_000.0
STYLE = SimpleNamespace(
    key="portrait",
    positive="portrait of a person",
    negative="low quality",
    lora_filename="lah-cyberpunk-sdxl.safetensors",
    lora_strength=1.0,
)


def configured(tmp_path, **overrides):
    return Settings(
        api_key="a" * 32,
        _env_file=None,
        data_dir=tmp_path / "data",
        runtime_dir=tmp_path / "runtime",
        comfy_dir=tmp_path / "comfy",
        retention_days=7,
        **overrides,
    )


def repository(tmp_path):
    repo = TaskRepository(tmp_path / "tasks.db")
    repo.create_schema()
    return repo


def finish(repo, task_id, when=NOW - 8 * 86400):
    task = repo.claim("worker", 60)
    assert task.task_id == task_id
    assert repo.succeed(task_id, task.claim_token, {})
    with connect(repo.path) as connection:
        with connection:
            connection.execute("UPDATE tasks SET finished_at=? WHERE id=?", (when, task_id))


def test_worker_lifecycle_runs_initial_and_hourly_real_retention(tmp_path, monkeypatch):
    monkeypatch.setattr("aigc_photomaker.worker.POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr("aigc_photomaker.retention.RETENTION_INTERVAL_SECONDS", 0.01)
    settings = configured(tmp_path)
    repo = repository(tmp_path)
    task = repo.create("expired", "inputs/expired.png", "portrait", "standard", 1)
    # This test uses the cleaner's real clock rather than cleanup(NOW).
    finish(repo, task.task_id, when=time.time() - 8 * 86400)
    source = settings.data_dir / task.input_path
    source.parent.mkdir(parents=True)
    source.write_bytes(b"image")
    processor = SimpleNamespace(on_heartbeat=None, stop_requested=None, process=lambda task: None)
    worker = Worker(settings, repo, processor)
    thread = threading.Thread(target=worker.run_forever)
    thread.start()
    try:
        deadline = time.monotonic() + 3
        while repo.get(task.task_id) is not None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert repo.get(task.task_id) is None
        assert not source.exists()
        first_pass = worker.retention_cleaner.pass_count
        deadline = time.monotonic() + 3
        while worker.retention_cleaner.pass_count <= first_pass and time.monotonic() < deadline:
            time.sleep(0.01)
        assert worker.retention_cleaner.pass_count > first_pass
    finally:
        worker.stop()
        thread.join(3)
    assert not thread.is_alive()
    assert worker.retention_cleaner._stop.is_set()


def test_retention_deletes_comfy_input_and_all_prefixed_outputs(tmp_path):
    settings = configured(tmp_path)
    settings.data_dir.mkdir(parents=True)
    repo = repository(tmp_path)
    task = repo.create("task-a", "inputs/task-a.png", "portrait", "standard", 1)
    claimed = repo.claim("worker", 60)
    assert repo.prepare_submission(
        task.task_id, claimed.claim_token, "intent-a", "aigc-photomaker/task-a"
    )
    assert repo.record_execution_ref(
        task.task_id, claimed.claim_token,
        {"name": "face.png", "subfolder": "uploads", "type": "input"},
    )
    assert repo.succeed(task.task_id, claimed.claim_token, {})
    with connect(repo.path) as connection:
        with connection:
            connection.execute("UPDATE tasks SET finished_at=? WHERE id=?", (NOW - 8 * 86400, task.task_id))
    stored = repo.get(task.task_id)
    data_file = settings.data_dir / stored.input_path
    data_file.parent.mkdir(parents=True)
    data_file.write_bytes(b"input")
    comfy_input = settings.comfy_dir / "input" / "uploads" / "face.png"
    comfy_input.parent.mkdir(parents=True)
    comfy_input.write_bytes(b"face")
    output_dir = settings.comfy_dir / "output" / "aigc-photomaker"
    output_dir.mkdir(parents=True)
    outputs = [output_dir / "task-a_00001_.png", output_dir / "task-a_00002_.png"]
    for path in outputs:
        path.write_bytes(b"output")
    other = output_dir / "task-ab_00001_.png"
    other.write_bytes(b"other")

    assert RetentionCleaner(settings, repo).cleanup(NOW) == 1
    assert not data_file.exists()
    assert not comfy_input.exists()
    assert all(not path.exists() for path in outputs)
    assert other.read_bytes() == b"other"


def test_retention_rejects_unsafe_comfy_reference_before_any_unlink(tmp_path):
    settings = configured(tmp_path)
    settings.data_dir.mkdir(parents=True)
    repo = repository(tmp_path)
    repo.create("unsafe", "inputs/unsafe.png", "portrait", "standard", 1)
    finish(repo, "unsafe")
    task = repo.get("unsafe")
    source = settings.data_dir / task.input_path
    source.parent.mkdir(parents=True)
    source.write_bytes(b"keep")
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"private")
    with connect(repo.path) as connection:
        with connection:
            connection.execute(
                "UPDATE tasks SET execution_refs_json=? WHERE id='unsafe'",
                (json.dumps([{"name": "face.png", "subfolder": "../../", "type": "input"}]),),
            )
    assert RetentionCleaner(settings, repo).cleanup(NOW) == 0
    assert source.exists()
    assert outside.exists()


def test_retention_comfy_unlink_failure_isolated_and_record_retained(tmp_path, monkeypatch):
    settings = configured(tmp_path)
    settings.data_dir.mkdir(parents=True)
    repo = repository(tmp_path)
    task = repo.create("failed", "inputs/failed.png", "portrait", "standard", 1)
    claimed = repo.claim("worker", 60)
    repo.prepare_submission(task.task_id, claimed.claim_token, "intent", "aigc-photomaker/failed")
    repo.record_execution_ref(task.task_id, claimed.claim_token, {"name": "in.png", "subfolder": "", "type": "input"})
    repo.fail(task.task_id, claimed.claim_token, {"code": "X"})
    with connect(repo.path) as connection:
        with connection:
            connection.execute("UPDATE tasks SET finished_at=? WHERE id='failed'", (NOW - 8 * 86400,))
    input_copy = settings.comfy_dir / "input" / "in.png"
    input_copy.parent.mkdir(parents=True)
    input_copy.write_bytes(b"in")
    output = settings.comfy_dir / "output" / "aigc-photomaker" / "failed_00001_.png"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"out")
    original = Path.unlink
    def fail_input(path, *args, **kwargs):
        if path == input_copy:
            raise OSError("busy")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", fail_input)

    assert RetentionCleaner(settings, repo).cleanup(NOW) == 0
    assert input_copy.exists()
    assert not output.exists()
    assert repo.get(task.task_id) is not None


class ReconcileComfy:
    def __init__(self, found=None, submit_error=None):
        self.found = found
        self.submit_error = submit_error
        self.submit_calls = 0
        self.upload_calls = []
        self.cancel_calls = []
        self.reconcile_calls = 0

    def upload_image(self, path):
        self.upload_calls.append(path)
        return ComfyImageRef("uploaded.png", "", "input")

    def find_submission(self, client_id, output_prefix):
        self.reconcile_calls += 1
        return self.found

    def submit(self, graph, client_id):
        self.submit_calls += 1
        if self.submit_error:
            raise self.submit_error
        return "prompt-new"

    def wait_for_result(self, *args):
        return ExecutionResult("prompt", [], {})

    def cancel(self, prompt_id):
        self.cancel_calls.append(prompt_id)


def processing_setup(tmp_path, comfy):
    settings = configured(tmp_path)
    settings.data_dir.mkdir(parents=True)
    repo = repository(tmp_path)
    task = repo.create("crash", "inputs/crash.png", "portrait", "standard", 1)
    (settings.data_dir / task.input_path).parent.mkdir(parents=True)
    (settings.data_dir / task.input_path).write_bytes(b"image")
    claimed = repo.claim("worker", 60)
    processor = TaskProcessor(settings, repo, comfy, {"portrait": STYLE})
    return repo, claimed, processor


def test_uncertain_accept_is_reconciled_without_second_submission(tmp_path):
    comfy = ReconcileComfy("prompt-accepted", AppError(503, "COMFYUI_UNAVAILABLE", "down", True))
    repo, task, processor = processing_setup(tmp_path, comfy)
    processor.process(task)
    stored = repo.get(task.task_id)
    assert comfy.submit_calls == 1
    assert stored.prompt_id == "prompt-accepted"
    assert stored.error["code"] == "RESULT_MISSING"


def test_uncertain_submit_not_found_fails_reviewably(tmp_path):
    comfy = ReconcileComfy(None, AppError(503, "COMFYUI_UNAVAILABLE", "down", True))
    repo, task, processor = processing_setup(tmp_path, comfy)
    processor.process(task)
    stored = repo.get(task.task_id)
    assert comfy.submit_calls == 1
    assert stored.status == "failed"
    assert stored.error["code"] == "SUBMISSION_UNCERTAIN"
    assert stored.prompt_id is None


def test_mark_submitted_database_failure_cancels_accepted_prompt(tmp_path, monkeypatch):
    comfy = ReconcileComfy()
    repo, task, processor = processing_setup(tmp_path, comfy)
    def fail_persistence(*args):
        raise OSError("database write failed")
    monkeypatch.setattr(repo, "mark_submitted", fail_persistence)
    processor.process(task)
    assert comfy.cancel_calls == ["prompt-new"]
    assert repo.get(task.task_id).error["code"] == "SUBMISSION_PERSISTENCE_FAILED"


def test_recovery_without_match_is_uncertain_and_never_resubmits(tmp_path):
    comfy = ReconcileComfy(found=None)
    repo, task, processor = processing_setup(tmp_path, comfy)
    repo.prepare_submission(
        task.task_id, task.claim_token, "intent", "aigc-photomaker/crash"
    )
    output = processor.settings.data_dir / task.output_path

    processor.process(repo.get(task.task_id))

    stored = repo.get(task.task_id)
    assert comfy.reconcile_calls == 1
    assert comfy.upload_calls == []
    assert comfy.submit_calls == 0
    assert stored.status == "failed"
    assert stored.error["code"] == "SUBMISSION_UNCERTAIN"
    assert stored.prompt_id is None
    assert not output.exists()


def test_recovery_reconciles_existing_prompt_before_upload_or_submit(tmp_path):
    comfy = ReconcileComfy("prompt-existing")
    repo, task, processor = processing_setup(tmp_path, comfy)
    repo.prepare_submission(task.task_id, task.claim_token, "intent", "aigc-photomaker/crash")
    task = repo.get(task.task_id)
    processor.process(task)
    assert comfy.reconcile_calls == 1
    assert comfy.submit_calls == 0
    assert repo.get(task.task_id).prompt_id == "prompt-existing"


def raw_asgi_call(app, messages, headers=()):
    sent = []
    incoming = iter(messages)
    async def receive():
        return next(incoming, {"type": "http.disconnect"})
    async def send(message):
        sent.append(message)
    scope = {"type": "http", "method": "POST", "path": "/api/v1/tasks", "headers": list(headers)}
    asyncio.run(app(scope, receive, send))
    return sent


def test_raw_body_content_length_rejected_before_receive():
    called = []
    async def downstream(scope, receive, send):
        called.append(True)
    app = RawBodyLimitMiddleware(downstream, 100)
    sent = raw_asgi_call(app, [], [(b"content-length", b"101")])
    assert not called
    assert sent[0]["status"] == 413
    assert (b"connection", b"close") in sent[0]["headers"]


def test_raw_body_chunked_and_lying_content_length_are_bounded():
    async def downstream(scope, receive, send):
        while (await receive())["more_body"]:
            pass
    app = RawBodyLimitMiddleware(downstream, 5)
    sent = raw_asgi_call(
        app,
        [{"type": "http.request", "body": b"1234", "more_body": True},
         {"type": "http.request", "body": b"56", "more_body": False}],
        [(b"content-length", b"2")],
    )
    assert sent[0]["status"] == 413
    assert (b"connection", b"close") in sent[0]["headers"]


def test_readiness_disk_check_is_boolean_and_enospc_is_structured(tmp_path, monkeypatch):
    settings = configured(tmp_path, min_free_disk_bytes=1024)
    monkeypatch.setattr("aigc_photomaker.app.ModelManager", lambda *a: SimpleNamespace(verify=lambda full_hash: [SimpleNamespace(ok=True)]))
    monkeypatch.setattr("aigc_photomaker.app.ComfyUIClient", lambda *a: SimpleNamespace(system_stats=lambda: {}, close=lambda: None))
    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/health/ready")
        assert isinstance(response.json()["checks"]["disk"], bool)
        assert response.json()["checks"]["disk"] is True
        monkeypatch.setattr("aigc_photomaker.app.shutil.disk_usage", lambda path: SimpleNamespace(free=0))
        response = client.get("/api/v1/health/ready")
        assert response.json()["checks"]["disk"] is False


def test_api_upload_write_enospc_is_disk_space_low(tmp_path, monkeypatch):
    settings = configured(tmp_path)
    monkeypatch.setattr("aigc_photomaker.app.ModelManager", lambda *a: SimpleNamespace(verify=lambda full_hash: [SimpleNamespace(ok=True)]))
    monkeypatch.setattr("aigc_photomaker.app.ComfyUIClient", lambda *a: SimpleNamespace(system_stats=lambda: {}, close=lambda: None))
    real_temporary = tempfile.NamedTemporaryFile
    class FullFile:
        def __init__(self):
            self.wrapped = real_temporary(mode="wb", dir=settings.data_dir / "inputs", delete=False)
            self.name = self.wrapped.name
        def write(self, chunk):
            raise OSError(errno.ENOSPC, "full")
        def close(self):
            self.wrapped.close()
    monkeypatch.setattr("aigc_photomaker.task_service.tempfile.NamedTemporaryFile", lambda *a, **k: FullFile())
    payload = io.BytesIO()
    Image.new("RGB", (128, 128)).save(payload, format="JPEG")
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/tasks",
            headers={"X-API-Key": "a" * 32},
            files={"file": ("face.jpg", payload.getvalue(), "image/jpeg")},
            data={"style": "cyberpunk", "mode": "standard"},
        )
    assert response.status_code == 507
    assert response.json()["error"]["code"] == "DISK_SPACE_LOW"


def test_normalization_enospc_is_disk_space_low(tmp_path, monkeypatch):
    source = tmp_path / "source.png"
    Image.new("RGB", (128, 128)).save(source)
    def full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "full")
    monkeypatch.setattr(Image.Image, "save", full)
    with pytest.raises(AppError) as caught:
        normalize_image(source, tmp_path / "out.png", 20_000_000, 2048)
    assert caught.value.code == "DISK_SPACE_LOW"


def test_sqlite_full_is_disk_space_low():
    error = disk_space_error(sqlite3.OperationalError("database or disk is full"))
    assert error is not None
    assert error.code == "DISK_SPACE_LOW"


def test_worker_enospc_is_disk_space_low(tmp_path):
    comfy = ReconcileComfy()
    repo, task, processor = processing_setup(tmp_path, comfy)
    def full(path):
        raise OSError(errno.ENOSPC, "full")
    comfy.upload_image = full
    processor.process(task)
    assert repo.get(task.task_id).error["code"] == "DISK_SPACE_LOW"


def test_existing_database_migrates_execution_metadata(tmp_path):
    path = tmp_path / "legacy.db"
    with connect(path) as connection:
        connection.execute("""
            CREATE TABLE tasks (
                id TEXT PRIMARY KEY, input_path TEXT NOT NULL, output_path TEXT NOT NULL,
                style TEXT NOT NULL, mode TEXT NOT NULL, seed INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued', worker_id TEXT, prompt_id TEXT,
                claim_token TEXT UNIQUE, lease_expires_at REAL, heartbeat_at REAL,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                progress_phase TEXT NOT NULL DEFAULT 'queued', progress_value INTEGER NOT NULL DEFAULT 0,
                progress_max INTEGER NOT NULL DEFAULT 0, timings_json TEXT NOT NULL DEFAULT '{}',
                error_json TEXT, created_at REAL NOT NULL, started_at REAL, finished_at REAL
            )
        """)
    repo = TaskRepository(path)
    repo.create_schema()
    with connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(tasks)")}
    assert {"submission_client_id", "output_prefix", "execution_refs_json"} <= columns


@pytest.mark.parametrize("value", ["   ", "''", '""', "'   '", '\"   \"'])
def test_install_replaces_every_empty_api_key_form_without_duplicates(tmp_path, value):
    installer = Path("scripts/install.sh").read_text()
    program = installer.split('python - "$ROOT/.env" <<\'PY\'\n', 1)[1].split("\nPY\n", 1)[0]
    env = tmp_path / ".env"
    env.write_text(f"AIGC_API_KEY={value}\nAIGC_API_KEY={value}\nOTHER=value\n")
    completed = subprocess.run(
        ["python", "-", str(env)], input=program, text=True, capture_output=True, check=True
    )
    lines = env.read_text().splitlines()
    keys = [line for line in lines if line.startswith("AIGC_API_KEY=")]
    assert len(keys) == 1
    generated = keys[0].split("=", 1)[1]
    assert len(generated) >= 48
    assert generated not in completed.stdout + completed.stderr
    assert "OTHER=value" in lines


def test_smoke_recovery_instructions_use_the_actual_api_url(monkeypatch, capsys):
    from aigc_photomaker import smoke

    report = smoke.SmokeReport(
        True, [], {}, {}, Path("report.json"), Path("report.md")
    )
    monkeypatch.setattr(smoke, "run_smoke", lambda *args, **kwargs: report)
    monkeypatch.setenv("AIGC_API_KEY", "not-printed")

    assert smoke.main(["--api-url", "http://127.0.0.1:8186"]) == 0
    output = capsys.readouterr().out
    assert "http://127.0.0.1:8186/api/v1/tasks/TASK_ID/cancel" in output
    assert "http://127.0.0.1:8000" not in output
    assert "not-printed" not in output


def test_release_script_and_ui_findings_are_statically_fixed():
    smoke = Path("src/aigc_photomaker/smoke.py").read_text()
    assert "SELECT id AS task_id,prompt_id,status" in smoke
    assert "bash scripts/stop.sh && bash scripts/start.sh" in smoke
    assert "kill -TERM" not in smoke
    status = Path("scripts/status.sh").read_text()
    assert "ss is unavailable" in status
    assert "|| true" not in status
    assert "10 MiB" in Path("web/index.html").read_text()
    installer = Path("scripts/install.sh").read_text()
    assert "token_urlsafe(48)" in installer
