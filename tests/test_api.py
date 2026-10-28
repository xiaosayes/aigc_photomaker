# Server-pending: run these FastAPI/SQLite behavior tests on the GPU server.
import asyncio
import io
import json
import tempfile
import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from aigc_photomaker.app import create_app
from aigc_photomaker.config import Settings
from aigc_photomaker.db import connect
from aigc_photomaker.errors import AppError
from aigc_photomaker.repository import TaskRepository
from aigc_photomaker.task_service import MAX_SEED, TaskService

API_KEY = "a" * 32
AUTH = {"X-API-Key": API_KEY}


class FakeModelManager:
    instances = []

    def __init__(self, *args, **kwargs):
        self.full_hash_calls = []
        self.ok = True
        self.__class__.instances.append(self)

    def verify(self, full_hash, include_optional=False):
        self.full_hash_calls.append(full_hash)
        return [SimpleNamespace(ok=self.ok)]


class FakeComfyUIClient:
    def __init__(self, *args, **kwargs):
        self.available = True

    def system_stats(self):
        if not self.available:
            raise RuntimeError("private ComfyUI connection detail")
        return {}

    def close(self):
        pass


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setattr("aigc_photomaker.app.ModelManager", FakeModelManager)
    monkeypatch.setattr("aigc_photomaker.app.ComfyUIClient", FakeComfyUIClient)
    FakeModelManager.instances.clear()
    return Settings(
        api_key=API_KEY,
        data_dir=tmp_path / "data",
        runtime_dir=tmp_path / "runtime",
        comfy_dir=tmp_path / "comfy",
        max_upload_bytes=1024 * 1024,
        _env_file=None,
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def jpeg_bytes():
    output = io.BytesIO()
    Image.new("RGB", (256, 256), "navy").save(output, format="JPEG")
    return output.getvalue()


def create_task(client, jpeg_bytes, **overrides):
    data = {"style": "cyberpunk", "mode": "standard", "seed": "42"}
    data.update(overrides)
    return client.post(
        "/api/v1/tasks",
        headers=AUTH,
        files={"file": ("face.jpg", jpeg_bytes, "image/jpeg")},
        data=data,
    )


def test_task_and_result_endpoints_require_api_key(client, jpeg_bytes):
    response = client.post(
        "/api/v1/tasks",
        files={"file": ("face.jpg", jpeg_bytes, "image/jpeg")},
        data={"style": "cyberpunk"},
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"
    assert client.get("/api/v1/tasks/missing").status_code == 401
    assert client.get("/api/v1/tasks/missing/result").status_code == 401
    assert client.post("/api/v1/tasks/missing/cancel").status_code == 401


def test_wrong_api_key_receives_unauthorized(client, jpeg_bytes):
    response = client.post(
        "/api/v1/tasks",
        headers={"X-API-Key": "b" * 32},
        files={"file": ("face.jpg", jpeg_bytes, "image/jpeg")},
        data={"style": "cyberpunk", "mode": "standard", "seed": "42"},
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


def test_generation_limit_rejects_first_excess_request_without_artifacts(
    settings, jpeg_bytes
):
    settings.generation_rate_per_minute = 2
    with TestClient(create_app(settings)) as client:
        assert create_task(client, jpeg_bytes).status_code == 202
        assert create_task(client, jpeg_bytes).status_code == 202
        inputs = settings.data_dir / "inputs"
        artifacts_before = set(inputs.iterdir())

        excess = create_task(client, jpeg_bytes)

        assert excess.status_code == 429
        assert excess.json()["error"]["code"] == "RATE_LIMITED"
        assert set(inputs.iterdir()) == artifacts_before
        with connect(settings.data_dir / "tasks.db") as connection:
            count = connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert count == 2


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("style", "unknown", "INVALID_STYLE"),
        ("mode", "fast", "INVALID_MODE"),
        ("seed", "-1", "INVALID_PARAMETER"),
        ("seed", str(MAX_SEED + 1), "INVALID_PARAMETER"),
        ("seed", "not-an-integer", "INVALID_PARAMETER"),
    ],
)
def test_invalid_generation_parameters_fail_before_creating_files(
    client, settings, field, value, code
):
    response = create_task(client, b"not even an image", **{field: value})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == code
    inputs = settings.data_dir / "inputs"
    assert not inputs.exists() or not list(inputs.iterdir())


def test_valid_jpeg_creates_queued_task_and_sanitized_status(
    client, settings, jpeg_bytes
):
    response = create_task(client, jpeg_bytes)

    assert response.status_code == 202
    created = response.json()
    assert created["status"] == "queued"
    assert created["seed"] == 42
    assert created["style"] == "cyberpunk"
    assert created["mode"] == "standard"
    task = TaskRepository(settings.data_dir / "tasks.db").get(created["task_id"])
    assert task.status == "queued"
    assert (settings.data_dir / task.input_path).is_file()

    status = client.get(f"/api/v1/tasks/{task.task_id}", headers=AUTH)
    assert status.status_code == 200
    payload = status.json()
    assert payload["seed"] == 42
    assert payload["style"] == "cyberpunk"
    assert payload["mode"] == "standard"
    assert payload["timings"] == {}
    assert payload["progress"]["phase"] == "queued"
    serialized = json.dumps(payload)
    for forbidden in (
        "claim_token",
        "worker_id",
        "lease_expires_at",
        "input_path",
        "output_path",
        str(settings.data_dir),
    ):
        assert forbidden not in serialized


def test_missing_seed_uses_cryptographic_53_bit_default(client, jpeg_bytes, monkeypatch):
    randbits_calls = []

    def random_seed(bits):
        randbits_calls.append(bits)
        return MAX_SEED

    monkeypatch.setattr("aigc_photomaker.task_service.secrets.randbits", random_seed)
    response = create_task(client, jpeg_bytes, seed="")
    assert response.status_code == 400  # An explicitly empty, non-null seed is invalid.

    response = create_task(client, jpeg_bytes)
    assert response.status_code == 202
    assert response.json()["seed"] == 42

    # Omitting the form field exercises the production default independently.
    response = client.post(
        "/api/v1/tasks",
        headers=AUTH,
        files={"file": ("face.jpg", jpeg_bytes, "image/jpeg")},
        data={"style": "cyberpunk", "mode": "standard"},
    )
    assert response.status_code == 202
    assert response.json()["seed"] == MAX_SEED
    assert randbits_calls == [53]


def test_upload_is_read_in_1mib_chunks_and_rejected_before_normalization(
    settings, monkeypatch
):
    class Upload:
        def __init__(self):
            self.read_sizes = []
            self.chunks = iter((b"123", b"456"))

        async def read(self, size):
            self.read_sizes.append(size)
            return next(self.chunks, b"")

        async def close(self):
            pass

    settings.max_upload_bytes = 5
    upload = Upload()
    service = TaskService(
        settings,
        TaskRepository(settings.data_dir / "tasks.db"),
        {"cyberpunk": SimpleNamespace()},
    )
    monkeypatch.setattr(
        "aigc_photomaker.task_service.normalize_image",
        lambda *args: pytest.fail("oversized upload reached image normalization"),
    )

    with pytest.raises(AppError) as raised:
        asyncio.run(service.create(upload, "cyberpunk", "standard", 1))

    assert raised.value.code == "IMAGE_TOO_LARGE"
    assert upload.read_sizes == [1024 * 1024, 1024 * 1024]
    assert not (settings.data_dir / "inputs").exists() or not list(
        (settings.data_dir / "inputs").iterdir()
    )


def test_temporary_upload_filesystem_operations_run_off_event_loop_thread(
    settings, monkeypatch
):
    event_loop_thread = threading.get_ident()
    operations = []
    read_threads = []
    real_named_temporary_file = tempfile.NamedTemporaryFile

    class RecordingTemporary:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.name = wrapped.name

        def write(self, chunk):
            operations.append(("write", threading.get_ident()))
            return self.wrapped.write(chunk)

        def close(self):
            operations.append(("close", threading.get_ident()))
            return self.wrapped.close()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.close()

    def named_temporary_file(*args, **kwargs):
        operations.append(("create", threading.get_ident()))
        return RecordingTemporary(real_named_temporary_file(*args, **kwargs))

    class Upload:
        def __init__(self):
            self.chunks = iter((b"first", b"second", b""))

        async def read(self, size):
            assert size == 1024 * 1024
            read_threads.append(threading.get_ident())
            return next(self.chunks)

    monkeypatch.setattr(
        "aigc_photomaker.task_service.tempfile.NamedTemporaryFile",
        named_temporary_file,
    )

    def fail_normalization(*args, **kwargs):
        raise AppError(400, "INVALID_IMAGE", "stop after upload", False)

    monkeypatch.setattr(
        "aigc_photomaker.task_service.normalize_image", fail_normalization
    )
    service = TaskService(
        settings,
        TaskRepository(settings.data_dir / "tasks.db"),
        {"cyberpunk": SimpleNamespace()},
    )

    with pytest.raises(AppError):
        asyncio.run(service.create(Upload(), "cyberpunk", "standard", 1))

    assert read_threads == [event_loop_thread] * 3
    assert [name for name, _ in operations].count("write") == 2
    assert "close" in [name for name, _ in operations]
    assert operations[0][0] == "create"
    assert all(thread_id != event_loop_thread for _, thread_id in operations)


def test_multipart_oversize_returns_structured_error_without_artifacts(
    settings, monkeypatch
):
    settings.max_upload_bytes = 64
    monkeypatch.setattr(
        "aigc_photomaker.task_service.normalize_image",
        lambda *args, **kwargs: pytest.fail("oversized upload reached normalization"),
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/tasks",
            headers=AUTH,
            files={"file": ("face.jpg", b"x" * 65, "image/jpeg")},
            data={"style": "cyberpunk", "mode": "standard", "seed": "42"},
        )

        assert response.status_code == 413
        assert response.json() == {
            "error": {
                "code": "IMAGE_TOO_LARGE",
                "message": "图片文件超过大小限制",
                "retryable": False,
            },
            "task_id": None,
        }
        inputs = settings.data_dir / "inputs"
        assert not inputs.exists() or not list(inputs.iterdir())
        with connect(settings.data_dir / "tasks.db") as connection:
            count = connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert count == 0


def test_failed_normalization_removes_temporary_and_normalized_files(
    client, settings, jpeg_bytes, monkeypatch
):
    def fail_normalization(source, destination, **kwargs):
        source.write_bytes(b"changed")
        destination.write_bytes(b"partial")
        raise AppError(400, "INVALID_IMAGE", "图片无法解码", False)

    monkeypatch.setattr(
        "aigc_photomaker.task_service.normalize_image", fail_normalization
    )
    response = create_task(client, jpeg_bytes)

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_IMAGE"
    assert list((settings.data_dir / "inputs").iterdir()) == []


def test_result_is_conflict_until_successful_then_requires_safe_existing_file(
    client, settings, jpeg_bytes
):
    task_id = create_task(client, jpeg_bytes).json()["task_id"]
    result_url = f"/api/v1/tasks/{task_id}/result"

    assert client.get(result_url).status_code == 401
    pending = client.get(result_url, headers=AUTH)
    assert pending.status_code == 409
    assert pending.json()["error"]["code"] == "RESULT_NOT_READY"

    repository = TaskRepository(settings.data_dir / "tasks.db")
    claimed = repository.claim("private-worker", 60)
    output = settings.data_dir / claimed.output_path
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(b"png-result")
    assert repository.succeed(claimed.task_id, claimed.claim_token, {"total": 1.0})

    result = client.get(result_url, headers=AUTH)
    assert result.status_code == 200
    assert result.content == b"png-result"

    output.unlink()
    missing = client.get(result_url, headers=AUTH)
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "RESULT_MISSING"
    assert str(settings.data_dir) not in missing.text


def test_result_rejects_path_outside_data_directory(client, settings, jpeg_bytes):
    task_id = create_task(client, jpeg_bytes).json()["task_id"]
    repository = TaskRepository(settings.data_dir / "tasks.db")
    claimed = repository.claim("worker", 60)
    assert repository.succeed(claimed.task_id, claimed.claim_token, {})
    with connect(settings.data_dir / "tasks.db") as connection:
        with connection:
            connection.execute(
                "UPDATE tasks SET output_path=? WHERE id=?", ("../outside.png", task_id)
            )

    response = client.get(f"/api/v1/tasks/{task_id}/result", headers=AUTH)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_TASK_PATH"


def test_unknown_task_and_queued_cancellation(client, settings, jpeg_bytes):
    unknown = client.get("/api/v1/tasks/not-there", headers=AUTH)
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "TASK_NOT_FOUND"
    assert unknown.json()["task_id"] == "not-there"

    task_id = create_task(client, jpeg_bytes).json()["task_id"]
    cancelled = client.post(f"/api/v1/tasks/{task_id}/cancel", headers=AUTH)
    assert cancelled.status_code == 200
    assert cancelled.json()["status"] == "cancelled"
    assert TaskRepository(settings.data_dir / "tasks.db").get(task_id).status == "cancelled"


def write_ready_files(settings):
    runtime = settings.runtime_dir
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "warmup.json").write_text(
        json.dumps(
            {
                "status": "succeeded",
                "completed_at": time.time(),
                "modes": {"standard": {}, "turbo": {}},
            }
        ),
        encoding="utf-8",
    )
    (runtime / "worker-heartbeat.json").write_text(
        json.dumps({"worker_id": "private", "heartbeat_at": time.time()}),
        encoding="utf-8",
    )


def test_readiness_requires_successful_warmup_and_fresh_worker(client, settings):
    assert client.get("/api/v1/health/ready").status_code == 503

    runtime = settings.runtime_dir
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "warmup.json").write_text(
        json.dumps(
            {"status": "succeeded", "modes": {"standard": {}, "turbo": {}}}
        ),
        encoding="utf-8",
    )
    missing_heartbeat = client.get("/api/v1/health/ready")
    assert missing_heartbeat.status_code == 503
    assert missing_heartbeat.json()["checks"]["warmup"] is True
    assert missing_heartbeat.json()["checks"]["worker"] is False

    write_ready_files(settings)
    ready = client.get("/api/v1/health/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert all(ready.json()["checks"].values())
    assert FakeModelManager.instances[-1].full_hash_calls[-1] is False

    heartbeat = json.loads((runtime / "worker-heartbeat.json").read_text())
    heartbeat["heartbeat_at"] = time.time() - settings.worker_lease_seconds - 1
    (runtime / "worker-heartbeat.json").write_text(json.dumps(heartbeat))
    assert client.get("/api/v1/health/ready").status_code == 503


def test_live_styles_static_hosting_and_unified_internal_error(
    settings, jpeg_bytes, monkeypatch
):
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        live = client.get("/api/v1/health/live")
        assert live.status_code == 200
        assert API_KEY not in live.text
        assert "comfy" not in live.text.lower()

        styles = client.get("/api/v1/styles")
        assert styles.status_code == 200
        assert "cyberpunk" in styles.text
        assert "standard" in styles.text
        assert client.get("/").status_code == 200

        task_id = create_task(client, jpeg_bytes).json()["task_id"]
        monkeypatch.setattr(
            client.app.state.task_service,
            "get_status",
            lambda task_id: (_ for _ in ()).throw(RuntimeError("secret stack detail")),
        )
        failed = client.get(f"/api/v1/tasks/{task_id}", headers=AUTH)
        assert failed.status_code == 500
        assert failed.json()["error"]["code"] == "INTERNAL_ERROR"
        assert "secret stack detail" not in failed.text
        assert "Traceback" not in failed.text
