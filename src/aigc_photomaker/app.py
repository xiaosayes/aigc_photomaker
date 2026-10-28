import asyncio
import hashlib
import json
import logging
import shutil
import time

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .comfy import ComfyUIClient
from .config import PROJECT_ROOT, Settings, get_settings
from .errors import AppError, disk_space_error
from .model_manager import ModelManager
from .repository import TaskRepository
from .security import GenerationRateLimiter, verify_api_key
from .styles import load_styles
from .task_service import TaskService

LOGGER = logging.getLogger(__name__)


class RawBodyLimitMiddleware:
    """Bound multipart request bodies before Starlette can spool them to disk."""

    def __init__(self, app, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    @staticmethod
    async def _too_large(send) -> None:
        error = AppError(413, "IMAGE_TOO_LARGE", "图片文件超过大小限制", False)
        body = json.dumps(
            {"error": error.to_dict(), "task_id": None}, ensure_ascii=False
        ).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": 413,
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"connection", b"close"),
            ],
        })
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method", "").upper() != "POST"
            or scope.get("path") != "/api/v1/tasks"
        ):
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = -1
            if declared > self.max_body_bytes:
                await self._too_large(send)
                return

        # Replaying a bounded raw body preserves normal multipart parsing while also
        # covering chunked requests and clients whose Content-Length is false.
        messages = []
        total = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            body = message.get("body", b"")
            total += len(body)
            if total > self.max_body_bytes:
                await self._too_large(send)
                return
            messages.append({
                "type": "http.request",
                "body": body,
                "more_body": bool(message.get("more_body", False)),
            })
            if not message.get("more_body", False):
                break
        position = 0

        async def replay():
            nonlocal position
            if position < len(messages):
                message = messages[position]
                position += 1
                return message
            return {"type": "http.disconnect"}

        await self.app(scope, replay, send)


def require_api_key(
    request: Request, settings: Settings = Depends(get_settings)
) -> None:
    if not verify_api_key(request.headers.get("X-API-Key", ""), settings.api_key):
        raise AppError(401, "UNAUTHORIZED", "API Key 无效", False)


def client_key(request: Request) -> str:
    raw_key = request.headers.get("X-API-Key", "").encode()
    fingerprint = hashlib.sha256(raw_key).hexdigest()[:16]
    host = request.client.host if request.client is not None else "unknown"
    return f"{host}:{fingerprint}"


def _worker_ready(settings: Settings) -> bool:
    path = settings.resolve(settings.runtime_dir) / "worker-heartbeat.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        heartbeat_at = report["heartbeat_at"]
        if isinstance(heartbeat_at, bool) or not isinstance(heartbeat_at, (int, float)):
            return False
        age = time.time() - heartbeat_at
        return 0 <= age <= settings.worker_lease_seconds
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _warmup_ready(settings: Settings) -> bool:
    path = settings.resolve(settings.runtime_dir) / "warmup.json"
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        modes = report.get("modes")
        return (
            report.get("status") == "succeeded"
            and isinstance(modes, dict)
            and {"standard", "turbo"}.issubset(modes)
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def _disk_ready(settings: Settings) -> bool:
    data_dir = settings.resolve(settings.data_dir)
    return shutil.disk_usage(data_dir).free >= settings.min_free_disk_bytes


def _models_ready(manager: ModelManager) -> bool:
    # Full SHA256 verification belongs to installation/startup, not a health poll.
    return all(status.ok for status in manager.verify(full_hash=False))


def _comfy_ready(client: ComfyUIClient) -> bool:
    client.system_stats()
    return True


def _task_id(request: Request) -> str | None:
    value = request.path_params.get("task_id")
    return value if isinstance(value, str) else None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="AIGC PhotoMaker")
    app.dependency_overrides[get_settings] = lambda: settings

    data_dir = settings.resolve(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    repository = TaskRepository(data_dir / "tasks.db")
    repository.create_schema()
    styles = load_styles(PROJECT_ROOT / "config" / "styles.json")
    task_service = TaskService(settings, repository, styles)
    app.add_middleware(
        RawBodyLimitMiddleware,
        max_body_bytes=settings.max_upload_bytes + settings.multipart_overhead_bytes,
    )
    generation_limiter = GenerationRateLimiter(settings.generation_rate_per_minute)
    model_manager = ModelManager(
        PROJECT_ROOT / "config" / "models.lock.json",
        settings.resolve(settings.comfy_dir) / "models",
    )
    comfy = ComfyUIClient(settings.comfy_url)
    app.state.task_service = task_service
    app.state.repository = repository
    app.state.model_manager = model_manager
    app.state.comfy = comfy
    app.state.generation_limiter = generation_limiter
    app.add_event_handler("shutdown", comfy.close)

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.to_dict(), "task_id": _task_id(request)},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        error = AppError(400, "INVALID_PARAMETER", "请求参数无效", False)
        return JSONResponse(
            status_code=error.status_code,
            content={"error": error.to_dict(), "task_id": _task_id(request)},
        )

    @app.exception_handler(Exception)
    async def handle_internal_error(request: Request, exc: Exception) -> JSONResponse:
        LOGGER.error("Unhandled API error for task %s", _task_id(request), exc_info=True)
        error = disk_space_error(exc) or AppError(
            500, "INTERNAL_ERROR", "服务器内部错误", False
        )
        return JSONResponse(
            status_code=error.status_code,
            content={"error": error.to_dict(), "task_id": _task_id(request)},
        )

    @app.get("/api/v1/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @app.get("/api/v1/health/ready")
    async def ready() -> JSONResponse:
        checks: dict[str, bool] = {}
        for name, check in (
            ("database", repository.ping),
            ("disk", lambda: _disk_ready(settings)),
            ("models", lambda: _models_ready(model_manager)),
            ("comfyui", lambda: _comfy_ready(comfy)),
            ("worker", lambda: _worker_ready(settings)),
            ("warmup", lambda: _warmup_ready(settings)),
        ):
            try:
                result = await asyncio.to_thread(check)
                checks[name] = bool(result)
            except Exception:
                checks[name] = False
        is_ready = all(checks.values())
        return JSONResponse(
            status_code=200 if is_ready else 503,
            content={"status": "ready" if is_ready else "not_ready", "checks": checks},
        )

    @app.get("/api/v1/styles")
    async def get_styles() -> dict:
        return {
            "styles": [
                {
                    "key": style.key,
                    "label": style.label,
                    "description": style.description,
                }
                for style in styles.values()
            ],
            "modes": {
                "standard": {
                    "steps": settings.standard_steps,
                    "cfg": settings.standard_cfg,
                },
                "turbo": {
                    "steps": settings.turbo_steps,
                    "cfg": settings.turbo_cfg,
                },
            },
        }

    @app.post("/api/v1/tasks", status_code=202, dependencies=[Depends(require_api_key)])
    async def create_task(
        request: Request,
        file: UploadFile = File(...),
        style: str = Form(...),
        mode: str = Form("standard"),
        seed: str | None = Form(None),
    ) -> dict:
        generation_limiter.allow_or_raise(client_key(request), time.time())
        return await task_service.create(file, style, mode, seed)

    @app.get("/api/v1/tasks/{task_id}", dependencies=[Depends(require_api_key)])
    async def get_task(task_id: str) -> dict:
        return await asyncio.to_thread(task_service.get_status, task_id)

    @app.get("/api/v1/tasks/{task_id}/result", dependencies=[Depends(require_api_key)])
    async def get_task_result(task_id: str) -> FileResponse:
        result = await asyncio.to_thread(task_service.get_result, task_id)
        return FileResponse(result, media_type="image/png", filename=f"{task_id}.png")

    @app.post("/api/v1/tasks/{task_id}/cancel", dependencies=[Depends(require_api_key)])
    async def cancel_task(task_id: str) -> dict:
        return await asyncio.to_thread(task_service.cancel, task_id)

    app.mount(
        "/",
        StaticFiles(directory=PROJECT_ROOT / "web", html=True),
        name="web",
    )
    return app
