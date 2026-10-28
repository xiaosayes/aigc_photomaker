import asyncio
import secrets
import tempfile
import uuid
from collections.abc import Mapping
from pathlib import Path

from fastapi import UploadFile

from .config import Settings
from .errors import AppError
from .images import normalize_image
from .models import TaskRecord
from .repository import TaskRepository
from .styles import StylePreset

UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_SEED = 2**53 - 1
_VALID_MODES = {"standard", "turbo"}


def _invalid_parameter(message: str) -> AppError:
    return AppError(400, "INVALID_PARAMETER", message, False)


def _task_not_found(task_id: str) -> AppError:
    return AppError(404, "TASK_NOT_FOUND", "任务不存在", False)


class TaskService:
    def __init__(
        self,
        settings: Settings,
        repository: TaskRepository,
        styles: Mapping[str, StylePreset],
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.styles = styles

    def _validate_style(self, style: str) -> None:
        if not isinstance(style, str) or style not in self.styles:
            raise AppError(400, "INVALID_STYLE", "风格不存在", False)

    @staticmethod
    def _validate_mode(mode: str) -> None:
        if mode not in _VALID_MODES:
            raise AppError(400, "INVALID_MODE", "生成模式仅支持 standard 或 turbo", False)

    @staticmethod
    def _validate_seed(seed: int | str | None) -> int:
        if seed is None:
            return secrets.randbits(53)
        if isinstance(seed, bool):
            raise _invalid_parameter("seed 必须是 0 到 2^53-1 的整数")
        try:
            value = int(seed) if isinstance(seed, str) else seed
        except (TypeError, ValueError) as exc:
            raise _invalid_parameter("seed 必须是 0 到 2^53-1 的整数") from exc
        if not isinstance(value, int) or not 0 <= value <= MAX_SEED:
            raise _invalid_parameter("seed 必须是 0 到 2^53-1 的整数")
        return value

    @staticmethod
    def _remove_files(paths: tuple[Path | None, ...]) -> None:
        for path in paths:
            if path is None:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    async def create(
        self,
        file: UploadFile,
        style: str,
        mode: str,
        seed: int | str | None,
    ) -> dict:
        # Reject invalid parameters before consuming or writing any uploaded content.
        self._validate_style(style)
        self._validate_mode(mode)
        validated_seed = self._validate_seed(seed)

        task_id = str(uuid.uuid4())
        data_dir = self.settings.resolve(self.settings.data_dir)
        inputs_dir = data_dir / "inputs"
        await asyncio.to_thread(inputs_dir.mkdir, parents=True, exist_ok=True)
        source: Path | None = None
        destination = inputs_dir / f"{task_id}.png"
        normalization_staging = destination.with_suffix(".part")
        temporary = None
        try:
            temporary = await asyncio.to_thread(
                tempfile.NamedTemporaryFile,
                mode="wb",
                dir=inputs_dir,
                prefix=f"{task_id}.",
                suffix=".upload",
                delete=False,
            )
            source = Path(temporary.name)
            try:
                size = 0
                while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                    size += len(chunk)
                    if size > self.settings.max_upload_bytes:
                        raise AppError(
                            413,
                            "IMAGE_TOO_LARGE",
                            "图片文件超过大小限制",
                            False,
                        )
                    await asyncio.to_thread(temporary.write, chunk)
            finally:
                await asyncio.to_thread(temporary.close)

            await asyncio.to_thread(
                normalize_image,
                source,
                destination,
                max_pixels=self.settings.max_image_pixels,
                max_side=self.settings.max_input_side,
            )
            task = await asyncio.to_thread(
                self.repository.create,
                task_id,
                f"inputs/{task_id}.png",
                style,
                mode,
                validated_seed,
            )
            return self._status(task)
        except BaseException:
            await asyncio.to_thread(
                self._remove_files, (source, destination, normalization_staging)
            )
            raise
        finally:
            await asyncio.to_thread(self._remove_files, (source,))

    def _get(self, task_id: str) -> TaskRecord:
        task = self.repository.get(task_id)
        if task is None:
            raise _task_not_found(task_id)
        return task

    @staticmethod
    def _status(task: TaskRecord) -> dict:
        return {
            "task_id": task.task_id,
            "status": task.status,
            "style": task.style,
            "mode": task.mode,
            "seed": task.seed,
            "progress": {
                "phase": task.progress_phase,
                "value": task.progress_value,
                "max": task.progress_max,
            },
            "cancel_requested": task.cancel_requested,
            "timings": task.timings,
            "error": task.error,
            "created_at": task.created_at,
            "started_at": task.started_at,
            "finished_at": task.finished_at,
        }

    def get_status(self, task_id: str) -> dict:
        return self._status(self._get(task_id))

    def get_result(self, task_id: str) -> Path:
        task = self._get(task_id)
        if task.status != "succeeded":
            raise AppError(409, "RESULT_NOT_READY", "任务尚未成功完成", True)

        root = self.settings.resolve(self.settings.data_dir).resolve()
        try:
            result = (root / task.output_path).resolve()
            result.relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise AppError(
                400, "INVALID_TASK_PATH", "任务结果路径无效", False
            ) from exc
        if not result.is_file():
            raise AppError(404, "RESULT_MISSING", "任务结果文件不存在", True)
        return result

    def cancel(self, task_id: str) -> dict:
        self._get(task_id)
        task = self.repository.request_cancel(task_id)
        if task is None:  # Defensive against deletion between the two short transactions.
            raise _task_not_found(task_id)
        return self._status(task)
