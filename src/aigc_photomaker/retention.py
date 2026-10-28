import logging
import threading
import time
from pathlib import Path, PurePosixPath, PureWindowsPath

from .config import Settings
from .models import TaskRecord
from .repository import TaskRepository


RETENTION_INTERVAL_SECONDS = 3600.0
LOGGER = logging.getLogger(__name__)


def _safe_relative(value: str, *, allow_empty: bool = False) -> PurePosixPath:
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        (not value and not allow_empty)
        or "\\" in value
        or windows_path.drive
        or windows_path.root
        or posix_path.is_absolute()
        or ".." in posix_path.parts
        or "." in posix_path.parts
    ):
        raise ValueError("Unsafe relative retention path")
    return posix_path


class RetentionCleaner:
    def __init__(self, settings: Settings, repository: TaskRepository) -> None:
        self.settings = settings
        self.repository = repository
        self._stop = threading.Event()
        self.pass_count = 0

    def stop(self) -> None:
        self._stop.set()

    def _path(self, relative: str) -> Path:
        root = self.settings.resolve(self.settings.data_dir).resolve()
        try:
            _safe_relative(relative)
            target = (root / relative).resolve()
            target.relative_to(root)
        except (OSError, ValueError) as exc:
            raise ValueError("Retention path is outside the data directory") from exc
        return target

    def _task_paths(self, task: TaskRecord) -> list[Path]:
        # Validate the complete task before unlinking anything, including sidecars.
        return [
            self._path(relative)
            for original in (task.input_path, task.output_path)
            for relative in (original, f"{original}.part")
        ]

    def _comfy_path(self, ref: dict[str, str]) -> Path:
        if set(ref) != {"name", "subfolder", "type"} or ref["type"] not in {
            "input", "output", "temp"
        }:
            raise ValueError("Invalid ComfyUI retention reference")
        name = ref["name"]
        subfolder = ref["subfolder"]
        if (
            not isinstance(name, str)
            or not isinstance(subfolder, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
        ):
            raise ValueError("Invalid ComfyUI retention filename")
        _safe_relative(subfolder, allow_empty=True)
        root = (self.settings.resolve(self.settings.comfy_dir) / {
            "input": "input", "output": "output", "temp": "temp"
        }[ref["type"]]).resolve()
        target = (root / subfolder / name).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError("ComfyUI retention path escapes its type root") from exc
        return target

    @staticmethod
    def _prefixed_name(name: str, stem: str) -> bool:
        # The separators prevent a short task ID from matching another task's files.
        return name == stem or name.startswith(f"{stem}_") or name.startswith(f"{stem}.")

    def _prefixed_outputs(self, task: TaskRecord, image_type: str = "output") -> list[Path]:
        prefix_value = task.output_prefix or f"aigc-photomaker/{task.task_id}"
        prefix = _safe_relative(prefix_value)
        if not prefix.name:
            raise ValueError("Invalid ComfyUI output prefix")
        root = (self.settings.resolve(self.settings.comfy_dir) / image_type).resolve()
        directory = (root / prefix.parent).resolve()
        try:
            directory.relative_to(root)
        except ValueError as exc:
            raise ValueError("ComfyUI output prefix escapes its type root") from exc
        if not directory.exists():
            return []
        return [
            child
            for child in directory.iterdir()
            if self._prefixed_name(child.name, prefix.name)
        ]

    def _comfy_paths(self, task: TaskRecord) -> list[Path]:
        paths: list[Path] = []
        for ref in task.execution_refs:
            target = self._comfy_path(ref)
            if ref["type"] != "input":
                prefix = _safe_relative(task.output_prefix or f"aigc-photomaker/{task.task_id}")
                if PurePosixPath(ref["subfolder"]) != prefix.parent or not self._prefixed_name(
                    ref["name"], prefix.name
                ):
                    raise ValueError("ComfyUI output reference belongs to another task")
            paths.append(target)
        paths.extend(self._prefixed_outputs(task))
        # Failed tasks can have partial histories, so retained temp references are also exact,
        # validated task-prefixed paths rather than arbitrary server-provided locations.
        for ref in task.execution_refs:
            if ref["type"] == "temp":
                paths.append(self._comfy_path(ref))
        return list(dict.fromkeys(paths))

    def cleanup(self, now: float) -> int:
        cutoff = now - self.settings.retention_days * 86400
        expired = self.repository.list_expired_terminal(cutoff)
        removed = 0
        for task in expired:
            try:
                paths = self._task_paths(task) + self._comfy_paths(task)
                failed = False
                for target in paths:
                    try:
                        target.unlink(missing_ok=True)
                    except OSError:
                        failed = True
                        LOGGER.exception("Retention file removal failed for task %s", task.task_id)
                if not failed and self.repository.delete_terminal(task.task_id, cutoff):
                    removed += 1
            except Exception:
                LOGGER.exception("Retention cleanup failed for task %s", task.task_id)
        return removed

    def run_forever(self) -> None:
        while not self._stop.is_set():
            try:
                self.cleanup(time.time())
            except Exception:
                LOGGER.exception("Retention cleanup pass failed")
            finally:
                self.pass_count += 1
            self._stop.wait(RETENTION_INTERVAL_SECONDS)
