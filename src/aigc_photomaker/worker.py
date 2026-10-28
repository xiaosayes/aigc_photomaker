import json
import logging
import os
import signal
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from .comfy import ComfyUIClient
from .config import PROJECT_ROOT, Settings, get_settings
from .errors import AppError, disk_space_error
from .logging_config import configure_logging
from .models import TaskRecord
from .repository import TaskRepository
from .retention import RetentionCleaner
from .styles import StylePreset, load_styles
from .workflow import ComfyImageRef, prepare_workflow


POLL_INTERVAL_SECONDS = 0.5
HEARTBEAT_INTERVAL_SECONDS = 2.0
LOGGER = logging.getLogger(__name__)


class _ClaimLost(Exception):
    pass


def _claim_lost() -> AppError:
    # Internal control flow only: fenced workers must not persist this as a task failure.
    return AppError(409, "WORKER_CLAIM_LOST", "Worker lease was reclaimed", True)


def _data_path(settings: Settings, relative: str) -> Path:
    root = settings.resolve(settings.data_dir).resolve()
    try:
        target = (root / relative).resolve()
        target.relative_to(root)
    except (OSError, ValueError) as exc:
        raise AppError(400, "INVALID_TASK_PATH", "Task path is outside the data directory", False) from exc
    return target


class TaskProcessor:
    def __init__(
        self,
        settings: Settings,
        repository: TaskRepository,
        comfy: ComfyUIClient,
        styles: Mapping[str, StylePreset],
        worker_id: str = "worker",
        on_heartbeat: Callable[[], None] | None = None,
        stop_requested: Callable[[], bool] | None = None,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.comfy = comfy
        self.styles = styles
        self.worker_id = worker_id
        self.on_heartbeat = on_heartbeat or (lambda: None)
        self.stop_requested = stop_requested or (lambda: False)

    def _cancel_quietly(self, prompt_id: str | None) -> None:
        if prompt_id:
            try:
                self.comfy.cancel(prompt_id)
            except Exception:
                # Cancellation remains terminal even if ComfyUI has already finished or stopped.
                pass

    def _checkpoint(self, task: TaskRecord, claim_token: str) -> None:
        current = self.repository.get(task.task_id)
        if (
            current is None
            or current.status != "running"
            or current.claim_token != claim_token
        ):
            raise _ClaimLost()
        if self.stop_requested() or current.cancel_requested:
            raise AppError(409, "CANCELLED", "Generation was cancelled", False)

    def _finish_cancellation(
        self, task: TaskRecord, claim_token: str, prompt_id: str | None
    ) -> None:
        self._cancel_quietly(prompt_id)
        self.repository.mark_cancelled(task.task_id, claim_token)

    def _finish_error(
        self,
        task: TaskRecord,
        claim_token: str,
        prompt_id: str | None,
        error: dict[str, object],
        cancellation: bool = False,
    ) -> None:
        current = self.repository.get(task.task_id)
        if (
            current is None
            or current.status != "running"
            or current.claim_token != claim_token
        ):
            return
        if cancellation or self.stop_requested() or current.cancel_requested:
            self._finish_cancellation(task, claim_token, prompt_id)
            return
        if self.repository.fail(task.task_id, claim_token, error):
            return
        # fail() is cancellation-guarded; a racing request must still finish as cancelled.
        latest = self.repository.get(task.task_id)
        if (
            latest is not None
            and latest.status == "running"
            and latest.claim_token == claim_token
            and (self.stop_requested() or latest.cancel_requested)
        ):
            self._finish_cancellation(task, claim_token, prompt_id)

    @staticmethod
    def _submission_uncertain() -> AppError:
        return AppError(
            503,
            "SUBMISSION_UNCERTAIN",
            "ComfyUI submission outcome is uncertain; manual review is required",
            False,
        )

    def _persist_prompt_id(
        self, task: TaskRecord, claim_token: str, prompt_id: str
    ) -> None:
        try:
            persisted = self.repository.mark_submitted(task.task_id, claim_token, prompt_id)
        except Exception as exc:
            self._cancel_quietly(prompt_id)
            disk_error = disk_space_error(exc)
            if disk_error is not None:
                raise disk_error from exc
            raise AppError(
                500,
                "SUBMISSION_PERSISTENCE_FAILED",
                "ComfyUI prompt ID persistence failed; the prompt was cancelled",
                False,
            ) from exc
        if not persisted:
            self._cancel_quietly(prompt_id)
            raise _ClaimLost()

    def _reconcile_submission(
        self, client_id: str, output_prefix: str
    ) -> str | None:
        return self.comfy.find_submission(client_id, output_prefix)

    def process(self, task: TaskRecord) -> None:
        claim_token = task.claim_token
        if claim_token is None:
            raise ValueError("A claimed task must have a claim token")
        prompt_id = task.prompt_id
        started = time.monotonic()
        timings: dict[str, float] = {}
        staging: Path | None = None
        try:
            self._checkpoint(task, claim_token)
            if prompt_id is None:
                output_prefix = task.output_prefix or f"aigc-photomaker/{task.task_id}"
                client_id = task.submission_client_id or f"aigc-photomaker-{task.task_id}"
                recovering = task.submission_client_id is not None
                self._checkpoint(task, claim_token)
                if recovering:
                    try:
                        prompt_id = self._reconcile_submission(client_id, output_prefix)
                    except Exception as exc:
                        raise self._submission_uncertain() from exc
                    if prompt_id is None:
                        raise self._submission_uncertain()
                    self._persist_prompt_id(task, claim_token, prompt_id)
                elif not self.repository.prepare_submission(
                    task.task_id, claim_token, client_id, output_prefix
                ):
                    raise _ClaimLost()

                if prompt_id is None:
                    image_ref = next(
                        (
                            ref
                            for ref in task.execution_refs
                            if ref.get("type") == "input"
                        ),
                        None,
                    )
                    if image_ref is None:
                        stage_started = time.monotonic()
                        uploaded = self.comfy.upload_image(
                            _data_path(self.settings, task.input_path)
                        )
                        timings["upload"] = time.monotonic() - stage_started
                        image_ref = {
                            "name": uploaded.name,
                            "subfolder": uploaded.subfolder,
                            "type": uploaded.type,
                        }
                        if not self.repository.record_execution_ref(
                            task.task_id, claim_token, image_ref
                        ):
                            raise _ClaimLost()
                    self._checkpoint(task, claim_token)
                    graph = prepare_workflow(
                        task.mode,
                        ComfyImageRef(**image_ref),
                        self.styles[task.style],
                        task.seed,
                        output_prefix,
                        self.settings,
                    )
                    self._checkpoint(task, claim_token)
                    stage_started = time.monotonic()
                    try:
                        prompt_id = self.comfy.submit(graph, client_id)
                    except AppError as exc:
                        if exc.code != "COMFYUI_UNAVAILABLE":
                            raise
                        try:
                            prompt_id = self._reconcile_submission(client_id, output_prefix)
                        except Exception as reconcile_exc:
                            raise self._submission_uncertain() from reconcile_exc
                        if prompt_id is None:
                            raise self._submission_uncertain() from exc
                    timings["submit"] = time.monotonic() - stage_started
                    self._persist_prompt_id(task, claim_token, prompt_id)

            def heartbeat() -> None:
                if not self.repository.heartbeat(
                    task.task_id, claim_token, self.settings.worker_lease_seconds
                ):
                    raise _claim_lost()
                self.on_heartbeat()

            def cancel_requested() -> bool:
                if self.stop_requested():
                    return True
                current = self.repository.get(task.task_id)
                return bool(current and current.cancel_requested)

            def progress(value: float, maximum: float) -> None:
                if not self.repository.update_progress(
                    task.task_id,
                    claim_token,
                    "executing",
                    int(value),
                    int(maximum),
                ):
                    raise _ClaimLost()

            stage_started = time.monotonic()
            result = self.comfy.wait_for_result(
                prompt_id,
                self.settings.execution_timeout_seconds,
                heartbeat,
                cancel_requested,
                progress,
            )
            timings["execute"] = time.monotonic() - stage_started
            self._checkpoint(task, claim_token)
            for image in result.images:
                if not self.repository.record_execution_ref(
                    task.task_id,
                    claim_token,
                    {"name": image.name, "subfolder": image.subfolder, "type": image.type},
                ):
                    raise _ClaimLost()
            if not result.images:
                raise AppError(500, "RESULT_MISSING", "ComfyUI 未返回输出图片", True)

            output = _data_path(self.settings, task.output_path)
            staging = output.with_name(f"{output.name}.{claim_token}.part")
            self._checkpoint(task, claim_token)
            stage_started = time.monotonic()
            self.comfy.download_image(result.images[0], staging)
            timings["download"] = time.monotonic() - stage_started
            self._checkpoint(task, claim_token)
            # The final ownership check is deliberately adjacent to the only replacement.
            staging.replace(output)
            staging = None
            timings["total"] = time.monotonic() - started

            self._checkpoint(task, claim_token)
            if not self.repository.succeed(task.task_id, claim_token, timings):
                latest = self.repository.get(task.task_id)
                if (
                    latest is not None
                    and latest.status == "running"
                    and latest.claim_token == claim_token
                    and (self.stop_requested() or latest.cancel_requested)
                ):
                    self._finish_cancellation(task, claim_token, prompt_id)
        except _ClaimLost:
            return
        except AppError as exc:
            if exc.code == "WORKER_CLAIM_LOST":
                return
            self._finish_error(
                task,
                claim_token,
                prompt_id,
                exc.to_dict(),
                cancellation=exc.code == "CANCELLED",
            )
        except Exception as exc:
            mapped = disk_space_error(exc)
            error = mapped.to_dict() if mapped is not None else {
                "code": "INTERNAL_ERROR",
                "message": "Worker execution failed",
                "retryable": False,
            }
            self._finish_error(task, claim_token, prompt_id, error)
        finally:
            if staging is not None:
                for temporary in (staging, Path(f"{staging}.part")):
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        LOGGER.warning("Unable to remove stale staging output", exc_info=True)


class _ActiveTaskSupervision:
    def __init__(self, worker: "Worker", task: TaskRecord) -> None:
        self.worker = worker
        self.task = task
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"heartbeat-{task.task_id}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if not self.worker.repository.heartbeat(
                    self.task.task_id,
                    self.task.claim_token,
                    self.worker.settings.worker_lease_seconds,
                ):
                    return
                self.worker._heartbeat(force=True)
            except Exception:
                LOGGER.exception("Active task heartbeat failed")
            if self._stop.wait(HEARTBEAT_INTERVAL_SECONDS):
                return


class Worker:
    def __init__(
        self,
        settings: Settings,
        repository: TaskRepository,
        processor: TaskProcessor,
        worker_id: str = "worker",
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.processor = processor
        self.worker_id = worker_id
        self._stop = threading.Event()
        self._last_heartbeat = float("-inf")
        self._heartbeat_lock = threading.Lock()
        self._current_task_id: str | None = None
        self.retention_cleaner = RetentionCleaner(settings, repository)
        self._retention_thread: threading.Thread | None = None

    def stop(self) -> None:
        self._stop.set()
        self.retention_cleaner.stop()

    def _write_heartbeat(self) -> None:
        runtime_dir = self.settings.resolve(self.settings.runtime_dir)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        destination = runtime_dir / "worker-heartbeat.json"
        temporary_name = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=runtime_dir,
                prefix=f"{destination.name}.",
                suffix=".part",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                json.dump(
                    {
                        "worker_id": self.worker_id,
                        "heartbeat_at": time.time(),
                        "task_id": self._current_task_id,
                    },
                    temporary,
                    sort_keys=True,
                )
                temporary.flush()
                os.fsync(temporary.fileno())
            Path(temporary_name).replace(destination)
            temporary_name = None
        finally:
            if temporary_name is not None:
                Path(temporary_name).unlink(missing_ok=True)

    def _heartbeat(self, force: bool = False) -> None:
        with self._heartbeat_lock:
            now = time.monotonic()
            if force or now - self._last_heartbeat >= HEARTBEAT_INTERVAL_SECONDS:
                self._write_heartbeat()
                self._last_heartbeat = now

    def run_forever(self) -> None:
        self.repository.create_schema()
        previous_term_handler = None
        try:
            previous_term_handler = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, lambda signum, frame: self.stop())
        except ValueError:
            # Signal handlers are available only in the main thread (including tests).
            previous_term_handler = None

        previous_heartbeat = self.processor.on_heartbeat
        previous_stop_requested = self.processor.stop_requested
        self.processor.on_heartbeat = self._heartbeat
        self.processor.stop_requested = self._stop.is_set
        self._retention_thread = threading.Thread(
            target=self.retention_cleaner.run_forever,
            name="retention-cleaner",
            daemon=True,
        )
        self._retention_thread.start()
        try:
            self._heartbeat(force=True)
            while not self._stop.is_set():
                task = self.repository.claim(self.worker_id, self.settings.worker_lease_seconds)
                if task is None:
                    self._heartbeat()
                    self._stop.wait(POLL_INTERVAL_SECONDS)
                    continue
                self._current_task_id = task.task_id
                supervision = _ActiveTaskSupervision(self, task)
                supervision.start()
                try:
                    self.processor.process(task)
                finally:
                    supervision.stop()
                    self._current_task_id = None
                self._heartbeat()
        finally:
            self.retention_cleaner.stop()
            if self._retention_thread is not None:
                self._retention_thread.join()
            self.processor.on_heartbeat = previous_heartbeat
            self.processor.stop_requested = previous_stop_requested
            self._heartbeat(force=True)
            if previous_term_handler is not None:
                signal.signal(signal.SIGTERM, previous_term_handler)


def main() -> None:
    configure_logging("worker")
    settings = get_settings()
    repository = TaskRepository(settings.resolve(settings.data_dir) / "tasks.db")
    comfy = ComfyUIClient(settings.comfy_url)
    worker = Worker(
        settings,
        repository,
        TaskProcessor(
            settings,
            repository,
            comfy,
            load_styles(PROJECT_ROOT / "config" / "styles.json"),
        ),
    )
    try:
        worker.run_forever()
    finally:
        comfy.close()


if __name__ == "__main__":
    main()
