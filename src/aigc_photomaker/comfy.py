import json
import time
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlencode, urlsplit

import httpx
import websocket
from PIL import Image

from .errors import AppError, disk_space_error
from .workflow import ComfyImageRef


_POLL_SECONDS = 0.5
_DEFAULT_TIMEOUT_SECONDS = 120.0
_SUPPORTED_IMAGE_TYPES = {
    "image/png": "PNG",
    "image/jpeg": "JPEG",
    "image/webp": "WEBP",
}


@dataclass(frozen=True)
class ExecutionResult:
    prompt_id: str
    images: list[ComfyImageRef]
    history: dict[str, Any]


def _unavailable() -> AppError:
    return AppError(503, "COMFYUI_UNAVAILABLE", "ComfyUI is unavailable", True)


def _invalid_result() -> AppError:
    return AppError(502, "RESULT_MISSING", "ComfyUI returned an invalid image result", True)


def _invalid_workflow() -> AppError:
    return AppError(400, "WORKFLOW_INVALID", "ComfyUI rejected the workflow", False)


class ComfyUIClient:
    def __init__(
        self,
        base_url: str,
        http_client: httpx.Client | None = None,
        websocket_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("ComfyUI base URL must be an HTTP(S) URL with a host")
        self._owns_http = http_client is None
        self._http = http_client or httpx.Client(base_url=self.base_url, trust_env=False)
        self._websocket_factory = websocket_factory or websocket.create_connection
        self._client_id: str | None = None

    def close(self) -> None:
        if self._owns_http:
            self._http.close()

    def __enter__(self) -> "ComfyUIClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    @staticmethod
    def _check_response(response: httpx.Response, workflow: bool = False) -> None:
        if 200 <= response.status_code < 300:
            return
        if workflow and response.status_code in {400, 422}:
            raise _invalid_workflow()
        raise _unavailable()

    def _request(self, method: str, path: str, workflow: bool = False, **kwargs) -> httpx.Response:
        try:
            response = self._http.request(method, self._url(path), **kwargs)
            self._check_response(response, workflow=workflow)
            return response
        except httpx.HTTPError as exc:
            raise _unavailable() from exc

    @staticmethod
    def _json_object(response: httpx.Response) -> dict[str, Any]:
        try:
            value = response.json()
        except (ValueError, UnicodeError) as exc:
            raise _unavailable() from exc
        if not isinstance(value, dict):
            raise _unavailable()
        return value

    def system_stats(self) -> dict[str, Any]:
        return self._json_object(self._request("GET", "/system_stats"))

    def upload_image(self, path: Path) -> ComfyImageRef:
        source = Path(path)
        try:
            with source.open("rb") as image:
                response = self._request(
                    "POST",
                    "/upload/image",
                    files={"image": (source.name, image, "application/octet-stream")},
                )
        except OSError as exc:
            disk_error = disk_space_error(exc)
            if disk_error is not None:
                raise disk_error from exc
            raise AppError(400, "INVALID_IMAGE", "Unable to read the upload image", False) from exc

        try:
            payload = self._json_object(response)
            name = payload["name"]
        except (AppError, KeyError) as exc:
            raise _unavailable() from exc
        subfolder = payload.get("subfolder", "")
        image_type = payload.get("type", "input")
        if not all(isinstance(value, str) for value in (name, subfolder, image_type)) or not name:
            raise _unavailable()
        return ComfyImageRef(name=name, subfolder=subfolder, type=image_type)

    def submit(self, graph: dict[str, Any], client_id: str) -> str:
        response = self._request(
            "POST",
            "/prompt",
            workflow=True,
            json={"prompt": graph, "client_id": client_id},
        )
        try:
            payload = self._json_object(response)
        except AppError as exc:
            raise _invalid_workflow() from exc
        prompt_id = payload.get("prompt_id")
        if payload.get("error") or not isinstance(prompt_id, str) or not prompt_id:
            raise _invalid_workflow()
        self._client_id = client_id
        return prompt_id

    @staticmethod
    def _matches_submission(
        graph: Any, extra_data: Any, client_id: str, output_prefix: str
    ) -> bool:
        if not isinstance(extra_data, dict) or extra_data.get("client_id") != client_id:
            return False
        if not isinstance(graph, dict):
            return False
        for node in graph.values():
            if not isinstance(node, dict) or node.get("class_type") != "SaveImage":
                continue
            inputs = node.get("inputs")
            if isinstance(inputs, dict) and inputs.get("filename_prefix") == output_prefix:
                return True
        return False

    def find_submission(self, client_id: str, output_prefix: str) -> str | None:
        """Reconcile an uncertain submission without ever creating a new prompt."""
        unavailable = False
        try:
            queue = self._json_object(self._request("GET", "/queue"))
            for key in ("queue_running", "queue_pending"):
                entries = queue.get(key, [])
                if not isinstance(entries, list):
                    continue
                for item in entries:
                    if not isinstance(item, (list, tuple)) or len(item) < 4:
                        continue
                    prompt_id = item[1]
                    if (
                        isinstance(prompt_id, str)
                        and prompt_id
                        and self._matches_submission(item[2], item[3], client_id, output_prefix)
                    ):
                        return prompt_id
        except AppError as exc:
            if exc.code != "COMFYUI_UNAVAILABLE":
                raise
            unavailable = True

        try:
            histories = self._json_object(self._request("GET", "/history"))
            for prompt_id, entry in histories.items():
                if not isinstance(prompt_id, str) or not isinstance(entry, dict):
                    continue
                prompt = entry.get("prompt")
                if (
                    isinstance(prompt, (list, tuple))
                    and len(prompt) >= 4
                    and self._matches_submission(prompt[2], prompt[3], client_id, output_prefix)
                ):
                    return prompt_id
        except AppError as exc:
            if exc.code != "COMFYUI_UNAVAILABLE":
                raise
            unavailable = True
        if unavailable:
            raise _unavailable()
        return None

    def history(self, prompt_id: str) -> dict[str, Any]:
        path = f"/history/{quote(prompt_id, safe='')}"
        payload = self._json_object(self._request("GET", path))
        if prompt_id not in payload:
            return {}
        entry = payload[prompt_id]
        if not isinstance(entry, dict):
            raise _unavailable()
        return entry

    def cancel(self, prompt_id: str) -> None:
        queue_error = None
        try:
            self._request("POST", "/queue", json={"delete": [prompt_id]})
        except AppError as exc:
            queue_error = exc
        try:
            self._request("POST", "/interrupt")
        except AppError as interrupt_error:
            if queue_error is not None:
                raise queue_error from interrupt_error
            raise
        if queue_error is not None:
            raise queue_error

    def download_image(self, ref: ComfyImageRef, dst: Path) -> None:
        destination = Path(dst)
        temporary = Path(f"{destination}.part")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with self._http.stream(
                "GET",
                self._url("/view"),
                params={"filename": ref.name, "subfolder": ref.subfolder, "type": ref.type},
            ) as response:
                if response.status_code == 404:
                    raise _invalid_result()
                self._check_response(response)
                media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                expected_format = _SUPPORTED_IMAGE_TYPES.get(media_type)
                if expected_format is None:
                    raise _invalid_result()
                length_header = response.headers.get("content-length")
                expected_size = None
                if length_header is not None:
                    try:
                        expected_size = int(length_header)
                    except ValueError as exc:
                        raise _invalid_result() from exc
                    if expected_size < 0:
                        raise _invalid_result()

                size = 0
                with temporary.open("wb") as output:
                    for chunk in response.iter_bytes():
                        output.write(chunk)
                        size += len(chunk)
                if size == 0 or (expected_size is not None and size != expected_size):
                    raise _invalid_result()

            with Image.open(temporary) as image:
                if image.format != expected_format or image.width <= 0 or image.height <= 0:
                    raise _invalid_result()
                image.verify()
            temporary.replace(destination)
        except AppError:
            raise
        except httpx.HTTPError as exc:
            raise _unavailable() from exc
        except OSError as exc:
            disk_error = disk_space_error(exc)
            if disk_error is not None:
                raise disk_error from exc
            raise _invalid_result() from exc
        except (SyntaxError, ValueError, Image.DecompressionBombError) as exc:
            raise _invalid_result() from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def map_execution_error(message: str) -> AppError:
        normalized = message.lower()
        if any(marker in normalized for marker in (
            "enospc", "no space left", "database or disk is full", "database is full"
        )):
            return AppError(507, "DISK_SPACE_LOW", "Server disk space is exhausted", True)
        if "out of memory" in normalized or "cuda oom" in normalized:
            return AppError(500, "GPU_OOM", "ComfyUI ran out of GPU memory", False)
        if "output_no_face_detected" in normalized:
            return AppError(
                422, "OUTPUT_NO_FACE_DETECTED", "No face was detected in the generated image", True
            )
        if "output_multiple_faces_detected" in normalized:
            return AppError(
                422,
                "OUTPUT_MULTIPLE_FACES_DETECTED",
                "Multiple faces were detected in the generated image",
                True,
            )
        if "no_face_detected" in normalized:
            return AppError(422, "NO_FACE_DETECTED", "No face was detected in the input image", False)
        if "multiple_faces_detected" in normalized:
            return AppError(
                422,
                "MULTIPLE_FACES_DETECTED",
                "Multiple faces were detected in the input image",
                False,
            )
        if "comfyui_timeout" in normalized or "timed out" in normalized or "timeout" in normalized:
            return AppError(504, "COMFYUI_TIMEOUT", "ComfyUI execution timed out", True)
        if any(
            marker in normalized
            for marker in (
                "invalid prompt",
                "invalid workflow",
                "outputs failed validation",
                "prompt validation",
                "required input",
                "node type",
            )
        ):
            return _invalid_workflow()
        return AppError(500, "COMFYUI_EXECUTION_ERROR", "ComfyUI workflow execution failed", False)

    @staticmethod
    def _history_error(entry: dict[str, Any]) -> str:
        status = entry.get("status")
        if not isinstance(status, dict):
            return ""
        messages = status.get("messages", [])
        if not isinstance(messages, list):
            return ""
        collected = []
        for message in messages:
            data = None
            if isinstance(message, (list, tuple)) and len(message) >= 2:
                data = message[1]
            elif isinstance(message, dict):
                data = message
            if isinstance(data, dict):
                for key in ("exception_message", "exception_type", "message"):
                    value = data.get(key)
                    if isinstance(value, str):
                        collected.append(value)
        return "\n".join(collected)

    @staticmethod
    def _save_image_ids(entry: dict[str, Any]) -> set[str]:
        prompt = entry.get("prompt")
        # ComfyUI history: [number, prompt_id, prompt_graph, extra_data, outputs_to_execute]
        if not isinstance(prompt, (list, tuple)) or len(prompt) < 3 or not isinstance(prompt[2], dict):
            return set()
        return {
            str(node_id)
            for node_id, node in prompt[2].items()
            if isinstance(node, dict) and node.get("class_type") == "SaveImage"
        }

    @classmethod
    def _result_from_history(cls, prompt_id: str, entry: dict[str, Any]) -> ExecutionResult | None:
        if not entry:
            return None
        status = entry.get("status")
        status = status if isinstance(status, dict) else {}
        if status.get("status_str") == "error":
            raise cls.map_execution_error(cls._history_error(entry))
        if status.get("status_str") != "success" or status.get("completed") is not True:
            return None

        images = []
        save_image_ids = cls._save_image_ids(entry)
        outputs = entry.get("outputs")
        if isinstance(outputs, dict):
            for node_id, node_output in outputs.items():
                if str(node_id) not in save_image_ids:
                    continue
                if not isinstance(node_output, dict) or not isinstance(node_output.get("images"), list):
                    continue
                for image in node_output["images"]:
                    if not isinstance(image, dict):
                        continue
                    name = image.get("filename")
                    subfolder = image.get("subfolder", "")
                    image_type = image.get("type", "output")
                    if (
                        isinstance(name, str)
                        and name
                        and isinstance(subfolder, str)
                        and image_type in {"output", "temp"}
                    ):
                        images.append(ComfyImageRef(name, subfolder, image_type))

        return ExecutionResult(prompt_id=prompt_id, images=images, history=entry)

    @staticmethod
    def _close_websocket(socket: Any) -> None:
        try:
            socket.close()
        except (websocket.WebSocketException, OSError):
            pass

    def _cancel_quietly(self, prompt_id: str) -> None:
        try:
            self.cancel(prompt_id)
        except Exception:
            # The terminal cancellation/timeout error must not be replaced by cleanup errors.
            pass

    def _websocket_url(self, prompt_id: str) -> tuple[str, str]:
        parsed = urlsplit(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        client_id = self._client_id or prompt_id
        url = f"{scheme}://{parsed.netloc}/ws?{urlencode({'client_id': client_id})}"
        return url, parsed.hostname or ""

    def wait_for_result(
        self,
        prompt_id: str,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
        heartbeat: Callable[[], None] | None = None,
        is_cancel_requested: Callable[[], bool] | None = None,
        on_progress: Callable[[Real, Real], None] | None = None,
    ) -> ExecutionResult:
        heartbeat = heartbeat or (lambda: None)
        is_cancel_requested = is_cancel_requested or (lambda: False)
        on_progress = on_progress or (lambda value, maximum: None)
        deadline = time.monotonic() + timeout_seconds
        socket = None
        websocket_attempted = False
        history_due = 0.0

        try:
            while True:
                heartbeat()
                if is_cancel_requested():
                    self._cancel_quietly(prompt_id)
                    raise AppError(409, "CANCELLED", "Generation was cancelled", False)
                if time.monotonic() >= deadline:
                    self._cancel_quietly(prompt_id)
                    raise AppError(504, "COMFYUI_TIMEOUT", "ComfyUI execution timed out", True)

                poll_history = socket is None and websocket_attempted
                if not websocket_attempted:
                    websocket_attempted = True
                    url, hostname = self._websocket_url(prompt_id)
                    try:
                        socket = self._websocket_factory(
                            url,
                            timeout=_POLL_SECONDS,
                            proxy_type=None,
                            http_proxy_host=None,
                            http_proxy_port=None,
                            http_no_proxy=[hostname],
                        )
                    except (websocket.WebSocketException, OSError):
                        socket = None
                        poll_history = True

                if socket is not None:
                    try:
                        raw_event = socket.recv()
                    except websocket.WebSocketTimeoutException:
                        poll_history = True
                    except (websocket.WebSocketException, OSError):
                        self._close_websocket(socket)
                        socket = None
                        poll_history = True
                    else:
                        try:
                            event = json.loads(raw_event)
                        except (TypeError, ValueError, UnicodeError):
                            event = None
                        if isinstance(event, dict):
                            event_type = event.get("type")
                            data = event.get("data")
                            data = data if isinstance(data, dict) else {}
                            event_prompt_id = data.get("prompt_id")
                            matches_prompt = event_prompt_id == prompt_id
                            if event_type == "progress" and matches_prompt:
                                value, maximum = data.get("value"), data.get("max")
                                if (
                                    isinstance(value, Real)
                                    and not isinstance(value, bool)
                                    and isinstance(maximum, Real)
                                    and not isinstance(maximum, bool)
                                ):
                                    on_progress(value, maximum)
                            elif event_type == "execution_error" and matches_prompt:
                                message = "\n".join(
                                    value
                                    for key in ("exception_message", "exception_type", "message")
                                    if isinstance((value := data.get(key)), str)
                                )
                                raise self.map_execution_error(message)
                            elif (
                                event_type == "execution_success"
                                or (event_type == "executing" and data.get("node") is None)
                            ) and matches_prompt:
                                self._close_websocket(socket)
                                socket = None
                                poll_history = True
                                history_due = 0.0

                if poll_history and time.monotonic() >= history_due:
                    try:
                        entry = self.history(prompt_id)
                    except AppError as exc:
                        if exc.code != "COMFYUI_UNAVAILABLE":
                            raise
                        entry = {}
                    result = self._result_from_history(prompt_id, entry)
                    if result is not None:
                        return result
                    history_due = time.monotonic() + _POLL_SECONDS

                if poll_history:
                    time.sleep(_POLL_SECONDS)
        finally:
            if socket is not None:
                self._close_websocket(socket)
