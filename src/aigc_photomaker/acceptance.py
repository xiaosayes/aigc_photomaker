"""Shared, server-side API and GPU helpers for smoke and benchmark acceptance.

Nothing in this module is invoked by application startup.  The command-line entry
points are intended to run on the GPU server against the real HTTP API.
"""

from __future__ import annotations

import csv
import io
import json
import os
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .config import PROJECT_ROOT

POLL_INTERVAL_SECONDS = 0.5
GPU_SAMPLE_INTERVAL_SECONDS = 0.1
GPU_ID = 1
MAX_ERROR_LENGTH = 500


class AcceptanceError(RuntimeError):
    """A sanitized acceptance failure which must not contain the API key."""

    def __init__(self, code: str, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "http_status": self.status}


class APIError(AcceptanceError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    task_id: str
    mode: str
    seed: int
    client_total_seconds: float
    server_stages: dict[str, float]
    image: dict[str, Any]
    style: str = "cyberpunk"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GpuMonitor:
    """Sample physical GPU 1 every 100 ms without a shell or inherited environment."""

    interval_seconds: float = GPU_SAMPLE_INTERVAL_SECONDS
    command_timeout_seconds: float = 5.0
    samples: list[dict[str, float]] = field(default_factory=list, init=False)
    errors: list[str] = field(default_factory=list, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    @property
    def peak_vram_mib(self) -> float | None:
        with self._lock:
            values = [sample["memory_used_mib"] for sample in self.samples]
        return max(values) if values else None

    @property
    def peak_utilization_percent(self) -> float | None:
        with self._lock:
            values = [sample["utilization_percent"] for sample in self.samples]
        return max(values) if values else None

    def __enter__(self) -> "GpuMonitor":
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("GPU monitor already started")
        self._thread = threading.Thread(target=self._run, name="acceptance-gpu1", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self) -> None:
        # Take one sample immediately, then use a monotonic schedule to avoid drift.
        next_sample = time.monotonic()
        while not self._stop.is_set():
            self.sample_once()
            next_sample = max(next_sample + self.interval_seconds, time.monotonic())
            if self._stop.wait(max(0.0, next_sample - time.monotonic())):
                return

    def sample_once(self) -> None:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--id=1",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.command_timeout_seconds,
            )
            row = next(csv.reader(completed.stdout.strip().splitlines()))
            if len(row) != 2:
                raise ValueError("unexpected nvidia-smi column count")
            sample = {
                "memory_used_mib": float(row[0].strip()),
                "utilization_percent": float(row[1].strip()),
            }
            with self._lock:
                self.samples.append(sample)
        except Exception as exc:
            # Do not include command environments or stderr, which can contain local details.
            with self._lock:
                if len(self.errors) < 20:
                    self.errors.append(f"GPU 1 sampling failed: {type(exc).__name__}")


def _redact(value: object, api_key: str) -> str:
    text = str(value)
    if api_key:
        text = text.replace(api_key, "[redacted]")
    return text[:MAX_ERROR_LENGTH]


def _base_url(api_url: str) -> str:
    value = api_url.rstrip("/")
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("api_url must be an absolute HTTP(S) URL")
    if parsed.query or parsed.fragment:
        raise ValueError("api_url must not contain a query or fragment")
    return value


def _request(
    api_url: str,
    api_key: str,
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    content_type: str | None = None,
    timeout_seconds: float = 60.0,
) -> tuple[int, bytes, dict[str, str]]:
    url = _base_url(api_url) + path
    headers = {"X-API-Key": api_key, "Accept": "application/json"}
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.status, response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        # Error bodies are parsed below and never logged by this module.
        return exc.code, exc.read(), dict(exc.headers.items()) if exc.headers else {}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise AcceptanceError("API_UNREACHABLE", f"API request failed: {type(exc).__name__}") from exc


def _json_payload(payload: bytes, api_key: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcceptanceError("INVALID_API_RESPONSE", "API returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise AcceptanceError("INVALID_API_RESPONSE", "API returned a non-object JSON response")
    return value


def multipart_body(
    input_bytes: bytes,
    filename: str,
    *,
    style: str,
    mode: str,
    seed: int,
) -> tuple[bytes, str]:
    """Create the complete multipart request; callers time this function itself."""
    boundary = f"aigc-{uuid.uuid4().hex}"
    fields = {"style": style, "mode": mode, "seed": str(seed)}
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    safe_filename = Path(filename).name.replace('"', "_")
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="file"; filename="{safe_filename}"\r\n'
                "Content-Type: image/png\r\n\r\n"
            ).encode(),
            input_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def create_task(
    api_url: str,
    api_key: str,
    input_bytes: bytes,
    filename: str,
    *,
    style: str,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    body, content_type = multipart_body(
        input_bytes, filename, style=style, mode=mode, seed=seed
    )
    status, payload, _ = _request(
        api_url,
        api_key,
        "/api/v1/tasks",
        method="POST",
        body=body,
        content_type=content_type,
    )
    response = _json_payload(payload, api_key)
    if status != 202:
        error = response.get("error")
        if not isinstance(error, dict):
            raise AcceptanceError("API_REQUEST_FAILED", f"Task creation returned HTTP {status}", status=status)
        raise APIError(
            _redact(error.get("code", "API_REQUEST_FAILED"), api_key),
            _redact(error.get("message", "Task creation failed"), api_key),
            status=status,
        )
    if not isinstance(response.get("task_id"), str):
        raise AcceptanceError("INVALID_API_RESPONSE", "Created task has no task_id", status=status)
    return response


def _task_status(api_url: str, api_key: str, task_id: str) -> dict[str, Any]:
    quoted_id = urllib.parse.quote(task_id, safe="")
    status, payload, _ = _request(api_url, api_key, f"/api/v1/tasks/{quoted_id}")
    response = _json_payload(payload, api_key)
    if status != 200:
        error = response.get("error")
        code = error.get("code", "TASK_QUERY_FAILED") if isinstance(error, dict) else "TASK_QUERY_FAILED"
        message = error.get("message", "Task query failed") if isinstance(error, dict) else "Task query failed"
        raise AcceptanceError(_redact(code, api_key), _redact(message, api_key), status=status)
    return response


def decode_result_png(payload: bytes) -> dict[str, Any]:
    """Fully decode the downloaded image, not merely its header, for display readiness."""
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            metadata = {
                "format": image.format,
                "mode": image.mode,
                "width": image.width,
                "height": image.height,
                "bytes": len(payload),
            }
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise AcceptanceError("INVALID_RESULT_IMAGE", "Result image cannot be fully decoded") from exc
    if metadata["format"] != "PNG" or metadata["mode"] != "RGB":
        raise AcceptanceError("INVALID_RESULT_IMAGE", "Result must be an RGB PNG")
    if (metadata["width"], metadata["height"]) != (1024, 1024):
        raise AcceptanceError("INVALID_RESULT_IMAGE", "Result must be 1024x1024")
    return metadata


def wait_for_terminal_task(
    api_url: str,
    api_key: str,
    task_id: str,
    *,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    """Poll a real API task every 500 ms until it reaches a terminal state."""
    deadline = time.monotonic() + timeout_seconds
    next_poll = time.monotonic()
    while True:
        if time.monotonic() > deadline:
            raise AcceptanceError("TASK_TIMEOUT", f"Task {task_id} did not finish within {timeout_seconds:g}s")
        current = _task_status(api_url, api_key, task_id)
        state = current.get("status")
        if state in {"succeeded", "failed", "cancelled"}:
            return current
        if state not in {"queued", "running"}:
            raise AcceptanceError("INVALID_API_RESPONSE", f"Unknown task status: {_redact(state, api_key)}")
        next_poll = max(next_poll + POLL_INTERVAL_SECONDS, time.monotonic())
        delay = max(0.0, next_poll - time.monotonic())
        if delay:
            time.sleep(delay)


def run_generation(
    api_url: str,
    api_key: str,
    input_path: str | Path,
    mode: str,
    seed: int,
    style: str = "cyberpunk",
    *,
    timeout_seconds: float = 180.0,
) -> GenerationResult:
    """Measure multipart creation through complete download and full image decode."""
    started = time.perf_counter()
    input_bytes = Path(input_path).read_bytes()
    # The timer deliberately starts before reading and constructing multipart content.
    body, content_type = multipart_body(
        input_bytes, Path(input_path).name, style=style, mode=mode, seed=seed
    )
    status, payload, _ = _request(
        api_url,
        api_key,
        "/api/v1/tasks",
        method="POST",
        body=body,
        content_type=content_type,
    )
    created = _json_payload(payload, api_key)
    if status != 202:
        error = created.get("error")
        if isinstance(error, dict):
            raise APIError(
                _redact(error.get("code", "TASK_CREATE_FAILED"), api_key),
                _redact(error.get("message", "Task creation failed"), api_key),
                status=status,
            )
        raise AcceptanceError("TASK_CREATE_FAILED", f"Task creation returned HTTP {status}", status=status)
    task_id = created.get("task_id")
    if not isinstance(task_id, str):
        raise AcceptanceError("INVALID_API_RESPONSE", "Created task has no task_id")

    current = wait_for_terminal_task(
        api_url, api_key, task_id, timeout_seconds=timeout_seconds
    )
    state = current.get("status")
    if state != "succeeded":
        error = current.get("error")
        if isinstance(error, dict):
            raise AcceptanceError(
                _redact(error.get("code", "TASK_FAILED"), api_key),
                _redact(error.get("message", "Task failed"), api_key),
            )
        raise AcceptanceError("TASK_FAILED", f"Task ended with status {state}")

    quoted_id = urllib.parse.quote(task_id, safe="")
    result_status, result_payload, _ = _request(
        api_url, api_key, f"/api/v1/tasks/{quoted_id}/result", timeout_seconds=120.0
    )
    if result_status != 200:
        response = _json_payload(result_payload, api_key)
        error = response.get("error")
        code = error.get("code", "RESULT_DOWNLOAD_FAILED") if isinstance(error, dict) else "RESULT_DOWNLOAD_FAILED"
        message = error.get("message", "Result download failed") if isinstance(error, dict) else "Result download failed"
        raise AcceptanceError(_redact(code, api_key), _redact(message, api_key), status=result_status)
    image = decode_result_png(result_payload)
    elapsed = time.perf_counter() - started
    timings = current.get("timings", {})
    stages = {
        str(name): float(value)
        for name, value in timings.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    } if isinstance(timings, dict) else {}
    return GenerationResult(task_id, mode, seed, elapsed, stages, image, style)


def expect_task_error(
    api_url: str,
    api_key: str,
    input_bytes: bytes,
    filename: str,
    *,
    style: str,
    mode: str,
    seed: int,
    expected_code: str,
) -> dict[str, Any]:
    try:
        created = create_task(
            api_url,
            api_key,
            input_bytes,
            filename,
            style=style,
            mode=mode,
            seed=seed,
        )
    except APIError as exc:
        if exc.code != expected_code:
            raise AcceptanceError(
                "UNEXPECTED_ERROR_CODE",
                f"Expected {expected_code}, received {_redact(exc.code, api_key)}",
            ) from exc
        return {"expected_code": expected_code, "received_code": exc.code, "passed": True}
    # Face-policy failures are normally produced by ComfyUI and surfaced by the
    # worker asynchronously, so task acceptance alone is not a failed expectation.
    task_id = created["task_id"]
    terminal = wait_for_terminal_task(api_url, api_key, task_id)
    if terminal.get("status") != "failed":
        raise AcceptanceError(
            "EXPECTED_ERROR_MISSING",
            f"Expected {expected_code}, but task ended as {terminal.get('status')}",
        )
    error = terminal.get("error")
    received_code = error.get("code") if isinstance(error, dict) else None
    if received_code != expected_code:
        raise AcceptanceError(
            "UNEXPECTED_ERROR_CODE",
            f"Expected {expected_code}, received {_redact(received_code, api_key)}",
        )
    return {
        "expected_code": expected_code,
        "received_code": received_code,
        "task_id": task_id,
        "passed": True,
    }


def _expected_project_pids() -> dict[str, int | None]:
    """Read service-owned PID files; ComfyUI is the required GPU generation process."""
    expected: dict[str, int | None] = {}
    for role in ("worker", "comfy"):
        path = PROJECT_ROOT / ".runtime" / "app" / "pids" / f"{role}.pid"
        try:
            pid = int(path.read_text(encoding="ascii").strip())
            expected[role] = pid if pid > 1 else None
        except (OSError, ValueError):
            expected[role] = None
    return expected


def _read_process_details(pid: int) -> tuple[str, str]:
    """Return readable command/cwd evidence; an unreadable command is indeterminate."""
    if pid <= 1:
        raise OSError("invalid process ID")
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
        "utf-8", errors="replace"
    ).strip()
    if not cmdline:
        raise OSError("empty or unreadable command line")
    try:
        cwd = os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        cwd = ""
    return cmdline, cwd


_SAFE_UNRELATED_EXECUTABLES = {"ffmpeg"}


def _classify_command(cmdline: str, cwd: str) -> str:
    """Classify the full command; unknown commands always fail closed."""
    root = str(PROJECT_ROOT)
    normalized_command = cmdline.replace("\\", "/")
    try:
        in_checkout = bool(cwd) and Path(cwd).resolve().is_relative_to(PROJECT_ROOT.resolve())
    except (OSError, RuntimeError, ValueError):
        in_checkout = False
    associated_with_checkout = in_checkout or root in cmdline
    project_markers = ("aigc_photomaker.worker", "aigc_photomaker.app:create_app")
    comfy_marker = "ComfyUI/main.py" in normalized_command
    if associated_with_checkout and (
        any(marker in cmdline for marker in project_markers) or comfy_marker
    ):
        return "project"
    try:
        tokens = shlex.split(normalized_command)
    except ValueError:
        return "indeterminate"
    if not tokens:
        return "indeterminate"
    # Wrappers can hide the actual runtime in later tokens. Inspect the complete
    # command for Python/uv evidence instead of trusting only its first token.
    executable_names = {Path(token).name.lower() for token in tokens}
    if "uv" in executable_names or any(
        name.startswith("python") for name in executable_names
    ):
        return "indeterminate"
    if Path(tokens[0]).name.lower() in _SAFE_UNRELATED_EXECUTABLES:
        return "unrelated"
    return "indeterminate"


def _attribute_process(pid: int, expected_pids: dict[int, str]) -> str:
    """Fail closed: expected/known project, clearly unrelated, or indeterminate."""
    if pid in expected_pids:
        # The service PID file positively attributes custom ComfyUI paths even when
        # another user's /proc entries are not readable.
        return "project"
    try:
        cmdline, cwd = _read_process_details(pid)
    except (OSError, UnicodeError):
        return "indeterminate"
    return _classify_command(cmdline, cwd)


def classify_gpu_processes(
    gpu_indices: dict[str, int],
    processes: list[dict[str, Any]],
    attribute_process: Callable[[int], str],
    expected_gpu_processes: dict[str, int | None],
) -> dict[str, Any]:
    expected_pids = {
        pid: role for role, pid in expected_gpu_processes.items() if pid is not None
    }
    expected_roles_by_pid = expected_pids
    project_processes: list[dict[str, Any]] = []
    indeterminate_processes: list[dict[str, Any]] = []
    observed_expected_roles: set[str] = set()
    ignored = 0
    for process in processes:
        gpu_index = gpu_indices.get(str(process.get("gpu_uuid", "")), -1)
        try:
            pid = int(process["pid"])
            if pid <= 1:
                raise ValueError("invalid PID")
        except (KeyError, TypeError, ValueError):
            indeterminate_processes.append(
                {"pid": -1, "gpu_index": gpu_index, "reason": "invalid_pid"}
            )
            continue
        attribution = attribute_process(pid)
        if attribution not in {"project", "unrelated", "indeterminate"}:
            attribution = "indeterminate"
        if attribution == "unrelated":
            ignored += 1
            continue
        if attribution == "indeterminate":
            indeterminate_processes.append(
                {"pid": pid, "gpu_index": gpu_index, "reason": "incomplete_attribution"}
            )
            continue
        role = expected_roles_by_pid.get(pid, "project")
        if pid in expected_pids:
            observed_expected_roles.add(expected_roles_by_pid[pid])
        project_processes.append({"pid": pid, "gpu_index": gpu_index, "role": role})

    missing = [
        role
        for role, pid in expected_gpu_processes.items()
        if pid is None or role not in observed_expected_roles
    ]
    misplaced = [process for process in project_processes if process["gpu_index"] != GPU_ID]
    # Unattributed processes remain visible as warnings but are not project processes.
    # Managed PID ownership and every positively identified project process stay strict.
    if misplaced or missing or not expected_gpu_processes:
        status = "failed"
    else:
        status = "passed"
    return {
        "passed": status == "passed",
        "status": status,
        "physical_gpu_id": GPU_ID,
        "project_processes": project_processes,
        "indeterminate_processes": indeterminate_processes,
        "missing_expected_gpu_processes": missing,
        "ignored_unrelated_processes": ignored,
    }


def check_project_gpu_processes() -> dict[str, Any]:
    try:
        gpu_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        process_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        gpu_indices = {
            row[1].strip(): int(row[0].strip())
            for row in csv.reader(gpu_result.stdout.strip().splitlines())
            if len(row) >= 2
        }
        processes = [
            {"gpu_uuid": row[0].strip(), "pid": row[1].strip()}
            for row in csv.reader(process_result.stdout.strip().splitlines())
            if len(row) >= 2
        ]
        expected = _expected_project_pids()
        expected_pids = {pid: role for role, pid in expected.items() if pid is not None}
        return classify_gpu_processes(
            gpu_indices,
            processes,
            lambda pid: _attribute_process(pid, expected_pids),
            {"comfy": expected["comfy"]},
        )
    except Exception as exc:
        return {
            "passed": False,
            "status": "indeterminate",
            "physical_gpu_id": GPU_ID,
            "project_processes": [],
            "indeterminate_processes": [],
            "missing_expected_gpu_processes": ["comfy"],
            "ignored_unrelated_processes": 0,
            "error": f"Unable to inspect GPU processes: {type(exc).__name__}",
        }


def log_offsets(paths: list[Path] | None = None) -> dict[str, int]:
    paths = paths or [
        PROJECT_ROOT / ".runtime" / "app" / "logs" / "worker.log",
        PROJECT_ROOT / ".runtime" / "app" / "logs" / "comfy.log",
    ]
    offsets = {}
    for path in paths:
        try:
            offsets[str(path)] = path.stat().st_size
        except OSError:
            offsets[str(path)] = 0
    return offsets


def detect_oom_since(offsets: dict[str, int]) -> bool:
    """Inspect only new log bytes; report a boolean rather than potentially secret log lines."""
    for filename, offset in offsets.items():
        try:
            with open(filename, "rb") as stream:
                stream.seek(offset)
                content = stream.read().lower()
            if b"out of memory" in content or b"cuda oom" in content:
                return True
        except OSError:
            continue
    return False


def error_details(exc: Exception, api_key: str) -> dict[str, Any]:
    if isinstance(exc, AcceptanceError):
        return exc.to_dict()
    return {"code": "ACCEPTANCE_ERROR", "message": _redact(type(exc).__name__, api_key), "http_status": None}


def contains_oom(error: dict[str, Any]) -> bool:
    text = json.dumps(error, ensure_ascii=False).lower()
    return any(
        marker in text
        for marker in ("out of memory", "cuda oom", "cuda_oom", "gpu_oom")
    )
