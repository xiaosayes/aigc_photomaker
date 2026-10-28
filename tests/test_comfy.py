# Server-pending: run these HTTP/WebSocket behavior tests on the GPU server.
import io
import json
from pathlib import Path

import httpx
import pytest
import websocket
from PIL import Image

from aigc_photomaker.comfy import ComfyUIClient
from aigc_photomaker.errors import AppError
from aigc_photomaker.workflow import ComfyImageRef


BASE_URL = "http://127.0.0.1:8188"


def client_for(handler, **kwargs):
    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return ComfyUIClient(BASE_URL, http_client=http_client, **kwargs)


def history_entry(status="success", message=None):
    entry = {
        # ComfyUI: [number, prompt_id, prompt_graph, extra_data, outputs_to_execute]
        "prompt": [2, "p1", {"12": {"class_type": "SaveImage"}}, {}, ["12"]],
        "outputs": {
            "12": {
                "images": [
                    {"filename": "one.png", "subfolder": "aigc", "type": "output"},
                    {"filename": "two.png", "subfolder": "", "type": "output"},
                ]
            },
            "99": {"images": [{"filename": "untrusted.png", "type": "output"}]},
        },
        "status": {"status_str": status, "completed": status == "success", "messages": []},
    }
    if message is not None:
        entry["status"]["messages"] = [["execution_error", {"exception_message": message}]]
    return entry


def png_bytes():
    output = io.BytesIO()
    Image.new("RGB", (2, 3), "red").save(output, format="PNG")
    return output.getvalue()


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture(autouse=True)
def fake_clock(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr("aigc_photomaker.comfy.time.monotonic", clock.monotonic)
    monkeypatch.setattr("aigc_photomaker.comfy.time.sleep", clock.sleep)
    return clock


class FakeWebSocket:
    def __init__(self, events):
        self.events = list(events)
        self.closed = False

    def recv(self):
        if not self.events:
            raise websocket.WebSocketTimeoutException("timed out")
        event = self.events.pop(0)
        if isinstance(event, BaseException):
            raise event
        return json.dumps(event)

    def close(self):
        self.closed = True


def test_internal_http_client_bypasses_environment(monkeypatch):
    captured = {}
    injected = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200)))

    def fake_client(**kwargs):
        captured.update(kwargs)
        return injected

    monkeypatch.setattr("aigc_photomaker.comfy.httpx.Client", fake_client)
    comfy = ComfyUIClient(BASE_URL)

    assert comfy._http is injected
    assert captured["trust_env"] is False
    assert captured["base_url"] == BASE_URL


def test_system_stats_and_upload_image_closes_file(tmp_path, monkeypatch):
    source = tmp_path / "face.png"
    source.write_bytes(b"image bytes")
    opened = []
    original_open = Path.open

    def tracking_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path == source:
            opened.append(stream)
        return stream

    monkeypatch.setattr(Path, "open", tracking_open)
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/system_stats":
            return httpx.Response(200, json={"system": {"os": "linux"}})
        assert request.headers["content-type"].startswith("multipart/form-data")
        assert b'name="image"' in request.content
        assert b"image bytes" in request.content
        return httpx.Response(200, json={"name": "face.png", "subfolder": "uploads", "type": "input"})

    comfy = client_for(handler)

    assert comfy.system_stats() == {"system": {"os": "linux"}}
    assert comfy.upload_image(source) == ComfyImageRef("face.png", "uploads", "input")
    assert paths == ["/system_stats", "/upload/image"]
    assert len(opened) == 1 and opened[0].closed


def test_upload_read_error_does_not_expose_local_path(tmp_path):
    missing = tmp_path / "private" / "secret.png"
    comfy = client_for(lambda request: httpx.Response(500))

    with pytest.raises(AppError) as caught:
        comfy.upload_image(missing)

    assert caught.value.code == "INVALID_IMAGE"
    assert str(tmp_path) not in str(caught.value)


def test_submit_returns_prompt_id_and_sends_graph_and_client_id():
    graph = {"1": {}}

    def handler(request):
        assert request.url.path == "/prompt"
        assert json.loads(request.content) == {"prompt": graph, "client_id": "worker"}
        return httpx.Response(200, json={"prompt_id": "p1"})

    comfy = client_for(handler)

    assert comfy.submit(graph, "worker") == "p1"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(400, json={"error": {"message": "/private/workflow"}, "node_errors": {}}),
        httpx.Response(200, json={"error": "invalid prompt"}),
        httpx.Response(200, json={}),
    ],
)
def test_submit_maps_invalid_workflow_without_echoing_response(response):
    comfy = client_for(lambda request: response)

    with pytest.raises(AppError) as caught:
        comfy.submit({"1": {}}, "worker")

    assert caught.value.code == "WORKFLOW_INVALID"
    assert caught.value.retryable is False
    assert "/private/workflow" not in str(caught.value)


def test_find_submission_matches_stable_client_and_output_prefix_in_queue():
    graph = {"12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "aigc/task-1"}}}
    def handler(request):
        assert request.url.path == "/queue"
        return httpx.Response(200, json={
            "queue_running": [[1, "other", graph, {"client_id": "other"}]],
            "queue_pending": [[2, "p1", graph, {"client_id": "intent-1"}]],
        })
    assert client_for(handler).find_submission("intent-1", "aigc/task-1") == "p1"


def test_find_submission_uses_history_and_does_not_match_prefix_alone():
    graph = {"12": {"class_type": "SaveImage", "inputs": {"filename_prefix": "aigc/task-1"}}}
    def handler(request):
        if request.url.path == "/queue":
            return httpx.Response(200, json={"queue_running": [], "queue_pending": []})
        assert request.url.path == "/history"
        return httpx.Response(200, json={
            "wrong": {"prompt": [1, "wrong", graph, {"client_id": "other"}, ["12"]]},
            "found": {"prompt": [2, "found", graph, {"client_id": "intent-1"}, ["12"]]},
        })
    comfy = client_for(handler)
    assert comfy.find_submission("intent-1", "aigc/task-1") == "found"
    assert comfy.find_submission("missing", "aigc/task-1") is None


def test_history_returns_only_requested_prompt():
    entry = history_entry()

    def handler(request):
        assert request.url.path == "/history/p1"
        return httpx.Response(200, json={"p1": entry, "other": {"outputs": {}}})

    assert client_for(handler).history("p1") == entry


def test_history_missing_prompt_is_empty():
    comfy = client_for(lambda request: httpx.Response(200, json={}))
    assert comfy.history("missing") == {}


@pytest.mark.parametrize(
    "message,code,retryable",
    [
        ("CUDA out of memory", "GPU_OOM", False),
        ("NO_FACE_DETECTED: none", "NO_FACE_DETECTED", False),
        ("MULTIPLE_FACES_DETECTED: two", "MULTIPLE_FACES_DETECTED", False),
        ("OUTPUT_NO_FACE_DETECTED: none", "OUTPUT_NO_FACE_DETECTED", True),
        ("OUTPUT_MULTIPLE_FACES_DETECTED: two", "OUTPUT_MULTIPLE_FACES_DETECTED", True),
        ("execution timed out", "COMFYUI_TIMEOUT", True),
        ("Prompt outputs failed validation", "WORKFLOW_INVALID", False),
        ("unexpected execution failure /home/private/model", "COMFYUI_EXECUTION_ERROR", False),
    ],
)
def test_execution_errors_are_mapped_without_sensitive_details(message, code, retryable):
    error = ComfyUIClient.map_execution_error(message)

    assert isinstance(error, AppError)
    assert error.code == code
    assert error.retryable is retryable
    assert str(error).startswith(code)
    assert "/home/private" not in str(error)


def test_wait_for_result_handles_real_progress_and_save_image_outputs():
    events = [
        {"type": "progress", "data": {"prompt_id": "p1", "value": 3, "max": 25}},
        {"type": "progress", "data": {"value": 7, "max": 20}},
        {"type": "progress", "data": {"prompt_id": "other", "value": 9, "max": 10}},
        {"type": "executing", "data": {"prompt_id": "p1", "node": None}},
    ]
    socket = FakeWebSocket(events)
    urls = []
    options = {}

    def websocket_factory(url, **kwargs):
        urls.append(url)
        options.update(kwargs)
        return socket

    def handler(request):
        assert request.url.path == "/history/p1"
        return httpx.Response(200, json={"p1": history_entry()})

    comfy = client_for(handler, websocket_factory=websocket_factory)
    progress = []
    heartbeats = []
    result = comfy.wait_for_result(
        "p1",
        120,
        lambda: heartbeats.append("beat"),
        lambda: False,
        lambda value, maximum: progress.append((value, maximum)),
    )

    assert result.prompt_id == "p1"
    assert result.images == [
        ComfyImageRef("one.png", "aigc", "output"),
        ComfyImageRef("two.png", "", "output"),
    ]
    assert result.history["status"]["status_str"] == "success"
    assert progress == [(3, 25)]
    assert heartbeats
    assert urls == ["ws://127.0.0.1:8188/ws?client_id=p1"]
    assert options["http_no_proxy"] == ["127.0.0.1"]
    assert options["proxy_type"] is None
    assert socket.closed


def test_wait_uses_submitted_worker_id_for_websocket():
    socket = FakeWebSocket([{"type": "execution_success", "data": {"prompt_id": "p1"}}])
    seen = []

    def handler(request):
        if request.url.path == "/prompt":
            return httpx.Response(200, json={"prompt_id": "p1"})
        return httpx.Response(200, json={"p1": history_entry()})

    comfy = client_for(handler, websocket_factory=lambda url, **kwargs: (seen.append(url) or socket))
    comfy.submit({"1": {}}, "worker-7")
    comfy.wait_for_result("p1", 120, lambda: None, lambda: False, lambda value, maximum: None)

    assert seen == ["ws://127.0.0.1:8188/ws?client_id=worker-7"]


def test_wait_maps_websocket_execution_error():
    socket = FakeWebSocket([
        {"type": "execution_error", "data": {"prompt_id": "p1", "exception_message": "CUDA out of memory"}}
    ])
    comfy = client_for(
        lambda request: httpx.Response(500),
        websocket_factory=lambda url, **kwargs: socket,
    )

    with pytest.raises(AppError) as caught:
        comfy.wait_for_result("p1", 120, lambda: None, lambda: False, lambda value, maximum: None)

    assert caught.value.code == "GPU_OOM"
    assert socket.closed


def test_wait_ignores_execution_error_without_prompt_id():
    socket = FakeWebSocket([
        {"type": "execution_error", "data": {"exception_message": "CUDA out of memory"}},
        {"type": "executing", "data": {"prompt_id": "p1", "node": None}},
    ])
    comfy = client_for(
        lambda request: httpx.Response(200, json={"p1": history_entry()}),
        websocket_factory=lambda url, **kwargs: socket,
    )

    result = comfy.wait_for_result(
        "p1", 120, lambda: None, lambda: False, lambda value, maximum: None
    )

    assert result.prompt_id == "p1"


def test_wait_ignores_terminal_event_without_prompt_id():
    socket = FakeWebSocket([
        {"type": "execution_success", "data": {}},
        {"type": "executing", "data": {"prompt_id": "p1", "node": None}},
    ])

    def handler(request):
        assert socket.events == [], "history must follow the matching terminal event"
        return httpx.Response(200, json={"p1": history_entry()})

    comfy = client_for(handler, websocket_factory=lambda url, **kwargs: socket)
    result = comfy.wait_for_result(
        "p1", 120, lambda: None, lambda: False, lambda value, maximum: None
    )

    assert result.prompt_id == "p1"


def test_websocket_disconnect_falls_back_to_history_and_heartbeats():
    socket = FakeWebSocket([OSError("disconnected")])
    calls = []

    def handler(request):
        calls.append(request.url.path)
        entry = {} if len(calls) == 1 else history_entry()
        return httpx.Response(200, json={} if not entry else {"p1": entry})

    comfy = client_for(handler, websocket_factory=lambda url, **kwargs: socket)
    heartbeats = []
    result = comfy.wait_for_result(
        "p1", 120, lambda: heartbeats.append(1), lambda: False, lambda value, maximum: None
    )

    assert result.images[0].name == "one.png"
    assert calls == ["/history/p1", "/history/p1"]
    assert len(heartbeats) >= 2
    assert socket.closed


def test_history_images_remain_pending_until_explicit_terminal_success():
    socket = FakeWebSocket([OSError("disconnected")])
    entries = [history_entry("running"), history_entry()]
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"p1": entries[len(calls) - 1]})

    comfy = client_for(handler, websocket_factory=lambda url, **kwargs: socket)
    result = comfy.wait_for_result(
        "p1", 120, lambda: None, lambda: False, lambda value, maximum: None
    )

    assert calls == ["/history/p1", "/history/p1"]
    assert result.history["status"]["status_str"] == "success"
    assert result.history["status"]["completed"] is True


def test_history_error_is_mapped_from_status_messages():
    comfy = client_for(
        lambda request: httpx.Response(200, json={"p1": history_entry("error", "NO_FACE_DETECTED: none")}),
        websocket_factory=lambda url, **kwargs: (_ for _ in ()).throw(OSError("offline")),
    )

    with pytest.raises(AppError) as caught:
        comfy.wait_for_result("p1", 120, lambda: None, lambda: False, lambda value, maximum: None)

    assert caught.value.code == "NO_FACE_DETECTED"


def test_wait_timeout_cancels_before_raising():
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path, request.content))
        if request.url.path.startswith("/history/"):
            return httpx.Response(200, json={})
        if request.url.path == "/queue":
            return httpx.Response(503, json={})
        return httpx.Response(200, json={})

    comfy = client_for(
        handler,
        websocket_factory=lambda url, **kwargs: (_ for _ in ()).throw(OSError("offline")),
    )

    with pytest.raises(AppError) as caught:
        comfy.wait_for_result("p1", 1, lambda: None, lambda: False, lambda value, maximum: None)

    assert caught.value.code == "COMFYUI_TIMEOUT"
    assert caught.value.retryable is True
    assert [(method, path) for method, path, content in calls[-2:]] == [
        ("POST", "/queue"),
        ("POST", "/interrupt"),
    ]
    assert json.loads(calls[-2][2]) == {"delete": ["p1"]}


def test_wait_cancellation_cancels_and_never_connects_websocket():
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={})

    comfy = client_for(
        handler,
        websocket_factory=lambda url, **kwargs: pytest.fail("WebSocket must not connect after cancellation"),
    )

    with pytest.raises(AppError) as caught:
        comfy.wait_for_result("p1", 120, lambda: None, lambda: True, lambda value, maximum: None)

    assert caught.value.code == "CANCELLED"
    assert calls == ["/queue", "/interrupt"]


def test_cancel_deletes_queued_prompt_then_interrupts_running_prompt():
    calls = []

    def handler(request):
        calls.append((request.url.path, request.content))
        return httpx.Response(200, json={})

    client_for(handler).cancel("p1")

    assert calls[0][0] == "/queue"
    assert json.loads(calls[0][1]) == {"delete": ["p1"]}
    assert calls[1] == ("/interrupt", b"")


def test_cancel_attempts_interrupt_after_queue_delete_failure():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/queue":
            return httpx.Response(503, json={})
        return httpx.Response(200, json={})

    with pytest.raises(AppError) as caught:
        client_for(handler).cancel("p1")

    assert caught.value.code == "COMFYUI_UNAVAILABLE"
    assert paths == ["/queue", "/interrupt"]


def test_download_image_streams_to_part_then_replaces(tmp_path):
    payload = png_bytes()
    destination = tmp_path / "nested" / "result.png"

    def handler(request):
        assert request.url.path == "/view"
        assert request.url.params == httpx.QueryParams(
            {"filename": "one.png", "subfolder": "aigc", "type": "output"}
        )
        return httpx.Response(
            200,
            content=payload,
            headers={"Content-Type": "image/png", "Content-Length": str(len(payload))},
        )

    client_for(handler).download_image(ComfyImageRef("one.png", "aigc", "output"), destination)

    assert destination.read_bytes() == payload
    assert not Path(f"{destination}.part").exists()
    with Image.open(destination) as image:
        assert image.size == (2, 3)


@pytest.mark.parametrize(
    "payload,content_type,extra_size",
    [
        (b"not an image", "image/png", 0),
        (png_bytes(), "text/plain", 0),
        (png_bytes(), "image/png", 1),
        (b"", "image/png", 0),
    ],
)
def test_download_rejects_invalid_size_or_content_without_replacing_existing_file(
    tmp_path, payload, content_type, extra_size
):
    destination = tmp_path / "result.png"
    destination.write_bytes(b"existing")

    def handler(request):
        return httpx.Response(
            200,
            content=payload,
            headers={"Content-Type": content_type, "Content-Length": str(len(payload) + extra_size)},
        )

    comfy = client_for(handler)
    with pytest.raises(AppError) as caught:
        comfy.download_image(ComfyImageRef("one.png", type="output"), destination)

    assert caught.value.code == "RESULT_MISSING"
    assert destination.read_bytes() == b"existing"
    assert not Path(f"{destination}.part").exists()
    assert str(tmp_path) not in str(caught.value)
