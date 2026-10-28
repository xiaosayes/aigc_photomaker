# Server-pending: run these real SQLite behavior tests on the GPU server.
from concurrent.futures import ThreadPoolExecutor

import pytest

from aigc_photomaker.db import connect
from aigc_photomaker.repository import TaskRepository


def test_sqlite_durability_pragmas(tmp_path):
    with connect(tmp_path / "tasks.db") as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 1
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def make_repository(tmp_path):
    repo = TaskRepository(tmp_path / "tasks.db")
    repo.create_schema()
    return repo


def test_claim_is_atomic_and_recovers_expired_running_task(tmp_path):
    repo = make_repository(tmp_path)
    created = repo.create(
        task_id="t1",
        input_path="inputs/t1.png",
        style="cyberpunk",
        mode="standard",
        seed=42,
    )
    assert created.status == "queued"
    assert created.output_path == "outputs/t1.png"
    assert created.timings == {}
    assert created.error is None

    first = repo.claim("worker-a", lease_seconds=1)
    assert first.status == "running"
    assert first.claim_token
    assert repo.claim("worker-b", lease_seconds=1) is None
    assert repo.mark_submitted("t1", first.claim_token, "prompt-123")

    repo.force_lease_expiry("t1")
    recovered = repo.claim("worker-b", lease_seconds=60)
    assert recovered.task_id == "t1"
    assert recovered.claim_token != first.claim_token
    assert recovered.prompt_id == "prompt-123"
    assert not repo.heartbeat("t1", first.claim_token, 60)


def test_claim_prioritizes_queued_work_over_expired_running_work(tmp_path):
    repo = make_repository(tmp_path)
    repo.create("expired", "inputs/expired.png", "anime", "turbo", 1)
    first = repo.claim("worker-a", 1)
    repo.create("queued", "inputs/queued.png", "ink", "standard", 2)
    repo.force_lease_expiry(first.task_id)

    queued = repo.claim("worker-b", 60)
    recovered = repo.claim("worker-c", 60)

    assert queued.task_id == "queued"
    assert recovered.task_id == "expired"
    assert repo.claim("worker-d", 60) is None


def test_concurrent_claimers_receive_only_one_claim(tmp_path):
    make_repository(tmp_path).create("t1", "inputs/t1.png", "anime", "turbo", 7)

    def claim(_):
        repo = TaskRepository(tmp_path / "tasks.db")
        return repo.claim("worker", 60)

    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(claim, range(8)))

    claimed = [task for task in claims if task is not None]
    assert len(claimed) == 1
    assert claimed[0].task_id == "t1"


def test_new_claim_fences_every_old_worker_mutation(tmp_path):
    repo = make_repository(tmp_path)
    repo.create("t1", "inputs/t1.png", "oil", "standard", 11)
    first = repo.claim("worker-a", 1)
    repo.force_lease_expiry("t1")
    current = repo.claim("worker-b", 60)

    assert not repo.heartbeat("t1", first.claim_token, 60)
    assert not repo.mark_submitted("t1", first.claim_token, "old-prompt")
    assert not repo.update_progress("t1", first.claim_token, "executing", 2, 8)
    assert not repo.succeed("t1", first.claim_token, {"total": 1.0})
    assert not repo.fail(
        "t1",
        first.claim_token,
        {"code": "INTERNAL_ERROR", "message": "old worker", "retryable": True},
    )
    assert not repo.mark_cancelled("t1", first.claim_token)
    assert repo.request_cancel("t1", first.claim_token).cancel_requested is False

    task = repo.get("t1")
    assert task.status == "running"
    assert task.claim_token == current.claim_token
    assert task.prompt_id is None
    assert task.progress_phase == "queued"
    assert task.timings == {}
    assert task.error is None


def test_current_worker_can_heartbeat_submit_and_update_progress(tmp_path):
    repo = make_repository(tmp_path)
    repo.create("t1", "inputs/t1.png", "cyberpunk", "standard", 42)
    task = repo.claim("worker", 1)
    initial_lease = task.lease_expires_at

    assert repo.heartbeat("t1", task.claim_token, 60)
    assert repo.mark_submitted("t1", task.claim_token, "prompt-1")
    assert repo.update_progress("t1", task.claim_token, "executing", 3, 25)

    updated = repo.get("t1")
    assert updated.prompt_id == "prompt-1"
    assert updated.lease_expires_at > initial_lease
    assert updated.progress_phase == "executing"
    assert updated.progress_value == 3
    assert updated.progress_max == 25


def test_succeed_persists_relative_output_and_timings_and_is_terminal(tmp_path):
    repo = make_repository(tmp_path)
    repo.create("t1", "inputs/t1.png", "anime", "turbo", 7)
    task = repo.claim("worker", 60)
    timings = {"submit": 0.2, "execute": 4.5, "total": 5.0}

    assert repo.succeed("t1", task.claim_token, timings, "outputs/custom.png")
    succeeded = repo.get("t1")
    assert succeeded.status == "succeeded"
    assert succeeded.output_path == "outputs/custom.png"
    assert succeeded.timings == timings
    assert succeeded.finished_at is not None

    assert not repo.succeed("t1", task.claim_token, {"total": 99.0})
    assert not repo.fail(
        "t1", task.claim_token, {"code": "INTERNAL_ERROR", "message": "late", "retryable": False}
    )
    assert not repo.mark_cancelled("t1", task.claim_token)
    assert repo.request_cancel("t1").status == "succeeded"
    assert repo.get("t1") == succeeded


def test_fail_persists_structured_error_and_is_terminal(tmp_path):
    repo = make_repository(tmp_path)
    repo.create("t1", "inputs/t1.png", "ink", "standard", 9)
    task = repo.claim("worker", 60)
    error = {"code": "GPU_OOM", "message": "Out of memory", "retryable": False}

    assert repo.fail("t1", task.claim_token, error)
    failed = repo.get("t1")
    assert failed.status == "failed"
    assert failed.error == error
    assert failed.finished_at is not None

    assert not repo.update_progress("t1", task.claim_token, "executing", 1, 2)
    assert repo.request_cancel("t1").status == "failed"
    assert repo.get("t1") == failed


def test_cancel_queued_task(tmp_path):
    repo = make_repository(tmp_path)
    created = repo.create("t2", "inputs/t2.png", "anime", "turbo", 7)

    cancelled = repo.request_cancel("t2")
    assert cancelled.status == "cancelled"
    assert cancelled.cancel_requested is True
    assert cancelled.finished_at is not None
    assert repo.claim("worker", 60) is None
    assert repo.request_cancel("t2") == cancelled
    assert repo.get("t2") == cancelled
    assert created.status == "queued"


def test_running_cancellation_requires_current_token_to_finish(tmp_path):
    repo = make_repository(tmp_path)
    repo.create("t1", "inputs/t1.png", "oil", "standard", 3)
    first = repo.claim("worker-a", 1)
    repo.force_lease_expiry("t1")
    current = repo.claim("worker-b", 60)

    requested = repo.request_cancel("t1")
    assert requested.status == "running"
    assert requested.cancel_requested is True
    assert not repo.mark_cancelled("t1", first.claim_token)
    assert repo.mark_cancelled("t1", current.claim_token)

    cancelled = repo.get("t1")
    assert cancelled.status == "cancelled"
    assert not repo.fail(
        "t1", current.claim_token, {"code": "INTERNAL_ERROR", "message": "late", "retryable": False}
    )
    assert repo.get("t1") == cancelled


INVALID_TASK_PATHS = [
    r"C:\face.png",
    "C:/face.png",
    r"C:face.png",
    r"\face.png",
    r"\\server\share\face.png",
    "//server/share/face.png",
    "/face.png",
    "../face.png",
    r"..\face.png",
    "inputs/../../face.png",
    r"inputs\face.png",
]


@pytest.mark.parametrize("path", INVALID_TASK_PATHS)
def test_rejects_non_relative_input_paths(tmp_path, path):
    repo = make_repository(tmp_path)

    with pytest.raises(ValueError):
        repo.create("invalid", path, "anime", "turbo", 1)

    assert repo.get("invalid") is None


@pytest.mark.parametrize("path", INVALID_TASK_PATHS)
def test_rejects_non_relative_output_paths(tmp_path, path):
    repo = make_repository(tmp_path)
    repo.create("valid", "inputs/foo.png", "anime", "turbo", 1)
    claimed = repo.claim("worker", 60)

    with pytest.raises(ValueError):
        repo.succeed("valid", claimed.claim_token, {}, path)

    task = repo.get("valid")
    assert task.status == "running"
    assert task.output_path == "outputs/valid.png"


def test_accepts_safe_posix_relative_input_and_output_paths(tmp_path):
    repo = make_repository(tmp_path)

    task = repo.create(
        "valid",
        "inputs/foo.png",
        "anime",
        "turbo",
        1,
        output_path="outputs/foo.png",
    )

    assert task.input_path == "inputs/foo.png"
    assert task.output_path == "outputs/foo.png"
