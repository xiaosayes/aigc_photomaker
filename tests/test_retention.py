# Server-pending: run these real-SQLite retention behavior tests on the GPU server.
import pytest

from aigc_photomaker.config import Settings
from aigc_photomaker.db import connect
from aigc_photomaker.repository import TaskRepository
from aigc_photomaker.retention import RetentionCleaner


NOW = 2_000_000_000.0


def setup_cleaner(tmp_path, retention_days=7):
    settings = Settings(
        api_key="a" * 32,
        _env_file=None,
        data_dir=tmp_path / "data",
        comfy_dir=tmp_path / "comfy",
        retention_days=retention_days,
    )
    data_dir = settings.resolve(settings.data_dir)
    data_dir.mkdir(parents=True)
    repo = TaskRepository(tmp_path / "tasks.db")
    repo.create_schema()
    return settings, data_dir, repo, RetentionCleaner(settings, repo)


def finish_task(repo, task_id, status, finished_at):
    claimed = repo.claim("worker", 60)
    assert claimed.task_id == task_id
    if status == "succeeded":
        assert repo.succeed(task_id, claimed.claim_token, {"total": 1.0})
    elif status == "failed":
        assert repo.fail(
            task_id,
            claimed.claim_token,
            {"code": "GPU_OOM", "message": "OOM", "retryable": False},
        )
    else:
        raise ValueError(status)
    with connect(repo.path) as connection:
        with connection:
            connection.execute(
                "UPDATE tasks SET finished_at=? WHERE id=?", (finished_at, task_id)
            )


def create_files(data_dir, task):
    paths = []
    for relative in (task.input_path, task.output_path):
        path = data_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
        paths.append(path)
        sidecar = path.with_name(path.name + ".part")
        sidecar.write_bytes(b"partial")
        paths.append(sidecar)
    return paths


def test_cleanup_deletes_only_expired_terminal_tasks_and_known_files(tmp_path):
    _, data_dir, repo, cleaner = setup_cleaner(tmp_path)
    expired = repo.create("expired", "inputs/expired.png", "portrait", "standard", 1)
    finish_task(repo, "expired", "succeeded", NOW - 8 * 86400)
    recent = repo.create("recent", "inputs/recent.png", "portrait", "standard", 2)
    finish_task(repo, "recent", "failed", NOW - 6 * 86400)
    running = repo.create("running", "inputs/running.png", "portrait", "standard", 3)
    claimed = repo.claim("worker", 60)
    assert claimed.task_id == "running"
    with connect(repo.path) as connection:
        with connection:
            connection.execute(
                "UPDATE tasks SET finished_at=? WHERE id='running'", (NOW - 30 * 86400,)
            )

    expired_files = create_files(data_dir, repo.get("expired"))
    recent_files = create_files(data_dir, repo.get("recent"))
    running_files = create_files(data_dir, repo.get("running"))
    unrelated = data_dir / "outputs" / "expired.png.tmp"
    unrelated.write_bytes(b"keep")

    assert cleaner.cleanup(NOW) == 1
    assert all(not path.exists() for path in expired_files)
    assert all(path.exists() for path in recent_files + running_files)
    assert unrelated.exists()
    assert repo.get("expired") is None
    assert repo.get("recent").status == "failed"
    assert repo.get("running").status == "running"


def test_cleanup_covers_cancelled_and_uses_strict_cutoff(tmp_path):
    _, data_dir, repo, cleaner = setup_cleaner(tmp_path)
    cancelled = repo.create("cancelled", "inputs/cancelled.png", "portrait", "turbo", 1)
    repo.request_cancel("cancelled")
    boundary = repo.create("boundary", "inputs/boundary.png", "portrait", "turbo", 2)
    finish_task(repo, "boundary", "succeeded", NOW - 7 * 86400)
    with connect(repo.path) as connection:
        with connection:
            connection.execute(
                "UPDATE tasks SET finished_at=? WHERE id='cancelled'",
                (NOW - 7 * 86400 - 0.001,),
            )
    cancelled_files = create_files(data_dir, repo.get(cancelled.task_id))
    boundary_files = create_files(data_dir, repo.get(boundary.task_id))

    assert cleaner.cleanup(NOW) == 1
    assert all(not path.exists() for path in cancelled_files)
    assert all(path.exists() for path in boundary_files)
    assert repo.get("cancelled") is None
    assert repo.get("boundary") is not None


def test_cleanup_does_not_delete_database_record_when_file_removal_fails(tmp_path, monkeypatch):
    _, data_dir, repo, cleaner = setup_cleaner(tmp_path)
    repo.create("expired", "inputs/expired.png", "portrait", "standard", 1)
    finish_task(repo, "expired", "succeeded", NOW - 8 * 86400)
    files = create_files(data_dir, repo.get("expired"))

    original_unlink = type(files[0]).unlink

    def fail_output(path, *args, **kwargs):
        if path == files[1]:
            raise OSError("disk busy")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(type(files[0]), "unlink", fail_output)
    assert cleaner.cleanup(NOW) == 0

    assert repo.get("expired") is not None
    assert files[1].exists()


def test_cleanup_rejects_unsafe_stored_path_without_deleting_anything(tmp_path):
    _, data_dir, repo, cleaner = setup_cleaner(tmp_path)
    repo.create("expired", "inputs/expired.png", "portrait", "standard", 1)
    finish_task(repo, "expired", "succeeded", NOW - 8 * 86400)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"private")
    with connect(repo.path) as connection:
        with connection:
            connection.execute(
                "UPDATE tasks SET input_path='../outside.png' WHERE id='expired'"
            )

    assert cleaner.cleanup(NOW) == 0

    assert outside.exists()
    assert repo.get("expired") is not None


def test_cleanup_validates_all_paths_and_isolates_failures_per_task(tmp_path, caplog):
    _, data_dir, repo, cleaner = setup_cleaner(tmp_path)
    for task_id, age in (("first", 30), ("unsafe", 20), ("last", 10)):
        repo.create(task_id, f"inputs/{task_id}.png", "portrait", "standard", 1)
        finish_task(repo, task_id, "succeeded", NOW - age * 86400)
    first_files = create_files(data_dir, repo.get("first"))
    unsafe_files = create_files(data_dir, repo.get("unsafe"))
    last_files = create_files(data_dir, repo.get("last"))
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"private")
    with connect(repo.path) as connection:
        with connection:
            connection.execute(
                "UPDATE tasks SET output_path='../outside.png' WHERE id='unsafe'"
            )

    assert cleaner.cleanup(NOW) == 2

    assert all(not path.exists() for path in first_files + last_files)
    assert all(path.exists() for path in unsafe_files)
    assert outside.read_bytes() == b"private"
    assert repo.get("first") is None
    assert repo.get("unsafe") is not None
    assert repo.get("last") is None
    assert "unsafe" in caplog.text


def test_hourly_cleanup_logs_and_survives_pass_failure(tmp_path, monkeypatch, caplog):
    _, _, _, cleaner = setup_cleaner(tmp_path)
    monkeypatch.setattr("aigc_photomaker.retention.RETENTION_INTERVAL_SECONDS", 0)
    calls = []

    def flaky_cleanup(now):
        calls.append(now)
        if len(calls) == 1:
            raise RuntimeError("database unavailable")
        cleaner.stop()
        return 0

    monkeypatch.setattr(cleaner, "cleanup", flaky_cleanup)
    cleaner.run_forever()

    assert len(calls) == 2
    assert "database unavailable" in caplog.text


def test_repository_expired_terminal_operations_are_cutoff_fenced(tmp_path):
    _, _, repo, _ = setup_cleaner(tmp_path)
    repo.create("old", "inputs/old.png", "portrait", "standard", 1)
    finish_task(repo, "old", "failed", NOW - 100)
    repo.create("new", "inputs/new.png", "portrait", "standard", 2)
    finish_task(repo, "new", "succeeded", NOW - 10)
    repo.create("running", "inputs/running.png", "portrait", "standard", 3)
    assert repo.claim("worker", 60).task_id == "running"

    expired = repo.list_expired_terminal(NOW - 50)
    assert [task.task_id for task in expired] == ["old"]
    assert not repo.delete_terminal("running", NOW - 50)
    assert not repo.delete_terminal("new", NOW - 50)
    assert repo.delete_terminal("old", NOW - 50)
    assert not repo.delete_terminal("old", NOW - 50)
    assert repo.get("new") is not None
    assert repo.get("running") is not None
