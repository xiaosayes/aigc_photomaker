import json
import time
import uuid
from collections.abc import Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath

from .db import SCHEMA_SQL, connect
from .models import TaskRecord


def _relative_data_path(value: str) -> str:
    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        not value
        or "\\" in value
        or windows_path.drive
        or windows_path.root
        or posix_path.is_absolute()
        or ".." in posix_path.parts
    ):
        raise ValueError("Task paths must be relative to the data directory")
    return value


class TaskRepository:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def create_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with connect(self.path) as connection:
            with connection:
                connection.executescript(SCHEMA_SQL)
                columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(tasks)")
                }
                migrations = {
                    "submission_client_id": "ALTER TABLE tasks ADD COLUMN submission_client_id TEXT",
                    "output_prefix": "ALTER TABLE tasks ADD COLUMN output_prefix TEXT",
                    "execution_refs_json": (
                        "ALTER TABLE tasks ADD COLUMN execution_refs_json TEXT NOT NULL DEFAULT '[]'"
                    ),
                }
                for name, statement in migrations.items():
                    if name not in columns:
                        connection.execute(statement)

    def create(
        self,
        task_id: str,
        input_path: str,
        style: str,
        mode: str,
        seed: int,
        output_path: str | None = None,
    ) -> TaskRecord:
        input_path = _relative_data_path(input_path)
        if output_path is None:
            output_path = f"outputs/{task_id}.png"
        output_path = _relative_data_path(output_path)
        now = time.time()
        with connect(self.path) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO tasks (
                        id, input_path, output_path, style, mode, seed, created_at
                    ) VALUES (
                        :task_id, :input_path, :output_path, :style, :mode, :seed, :created_at
                    )
                    """,
                    {
                        "task_id": task_id,
                        "input_path": input_path,
                        "output_path": output_path,
                        "style": style,
                        "mode": mode,
                        "seed": seed,
                        "created_at": now,
                    },
                )
        task = self.get(task_id)
        assert task is not None
        return task

    def ping(self) -> bool:
        """Return whether the task database can execute a minimal query."""
        with connect(self.path) as connection:
            connection.execute("SELECT 1").fetchone()
        return True

    def get(self, task_id: str) -> TaskRecord | None:
        with connect(self.path) as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return TaskRecord.from_row(row) if row is not None else None

    def claim(self, worker_id: str, lease_seconds: float) -> TaskRecord | None:
        token = uuid.uuid4().hex
        now = time.time()
        with connect(self.path) as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE tasks
                    SET status='running', worker_id=:worker_id, claim_token=:token,
                        lease_expires_at=:now + :lease, heartbeat_at=:now,
                        started_at=COALESCE(started_at, :now)
                    WHERE id=(
                      SELECT id FROM tasks
                      WHERE status='queued' OR (status='running' AND lease_expires_at < :now)
                      ORDER BY CASE status WHEN 'queued' THEN 0 ELSE 1 END, created_at
                      LIMIT 1
                    )
                    """,
                    {
                        "worker_id": worker_id,
                        "token": token,
                        "now": now,
                        "lease": lease_seconds,
                    },
                )
            row = connection.execute(
                "SELECT * FROM tasks WHERE claim_token = ?", (token,)
            ).fetchone()
        return TaskRecord.from_row(row) if row is not None else None

    def _fenced_update(
        self,
        task_id: str,
        claim_token: str,
        assignments: str,
        parameters: Mapping[str, object] | None = None,
        require_not_cancelled: bool = False,
    ) -> bool:
        values: dict[str, object] = {"task_id": task_id, "claim_token": claim_token}
        if parameters:
            values.update(parameters)
        with connect(self.path) as connection:
            with connection:
                cancel_guard = " AND cancel_requested=0" if require_not_cancelled else ""
                cursor = connection.execute(
                    f"""
                    UPDATE tasks SET {assignments}
                    WHERE id=:task_id AND status='running' AND claim_token=:claim_token
                    {cancel_guard}
                    """,
                    values,
                )
        return cursor.rowcount == 1

    def heartbeat(
        self, task_id: str, claim_token: str, lease_seconds: float
    ) -> bool:
        return self._fenced_update(
            task_id,
            claim_token,
            "lease_expires_at=:now + :lease, heartbeat_at=:now",
            {"now": time.time(), "lease": lease_seconds},
        )

    def prepare_submission(
        self,
        task_id: str,
        claim_token: str,
        client_id: str,
        output_prefix: str,
    ) -> bool:
        if not client_id or not output_prefix:
            raise ValueError("Submission intent and output prefix are required")
        return self._fenced_update(
            task_id,
            claim_token,
            "submission_client_id=:client_id, output_prefix=:output_prefix",
            {"client_id": client_id, "output_prefix": output_prefix},
        )

    def record_execution_ref(
        self,
        task_id: str,
        claim_token: str,
        ref: Mapping[str, str],
    ) -> bool:
        normalized = {key: ref.get(key) for key in ("name", "subfolder", "type")}
        if (
            not isinstance(normalized["name"], str)
            or not normalized["name"]
            or not isinstance(normalized["subfolder"], str)
            or normalized["type"] not in {"input", "output", "temp"}
        ):
            raise ValueError("Invalid ComfyUI execution reference")
        with connect(self.path) as connection:
            with connection:
                row = connection.execute(
                    "SELECT execution_refs_json FROM tasks "
                    "WHERE id=? AND status='running' AND claim_token=?",
                    (task_id, claim_token),
                ).fetchone()
                if row is None:
                    return False
                refs = json.loads(row[0])
                if normalized not in refs:
                    refs.append(normalized)
                cursor = connection.execute(
                    "UPDATE tasks SET execution_refs_json=? "
                    "WHERE id=? AND status='running' AND claim_token=?",
                    (json.dumps(refs), task_id, claim_token),
                )
        return cursor.rowcount == 1

    def mark_submitted(
        self, task_id: str, claim_token: str, prompt_id: str
    ) -> bool:
        return self._fenced_update(
            task_id,
            claim_token,
            "prompt_id=:prompt_id, progress_phase='submitted'",
            {"prompt_id": prompt_id},
        )

    def update_progress(
        self,
        task_id: str,
        claim_token: str,
        phase: str,
        value: int,
        maximum: int,
    ) -> bool:
        return self._fenced_update(
            task_id,
            claim_token,
            """progress_phase=:phase, progress_value=:value,
               progress_max=:maximum, heartbeat_at=:now""",
            {
                "phase": phase,
                "value": value,
                "maximum": maximum,
                "now": time.time(),
            },
        )

    def succeed(
        self,
        task_id: str,
        claim_token: str,
        timings: Mapping[str, float],
        output_path: str | None = None,
    ) -> bool:
        assignments = "status='succeeded', timings_json=:timings, finished_at=:now"
        parameters: dict[str, object] = {
            "timings": json.dumps(dict(timings)),
            "now": time.time(),
        }
        if output_path is not None:
            assignments += ", output_path=:output_path"
            parameters["output_path"] = _relative_data_path(output_path)
        return self._fenced_update(
            task_id, claim_token, assignments, parameters, require_not_cancelled=True
        )

    def fail(
        self,
        task_id: str,
        claim_token: str,
        error: Mapping[str, object],
    ) -> bool:
        return self._fenced_update(
            task_id,
            claim_token,
            "status='failed', error_json=:error, finished_at=:now",
            {"error": json.dumps(dict(error)), "now": time.time()},
            require_not_cancelled=True,
        )

    def request_cancel(
        self, task_id: str, claim_token: str | None = None
    ) -> TaskRecord | None:
        now = time.time()
        with connect(self.path) as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE tasks
                    SET status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END,
                        cancel_requested=1,
                        finished_at=CASE WHEN status='queued' THEN :now ELSE finished_at END
                    WHERE id=:task_id
                      AND status IN ('queued', 'running')
                      AND (:claim_token IS NULL OR claim_token=:claim_token)
                    """,
                    {"task_id": task_id, "claim_token": claim_token, "now": now},
                )
        return self.get(task_id)

    def mark_cancelled(self, task_id: str, claim_token: str) -> bool:
        return self._fenced_update(
            task_id,
            claim_token,
            "status='cancelled', cancel_requested=1, finished_at=:now",
            {"now": time.time()},
        )

    def list_expired_terminal(self, cutoff: float) -> list[TaskRecord]:
        with connect(self.path) as connection:
            rows = connection.execute(
                """
                SELECT * FROM tasks
                WHERE status IN ('succeeded', 'failed', 'cancelled')
                  AND finished_at IS NOT NULL AND finished_at < :cutoff
                ORDER BY finished_at, id
                """,
                {"cutoff": cutoff},
            ).fetchall()
        return [TaskRecord.from_row(row) for row in rows]

    def delete_terminal(self, task_id: str, cutoff: float) -> bool:
        with connect(self.path) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    DELETE FROM tasks
                    WHERE id=:task_id
                      AND status IN ('succeeded', 'failed', 'cancelled')
                      AND finished_at IS NOT NULL AND finished_at < :cutoff
                    """,
                    {"task_id": task_id, "cutoff": cutoff},
                )
        return cursor.rowcount == 1

    def force_lease_expiry(self, task_id: str) -> None:
        """Internal fault-injection hook; intentionally not exposed by an API."""
        with connect(self.path) as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE tasks SET lease_expires_at=0
                    WHERE id=:task_id AND status='running'
                    """,
                    {"task_id": task_id},
                )
