import json
import sqlite3
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TaskRecord:
    task_id: str
    input_path: str
    output_path: str
    style: str
    mode: str
    seed: int
    status: str
    worker_id: str | None
    prompt_id: str | None
    submission_client_id: str | None
    output_prefix: str | None
    execution_refs: list[dict[str, str]]
    claim_token: str | None
    lease_expires_at: float | None
    heartbeat_at: float | None
    cancel_requested: bool
    progress_phase: str
    progress_value: int
    progress_max: int
    timings: dict[str, float]
    error: dict[str, Any] | None
    created_at: float
    started_at: float | None
    finished_at: float | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "TaskRecord":
        return cls(
            task_id=row["id"],
            input_path=row["input_path"],
            output_path=row["output_path"],
            style=row["style"],
            mode=row["mode"],
            seed=row["seed"],
            status=row["status"],
            worker_id=row["worker_id"],
            prompt_id=row["prompt_id"],
            submission_client_id=row["submission_client_id"],
            output_prefix=row["output_prefix"],
            execution_refs=json.loads(row["execution_refs_json"]),
            claim_token=row["claim_token"],
            lease_expires_at=row["lease_expires_at"],
            heartbeat_at=row["heartbeat_at"],
            cancel_requested=bool(row["cancel_requested"]),
            progress_phase=row["progress_phase"],
            progress_value=row["progress_value"],
            progress_max=row["progress_max"],
            timings=json.loads(row["timings_json"]),
            error=json.loads(row["error_json"]) if row["error_json"] else None,
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )
