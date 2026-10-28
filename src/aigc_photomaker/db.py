import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    input_path TEXT NOT NULL,
    output_path TEXT NOT NULL,
    style TEXT NOT NULL,
    mode TEXT NOT NULL,
    seed INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    worker_id TEXT,
    prompt_id TEXT,
    submission_client_id TEXT,
    output_prefix TEXT,
    execution_refs_json TEXT NOT NULL DEFAULT '[]',
    claim_token TEXT UNIQUE,
    lease_expires_at REAL,
    heartbeat_at REAL,
    cancel_requested INTEGER NOT NULL DEFAULT 0 CHECK (cancel_requested IN (0, 1)),
    progress_phase TEXT NOT NULL DEFAULT 'queued',
    progress_value INTEGER NOT NULL DEFAULT 0,
    progress_max INTEGER NOT NULL DEFAULT 0,
    timings_json TEXT NOT NULL DEFAULT '{}',
    error_json TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);

CREATE INDEX IF NOT EXISTS tasks_claim_idx
    ON tasks(status, created_at);
"""


@contextmanager
def connect(path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open a short-lived connection with the task-store durability settings."""
    connection = sqlite3.connect(path, timeout=30, check_same_thread=False)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        yield connection
    finally:
        connection.close()
