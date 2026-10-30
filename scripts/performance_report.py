#!/usr/bin/env python3
"""Summarize real task timings from data/tasks.db without creating new tasks."""

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGES = ("upload", "submit", "execute", "download", "total")


def load_tasks(database, limit):
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            """SELECT id, style, mode, seed, status, prompt_id,
                      created_at, started_at, finished_at, timings_json
               FROM tasks ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    finally:
        connection.close()
    tasks = []
    for row in rows:
        item = dict(row)
        item["timings"] = json.loads(item.pop("timings_json") or "{}")
        tasks.append(item)
    return tasks


def summarize(tasks):
    grouped = defaultdict(list)
    for task in tasks:
        if task["status"] == "succeeded":
            grouped[task["mode"]].append(task["timings"])
    report = {}
    for mode, timings in grouped.items():
        stage_report = {}
        for stage in STAGES:
            values = [float(item[stage]) for item in timings if stage in item]
            if values:
                stage_report[stage] = {
                    "count": len(values),
                    "mean": statistics.fmean(values),
                    "min": min(values),
                    "max": max(values),
                    "median": statistics.median(values),
                }
        report[mode] = {"successful_tasks": len(timings), "stages": stage_report}
    return report


def print_report(tasks):
    print("MODE SUMMARY:")
    print(json.dumps(summarize(tasks), ensure_ascii=False, indent=2))
    print("\nRECENT TASKS (newest first):")
    for task in tasks:
        created = datetime.fromtimestamp(task["created_at"], tz=timezone.utc).isoformat()
        timings = " ".join(
            f"{stage}={task['timings'][stage]:.3f}s"
            for stage in STAGES if stage in task["timings"]
        )
        print(
            f"{created} | {task['mode']:8s} | {task['style']:12s} | "
            f"seed={task['seed']} | {task['status']:9s} | {timings}"
        )
        print(f"  task_id={task['id']} prompt_id={task['prompt_id']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20, help="Number of recent tasks (1-200)")
    args = parser.parse_args()
    if not 1 <= args.limit <= 200:
        parser.error("--limit must be between 1 and 200")
    database = PROJECT_ROOT / "data" / "tasks.db"
    if not database.is_file():
        parser.error(f"Task database is missing: {database}")
    tasks = load_tasks(database, args.limit)
    if not tasks:
        print("No tasks found.")
        return 0
    print_report(tasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
