# Server-pending: timing summary tests using synthetic task records.
import importlib.util
from pathlib import Path


path = Path(__file__).parents[1] / "scripts" / "performance_report.py"
spec = importlib.util.spec_from_file_location("performance_report", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_summary_groups_only_succeeded_tasks_and_computes_stage_statistics():
    tasks = [
        {"mode": "standard", "status": "succeeded", "timings": {"execute": 50.0, "total": 55.0}},
        {"mode": "standard", "status": "succeeded", "timings": {"execute": 70.0, "total": 75.0}},
        {"mode": "turbo", "status": "succeeded", "timings": {"execute": 8.0, "total": 10.0}},
        {"mode": "standard", "status": "failed", "timings": {"execute": 999.0}},
    ]

    report = module.summarize(tasks)

    standard = report["standard"]["stages"]
    assert standard["execute"] == {"count": 2, "mean": 60.0, "min": 50.0, "max": 70.0, "median": 60.0}
    assert standard["total"]["median"] == 65.0
    assert report["turbo"]["successful_tasks"] == 1
    assert "failed" not in report


def test_summary_handles_missing_stages_and_empty_tasks():
    assert module.summarize([]) == {}
    tasks = [{"mode": "standard", "status": "succeeded", "timings": {"total": 12.0}}]
    report = module.summarize(tasks)
    assert set(report["standard"]["stages"]) == {"total"}
