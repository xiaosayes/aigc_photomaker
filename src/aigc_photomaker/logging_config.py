import logging
import time

from pythonjsonlogger import jsonlogger


def configure_logging(component: str) -> None:
    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(levelname)s %(message)s %(component)s %(task_id)s %(duration_ms)s",
        rename_fields={
            "asctime": "timestamp",
            "levelname": "level",
            "message": "event",
        },
        defaults={"component": component, "task_id": None, "duration_ms": None},
        json_ensure_ascii=False,
    )
    formatter.converter = time.gmtime
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
