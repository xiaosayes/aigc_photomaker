import errno
import sqlite3


class AppError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        retryable: bool = False,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retryable = retryable
        super().__init__(f"{code}: {message}")

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }


def disk_space_error(exc: BaseException) -> AppError | None:
    """Map local disk exhaustion without exposing filesystem or database details."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, OSError) and current.errno == errno.ENOSPC:
            break
        if isinstance(current, sqlite3.OperationalError) and any(
            marker in str(current).lower()
            for marker in ("database or disk is full", "database is full", "disk is full")
        ):
            break
        current = current.__cause__ or current.__context__
    else:
        return None
    return AppError(507, "DISK_SPACE_LOW", "服务器磁盘空间不足", True)
