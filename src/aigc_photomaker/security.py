import math
import secrets
import threading

from pydantic import SecretStr

from aigc_photomaker.errors import AppError


def verify_api_key(provided: str, expected: SecretStr) -> bool:
    return secrets.compare_digest(provided, expected.get_secret_value())


class GenerationRateLimiter:
    def __init__(self, limit: int = 20, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._windows: dict[str, tuple[int, int]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, now: float) -> bool:
        window_index = math.floor(now / self.window_seconds)
        with self._lock:
            stale_keys = [
                stored_key
                for stored_key, (stored_window, _) in self._windows.items()
                if stored_window < window_index
            ]
            for stale_key in stale_keys:
                del self._windows[stale_key]

            stored_window, count = self._windows.get(key, (window_index, 0))
            if stored_window != window_index:
                count = 0
            if count >= self.limit:
                return False
            self._windows[key] = (window_index, count + 1)
            return True

    def allow_or_raise(self, key: str, now: float) -> None:
        if not self.allow(key, now):
            raise AppError(429, "RATE_LIMITED", "生成请求过于频繁，请稍后重试", True)
