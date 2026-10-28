# Server-pending: run these dependency/behavior tests on the GPU server.
import threading

import pytest
from pydantic import SecretStr

from aigc_photomaker.errors import AppError
from aigc_photomaker.security import GenerationRateLimiter, verify_api_key


def test_api_key_accepts_only_the_expected_non_empty_value(monkeypatch):
    expected = SecretStr("a" * 32)
    calls: list[tuple[str, str]] = []

    def fake_compare_digest(provided: str, wanted: str) -> bool:
        calls.append((provided, wanted))
        return provided == wanted

    monkeypatch.setattr(
        "aigc_photomaker.security.secrets.compare_digest", fake_compare_digest
    )

    assert verify_api_key("a" * 32, expected)
    assert not verify_api_key("wrong", expected)
    assert not verify_api_key("", expected)

    assert calls == [
        ("a" * 32, "a" * 32),
        ("wrong", "a" * 32),
        ("", "a" * 32),
    ]


def test_twenty_first_generation_in_window_is_rejected():
    limiter = GenerationRateLimiter()

    assert all(limiter.allow("client", second) for second in range(20))
    assert not limiter.allow("client", 20.0)

    with pytest.raises(AppError) as raised:
        limiter.allow_or_raise("client", 21.0)

    assert raised.value.status_code == 429
    assert raised.value.code == "RATE_LIMITED"
    assert raised.value.retryable is True


def test_next_fixed_window_restores_generation_capacity():
    limiter = GenerationRateLimiter()

    assert all(limiter.allow("client", second) for second in range(20))
    assert not limiter.allow("client", 59.999)
    assert limiter.allow("client", 60.0)
    assert limiter.allow_or_raise("client", 60.1) is None


def test_rate_limit_is_independent_per_key():
    limiter = GenerationRateLimiter(limit=1)

    assert limiter.allow("first", 0.0)
    assert not limiter.allow("first", 1.0)
    assert limiter.allow("second", 1.0)


def test_custom_limit_and_window_are_used():
    limiter = GenerationRateLimiter(limit=2, window_seconds=10.0)

    assert limiter.allow("client", 5.0)
    assert limiter.allow("client", 9.0)
    assert not limiter.allow("client", 9.999)
    assert limiter.allow("client", 10.0)


def test_old_windows_are_cleaned_up():
    limiter = GenerationRateLimiter()

    assert limiter.allow("old", 0.0)
    assert limiter.allow("new", 120.0)

    assert "old" not in limiter._windows


def test_allow_or_raise_does_not_consume_capacity_after_limit():
    limiter = GenerationRateLimiter(limit=1)

    assert limiter.allow("client", 0.0)
    for now in (1.0, 2.0):
        with pytest.raises(AppError):
            limiter.allow_or_raise("client", now)
    assert limiter.allow("client", 60.0)


def test_rate_limiter_is_thread_safe():
    limiter = GenerationRateLimiter()
    barrier = threading.Barrier(40)
    results: list[bool] = []
    results_lock = threading.Lock()

    def reserve() -> None:
        barrier.wait()
        allowed = limiter.allow("client", 0.0)
        with results_lock:
            results.append(allowed)

    threads = [threading.Thread(target=reserve) for _ in range(40)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sum(results) == 20
    assert not limiter.allow("client", 1.0)
