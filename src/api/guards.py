"""Abuse controls for the route that spends model credits (SPEC §9.4): optional API key, per-client hourly
limit and a daily budget of model calls. Counters live in memory, so they reset on restart and are per
process; that is enough for one small instance and is stated in the README."""

from __future__ import annotations

import math
import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from functools import lru_cache

from src.core.settings import Settings

HOUR_S = 3600.0


class GuardError(Exception):
    def __init__(self, status: int, code: str, detail: str, retry_after: int | None = None):
        super().__init__(detail)
        self.status, self.code, self.detail, self.retry_after = status, code, detail, retry_after


def check_api_key(settings: Settings, supplied: str | None) -> None:
    """No-op when no key is configured; otherwise the header must match (constant-time compare)."""
    if settings.api_key is None:
        return
    if supplied is None or not secrets.compare_digest(supplied.encode(), settings.api_key.encode()):
        raise GuardError(401, "UNAUTHORIZED", "a valid X-API-Key header is required for this route")


class UsageGuard:
    def __init__(self, clock=time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._day = ""
        self._calls_today = 0

    def _today(self) -> str:
        return datetime.fromtimestamp(self._clock(), UTC).strftime("%Y-%m-%d")

    def admit(self, client: str, settings: Settings) -> None:
        """Raise GuardError(429) if this client is over its hourly limit or today's budget is spent."""
        now = self._clock()
        with self._lock:
            if self._day != self._today():
                self._day, self._calls_today = self._today(), 0
            if self._calls_today >= settings.daily_llm_budget:
                raise GuardError(
                    429, "DAILY_BUDGET_EXHAUSTED", "today's model-call budget is used up; try again tomorrow",
                    retry_after=self._seconds_to_midnight(now),
                )  # fmt: skip
            hits = self._hits[client]
            while hits and now - hits[0] >= HOUR_S:
                hits.popleft()
            if len(hits) >= settings.per_ip_per_hour:
                wait = math.ceil(HOUR_S - (now - hits[0]))
                raise GuardError(
                    429, "RATE_LIMITED", f"limit of {settings.per_ip_per_hour} analyses per hour", wait
                )
            hits.append(now)

    def record_calls(self, n: int) -> None:
        """Add the model calls an analysis actually made to today's total."""
        with self._lock:
            if self._day != self._today():
                self._day, self._calls_today = self._today(), 0
            self._calls_today += n

    @staticmethod
    def _seconds_to_midnight(now: float) -> int:
        return int(86400 - (now % 86400)) + 1


@lru_cache(maxsize=1)
def get_guard() -> UsageGuard:
    return UsageGuard()
