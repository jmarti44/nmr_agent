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
    """Hourly limit per client and a daily budget of model calls.

    The budget is reserved when a request is admitted (up to `max_llm_calls`, or what is left of the day's
    budget) and settled when it ends with the calls it actually made, successful or not. Concurrent requests
    therefore cannot overshoot it, and a failed analysis still counts."""

    SWEEP_EVERY_S = 60.0

    def __init__(self, clock=time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._day = ""
        self._calls_today = 0  # settled calls
        self._reserved = 0  # calls promised to analyses still running
        self._last_sweep = 0.0

    def _today(self) -> str:
        return datetime.fromtimestamp(self._clock(), UTC).strftime("%Y-%m-%d")

    def _roll_day(self) -> None:
        if self._day != self._today():
            self._day, self._calls_today = self._today(), 0

    def _sweep(self, now: float) -> None:
        """Forget clients with no hit in the last hour, so the table cannot grow without bound."""
        if now - self._last_sweep < self.SWEEP_EVERY_S:
            return
        self._last_sweep = now
        for client in [c for c, hits in self._hits.items() if not hits or now - hits[-1] >= HOUR_S]:
            del self._hits[client]

    def admit(self, client: str, settings: Settings) -> int:
        """Reserve model calls for one analysis and return how many it may make.

        Raises GuardError(429) if the client is over its hourly limit or the day's budget is spent or fully
        reserved. The caller must call `settle` with the returned number when the analysis ends."""
        now = self._clock()
        with self._lock:
            self._roll_day()
            self._sweep(now)
            free = settings.daily_llm_budget - self._calls_today - self._reserved
            if free <= 0:
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
            reserved = min(settings.max_llm_calls, free)
            self._reserved += reserved
            return reserved

    def settle(self, reserved: int, calls_made: int) -> None:
        """Give back the reservation and add the calls actually made to today's total."""
        with self._lock:
            self._roll_day()
            self._reserved = max(0, self._reserved - reserved)
            self._calls_today += calls_made

    @staticmethod
    def _seconds_to_midnight(now: float) -> int:
        return int(86400 - (now % 86400)) + 1


@lru_cache(maxsize=1)
def get_guard() -> UsageGuard:
    return UsageGuard()
