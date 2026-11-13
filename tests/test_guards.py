import pytest
from fastapi.testclient import TestClient

from src.api.analyze_routes import get_llm_client
from src.api.guards import GuardError, UsageGuard, check_api_key, get_guard
from src.api.main import app
from src.core.settings import Settings, get_settings
from tests.llm.fake_api import FakeAPI, hypothesis_json, message
from tests.prompting.helpers import ASPIRIN_DMSO


class Clock:
    def __init__(self, t=1_800_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_hourly_limit_per_client_then_recovery():
    clock = Clock()
    g, s = UsageGuard(clock), Settings(per_ip_per_hour=2)
    g.admit("a", s)
    g.admit("a", s)
    with pytest.raises(GuardError) as e:
        g.admit("a", s)
    assert e.value.status == 429 and e.value.code == "RATE_LIMITED" and 0 < e.value.retry_after <= 3600
    g.admit("b", s)  # another client is unaffected
    clock.t += 3601
    g.admit("a", s)


def test_daily_budget_counts_settled_calls_and_resets_at_midnight_utc():
    clock = Clock()
    g, s = UsageGuard(clock), Settings(daily_llm_budget=3, per_ip_per_hour=100, max_llm_calls=4)
    r = g.admit("a", s)
    assert r == 3  # a reservation never exceeds what is left of the day's budget
    g.settle(r, 3)
    with pytest.raises(GuardError) as e:
        g.admit("b", s)
    assert e.value.code == "DAILY_BUDGET_EXHAUSTED" and e.value.retry_after > 0
    clock.t += 86400
    assert g.admit("b", s) == 3


def test_reservations_stop_concurrent_admissions_from_overshooting_the_budget():
    g, s = UsageGuard(Clock()), Settings(daily_llm_budget=6, per_ip_per_hour=100, max_llm_calls=4)
    first = g.admit("a", s)  # reserves 4
    second = g.admit("b", s)  # reserves the last 2
    assert (first, second) == (4, 2)
    with pytest.raises(GuardError):
        g.admit("c", s)  # everything is promised, though nothing has been spent yet
    g.settle(first, 1)  # the first analysis only needed one call: 3 calls are free again
    assert g.admit("c", s) == 3


def test_idle_clients_are_forgotten():
    clock = Clock()
    g, s = UsageGuard(clock), Settings(per_ip_per_hour=5, daily_llm_budget=1000)
    for i in range(50):
        g.settle(g.admit(f"client-{i}", s), 0)
    assert len(g._hits) == 50
    clock.t += 3601
    g.settle(g.admit("late", s), 0)
    assert list(g._hits) == ["late"]


def test_api_key_is_optional_and_compared_exactly():
    check_api_key(Settings(), None)  # none configured: open
    s = Settings(api_key="secret-key")
    check_api_key(s, "secret-key")
    for bad in (None, "", "secret-kez", "secret-key "):
        with pytest.raises(GuardError) as e:
            check_api_key(s, bad)
        assert e.value.status == 401


def _post(settings, replies, headers=None):
    api = FakeAPI(replies)
    app.dependency_overrides[get_llm_client] = lambda: api.client()
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        return TestClient(app).post("/v1/analyze", json=ASPIRIN_DMSO, headers=headers or {}), api
    finally:
        app.dependency_overrides.clear()


def test_route_requires_the_key_when_configured_and_never_calls_the_model_without_it():
    s = Settings(api_key="k1")
    r, api = _post(s, [], headers={})
    assert r.status_code == 401 and r.json()["error_code"] == "UNAUTHORIZED" and not api.requests
    r, _ = _post(s, [message(hypothesis_json())], headers={"X-API-Key": "k1"})
    assert r.status_code == 200


def test_route_returns_429_with_retry_after_after_the_hourly_limit():
    s = Settings(per_ip_per_hour=1)
    assert _post(s, [message(hypothesis_json())])[0].status_code == 200
    r, api = _post(s, [])
    assert r.status_code == 429 and r.json()["error_code"] == "RATE_LIMITED"
    assert int(r.headers["Retry-After"]) > 0 and not api.requests


def test_route_stops_when_the_daily_budget_is_spent():  # one analysis, one call
    s = Settings(daily_llm_budget=1)
    assert _post(s, [message(hypothesis_json())])[0].status_code == 200  # one call used
    r, api = _post(s, [])
    assert r.status_code == 429 and r.json()["error_code"] == "DAILY_BUDGET_EXHAUSTED" and not api.requests


def test_free_routes_are_not_limited():
    app.dependency_overrides[get_settings] = lambda: Settings(
        per_ip_per_hour=1, api_key="k", daily_llm_budget=0
    )
    try:
        c = TestClient(app)
        for _ in range(3):
            assert c.post("/v1/process", json=ASPIRIN_DMSO).status_code == 200
        assert c.get("/healthz").status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_the_guard_is_shared_across_requests():
    assert get_guard() is get_guard()


def test_failed_analyses_still_count_against_the_budget():
    """Three paid calls that never validate end in a 502; all three must be counted (review R4-1)."""
    s = Settings(daily_llm_budget=100, max_llm_calls=4)
    r, api = _post(s, [message("not json"), message("still not json"), message("{}")])
    assert r.status_code == 502 and len(api.requests) == 3
    g = get_guard()
    assert g._calls_today == 3 and g._reserved == 0


def test_a_timeout_after_one_call_counts_that_call():
    import httpx2

    s = Settings(daily_llm_budget=100, sdk_max_retries=0)
    r, api = _post(s, [message("not json"), httpx2.ReadTimeout("slow")])
    g = get_guard()
    assert r.status_code in (502, 504) and g._calls_today == 1 and g._reserved == 0


def test_a_request_that_fails_validation_before_any_call_counts_nothing():
    s = Settings(daily_llm_budget=100)
    app.dependency_overrides[get_settings] = lambda: s
    try:
        r = TestClient(app).post("/v1/analyze", json={"input_format": "peaklist"})
    finally:
        app.dependency_overrides.clear()
    g = get_guard()
    assert r.status_code == 400 and g._calls_today == 0 and g._reserved == 0


def test_concurrent_requests_cannot_overshoot_the_daily_budget():
    import threading
    import time

    lock = threading.Lock()
    made: list[int] = []

    class Slow(FakeAPI):
        def handler(self, request):
            time.sleep(0.4)
            with lock:
                made.append(1)
                self.replies = [message(hypothesis_json())]
            return super().handler(request)

    api = Slow([])
    app.dependency_overrides[get_llm_client] = lambda: api.client()
    app.dependency_overrides[get_settings] = lambda: Settings(daily_llm_budget=2, per_ip_per_hour=100)
    codes: list[int] = []

    def one():
        codes.append(TestClient(app).post("/v1/analyze", json=ASPIRIN_DMSO).status_code)

    try:
        threads = [threading.Thread(target=one) for _ in range(6)]
        [t.start() for t in threads]
        [t.join() for t in threads]
    finally:
        app.dependency_overrides.clear()
    assert len(made) <= 2
    assert codes.count(200) >= 1 and codes.count(429) == 6 - codes.count(200)
