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


def test_daily_budget_counts_actual_calls_and_resets_at_midnight_utc():
    clock = Clock()
    g, s = UsageGuard(clock), Settings(daily_llm_budget=3, per_ip_per_hour=100)
    g.admit("a", s)
    g.record_calls(3)
    with pytest.raises(GuardError) as e:
        g.admit("b", s)
    assert e.value.code == "DAILY_BUDGET_EXHAUSTED" and e.value.retry_after > 0
    clock.t += 86400
    g.admit("b", s)


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


def test_route_stops_when_the_daily_budget_is_spent():
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
