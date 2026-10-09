from fastapi.testclient import TestClient

from src.api.main import app


def test_healthz():
    r = TestClient(app).get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_openapi_served():
    r = TestClient(app).get("/openapi.json")
    assert r.status_code == 200
    assert r.json()["info"]["title"] == "NMR-Agent"
