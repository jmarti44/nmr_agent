import httpx2
import pytest
from fastapi.testclient import TestClient

from src.api.analyze_routes import get_llm_client
from src.api.main import app
from src.core.settings import Settings, get_settings
from tests.llm.fake_api import ASPIRIN, FakeAPI, error, hypothesis_json, message
from tests.prompting.helpers import ASPIRIN_DMSO


@pytest.fixture
def call():
    """POST /v1/analyze with a fake model behind it; returns (response, fake api)."""

    def _call(replies, body=None, **settings):
        api = FakeAPI(replies)
        app.dependency_overrides[get_llm_client] = lambda: api.client()
        app.dependency_overrides[get_settings] = lambda: Settings(**settings)
        try:
            return TestClient(app).post("/v1/analyze", json=body or ASPIRIN_DMSO), api
        finally:
            app.dependency_overrides.clear()

    return _call


def test_analyze_returns_a_verified_response(call):
    r, api = call([message(hypothesis_json())])
    assert r.status_code == 200
    j = r.json()
    assert j["request_id"].startswith("req_") and j["candidates"][0]["status"] == "accepted"
    assert j["candidates"][0]["candidate"]["smiles"] == ASPIRIN and j["usage"]["llm_calls"] == 1
    assert j["versions"]["prompt_version"] and j["peak_table"]["peaks"] and j["hypothesis_raw"]["candidates"]
    assert len(api.requests) == 1


def test_invalid_body_is_400_invalid_input(call):
    r, api = call([], body={"input_format": "peaklist"})
    assert r.status_code == 400 and r.json()["error_code"] == "INVALID_INPUT" and not api.requests


def test_raw_spectrum_formats_are_not_supported_yet(call):
    r, api = call([], body={"input_format": "jcampdx"})
    assert (
        r.status_code == 400
        and r.json()["error_code"] == "INVALID_INPUT"
        and "peaklist" in r.json()["detail"]
    )
    assert not api.requests


def test_refusal_is_451(call):
    r, _ = call(
        [
            message(
                "",
                stop_reason="refusal",
                stop_details={"type": "refusal", "category": "bio", "explanation": None},
            )
        ]
    )
    assert r.status_code == 451 and r.json()["error_code"] == "LLM_REFUSED" and "bio" in r.json()["detail"]


def test_never_valid_output_is_502_schema_failure(call):
    r, api = call([message("{}"), message("{}"), message("{}")])
    assert r.status_code == 502 and r.json()["error_code"] == "LLM_SCHEMA_FAILURE" and len(api.requests) == 3


def test_model_api_error_is_502_unavailable_and_keeps_retry_after(call):
    r, _ = call([error(429, "rate_limit_error", "slow", {"retry-after": "21"})], sdk_max_retries=0)
    assert (
        r.status_code == 502
        and r.json()["error_code"] == "LLM_UNAVAILABLE"
        and r.headers["retry-after"] == "21"
    )


def test_upstream_error_text_is_not_returned_to_the_caller_but_is_logged(call):
    from structlog.testing import capture_logs

    secret = "Your credit balance is too low: acct_ZX81"
    with capture_logs() as logs:
        r, _ = call([error(400, "invalid_request_error", secret)], sdk_max_retries=0)
    assert r.status_code == 502 and r.json()["error_code"] == "LLM_UNAVAILABLE"
    assert secret not in r.text and "status_400" in r.json()["detail"]
    entry = next(x for x in logs if x["event"] == "llm_api_error")
    assert secret in entry["upstream_message"] and entry["request_id"] == r.json()["request_id"]


def test_timeout_is_504(call):
    r, _ = call([httpx2.ReadTimeout("slow")], sdk_max_retries=0)
    assert r.status_code == 504 and r.json()["error_code"] == "TIMEOUT"


def test_missing_credentials_are_503(call, monkeypatch):
    import anthropic

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    api = FakeAPI([message(hypothesis_json())])
    http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(api.handler))
    app.dependency_overrides[get_llm_client] = lambda: anthropic.Anthropic(http_client=http)
    app.dependency_overrides[get_settings] = lambda: Settings()
    try:
        r = TestClient(app).post("/v1/analyze", json=ASPIRIN_DMSO)
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 503 and r.json()["error_code"] == "LLM_UNAVAILABLE"


def test_every_error_body_has_a_request_id(call):
    r, _ = call([message("", stop_reason="refusal", stop_details=None)])
    assert r.json()["request_id"].startswith("req_")


def test_openapi_documents_the_route_and_its_errors():
    spec = TestClient(app).get("/openapi.json").json()
    op = spec["paths"]["/v1/analyze"]["post"]
    assert {"200", "400", "451", "502", "503", "504"} <= set(op["responses"])
    assert "not proven correct" in op["description"]


def test_importing_the_app_loads_neither_the_model_sdk_nor_the_chemistry_libraries():
    import subprocess
    import sys

    mods = ("anthropic", "rdkit", "nmrglue")
    code = f"import sys; import src.api.main; print([m for m in {mods} if m in sys.modules])"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "[]"
