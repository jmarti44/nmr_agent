import json

import pytest
from fastapi.testclient import TestClient

from src.api import limits
from src.api.analyze_routes import get_llm_client
from src.api.main import app
from src.core.settings import Settings, get_settings
from src.ingest import safety
from tests.llm.fake_api import FakeAPI, hypothesis_json, message
from tests.processing.test_pipeline import zip_dir
from tests.prompting.helpers import ASPIRIN_DMSO
from tests.spectra import h1_spectrum, jcamp_text
from tests.synthetic import write_synthetic_bruker

client = TestClient(app)


@pytest.fixture(scope="module")
def bruker_zip(tmp_path_factory):
    return zip_dir(write_synthetic_bruker(tmp_path_factory.mktemp("api") / "x"))


def post(path, data, fmt, filename="a.zip", ctype="application/zip", **meta):
    body = {"input_format": fmt, **meta}
    return client.post(path, files={"file": (filename, data, ctype)}, data={"metadata": json.dumps(body)})


def test_process_returns_a_peak_table_for_a_bruker_zip(bruker_zip):
    r = post("/v1/process", bruker_zip, "bruker_zip")
    assert r.status_code == 200
    body = r.json()
    assert body["solvent"] == "CDCl3" and body["noise_sigma"]["1H"] > 0
    assert any("taken from the acquisition parameters" in line for line in body["processing_log"])
    assert [p["id"] for p in body["peaks"]] == list(range(1, len(body["peaks"]) + 1))


def test_process_accepts_a_jcamp_file():
    spec, ppm = h1_spectrum()
    r = post("/v1/process", jcamp_text(spec, ppm).encode(), "jcampdx", "a.jdx", "chemical/x-jcamp-dx")
    assert r.status_code == 200 and r.json()["referencing"] == "auto:TMS"


def test_process_also_takes_a_json_peak_list():
    r = client.post("/v1/process", json=ASPIRIN_DMSO)
    assert r.status_code == 200 and r.json()["solvent"] == "DMSO-d6"


@pytest.mark.parametrize(
    ("data", "fmt", "detail"),
    [
        (b"not a zip", "bruker_zip", "not a valid zip"),
        (b"", "bruker_zip", "empty"),
    ],
)
def test_bad_uploads_are_400_with_a_message(data, fmt, detail):
    r = post("/v1/process", data, fmt)
    assert r.status_code == 400 and r.json()["error_code"] == "INVALID_INPUT" and detail in r.json()["detail"]
    assert r.json()["request_id"].startswith("req_")


def test_oversized_upload_is_413(monkeypatch):
    monkeypatch.setattr(safety, "MAX_UPLOAD_BYTES", 100)
    monkeypatch.setattr(limits, "MULTIPART_OVERHEAD_BYTES", 0)
    r = post("/v1/process", b"x" * 5000, "bruker_zip")
    assert r.status_code == 413 and r.json()["error_code"] == "PAYLOAD_TOO_LARGE"


def test_multipart_without_metadata_or_file_is_400():
    assert client.post("/v1/process", files={"file": ("a.zip", b"x")}).status_code == 400
    assert client.post("/v1/process", data={"metadata": "{}"}, files={"x": ("a", b"x")}).status_code == 400


def test_bad_metadata_json_is_400(bruker_zip):
    r = client.post("/v1/process", files={"file": ("a.zip", bruker_zip)}, data={"metadata": "{nope"})
    assert r.status_code == 400 and "JSON" in r.json()["detail"]


def test_metadata_that_fails_validation_is_400(bruker_zip):
    r = post("/v1/process", bruker_zip, "bruker_zip", max_candidates=99)
    assert r.status_code == 400 and "max_candidates" in r.json()["detail"]


def test_zip_slip_is_400():
    import io
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../evil", "x")
    r = post("/v1/process", buf.getvalue(), "bruker_zip")
    assert r.status_code == 400 and "unsafe path" in r.json()["detail"]


def test_analyze_runs_the_whole_pipeline_on_an_uploaded_zip(bruker_zip):
    # One accepted answer is enough here, so a single reply is queued; the candidate is not checked for
    # chemistry, only that the uploaded spectrum reached the prompt and the response.
    api = FakeAPI([message(hypothesis_json()), message(hypothesis_json())])
    app.dependency_overrides[get_llm_client] = lambda: api.client()
    app.dependency_overrides[get_settings] = lambda: Settings()
    try:
        r = post("/v1/analyze", bruker_zip, "bruker_zip")
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["peak_table"]["solvent"] == "CDCl3" and body["candidates"]
    user_text = json.dumps(api.requests[0]["body"]["messages"])
    assert "referencing: auto:TMS" in user_text


def test_analyze_with_a_bad_upload_never_calls_the_model():
    api = FakeAPI([])
    app.dependency_overrides[get_llm_client] = lambda: api.client()
    app.dependency_overrides[get_settings] = lambda: Settings()
    try:
        r = post("/v1/analyze", b"junk", "bruker_zip")
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 400 and not api.requests


def test_openapi_documents_both_body_types():
    spec = client.get("/openapi.json").json()
    for path in ("/v1/analyze", "/v1/process"):
        content = spec["paths"][path]["post"]["requestBody"]["content"]
        assert set(content) == {"application/json", "multipart/form-data"}
