import sys

from fastapi.testclient import TestClient

from src.api.main import app
from tests.verify.cases import CASES

client = TestClient(app)


def _body(case, smiles):
    return {
        "peak_table": case.table().model_dump(mode="json"),
        "candidates": smiles,
        "context": {"molecular_formula": case.formula, "solvent": case.solvent},
    }


def test_verify_accepts_the_true_structure_and_rejects_a_wrong_formula():
    case = CASES["aspirin"]
    r = client.post("/v1/verify", json=_body(case, [case.smiles, "CCO", "C1CC"]))
    assert r.status_code == 200
    out = {v["candidate"]["rank"]: v for v in r.json()}
    assert out[1]["status"] == "accepted" and out[1]["inchikey"].startswith("BSYNRYMUTXBXSQ")
    assert out[1]["final_confidence"] == out[1]["verifier_score"] == 1.0  # no model involved
    assert out[2]["status"] == "rejected" and out[3]["status"] == "rejected"
    assert out[3]["canonical_smiles"] is None


def test_verify_returns_best_first():
    case = CASES["aspirin"]
    r = client.post("/v1/verify", json=_body(case, ["COC(=O)c1ccccc1C(=O)O", case.smiles]))
    assert [v["status"] for v in r.json()][0] == "accepted"


def test_invalid_requests_are_400_invalid_input_with_request_id():
    only_solvent = {"peaks": [{"id": 1, "nucleus": "1H", "shift_ppm": 7.27, "role": "solvent"}]}
    for body in (
        {},
        {"candidates": []},
        {"peak_table": {"peaks": []}, "candidates": ["CCO"]},
        {"peak_table": only_solvent, "candidates": ["CCO"]},
    ):
        r = client.post("/v1/verify", json=body)
        assert r.status_code == 400
        j = r.json()
        assert j["error_code"] == "INVALID_INPUT" and j["request_id"].startswith("req_") and j["detail"]


def test_reference_shifts_lists_tables_with_sources_and_stays_rdkit_free():
    r = client.get("/v1/reference/shifts")
    assert r.status_code == 200
    j = r.json()
    assert {x["key"] for x in j["h1_rules"]} >= {"h_aldehyde", "h_aromatic"}
    assert all(x["sources"] for x in j["h1_rules"] + j["c13_rules"])
    assert len(j["functional_groups"]) == 25 and "CDCl3" in j["solvents_and_impurities"]["residual_solvents"]
    assert j["version"]


def test_openapi_documents_the_new_routes():
    paths = client.get("/openapi.json").json()["paths"]
    assert "/v1/verify" in paths and "/v1/reference/shifts" in paths


def test_healthz_does_not_import_rdkit():
    """/healthz must answer before the heavy libraries load (SPEC F8). Run in a fresh interpreter."""
    import subprocess

    code = (
        "import sys; from fastapi.testclient import TestClient; from src.api.main import app;"
        "TestClient(app).get('/healthz'); TestClient(app).get('/v1/reference/shifts');"
        "print('rdkit' in sys.modules, 'nmrglue' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "False False"
