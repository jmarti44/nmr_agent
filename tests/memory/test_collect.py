"""select-large, the SD-record helper and collect. Fake API only; every file is in a temporary directory."""

from __future__ import annotations

import collections
import hashlib
import json
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import bench.memory as bm
from src.core.settings import Settings
from src.memory import outcome_config
from tests.llm.fake_api import ASPIRIN, ISOMER, FakeAPI, error, hypothesis_json, message

ROOT = Path(__file__).resolve().parents[2]
SD = ROOT / "bench" / "data" / "nmrshiftdb2" / "nmrshiftdb2withsignals.sd"
GIT = {"git_commit": "0" * 40, "git_dirty_code": False}


# --------------------------------------------------------------------------- synthetic memory lists


def entry(i: int, stratum: str, smiles: str, scaffold: str = "") -> dict:
    return {
        "position": i,
        "id": f"id{i}",
        "pool": "named" if i % 2 else "unreported",
        "smiles": smiles,
        "inchikey": f"KEY{i:011d}-UHFFFAOYSA-N",
        "heavy_atoms": {">20": 25, "11-20": 15, "<=10": 8}[stratum],
        "stratum": stratum,
        "scaffold": scaffold,
        "solvent": None,
    }


def write_selection(tmp: Path, per_stratum=(130, 70, 30), excluded=("id2", "id3")) -> dict[str, Path]:
    """memory_order.json, gonogo_samples.json and mem_eval.json with the three strata interleaved."""
    pattern = [">20", "11-20", ">20", "11-20", "<=10"]
    counts = dict(zip((">20", "11-20", "<=10"), per_stratum, strict=True))
    entries, i = [], 0
    while any(counts.values()):
        for s in pattern:
            if counts[s]:
                counts[s] -= 1
                i += 1
                entries.append(entry(i, s, "CCCCCCCCCC"))  # acyclic chain: no scaffold, clears every rule
    files = {
        "order": tmp / "memory_order.json",
        "samples": tmp / "gonogo_samples.json",
        "eval": tmp / "mem_eval.json",
    }
    files["order"].write_text(json.dumps({"entries": entries}))
    files["samples"].write_text(json.dumps({"excluded": [{"id": x} for x in excluded]}))
    files["eval"].write_text(json.dumps({"records": [{"id": "e1", "smiles": "c1ccc2ccccc2c1"}]}))
    return files


def run_select_large(tmp: Path, files: dict[str, Path], **kw) -> dict:
    return bm.select_large(
        out=tmp / "memory_large.json",
        order_file=files["order"],
        samples_file=files["samples"],
        eval_file=files["eval"],
        old_keys=set(),
        git=GIT,
        **kw,
    )


# --------------------------------------------------------------------------- select-large


def test_select_large_takes_120_60_20_in_memory_order_and_the_rest_are_replacements(tmp_path):
    files = write_selection(tmp_path)
    out = run_select_large(tmp_path, files)
    rows = out["entries"]
    assert len(rows) == 200
    assert collections.Counter(r["stratum"] for r in rows) == {">20": 120, "11-20": 60, "<=10": 20}
    entries = json.loads(files["order"].read_text())["entries"]
    usable = [e for e in entries if e["id"] not in {"id2", "id3"}]
    for s, n in bm.LARGE_QUOTAS.items():
        in_stratum = [e["id"] for e in usable if e["stratum"] == s]
        assert [r["id"] for r in rows if r["stratum"] == s] == in_stratum[:n]
        assert out["replacement_order"][s] == in_stratum[n:]
    positions = [r["position"] for r in rows]
    assert positions == sorted(positions) and [r["index"] for r in rows] == list(range(1, 201))
    assert {"stratum", "position", "id", "pool", "smiles", "inchikey", "scaffold", "solvent"} <= set(rows[0])


def test_select_large_leaves_out_the_excluded_molecules(tmp_path):
    files = write_selection(tmp_path, excluded=("id1", "id2"))  # id1 would be the first >20 entry
    out = run_select_large(tmp_path, files)
    ids = {r["id"] for r in out["entries"]} | {i for ids in out["replacement_order"].values() for i in ids}
    assert not ids & {"id1", "id2"} and out["excluded_ids"] == ["id1", "id2"]


def test_select_large_records_sources_commit_and_a_clean_leakage_result(tmp_path):
    files = write_selection(tmp_path)
    out = run_select_large(tmp_path, files)
    assert out["git_commit"] == GIT["git_commit"]
    assert set(out["sources"].values()) == {
        hashlib.sha256(p.read_bytes()).hexdigest() for p in files.values()
    }
    assert out["leakage"]["rules"] == [1, 2, 4] and out["leakage"]["clean"] is True
    assert out["leakage"]["memory_items"] == 200 and out["leakage"]["mem_eval_molecules"] == 1


def test_select_large_refuses_to_overwrite(tmp_path):
    files = write_selection(tmp_path)
    run_select_large(tmp_path, files)
    before = (tmp_path / "memory_large.json").read_bytes()
    with pytest.raises(SystemExit, match="refusing"):
        run_select_large(tmp_path, files)
    assert (tmp_path / "memory_large.json").read_bytes() == before


def test_select_large_writes_nothing_when_a_rule_is_broken(tmp_path):
    files = write_selection(tmp_path)
    files["eval"].write_text(
        json.dumps({"records": [{"id": "e1", "smiles": "CCCCCCCCCC"}]})
    )  # same structure
    with pytest.raises(SystemExit, match="leakage"):
        run_select_large(tmp_path, files)
    assert not (tmp_path / "memory_large.json").exists()


def test_select_large_needs_enough_usable_entries(tmp_path):
    files = write_selection(tmp_path, per_stratum=(120, 60, 19), excluded=())
    with pytest.raises(SystemExit, match="need 20"):
        run_select_large(tmp_path, files)


def test_the_committed_list_matches_the_frozen_inputs():
    large = bm.load_large()
    assert len(large["entries"]) == 200
    assert collections.Counter(r["stratum"] for r in large["entries"]) == {
        ">20": 120,
        "11-20": 60,
        "<=10": 20,
    }
    assert large["leakage"]["clean"] is True and len(large["excluded_ids"]) == 9
    for path, digest in large["sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    # from the frozen order, independently
    entries = json.loads(bm.FILES["memory_order"].read_text())["entries"]
    usable = [e for e in entries if e["id"] not in set(large["excluded_ids"])]
    for s, n in bm.LARGE_QUOTAS.items():
        ids = [e["id"] for e in usable if e["stratum"] == s]
        assert [r["id"] for r in large["entries"] if r["stratum"] == s] == ids[:n]
        assert large["replacement_order"][s] == ids[n:]


# --------------------------------------------------------------------------- the SD helper


# Aspirin's own spectra, keyed by atom index of "CC(=O)Oc1ccccc1C(=O)O": the verifier accepts them, so a
# fake answer of aspirin ends the analysis in one model call.
ASPIRIN_H = {0: 2.25, 5: 7.15, 6: 7.64, 7: 7.36, 8: 7.95, 12: 12.95}
ASPIRIN_C = {0: 20.9, 1: 169.4, 4: 150.5, 5: 124.0, 6: 134.0, 7: 126.5, 8: 132.0, 9: 122.5, 10: 165.2}


def make_record(
    rid: str, smiles: str, solvent: str | None = "CDCl3", index_h: bool = True, aspirin_spectra: bool = False
) -> bm.SdRecord:
    """A synthetic SD record: atom indices are those of the canonical SMILES' molecule."""
    mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smiles)
    if smiles == ASPIRIN or aspirin_spectra:
        h_shift, c_shift = ASPIRIN_H, ASPIRIN_C
    else:
        h_shift = {
            a.GetIdx(): 1.0 + 0.31 * a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetTotalNumHs() and a.GetSymbol() == "C"
        }
        c_shift = {a.GetIdx(): 20 + 7.3 * a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C"}
    h = [f"{v:.2f};0.0;{i if index_h else ''}" for i, v in h_shift.items()]
    c = [f"{v:.2f};0.0S;{i}" for i, v in c_shift.items()]
    return bm.SdRecord(
        id=rid,
        mol=mol,
        smiles=smiles,
        inchikey=Chem.MolToInchiKey(mol),
        formula=rdMolDescriptors.CalcMolFormula(mol),
        solvent=solvent,
        h_spec="|".join(h) + "|",
        c_spec="|".join(c) + "|",
        h_spectrum=1,
        c_spectrum=2,
    )


@pytest.mark.skipif(not SD.exists(), reason="bench/data (the NMRShiftDB2 export) is absent")
def test_sd_helper_gives_the_pair_that_pool_chooses(tmp_path, monkeypatch):
    from bench import benchmark

    head = tmp_path / "head.sd"
    n, kept = 0, []
    with open(SD, errors="replace") as fh:
        for line in fh:
            kept.append(line)
            if line.startswith("$$$$"):
                n += 1
                if n == 4000:
                    break
    head.write_text("".join(kept))
    monkeypatch.setattr(benchmark, "SD", str(head))
    for unreported in (False, True):
        pooled = benchmark.pool(unreported=unreported)
        assert pooled, "the sample must contain pool records"
        got = bm.read_sd_records({r["id"]: "unreported" if unreported else "named" for r in pooled}, head)
        assert set(got) == {r["id"] for r in pooled}
        for r in pooled:
            g = got[r["id"]]
            assert (g.h_spec, g.c_spec, g.solvent) == (r["h_spec"], r["c_spec"], r["solvent"])
            assert g.pool_record() == r
            assert g.mol.GetNumAtoms() > 0 and isinstance(g.h_spectrum, int) and isinstance(g.c_spectrum, int)


@pytest.mark.skipif(not SD.exists(), reason="bench/data (the NMRShiftDB2 export) is absent")
def test_sd_helper_finds_the_real_memory_molecules():
    rows = bm.load_large()["entries"][:3]
    got = bm.read_sd_records({r["id"]: r["pool"] for r in rows})
    for r in rows:
        g = got[r["id"]]
        assert g.smiles == r["smiles"] and g.inchikey == r["inchikey"] and g.h_spec and g.c_spec


def test_sd_helper_leaves_out_what_pool_would_not_keep(tmp_path):
    assert bm.read_sd_records({"nope": "named"}, _write(tmp_path, "")) == {}


def _write(tmp: Path, text: str) -> Path:
    p = tmp / "x.sd"
    p.write_text(text)
    return p


# --------------------------------------------------------------------------- collect with the fake API


def fake_large(tmp: Path, rows: list[tuple[str, str]]) -> tuple[Path, dict[str, bm.SdRecord]]:
    """Rows (id, smiles); a molecule that is not aspirin gets aspirin's spectra (same formula for ISOMER),
    so the fake model's answer of aspirin is accepted by the verifier and is wrong for that molecule."""
    recs = {rid: make_record(rid, smi, aspirin_spectra=True) for rid, smi in rows}
    entries = [
        {
            "index": i,
            "stratum": ">20",
            "position": i,
            "id": rid,
            "pool": "named",
            "smiles": recs[rid].smiles,
            "inchikey": recs[rid].inchikey,
            "heavy_atoms": recs[rid].mol.GetNumHeavyAtoms(),
            "scaffold": "",
            "solvent": "CDCl3",
        }
        for i, (rid, _) in enumerate(rows, 1)
    ]
    path = tmp / "memory_large.json"
    path.write_text(
        json.dumps({"entries": entries, "replacement_order": {">20": ["r1"], "11-20": [], "<=10": []}})
    )
    return path, recs


def reader(recs):
    return lambda wanted, path=None: {i: recs[i] for i in wanted if i in recs}


def run_collect(tmp: Path, large: Path, recs, api: FakeAPI, **kw):
    return bm.collect(
        large_file=large,
        out_dir=tmp / "collect",
        read_records=reader(recs),
        client_factory=lambda settings: api.client(),
        **kw,
    )


def result(tmp: Path, rid: str) -> dict:
    return json.loads((tmp / "collect" / f"{rid}.json").read_text())


def test_collect_records_solved_missed_and_no_answer(tmp_path):
    large, recs = fake_large(tmp_path, [("a1", ASPIRIN), ("a2", ISOMER), ("a3", ASPIRIN)])
    api = FakeAPI(
        [
            message(hypothesis_json([ASPIRIN])),  # truth aspirin: solved
            message(hypothesis_json([ASPIRIN])),  # truth is the isomer: missed
            error(400, "invalid_request_error", "bad"),  # no_answer
        ]
    )
    run_collect(tmp_path, large, recs, api)
    solved, missed, none = (result(tmp_path, i) for i in ("a1", "a2", "a3"))
    assert solved["status"] == "solved" and solved["candidates"][0]["smiles"] == ASPIRIN
    assert missed["status"] == "missed" and missed["candidates"][0]["smiles"] == ASPIRIN
    assert none["status"] == "no_answer" and none["candidates"] == [] and none["error"]
    assert (
        none["llm_calls"] == 0 and none["cost_usd"] == 0
    )  # CallMeter counts completed calls; the failed one is unpriced
    cfg = outcome_config(Settings()).model_dump()
    for r in (solved, missed, none):
        assert r["outcome_config"] == cfg and r["collected_on"]
    for r in (solved, missed):
        assert r["cost_usd"] > 0 and r["llm_calls"] == 1 and set(r["timing_ms"]) >= {"llm", "total"}
        assert r["trace"]["table"] and r["trace"]["memory_cases"] == [] and isinstance(r["warnings"], list)
        assert set(r["candidates"][0]) == {"smiles", "model_rank", "status", "score", "confidence"}


def test_collect_sends_no_examples_and_the_benchmark_request(tmp_path):
    from bench import benchmark

    large, recs = fake_large(tmp_path, [("a1", ASPIRIN)])
    api = FakeAPI([message(hypothesis_json([ASPIRIN]))])
    run_collect(tmp_path, large, recs, api)
    user = api.requests[0]["body"]["messages"][0]["content"]
    text = user if isinstance(user, str) else "".join(b.get("text", "") for b in user)
    req, _ = benchmark.request_for(recs["a1"].pool_record())
    assert str(req.peaks.peaks[0].shift_ppm) in text and "worked example" not in text.lower()


def test_collect_resumes_and_skips_finished_molecules(tmp_path):
    large, recs = fake_large(tmp_path, [("a1", ASPIRIN), ("a2", ASPIRIN)])
    run_collect(tmp_path, large, recs, FakeAPI([message(hypothesis_json([ASPIRIN]))]), ids=["a1"])
    first = (tmp_path / "collect" / "a1.json").read_bytes()
    api = FakeAPI([message(hypothesis_json([ASPIRIN]))])  # one reply only: a1 must not be called again
    run_collect(tmp_path, large, recs, api)
    assert len(api.requests) == 1 and (tmp_path / "collect" / "a1.json").read_bytes() == first
    assert result(tmp_path, "a2")["status"] == "solved"


def test_a_failed_molecule_stays_no_answer_on_a_rerun(tmp_path):
    large, recs = fake_large(tmp_path, [("a1", ASPIRIN)])
    run_collect(tmp_path, large, recs, FakeAPI([error(400, "invalid_request_error", "bad")]))
    api = FakeAPI([message(hypothesis_json([ASPIRIN]))])
    run_collect(tmp_path, large, recs, api)
    assert api.requests == [] and result(tmp_path, "a1")["status"] == "no_answer"


def test_collect_stops_at_the_spend_cap(tmp_path):
    large, recs = fake_large(tmp_path, [("a1", ASPIRIN), ("a2", ASPIRIN), ("a3", ASPIRIN)])
    api = FakeAPI([message(hypothesis_json([ASPIRIN])) for _ in range(3)])
    run_collect(tmp_path, large, recs, api, cap=0.0001)  # one analysis costs more than this
    assert len(api.requests) == 1 and sorted(p.name for p in (tmp_path / "collect").glob("*.json")) == [
        "a1.json"
    ]
    run_collect(tmp_path, large, recs, api, cap=0.0001)  # earlier spend counts: still over the cap
    assert len(api.requests) == 1


def test_default_cap_is_35_dollars():
    assert bm.SPEND_CAP_DEFAULT == 35.0


def test_dry_run_builds_every_request_and_calls_nothing(tmp_path, capsys):
    large, recs = fake_large(tmp_path, [("a1", ASPIRIN), ("a2", ASPIRIN)])

    def no_client(settings):
        raise AssertionError("the dry run must not create a client")

    report = bm.collect(
        dry_run=True,
        large_file=large,
        out_dir=tmp_path / "collect",
        read_records=reader(recs),
        client_factory=no_client,
    )
    out = capsys.readouterr().out
    assert report["molecules"] == 2 and report["buildable"] == 2 and report["failures"] == {}
    assert report["input_tokens_est"] > 0 and "estimated input tokens" in out and "2 buildable" in out
    assert not (tmp_path / "collect").exists()


def test_dry_run_lists_failures_and_the_replacement_it_would_use(tmp_path, capsys):
    large, recs = fake_large(tmp_path, [("a1", ASPIRIN), ("a2", ASPIRIN)])
    del recs["a2"]
    report = bm.collect(dry_run=True, large_file=large, out_dir=tmp_path / "c", read_records=reader(recs))
    assert list(report["failures"]) == ["a2"] and report["buildable"] == 1
    assert report["replacements"] == {"a2": "r1"} and "FAILS a2" in capsys.readouterr().out


def test_a_request_that_cannot_be_built_sends_nothing(tmp_path):
    large, recs = fake_large(tmp_path, [("a1", ASPIRIN)])
    recs["a1"] = make_record("a1", "CC(=O)Oc1ccccc1C(=O)O")
    bad = bm.SdRecord(**{**recs["a1"].__dict__, "h_spec": "99.0;0.0;1|"})  # 1H shift out of range
    api = FakeAPI([])
    run_collect(tmp_path, large, {"a1": bad}, api)
    assert api.requests == [] and not (tmp_path / "collect" / "a1.json").exists()


def test_collect_refuses_an_unknown_id(tmp_path):
    large, recs = fake_large(tmp_path, [("a1", ASPIRIN)])
    with pytest.raises(SystemExit, match="neither"):
        run_collect(tmp_path, large, recs, FakeAPI([]), ids=["zzz"])


def test_a_replacement_id_is_resolved_through_memory_order(tmp_path):
    large, recs = fake_large(tmp_path, [("a1", ASPIRIN)])
    order = tmp_path / "order.json"
    order.write_text(json.dumps({"entries": [entry(9, ">20", ASPIRIN) | {"id": "r1"}]}))
    rows = bm.resolve_rows(json.loads(large.read_text()), ["r1"], order)
    assert [r["id"] for r in rows] == ["r1"] and rows[0]["stratum"] == ">20"


def test_outcome_status_uses_the_connectivity_block():
    key = Chem.MolToInchiKey(Chem.MolFromSmiles(ASPIRIN))
    assert bm.outcome_status([ASPIRIN], key) == "solved"
    assert bm.outcome_status(["OC(=O)c1ccccc1OC(C)=O"], key) == "solved"  # same structure, other spelling
    assert bm.outcome_status([ISOMER, ASPIRIN], key) == "missed"
    assert bm.outcome_status(["not a smiles"], key) == "missed"
    assert bm.outcome_status([], key) == "missed"
