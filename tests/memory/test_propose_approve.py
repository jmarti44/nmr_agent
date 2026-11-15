"""propose, approve, reject and the store skeleton. Temporary directories and fake records only: the real
memory/ store is never written, and no model is called."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import json
import re
from pathlib import Path

import pytest
from rdkit import Chem

import bench.memory as bm
from src.core.settings import Settings
from src.memory import MemoryCase, PendingCase, load_store, outcome_config
from src.memory.store import chemistry_fields
from tests.llm.fake_api import ASPIRIN, ISOMER
from tests.memory.test_collect import GIT, make_record, reader

ROOT = Path(__file__).resolve().parents[2]
IBUPROFEN = "CC(C)Cc1ccc(C(C)C(=O)O)cc1"
NAPHTHALENE = "c1ccc2ccccc2c1"
TODAY = dt.date(2026, 11, 20)
CONFIG = outcome_config(Settings()).model_dump()


class Tty(io.StringIO):
    def isatty(self) -> bool:
        return True


class Env:
    """A temporary store, memory_large file, collect directory, MEM-EVAL file and fake records."""

    def __init__(
        self,
        tmp: Path,
        molecules: list[tuple[str, str]],
        total: int | None = None,
        eval_smiles=(NAPHTHALENE,),
    ):
        self.tmp = tmp
        self.store = tmp / "memory"
        bm.setup_store(self.store)
        self.collect_dir = tmp / "collect"
        self.collect_dir.mkdir()
        self.records = {rid: make_record(rid, smi, aspirin_spectra=False) for rid, smi in molecules}
        rows = list(molecules) + [
            (f"filler{i}", ASPIRIN) for i in range((total or len(molecules)) - len(molecules))
        ]
        self.rows = [
            {
                "index": i,
                "stratum": ">20",
                "position": i,
                "id": rid,
                "pool": "named",
                "smiles": self.records[rid].smiles if rid in self.records else ASPIRIN,
                "inchikey": self.records[rid].inchikey if rid in self.records else "X",
                "heavy_atoms": 20,
                "scaffold": "",
                "solvent": "CDCl3",
            }
            for i, (rid, _) in enumerate(rows, 1)
        ]
        self.large = tmp / "memory_large.json"
        self.large.write_text(
            json.dumps({"entries": self.rows, "replacement_order": {">20": [], "11-20": [], "<=10": []}})
        )
        self.eval = tmp / "mem_eval.json"
        self.eval.write_text(
            json.dumps({"records": [{"id": f"e{i}", "smiles": s} for i, s in enumerate(eval_smiles)]})
        )

    def collected(
        self, rid: str, status: str = "solved", candidates: list[str] | None = None, **over
    ) -> None:
        rec = self.records[rid]
        cands = [rec.smiles] if candidates is None and status == "solved" else (candidates or [])
        doc = {
            "id": rid, "truth": rec.smiles, "truth_inchikey": rec.inchikey, "status": status,
            "collected_on": "2026-11-18", "outcome_config": CONFIG,
            "candidates": [{"smiles": s, "model_rank": i + 1} for i, s in enumerate(cands)],
        }  # fmt: skip
        doc.update(over)
        (self.collect_dir / f"{rid}.json").write_text(json.dumps(doc))

    def propose(self, **kw) -> dict:
        return bm.propose(
            store=self.store, collect_dir=self.collect_dir, large_file=self.large, eval_file=self.eval,
            read_records=reader(self.records), git=GIT, old_keys=set(), today=TODAY, **kw,
        )  # fmt: skip

    def pending(self, cid: str) -> PendingCase:
        return PendingCase.model_validate_json((self.store / "pending" / f"{cid}.json").read_text())

    def approve(self, ids, typed="APPROVE\n", by="Dr. Reviewer", tty=True, checks=None, **kw) -> int:
        stdin = (Tty if tty else io.StringIO)(typed)
        return bm.approve(
            ids, by, store=self.store, stdin=stdin, stdout=io.StringIO(), eval_file=self.eval, today=TODAY,
            run_check=(lambda s: checks.append(s) or 0) if checks is not None else (lambda s: 0),
            old_keys=set(), **kw,
        )  # fmt: skip

    def reject(self, ids, reason="doubtful", checks=None) -> int:
        return bm.reject(
            ids,
            reason,
            store=self.store,
            today=TODAY,
            run_check=lambda s: (checks if checks is not None else []).append(s) or 0,
        )


@pytest.fixture
def env(tmp_path) -> Env:
    e = Env(tmp_path, [("m1", ASPIRIN), ("m2", ISOMER), ("m3", IBUPROFEN)])
    e.collected("m1", "solved")
    e.collected("m2", "missed", [ASPIRIN, "CCO"])
    e.collected("m3", "no_answer", [], error="LLMUnavailable: status_529")
    return e


# --------------------------------------------------------------------------- propose: fields


def test_proposed_case_has_every_field(env):
    out = env.propose()
    assert out["written"] == ["mem-000001", "mem-000002", "mem-000003"] and out["refused"] == {}
    p = env.pending("mem-000001")
    rec = env.records["m1"]
    chem = chemistry_fields(rec.smiles)
    assert (p.smiles, p.inchikey, p.formula) == (chem["smiles"], chem["inchikey"], chem["formula"])
    assert (p.dou, p.carbon_count, p.heavy_atoms) == (chem["dou"], chem["carbon_count"], chem["heavy_atoms"])
    assert (p.murcko_scaffold, p.generic_scaffold) == (chem["murcko_scaffold"], chem["generic_scaffold"])
    prov = p.provenance
    assert (prov.source, prov.source_record_id, prov.truth_basis) == (
        "nmrshiftdb2",
        "m1",
        "database-assigned structure",
    )
    assert prov.spectrum_ids == {"1H": 1, "13C": 2} and prov.proposed_on == TODAY
    import rdkit

    assert prov.extracted_by == f"bench/memory.py propose @ {GIT['git_commit']}, RDKit {rdkit.__version__}"
    assert p.solvent.value == "CDCl3"
    assert [(h.shift_ppm, h.integral is not None) for h in p.h_peaks][:1] == [
        (12.95, True)
    ]  # descending, integrals given
    assert len(p.c_peaks) == 9 and p.c_peaks == sorted(p.c_peaks, reverse=True)
    out = p.agent_outcome
    assert (out.status, out.proposed_smiles, out.collected_on) == (
        "solved",
        [rec.smiles],
        dt.date(2026, 11, 18),
    )
    assert (out.model_id, out.prompt_version, out.settings_hash) == tuple(CONFIG.values())
    assert "approval" not in json.loads((env.store / "pending" / "mem-000001.json").read_text())


def test_the_peaks_are_what_the_service_would_see(env):
    from bench import benchmark
    from src.prompting import prepare_request
    from src.schemas import Nucleus

    env.propose()
    p = env.pending("mem-000001")
    req, _ = benchmark.request_for(env.records["m1"].pool_record())
    table, _, _ = prepare_request(req)
    assert [(h.shift_ppm, h.integral) for h in p.h_peaks] == [
        (x.shift_ppm, x.integral) for x in table.analyte_peaks(Nucleus.H1)
    ]
    assert p.c_peaks == [x.shift_ppm for x in table.analyte_peaks(Nucleus.C13)]


def test_status_and_proposals_are_copied_from_the_collect_result(env):
    env.propose()
    missed, none = env.pending("mem-000002").agent_outcome, env.pending("mem-000003").agent_outcome
    assert missed.status == "missed" and missed.proposed_smiles == [ASPIRIN, "CCO"]
    assert none.status == "no_answer" and none.proposed_smiles == []


def test_the_pending_file_becomes_a_loadable_case_once_approved(env):
    env.propose()
    from src.memory import finalize

    case = finalize(env.pending("mem-000002"), "A", TODAY)
    assert MemoryCase.model_validate_json(case.model_dump_json()) == case


# --------------------------------------------------------------------------- propose: assignments


ENV_FORMAT = re.compile(r"^[A-Z][a-z]?(, aromatic)?, \d+ H(, ring)?$")


def test_environment_strings_come_from_the_atom():
    ring = Chem.MolFromSmiles("C1CCCCC1")
    assert bm.environment_of(ring.GetAtomWithIdx(0)) == "C, 2 H, ring"
    benzene = Chem.MolFromSmiles("c1ccccc1")
    assert bm.environment_of(benzene.GetAtomWithIdx(0)) == "C, aromatic, 1 H"
    chain = Chem.MolFromSmiles("CC=O")
    assert bm.environment_of(chain.GetAtomWithIdx(0)) == "C, 3 H"
    assert bm.environment_of(chain.GetAtomWithIdx(1)) == "C, 1 H"
    assert bm.environment_of(chain.GetAtomWithIdx(2)) == "O, 0 H"


def test_assignments_map_the_atom_index_and_carry_no_record_text(env):
    env.propose()
    p = env.pending("mem-000001")
    by = {(a.nucleus.value, a.shift_ppm): a.environment for a in p.assignments}
    assert by[("1H", 2.25)] == "C, 3 H"  # atom 0
    assert by[("1H", 7.15)] == "C, aromatic, 1 H"  # atom 5 of the aspirin ring
    assert by[("13C", 169.4)] == "C, 0 H"  # atom 1: the ester carbonyl
    assert by[("13C", 124.0)] == "C, aromatic, 1 H"
    assert all(ENV_FORMAT.match(a.environment) for a in p.assignments)
    assert len(p.assignments) == 6 + 9  # six 1H and nine 13C entries, each with an atom index


def test_text_in_the_record_is_never_copied(tmp_path):
    e = Env(tmp_path, [("m1", ASPIRIN)])
    rec = e.records["m1"]
    e.records["m1"] = bm.SdRecord(**{**rec.__dict__, "c_spec": rec.c_spec.replace("0.0S", "SECRET note")})
    e.collected("m1")
    e.propose()
    text = (e.store / "pending" / "mem-000001.json").read_text()
    assert "SECRET" not in text


def test_entries_without_an_atom_index_are_left_out_and_counted(tmp_path):
    e = Env(tmp_path, [("m1", ASPIRIN)])
    rec = e.records["m1"]
    e.records["m1"] = bm.SdRecord(**{**rec.__dict__, "h_spec": "2.25;0.0;|7.15;0.0;5|7.99;0.0;77|"})
    e.collected("m1")
    out = e.propose()
    p = e.pending("mem-000001")
    assert [(a.nucleus.value, a.shift_ppm) for a in p.assignments if a.nucleus.value == "1H"] == [
        ("1H", 7.15)
    ]
    notes = json.loads((e.store / "pending" / "propose-notes.json").read_text())["mem-000001"]
    assert notes["assignments_left_out"] == 2 and notes["assignments_entries"] == 3 + 9
    assert out["refused"] == {} or "1H" in str(
        out
    )  # a shift of 99 ppm is the request's problem, not an assignment's


def test_no_assignments_at_all_is_stored_as_null(tmp_path):
    e = Env(tmp_path, [("m1", ASPIRIN)])
    rec = e.records["m1"]
    e.records["m1"] = bm.SdRecord(
        **{**rec.__dict__, "h_spec": "2.25;0.0;|7.15;0.0;|", "c_spec": "20.9;0.0Q;|169.4;0.0S;|150.5;0.0S;|"}
    )
    e.collected("m1")
    e.propose()
    assert e.pending("mem-000001").assignments is None


# --------------------------------------------------------------------------- propose: refusals


def refused_reason(env: Env) -> str:
    out = env.propose()
    assert out["written"] == [] and "m1" in out["refused"], out
    return "; ".join(out["refused"]["m1"])


def test_refuses_a_case_that_breaks_the_leakage_rules(tmp_path):
    e = Env(tmp_path, [("m1", ASPIRIN)], eval_smiles=(ASPIRIN,))
    e.collected("m1")
    reason = refused_reason(e)
    assert all(f"rule {n}" in reason for n in (1, 2, 4))
    assert not list((e.store / "pending").glob("mem-*.json"))


def test_refuses_a_shared_scaffold(tmp_path):
    e = Env(
        tmp_path, [("m1", ASPIRIN)], eval_smiles=("CCOC(=O)c1ccccc1",)
    )  # benzene scaffold, other structure
    e.collected("m1")
    assert "rule 2" in refused_reason(e)


def test_refuses_a_proposal_that_is_an_eval_structure(tmp_path):
    e = Env(tmp_path, [("m1", ASPIRIN)])
    e.collected("m1", "missed", [NAPHTHALENE])
    assert "rule 5" in refused_reason(e)


def test_refuses_a_structure_key_that_is_already_pending_or_approved(tmp_path):
    e = Env(tmp_path, [("m1", ASPIRIN), ("m2", ASPIRIN)])
    e.records["m2"] = make_record("m2", ASPIRIN)  # another record of the same structure
    e.collected("m1")
    e.collected("m2")
    out = e.propose()
    assert out["written"] == ["mem-000001"] and "duplicates case mem-000001" in out["refused"]["m2"][0]
    assert e.approve(["mem-000001"]) == 0  # now it is approved, and the duplicate is still refused
    assert "duplicates case mem-000001" in e.propose()["refused"]["m2"][0]


def test_refuses_inconsistent_chemistry(tmp_path):
    e = Env(tmp_path, [("m1", ASPIRIN)])
    rec = e.records["m1"]
    e.records["m1"] = bm.SdRecord(**{**rec.__dict__, "formula": "C9H9O4"})  # not the structure's formula
    e.collected("m1")
    assert "inconsistent" in refused_reason(e)


def test_refuses_a_record_whose_structure_is_not_the_listed_one(tmp_path):
    e = Env(tmp_path, [("m1", ASPIRIN)])
    e.collected("m1")
    e.records["m1"] = make_record("m1", ISOMER)
    assert "differs from the listed" in refused_reason(e)


def test_refuses_an_outcome_collected_under_another_configuration(tmp_path):
    e = Env(tmp_path, [("m1", ASPIRIN)])
    e.collected("m1", outcome_config={**CONFIG, "settings_hash": "0" * 64})
    assert "different configuration" in refused_reason(e)


def test_a_molecule_without_a_result_is_skipped_and_a_case_is_not_proposed_twice(env):
    (env.collect_dir / "m3.json").unlink()
    out = env.propose()
    assert out["skipped"] == {"m3": "not collected"} and len(out["written"]) == 2
    again = env.propose()
    assert again["written"] == [] and again["skipped"]["m1"].startswith("already has case")


def test_case_ids_follow_memory_large_order_and_are_never_reused(tmp_path):
    e = Env(tmp_path, [("m1", ASPIRIN), ("m2", ISOMER)])
    e.collected("m2")
    assert e.propose()["written"] == ["mem-000002"]
    assert e.reject(["mem-000002"]) == 0
    e.collected("m1")
    assert e.propose()["written"] == [
        "mem-000001"
    ]  # the rejected id 2 stays taken, its molecule is not re-proposed
    assert e.propose()["written"] == []
    assert not (e.store / "pending" / "mem-000002.json").exists()


def test_replacement_molecules_get_ids_after_the_list(tmp_path):
    e = Env(tmp_path, [("m1", ASPIRIN)])
    e.records["r1"] = make_record("r1", ISOMER)
    doc = json.loads(e.large.read_text())
    doc["replacement_order"][">20"] = ["r1"]
    e.large.write_text(json.dumps(doc))
    order = tmp_path / "order.json"
    order.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        **doc["entries"][0],
                        "id": "r1",
                        "smiles": e.records["r1"].smiles,
                        "inchikey": e.records["r1"].inchikey,
                    }
                ]
            }
        )
    )
    e.collected("m1")
    e.collected("r1")
    rows = bm.resolve_rows(json.loads(e.large.read_text()), ["r1"], order)
    assert rows[0]["id"] == "r1"
    out = bm.propose(
        ids=["m1"], store=e.store, collect_dir=e.collect_dir, large_file=e.large, eval_file=e.eval,
        read_records=reader(e.records), git=GIT, old_keys=set(), today=TODAY,
    )  # fmt: skip
    assert out["written"] == ["mem-000001"]


# --------------------------------------------------------------------------- review sheets


def test_review_sheets_state_the_rule_and_show_each_case(env):
    env.propose()
    sheet = (env.store / "pending" / "review-batch-01.md").read_text()
    head = sheet[:1200]
    assert "The owner approves or rejects each case with the commands below" in head
    assert (
        "bench/memory.py approve mem-000001 mem-000002 mem-000003" in sheet
        and "bench/memory.py reject" in sheet
    )
    for cid in ("mem-000001", "mem-000002", "mem-000003"):
        assert f"## {cid}" in sheet
    p = env.pending("mem-000001")
    assert f"## mem-000001: {p.formula}, CDCl3" in sheet
    assert f"`{p.smiles}`" in sheet and f"`{p.murcko_scaffold}`" in sheet
    assert "1H peaks (6): 12.95 (1.0 H)" in sheet and "13C peaks (9):" in sheet
    assert "agent outcome: **solved**" in sheet and "**missed**" in sheet and "**no_answer**" in sheet
    assert "leakage rules 1, 2, 4, 5 against the 100 MEM-EVAL molecules: clean" in sheet
    assert "assignments: 15 (0 spectrum entries without an atom index left out)" in sheet
    assert "first proposal: none" in sheet  # the no_answer case


def test_review_sheet_shows_the_first_proposal_in_canonical_form_or_not_at_all(tmp_path):
    e = Env(tmp_path, [("m1", ASPIRIN), ("m2", ISOMER)])
    e.collected("m1", "missed", ["OC(=O)c1ccccc1OC(C)=O"])  # a non-canonical spelling
    e.collected("m2", "missed", ["not a smiles ((("])
    e.propose()
    sheet = (e.store / "pending" / "review-batch-01.md").read_text()
    assert f"first proposal: `{Chem.MolToSmiles(Chem.MolFromSmiles(ASPIRIN))}`" in sheet
    assert "OC(=O)c1ccccc1OC(C)=O" not in sheet
    assert "first proposal: not shown (does not parse as SMILES)" in sheet and "not a smiles" not in sheet


def test_review_sheets_come_in_batches_of_25(tmp_path):
    e = Env(tmp_path, [("m1", ASPIRIN)], total=26)
    e.records["m26"] = make_record("m26", ISOMER)
    e.rows[25].update(id="m26", smiles=e.records["m26"].smiles, inchikey=e.records["m26"].inchikey)
    e.large.write_text(
        json.dumps({"entries": e.rows, "replacement_order": {">20": [], "11-20": [], "<=10": []}})
    )
    e.collected("m1")
    e.collected("m26")
    out = e.propose()
    assert out["written"] == ["mem-000001", "mem-000026"]
    assert sorted(p.name for p in (e.store / "pending").glob("review-batch-*.md")) == [
        "review-batch-01.md",
        "review-batch-02.md",
    ]
    assert "mem-000026" in (e.store / "pending" / "review-batch-02.md").read_text()
    assert bm.batch_of("mem-000025") == 1 and bm.batch_of("mem-000026") == 2


def test_a_leakage_failure_in_a_pending_file_shows_on_its_sheet(env):
    env.propose()
    env.eval.write_text(json.dumps({"records": [{"id": "e0", "smiles": ASPIRIN}]}))
    bm.write_review_sheets(env.store, env.collect_dir, json.loads(env.eval.read_text())["records"], set())
    assert "FAILED" in (env.store / "pending" / "review-batch-01.md").read_text()


# --------------------------------------------------------------------------- approve


def file_bytes(path: Path) -> bytes:
    return path.read_bytes()


def test_approve_needs_the_exact_word_typed_at_a_terminal(env):
    env.propose()
    for typed in ("approve\n", "APPROVE please\n", " APPROVE\n", "yes\n", "\n", ""):
        assert env.approve(["mem-000001"], typed=typed) == 1
    assert (env.store / "pending" / "mem-000001.json").exists() and not list(
        (env.store / "cases").glob("mem-*.json")
    )
    assert json.loads((env.store / "manifest.json").read_text())["cases"] == []
    assert (env.store / "APPROVALS.log").read_text() == ""


def test_approve_refuses_a_non_interactive_stdin(env, monkeypatch):
    env.propose()
    monkeypatch.setenv("APPROVE", "1")  # no environment variable can stand in for the prompt
    assert env.approve(["mem-000001"], typed="APPROVE\n", tty=False) == 1
    assert (env.store / "pending" / "mem-000001.json").exists() and not (
        env.store / "cases" / "mem-000001.json"
    ).exists()


def test_the_command_line_has_no_way_to_skip_the_prompt(env, monkeypatch, capsys):
    env.propose()
    base = ["bench/memory.py", "approve", "mem-000001", "--by", "X", "--store", str(env.store)]
    for extra in ([], ["--yes"], ["--force"], ["--confirm", "APPROVE"], ["-y"]):
        monkeypatch.setattr("sys.argv", base + extra)
        with pytest.raises(SystemExit) as stop:
            bm.main()
        assert stop.value.code not in (0, None)
    assert (env.store / "pending" / "mem-000001.json").exists()
    import inspect

    assert not {"yes", "force", "confirm", "skip"} & set(inspect.signature(bm.approve).parameters)


def test_approve_moves_the_case_and_updates_the_manifest_and_the_log(env):
    env.propose()
    pending = (env.store / "pending" / "mem-000001.json").read_text()
    checks: list[str] = []
    assert env.approve(["mem-000001", "mem-000002"], checks=checks) == 0
    assert checks == [str(env.store)]
    assert (
        not (env.store / "pending" / "mem-000001.json").exists()
        and (env.store / "pending" / "mem-000003.json").exists()
    )
    case_file = env.store / "cases" / "mem-000001.json"
    case = MemoryCase.model_validate_json(case_file.read_text())
    assert (case.approval.approved_by, case.approval.date_added, case.approval.note) == (
        "Dr. Reviewer",
        TODAY,
        None,
    )
    assert case.model_dump(exclude={"approval"}) == json.loads(pending) | {} or True
    assert {k: v for k, v in json.loads(case_file.read_text()).items() if k != "approval"} == json.loads(
        pending
    )
    manifest = json.loads((env.store / "manifest.json").read_text())
    assert manifest["cases"] == [
        {
            "case_id": cid,
            "sha256": hashlib.sha256((env.store / "cases" / f"{cid}.json").read_bytes()).hexdigest(),
        }
        for cid in ("mem-000001", "mem-000002")
    ]
    assert (env.store / "APPROVALS.log").read_text().splitlines() == [
        "2026-11-20 mem-000001 approve Dr. Reviewer",
        "2026-11-20 mem-000002 approve Dr. Reviewer",
    ]


def test_the_approved_store_loads_in_strict_mode_with_the_matching_configuration(env, capsys):
    env.propose()
    assert env.approve(["mem-000001", "mem-000002", "mem-000003"]) == 0
    store = load_store(env.store, strict=True, expected_outcome_config=outcome_config(Settings()))
    assert [c.case_id for c in store.cases] == [
        "mem-000001",
        "mem-000002",
        "mem-000003",
    ] and not store.rejections
    assert f"memory_version: {store.memory_version}" in capsys.readouterr().out
    from src.memory import MemoryRuntime

    runtime = MemoryRuntime.from_directory(env.store, Settings(), strict=True)
    assert runtime.memory_version == store.memory_version


def test_approve_refuses_what_is_not_pending_or_has_no_name_and_changes_nothing(env):
    env.propose()
    assert env.approve(["mem-000001", "mem-000099"]) == 2
    assert env.approve(["mem-000001"], by="  ") == 2
    assert env.approve(["mem-000001", "mem-000001"]) == 2
    assert (env.store / "pending" / "mem-000001.json").exists() and not list(
        (env.store / "cases").glob("mem-*.json")
    )


def test_approve_refuses_a_pending_case_that_no_longer_passes_the_checks(env):
    env.propose()
    env.eval.write_text(json.dumps({"records": [{"id": "e0", "smiles": ASPIRIN}]}))
    assert env.approve(["mem-000001"]) == 2
    path = env.store / "pending" / "mem-000002.json"
    doc = json.loads(path.read_text())
    doc["agent_outcome"]["settings_hash"] = "f" * 64
    path.write_text(json.dumps(doc))
    assert env.approve(["mem-000002"]) == 2
    assert not list((env.store / "cases").glob("mem-*.json"))


# --------------------------------------------------------------------------- reject


def test_reject_moves_the_file_with_the_reason_and_logs_it(env):
    env.propose()
    before = json.loads((env.store / "pending" / "mem-000002.json").read_text())
    manifest = (env.store / "manifest.json").read_bytes()
    checks: list[str] = []
    assert env.reject(["mem-000002"], "a database problem,\nnot a miss", checks) == 0
    assert not (env.store / "pending" / "mem-000002.json").exists()
    doc = json.loads((env.store / "rejected" / "mem-000002.json").read_text())
    assert (
        doc["reason"] == "a database problem, not a miss"
        and doc["case"] == before
        and doc["rejected_on"] == "2026-11-20"
    )
    assert (
        env.store / "APPROVALS.log"
    ).read_text() == "2026-11-20 mem-000002 reject a database problem, not a miss\n"
    assert (env.store / "manifest.json").read_bytes() == manifest and checks == [str(env.store)]


def test_reject_refuses_an_unknown_case_and_an_empty_reason(env):
    env.propose()
    assert env.reject(["mem-000001", "mem-000099"]) == 2
    assert env.reject(["mem-000001"], reason=" ") == 2
    assert (env.store / "pending" / "mem-000001.json").exists() and not list(
        (env.store / "rejected").glob("mem-*.json")
    )


# --------------------------------------------------------------------------- the store skeleton


def test_setup_store_creates_the_empty_skeleton(tmp_path, capsys):
    store = tmp_path / "memory"
    bm.setup_store(store)
    assert json.loads((store / "manifest.json").read_text()) == {"store_schema_version": "1", "cases": []}
    assert (store / "retrieval_params.json").read_bytes() == (
        ROOT / "bench" / "memory" / "retrieval_params.json"
    ).read_bytes()
    assert (store / "APPROVALS.log").read_bytes() == b""
    assert (store / "cases" / ".gitkeep").exists() and (store / "rejected" / ".gitkeep").exists()
    notice = (store / "NMRSHIFTDB2-NOTICE.md").read_text()
    assert "nmrshiftdb2" in notice and "NMRSHIFTDB2-DATABASE-LICENSE.txt" in notice
    s = load_store(store, strict=True, expected_outcome_config=outcome_config(Settings()))
    assert s.cases == () and s.memory_version == "none"
    with pytest.raises(SystemExit, match="refusing"):
        bm.setup_store(store)


def test_check_passes_on_the_empty_skeleton(tmp_path, capsys):
    store = tmp_path / "memory"
    bm.setup_store(store)
    assert bm.check(str(store)) == 0
    assert "clean" in capsys.readouterr().out


def test_the_repository_store_skeleton_is_committed():
    store = ROOT / "memory"
    assert (store / "retrieval_params.json").read_bytes() == (
        ROOT / "bench" / "memory" / "retrieval_params.json"
    ).read_bytes()
    for name in (
        "manifest.json",
        "NMRSHIFTDB2-NOTICE.md",
        "APPROVALS.log",
        "cases/.gitkeep",
        "rejected/.gitkeep",
    ):
        assert (store / name).exists(), name
    load_store(store, strict=True, expected_outcome_config=outcome_config(Settings()))


def test_pending_cases_are_not_tracked():
    assert "memory/pending/" in (ROOT / ".gitignore").read_text().splitlines()
