# ruff: noqa: E501
"""bench.benchmark.request_for: the spectrum atom indices refer to the SD molblock, 0-based (review RM6-1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem

from bench import benchmark
from src.schemas import Nucleus

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "bench" / "data" / "nmrshiftdb2" / "nmrshiftdb2withsignals.sd"

# "CC(=O)OCC" in this order is the molblock's atom order; the canonical SMILES "CCOC(C)=O" orders them differently.
LAYOUT = "CC(=O)OCC"  # 0 CH3, 1 C=O, 2 O, 3 O, 4 CH2, 5 CH3
H_SPEC = "2.05;0.0;0|4.12;0.0;4|1.26;0.0;5|"
C_SPEC = "20.9;0.0Q;0|171.0;0.0S;1|60.4;0.0T;4|14.2;0.0Q;5|"


def sd_text(rid: str, mol: Chem.Mol, h_spec: str = H_SPEC, c_spec: str = C_SPEC) -> str:
    return (
        Chem.MolToMolBlock(mol)
        + f"> <nmrshiftdb2 ID>\n{rid}\n\n> <Spectrum 1H 1>\n{h_spec}\n\n> <Spectrum 13C 0>\n{c_spec}\n\n$$$$\n"
    )


def record(rid: str, mol: Chem.Mol, h_spec: str = H_SPEC) -> dict:
    return {
        "id": rid,
        "smiles": Chem.MolToSmiles(mol),
        "inchikey": Chem.MolToInchiKey(mol),
        "formula": "C4H8O2",
        "solvent": "CDCl3",
        "h_spec": h_spec,
        "c_spec": C_SPEC,
    }


@pytest.fixture
def sd_file(tmp_path, monkeypatch):
    """A tiny SD export holding the ethyl acetate record; benchmark.SD points at it."""
    mol = Chem.MolFromSmiles(LAYOUT)
    path = tmp_path / "x.sd"
    path.write_text(sd_text("777", mol))
    monkeypatch.setattr(benchmark, "SD", str(path))
    benchmark._MOLBLOCKS.pop(str(path), None)
    return path


def integrals(req) -> dict[float, float | None]:
    return {p.shift_ppm: p.integral for p in req.peaks.peaks if p.nucleus == Nucleus.H1}


def test_molblock_and_canonical_atom_orders_really_differ():
    mol = Chem.MolFromSmiles(LAYOUT)
    canon = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    assert [a.GetTotalNumHs() for a in mol.GetAtoms()] != [a.GetTotalNumHs() for a in canon.GetAtoms()]


def test_integrals_use_the_molblock_atom_order(sd_file):
    """Fails on the code before RM6-1: the indices then landed on the canonical-SMILES atoms (idx 5 is an O there),
    so the molecule counted as incomplete and no integral was given."""
    r = record("777", Chem.MolFromSmiles(LAYOUT))
    req, complete = benchmark.request_for(r)
    assert complete is True
    assert integrals(req) == {4.12: 2.0, 2.05: 3.0, 1.26: 3.0}
    assert [p.shift_ppm for p in req.peaks.peaks if p.nucleus == Nucleus.H1] == [4.12, 2.05, 1.26]


def test_a_molecule_can_be_passed_instead_of_looked_up():
    mol = Chem.MolFromSmiles(LAYOUT)
    req, complete = benchmark.request_for(record("not-in-any-file", mol), mol)
    assert complete and integrals(req)[4.12] == 2.0


def test_a_record_missing_from_the_export_raises(sd_file):
    r = record("999", Chem.MolFromSmiles(LAYOUT))
    with pytest.raises(benchmark.MolblockError, match="no molblock"):
        benchmark.request_for(r)


def test_a_molblock_that_does_not_match_the_stored_smiles_raises(sd_file):
    r = record("777", Chem.MolFromSmiles(LAYOUT)) | {"smiles": "CCCC(=O)O"}
    with pytest.raises(benchmark.MolblockError, match="does not match"):
        benchmark.request_for(r)
    with pytest.raises(benchmark.MolblockError, match="does not match"):
        benchmark.request_for(r, Chem.MolFromSmiles(LAYOUT))


def test_the_export_is_read_once_for_any_number_of_records(sd_file, monkeypatch):
    reads = []
    real = benchmark.records

    def counting():
        reads.append(1)
        return real()

    monkeypatch.setattr(benchmark, "records", counting)
    r = record("777", Chem.MolFromSmiles(LAYOUT))
    for _ in range(5):
        benchmark.request_for(r)
    with pytest.raises(benchmark.MolblockError):
        benchmark.request_for(record("888", Chem.MolFromSmiles(LAYOUT)))
    assert len(reads) == 1


def test_an_entry_on_an_atom_without_hydrogen_withholds_integrals_and_keeps_every_peak(sd_file):
    mol = Chem.MolFromSmiles(LAYOUT)
    h_spec = H_SPEC + "9.99;0.0;1|"  # atom 1 is the carbonyl carbon: no H
    req, complete = benchmark.request_for(record("777", mol, h_spec))
    assert complete is False
    assert integrals(req) == {9.99: None, 4.12: None, 2.05: None, 1.26: None}


def test_an_entry_without_an_atom_index_or_with_one_out_of_range_withholds_integrals(sd_file):
    mol = Chem.MolFromSmiles(LAYOUT)
    for extra in ("9.99;0.0;|", "9.99;0.0;77|"):
        req, complete = benchmark.request_for(record("777", mol, H_SPEC + extra))
        assert complete is False and set(integrals(req)) == {9.99, 4.12, 2.05, 1.26}
        assert set(integrals(req).values()) == {None}


def test_a_missing_hydrogen_bearing_carbon_withholds_integrals(sd_file):
    mol = Chem.MolFromSmiles(LAYOUT)
    req, complete = benchmark.request_for(record("777", mol, "2.05;0.0;0|4.12;0.0;4|"))
    assert complete is False and set(integrals(req).values()) == {None}


def test_a_missing_or_mismatched_molblock_is_a_harness_error_not_a_service_rejection(sd_file):
    """memory_gonogo.request_error turns ValueError into "the service rejects this record"; a lookup problem
    must propagate instead of excluding the molecule."""
    from bench.memory_gonogo import request_error

    assert not issubclass(benchmark.MolblockError, ValueError)
    mol = Chem.MolFromSmiles(LAYOUT)
    with pytest.raises(benchmark.MolblockError):
        request_error(record("999", mol))
    with pytest.raises(benchmark.MolblockError):
        request_error(record("777", mol) | {"smiles": "CCCC(=O)O"})
    assert request_error(record("777", mol)) is None


# --------------------------------------------------------------------------- real data


needs_sd = pytest.mark.skipif(not SD.exists(), reason="bench/data (the NMRShiftDB2 export) is absent")


@pytest.fixture(scope="module")
def main50():
    from bench.memory import read_sd_records

    recs = json.loads((ROOT / "bench" / "results_main" / "selection.json").read_text())["records"]
    sd = read_sd_records({r["id"]: "named" for r in recs}, SD)
    return recs, sd


@needs_sd
def test_main50_13c_indices_land_on_carbons(main50):
    recs, sd = main50
    total = carbons = 0
    for r in recs:
        mol = sd[r["id"]].mol
        for e in r["c_spec"].strip().strip("|").split("|"):
            p = e.split(";")
            if p[0] and len(p) >= 3 and p[2].strip().isdigit():
                total += 1
                carbons += int(p[2]) < mol.GetNumAtoms() and mol.GetAtomWithIdx(int(p[2])).GetSymbol() == "C"
    assert total > 500 and carbons / total >= 0.98


@needs_sd
def test_main50_integrals_counts(main50, monkeypatch):
    """The coordinator's probe (no inconsistency rule) gives 43 molecules whose H-bearing carbons all have a 1H
    shift. The rule that an entry on an atom without H makes the assignment inconsistent removes two of them."""
    recs, sd = main50
    monkeypatch.setattr(benchmark, "SD", str(SD))
    flag_only = given = 0
    for r in recs:
        mol = sd[r["id"]].mol
        hit = {
            int(e.split(";")[2])
            for e in r["h_spec"].strip().strip("|").split("|")
            if e.count(";") >= 2 and e.split(";")[2].strip().isdigit()
        }
        need = {a.GetIdx() for a in mol.GetAtoms() if a.GetTotalNumHs() and a.GetSymbol() == "C"}
        flag_only += need <= hit
        req, complete = benchmark.request_for(r, mol)
        given += complete
        assert complete == any(p.integral is not None for p in req.peaks.peaks)
    assert flag_only == 43
    assert given == 41
