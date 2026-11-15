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


# --------------------------------------------------------------------------- explicit H atoms in the molblock (RM6-2)


def explicit_h_mol() -> Chem.Mol:
    """Ethyl acetate with every H written right after its heavy atom, as some database molblocks do.
    File order: 0 C(H3) 1-3 H, 4 C 5 O 6 O 7 C(H2) 8-9 H, 10 C(H3) 11-13 H."""
    base = Chem.AddHs(Chem.MolFromSmiles(LAYOUT))
    order: list[int] = []
    for a in base.GetAtoms():
        if a.GetSymbol() != "H":
            order += [a.GetIdx(), *[n.GetIdx() for n in a.GetNeighbors() if n.GetSymbol() == "H"]]
    return Chem.RenumberAtoms(base, order)


# 13C: CH3 (0), C=O (4), CH2 (7), CH3 (10). 1H: the acetyl CH3 by its three H atoms, the CH2 by one H atom,
# the ethyl CH3 by its carbon.
EXPLICIT_H = "2.05;0.0;1|2.05;0.0;2|2.05;0.0;3|4.12;0.0;8|4.12;0.0;9|1.26;0.0;10|"
EXPLICIT_C = "20.9;0.0Q;0|171.0;0.0S;4|60.4;0.0T;7|14.2;0.0Q;10|"


def explicit_h_record(rid: str = "777") -> tuple[dict, str]:
    mol = explicit_h_mol()
    block = Chem.MolToMolBlock(mol)
    plain = Chem.MolFromMolBlock(block)
    r = record(rid, plain, EXPLICIT_H) | {"c_spec": EXPLICIT_C}
    return r, block


@pytest.fixture
def explicit_sd(tmp_path, monkeypatch):
    r, block = explicit_h_record()
    path = tmp_path / "h.sd"
    solv = "> <Solvent>\n0:Chloroform-D1 1:Chloroform-D1\n\n"
    path.write_text(
        block
        + f"> <nmrshiftdb2 ID>\n777\n\n> <Spectrum 1H 1>\n{EXPLICIT_H}\n\n> <Spectrum 13C 0>\n{EXPLICIT_C}\n\n{solv}$$$$\n"
    )
    monkeypatch.setattr(benchmark, "SD", str(path))
    benchmark._MOLBLOCKS.pop(str(path), None)
    return r, path


def test_the_explicit_h_layout_really_shifts_the_heavy_atoms():
    block = Chem.MolToMolBlock(explicit_h_mol())
    mol = Chem.MolFromMolBlock(block)
    assert (
        mol.GetNumAtoms() == 6
        and mol.GetNumAtoms() < Chem.MolFromMolBlock(block, removeHs=False).GetNumAtoms()
    )
    m = benchmark.atom_index_map(block, mol)
    assert len(m) == 14 and m[0] == 0 and m[1] == m[2] == m[3] == 0  # H atoms -> their carbon
    assert m[4] == 1 and m[7] == 4 and m[8] == m[9] == 4 and m[10] == 5


def test_explicit_hydrogens_before_heavy_atoms_give_correct_integrals(explicit_sd):
    """Fails on 3c2e9c8, which applied the file indices to the H-removed molecule (index 10 does not exist there):
    no integrals, because the molecule counted as inconsistent."""
    r, _ = explicit_sd
    req, complete = benchmark.request_for(r)
    assert complete is True
    assert integrals(req) == {4.12: 2.0, 2.05: 3.0, 1.26: 3.0}
    assert [p.shift_ppm for p in req.peaks.peaks if p.nucleus == Nucleus.H1] == [4.12, 2.05, 1.26]


def test_explicit_hydrogens_give_correct_assignment_environments(explicit_sd):
    """Fails on 3c2e9c8: the 13C indices 4, 7 and 10 name the wrong atoms or none in the H-removed molecule."""
    import bench.memory as bm

    r, path = explicit_sd
    rec = bm.read_sd_records({"777": "named"}, path)["777"]
    a, stats = bm.build_assignments(rec)
    got = {(x["nucleus"], x["shift_ppm"]): x["environment"] for x in a}
    assert got[("13C", 171.0)] == "C, 0 H" and got[("13C", 60.4)] == "C, 2 H"
    assert got[("13C", 20.9)] == "C, 3 H" and got[("13C", 14.2)] == "C, 3 H"
    assert got[("1H", 4.12)] == "C, 2 H" and got[("1H", 2.05)] == "C, 3 H" and got[("1H", 1.26)] == "C, 3 H"
    assert stats == {"entries": 10, "left_out": 0} and len(a) == 10


def test_an_index_that_maps_to_nothing_makes_the_assignment_inconsistent_and_is_left_out(explicit_sd):
    import bench.memory as bm

    r, path = explicit_sd
    bad = r | {"h_spec": EXPLICIT_H + "9.99;0.0;99|"}
    req, complete = benchmark.request_for(bad)
    assert complete is False and set(integrals(req).values()) == {None} and 9.99 in integrals(req)
    rec = bm.read_sd_records({"777": "named"}, path)["777"]
    rec = bm.SdRecord(**{**rec.__dict__, "h_spec": bad["h_spec"]})
    a, stats = bm.build_assignments(rec)
    assert stats == {"entries": 11, "left_out": 1} and len(a) == 10


def test_a_mapping_that_cannot_be_verified_raises():
    r, block = explicit_h_record()
    other = Chem.MolFromSmiles("CCOC(C)=O")  # same atoms, but not the molblock's order
    with pytest.raises(benchmark.MolblockError, match="one to one"):
        benchmark.atom_index_map(block, other)
    with pytest.raises(benchmark.MolblockError, match="one to one"):
        benchmark.atom_index_map(block, Chem.MolFromSmiles("CCCC"))


# --------------------------------------------------------------------------- real data


needs_sd = pytest.mark.skipif(not SD.exists(), reason="bench/data (the NMRShiftDB2 export) is absent")


@pytest.fixture(scope="module")
def main50():
    from bench.memory import read_sd_records

    recs = json.loads((ROOT / "bench" / "results_main" / "selection.json").read_text())["records"]
    sd = read_sd_records({r["id"]: "named" for r in recs}, SD)
    return recs, sd


def mapped(rec, k: int):
    return rec.index_map[k] if k < len(rec.index_map) else None


def c13_on_carbons(recs, sd) -> tuple[int, int]:
    total = carbons = 0
    for r in recs:
        rec = sd[r["id"]]
        for e in r["c_spec"].strip().strip("|").split("|"):
            p = e.split(";")
            if p[0] and len(p) >= 3 and p[2].strip().isdigit():
                total += 1
                m = mapped(rec, int(p[2]))
                carbons += m is not None and rec.mol.GetAtomWithIdx(m).GetSymbol() == "C"
    return carbons, total


@needs_sd
def test_main50_13c_indices_land_on_carbons(main50):
    recs, sd = main50
    assert c13_on_carbons(recs, sd) == (598, 598)


@needs_sd
def test_pilot_13c_indices_land_on_carbons():
    from bench.memory import read_sd_records

    recs = json.loads((ROOT / "bench" / "results" / "selection.json").read_text())["records"]
    sd = read_sd_records({r["id"]: "named" for r in recs}, SD)
    assert c13_on_carbons(recs, sd) == (283, 283)


@needs_sd
def test_main50_integrals_counts(main50):
    """49 of the main 50 get integrals (RM6-1 alone: 41 - the explicit-H molecules were withheld)."""
    recs, sd = main50
    given = 0
    for r in recs:
        rec = sd[r["id"]]
        req, complete = benchmark.request_for(r, rec.mol, rec.index_map)
        given += complete
        assert complete == any(p.integral is not None for p in req.peaks.peaks)
    assert given == 49


@needs_sd
def test_valine_with_explicit_hydrogens_maps_to_the_file_order():
    """10008327: file atoms C C C C H H H N O C O...; the carbonyl's 13C index is out of range in the H-removed
    molecule."""
    from bench.memory import read_sd_records

    rec = read_sd_records({"10008327": "named"}, SD)["10008327"]
    assert (
        max(int(e.split(";")[2]) for e in rec.c_spec.strip().strip("|").split("|")) >= rec.mol.GetNumAtoms()
    )
    c = [mapped(rec, int(e.split(";")[2])) for e in rec.c_spec.strip().strip("|").split("|")]
    assert None not in c and all(rec.mol.GetAtomWithIdx(m).GetSymbol() == "C" for m in c)
