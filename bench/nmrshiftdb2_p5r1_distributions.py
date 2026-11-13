"""Distributions behind the P5R-1 range changes, computed WITHOUT the benchmark molecules.

Usage: uv run python -m bench.nmrshiftdb2_p5r1_distributions
Leaves out every molecule in the three benchmark selections (by NMRShiftDB2 id and
InChIKey connectivity block) so the new ranges are not fitted to the molecules whose failures prompted
them. Experimental spectra only.
Prints, for the rules changed in P5R-1: aldehyde 1H and 13C, ring carbons of heteroaromatic rings, and how
often an amide N-H is listed for molecules that have one.
"""

from __future__ import annotations

import collections
import re

from rdkit import Chem, RDLogger

from bench.nmrshiftdb2_census import DEFAULT, TAG, records
from bench.nmrshiftdb2_shift_distributions import excluded_from, idx, q

RDLogger.DisableLog("rdApp.*")

SELECTIONS = [
    "bench/results/selection.json",
    "bench/results_main/selection.json",
    "bench/results_unreported/selection.json",
]
P_CHO = Chem.MolFromSmarts("[CX3H1](=O)[#6]")
P_CHO_ARYL = Chem.MolFromSmarts("[CX3H1](=O)a")
P_CHO_ORTHO_O = Chem.MolFromSmarts("[CX3H1](=O)c:c[OX2]")
# ring carbons of 5/6-membered heteroaromatic rings, split by what they are bonded to inside the ring
P_C_NEXT_N = Chem.MolFromSmarts("[c;r5,r6;$(c:n)]")  # C=N-type carbon (azole C2, azine C2, ...)
P_C_NEXT_OS = Chem.MolFromSmarts("[c;r5,r6;$(c:[o,s]);!$(c:n)]")
P_C_NEXT_N_AND_X = Chem.MolFromSmarts("[c;r5,r6;$(c(:n):[n,o,s])]")  # between two ring heteroatoms
# V09 ranges: sp3 CH2 whose neighbours are all sp3 carbons, in a cyclobutane (all-carbon 4-ring) / allylic
P_CYCLOBUTANE_CH2 = Chem.MolFromSmarts("[CX4;H2;r4;$([C]([CX4])[CX4]);!$(C~[!#6])]")
P_CYCLOBUTANE_CH = Chem.MolFromSmarts("[CX4;H1;r4;$([C]([CX4])([CX4])[CX4]);!$(C~[!#6])]")
P_ALLYLIC_CH2 = Chem.MolFromSmarts("[CX4;H2;!$(C~[!#6]);$(C(-[CX3]=[CX3])-[CX4])]")
P_ALLYLIC_CH2_RING = Chem.MolFromSmarts("[CX4;H2;R;!$(C~[!#6]);$(C(-[CX3;R]=[CX3;R])-[CX4;R])]")
P_AMIDE_NH = Chem.MolFromSmarts("[CX3](=O)[NX3;H1,H2]")


def main() -> None:
    ex_ids, ex_keys = excluded_from(SELECTIONS)
    out: dict[str, list[float]] = collections.defaultdict(list)
    amide = collections.Counter()
    mols = skipped = 0
    for rec in records(DEFAULT):
        end = rec.find("M  END")
        mol = Chem.MolFromMolBlock(rec[: end + 6])
        if mol is None:
            continue
        tags = {k: v for k, v in TAG.findall(rec[end:])}
        if (
            tags.get("nmrshiftdb2 ID", "").strip() in ex_ids
            or Chem.MolToInchiKey(mol).split("-")[0] in ex_keys
        ):
            skipped += 1
            continue
        mols += 1
        measured = set(idx(tags.get("Solvent", "")))
        for k, v in tags.items():
            m = re.fullmatch(r"Spectrum (1H|13C) (\d+)", k)
            if not m or int(m[2]) not in measured:
                continue
            by_atom: dict[int, list[float]] = collections.defaultdict(list)
            for e in v.strip().strip("|").split("|"):
                p = e.split(";")
                if len(p) >= 3 and p[0] and p[2].strip().isdigit() and int(p[2]) < mol.GetNumAtoms():
                    by_atom[int(p[2])].append(float(p[0]))
            if m[1] == "1H":
                for mt in mol.GetSubstructMatches(P_CHO):
                    out["aldehyde 1H, all"] += by_atom.get(mt[0], [])
                for mt in mol.GetSubstructMatches(P_CHO_ARYL):
                    out["aldehyde 1H, aryl"] += by_atom.get(mt[0], [])
                for mt in mol.GetSubstructMatches(P_CHO_ORTHO_O):
                    out["aldehyde 1H, aryl with ortho O"] += by_atom.get(mt[0], [])
                for patt, name in (
                    (P_CYCLOBUTANE_CH2, "cyclobutane CH2 (all-C ring, sp3 C neighbours)"),
                    (P_CYCLOBUTANE_CH, "cyclobutane CH (all-C ring, sp3 C neighbours)"),
                    (P_ALLYLIC_CH2, "allylic CH2 (no heteroatom neighbour)"),
                    (P_ALLYLIC_CH2_RING, "allylic CH2 in a ring, ring C=C"),
                ):
                    for mt in mol.GetSubstructMatches(patt):
                        out[f"1H {name}"] += by_atom.get(mt[0], [])
                for mt in mol.GetSubstructMatches(P_AMIDE_NH):
                    amide["amide N-H sites"] += 1
                    amide["with a listed 1H shift on N"] += bool(by_atom.get(mt[2]))
            else:
                for mt in mol.GetSubstructMatches(P_CHO):
                    out["aldehyde 13C, all"] += by_atom.get(mt[0], [])
                for mt in mol.GetSubstructMatches(P_CHO_ARYL):
                    out["aldehyde 13C, aryl"] += by_atom.get(mt[0], [])
                for patt, name in (
                    (P_C_NEXT_N, "ring C next to ring N"),
                    (P_C_NEXT_OS, "ring C next to ring O/S (no N)"),
                    (P_C_NEXT_N_AND_X, "ring C between two ring heteroatoms"),
                ):
                    for mt in mol.GetSubstructMatches(patt):
                        out[f"13C {name}"] += by_atom.get(mt[0], [])
    print("# P5R-1 distributions (experimental spectra, benchmark molecules excluded)")
    print(f"# molecules used: {mols}; benchmark molecules left out: {skipped}")
    for k in sorted(out):
        print(f"{k}: {q(out[k])}")
    for k, v in amide.items():
        print(f"{k}: {v}")
    for name, lo, hi in (
        ("aldehyde 1H, all", 9.0, 10.1),
        ("aldehyde 13C, all", 190, 208),
        ("13C ring C next to ring N", 100, 160),
        ("13C ring C between two ring heteroatoms", 100, 160),
        ("13C ring C next to ring O/S (no N)", 100, 160),
    ):
        xs = out[name]
        if xs:
            inside = sum(lo <= x <= hi for x in xs)
            print(f"  inside current range {lo}-{hi}: {inside}/{len(xs)} ({100 * inside / len(xs):.1f}%)")


if __name__ == "__main__":
    main()
