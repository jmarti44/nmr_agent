"""Census of nmrshiftdb2withsignals.sd (review R0-2): which records can feed the SPEC §11.4 benchmark?

Usage: uv run python bench/nmrshiftdb2_census.py [path/to/nmrshiftdb2withsignals.sd]

Spectrum N is treated as predicted when N appears with a non-empty value in the record's `Program` or
`NMRProgram` tag (e.g. "ACD/Labs ...", "Quantum ESPRESSO"); all other spectra are treated as experimental.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")
DEFAULT = Path(__file__).parent / "data/nmrshiftdb2/nmrshiftdb2withsignals.sd"
ALLOWED = {"C", "H", "N", "O", "S", "F", "Cl", "Br"}
TAG = re.compile(r"^> +<([^>]+)>\n(.*?)(?=\n\n|\Z)", re.M | re.S)
INDEXED = re.compile(r"(\d+):(.*?)(?=\s+\d+:|\s*$)", re.S)


def records(path: Path):
    buf: list[str] = []
    with path.open(errors="replace", newline=None) as fh:
        for line in fh:
            if line.startswith("$$$$"):
                yield "".join(buf)
                buf = []
            else:
                buf.append(line)


def indexed(value: str) -> dict[int, str]:
    return {int(i): v.strip() for i, v in INDEXED.findall(value.strip())}


def analyse(rec: str) -> dict:
    tags = {k: v for k, v in TAG.findall(rec[rec.find("M  END") :])}
    predicted = {i for t in ("Program", "NMRProgram") for i, v in indexed(tags.get(t, "")).items() if v}
    solvents = indexed(tags.get("Solvent", ""))
    spectra = {"1H": set(), "13C": set()}
    for k in tags:
        m = re.fullmatch(r"Spectrum (1H|13C) (\d+)", k)
        if m:
            spectra[m[1]].add(int(m[2]))
    exp = {n: {i for i in s if i not in predicted} for n, s in spectra.items()}
    mol = Chem.MolFromMolBlock(rec[: rec.find("M  END") + 6], sanitize=True, removeHs=False)
    ok = False
    if mol is not None:
        elements = {a.GetSymbol() for a in mol.GetAtoms()}
        heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
        ok = elements <= ALLOWED and heavy <= 30 and Descriptors.MolWt(mol) <= 450
    return {"spectra": spectra, "exp": exp, "solvent": solvents, "spec_filters": ok}


def main(path: Path) -> None:
    c: Counter = Counter()
    solv_h: Counter = Counter()
    solv_hc_same: Counter = Counter()
    for rec in records(path):
        r = analyse(rec)
        c["records"] += 1
        h, cc = bool(r["spectra"]["1H"]), bool(r["spectra"]["13C"])
        eh, ec = bool(r["exp"]["1H"]), bool(r["exp"]["13C"])
        c["any_1H"] += h
        c["any_13C"] += cc
        c["any_both"] += h and cc
        c["exp_1H"] += eh
        c["exp_13C"] += ec
        c["exp_both"] += eh and ec
        ok = r["spec_filters"]
        c["filtered_all"] += ok
        c["filtered_exp_1H"] += ok and eh
        c["filtered_exp_13C"] += ok and ec
        c["filtered_exp_both"] += ok and eh and ec
        for i in r["exp"]["1H"]:
            solv_h[r["solvent"].get(i, "?")] += 1
        if ok and eh and ec:
            for i in r["exp"]["1H"]:
                solv_hc_same[r["solvent"].get(i, "?")] += 1
    print("| metric | records |\n|---|---|")
    for k, v in c.items():
        print(f"| {k} | {v} |")
    print("\nSolvents of experimental 1H spectra (all records):")
    for k, v in solv_h.most_common(8):
        print(f"  {v:6d}  {k}")
    print("\nSolvents of experimental 1H spectra (filtered, both nuclei):")
    for k, v in solv_hc_same.most_common(8):
        print(f"  {v:6d}  {k}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT)
