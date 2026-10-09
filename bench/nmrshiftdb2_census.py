"""Census of nmrshiftdb2withsignals.sd (review R0-2): which records can feed the SPEC §11.4 benchmark?

Usage:
  uv run python bench/nmrshiftdb2_census.py [path/to/nmrshiftdb2withsignals.sd]
  uv run python bench/nmrshiftdb2_census.py --spotcheck N   # list N random "experimental" spectra

Spectrum N is treated as predicted when N appears with a non-empty value in the record's `Program` or
`NMRProgram` tag (e.g. "ACD/Labs ...", "Quantum ESPRESSO"); all other spectra are treated as experimental.
This is a blacklist, so the census also cross-checks it against a positive criterion: in this file only
measured spectra have entries in the per-spectrum `Solvent` tag (`solvent_key_disagreements` in the output).

A molecule is usable for the benchmark only if it has an experimental 1H spectrum and an experimental 13C
spectrum *measured in the same solvent* (reviewer follow-up on R0-2). Solvents compare as reported strings;
"Unreported" matching "Unreported" counts as a pair only for the separate robustness set.
"""

from __future__ import annotations

import random
import re
import sys
from collections import Counter
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
DEFAULT = Path(__file__).parent / "data/nmrshiftdb2/nmrshiftdb2withsignals.sd"
ALLOWED = {"C", "H", "N", "O", "S", "F", "Cl", "Br"}
UNREPORTED = "Unreported"
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
    spectra: dict[str, set[int]] = {"1H": set(), "13C": set()}
    for k in tags:
        m = re.fullmatch(r"Spectrum (1H|13C) (\d+)", k)
        if m:
            spectra[m[1]].add(int(m[2]))
    exp = {n: {i for i in s if i not in predicted} for n, s in spectra.items()}
    mol = Chem.MolFromMolBlock(rec[: rec.find("M  END") + 6], sanitize=True, removeHs=False)
    ok, dou = False, None
    if mol is not None:
        elements = {a.GetSymbol() for a in mol.GetAtoms()}
        heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
        ok = elements <= ALLOWED and heavy <= 30 and Descriptors.MolWt(mol) <= 450
        if ok:
            f = rdMolDescriptors.CalcMolFormula(mol)
            counts = {el: int(n or 1) for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", f)}
            x = sum(counts.get(k, 0) for k in ("F", "Cl", "Br"))
            dou = counts.get("C", 0) + 1 + counts.get("N", 0) / 2 - counts.get("H", 0) / 2 - x / 2
    return {
        "id": tags.get("nmrshiftdb2 ID", "?").strip(),
        "exp": exp,
        "solvent": {i: solvents.get(i, UNREPORTED) for i in spectra["1H"] | spectra["13C"]},
        "spec_filters": ok,
        "dou": dou,
        # Disagreements between the blacklist and the positive criterion (should both be 0).
        "exp_without_solvent_key": len({i for s in exp.values() for i in s} - set(solvents)),
        "pred_with_solvent_key": len((predicted & (spectra["1H"] | spectra["13C"])) & set(solvents)),
    }


def same_solvent_pairs(r: dict) -> set[str]:
    """Solvents shared by an experimental 1H and an experimental 13C spectrum of the record."""
    h = {r["solvent"][i] for i in r["exp"]["1H"]}
    c = {r["solvent"][i] for i in r["exp"]["13C"]}
    return h & c


def dou_bin(d: float) -> str:
    return "0-1" if d <= 1 else "2-3" if d <= 3 else "4-5" if d <= 5 else "6-7" if d <= 7 else "8+"


def main(path: Path) -> None:
    c: Counter = Counter()
    named_solvents: Counter = Counter()
    named_dou: Counter = Counter()
    for rec in records(path):
        r = analyse(rec)
        c["records"] += 1
        c["solvent_key_disagreements"] += r["exp_without_solvent_key"] + r["pred_with_solvent_key"]
        eh, ec = bool(r["exp"]["1H"]), bool(r["exp"]["13C"])
        c["exp_1H"] += eh
        c["exp_13C"] += ec
        c["exp_both_any_solvent"] += eh and ec
        ok = r["spec_filters"]
        c["filtered_exp_both_any_solvent"] += ok and eh and ec
        shared = same_solvent_pairs(r)
        named = shared - {UNREPORTED}
        c["exp_both_same_solvent"] += bool(shared)
        c["filtered_same_solvent_any"] += ok and bool(shared)
        c["filtered_same_solvent_named"] += ok and bool(named)
        c["filtered_same_solvent_unreported_only"] += ok and bool(shared) and not named
        if ok and named:
            for s in named:
                named_solvents[s] += 1
            named_dou[dou_bin(r["dou"])] += 1
    print("| metric | records |\n|---|---|")
    for k, v in c.items():
        print(f"| {k} | {v} |")
    print("\nNamed solvents shared by experimental 1H and 13C, filtered records (a record can list several):")
    for k, v in named_solvents.most_common():
        print(f"  {v:6d}  {k}")
    print("\nDoU bins, filtered named-solvent same-solvent records (by record):")
    for k in ("0-1", "2-3", "4-5", "6-7", "8+"):
        print(f"  {named_dou[k]:6d}  DoU {k}")


def spotcheck(path: Path, n: int, seed: int = 7) -> None:
    pool = []
    for rec in records(path):
        r = analyse(rec)
        for nuc in ("1H", "13C"):
            for i in sorted(r["exp"][nuc]):
                pool.append((r["id"], nuc, i, r["solvent"].get(i, UNREPORTED)))
    random.Random(seed).shuffle(pool)
    print(f"{len(pool)} spectra classified experimental; sample (seed {seed}):")
    for rid, nuc, i, s in pool[:n]:
        print(f"nmrshiftdb2 ID {rid}\t{nuc} spectrum {i}\t{s}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--spotcheck"]:
        spotcheck(DEFAULT, int(args[1]))
    else:
        main(Path(args[0]) if args else DEFAULT)
