"""Experimental NMRShiftDB2 shift distributions used to source verifier ranges (the NSDB2 source).

Usage: uv run python bench/nmrshiftdb2_shift_distributions.py
Only experimental spectra are used (spectrum index present in the per-spectrum `Solvent` tag, which agrees
exactly with the Program/NMRProgram blacklist; see nmrshiftdb2_census.py). Atom indices in the signal lists
are taken as indices into the RDKit molecule built from the record's molblock, the same convention as the
census; records whose indices fall outside the molecule are skipped per signal.
"""

from __future__ import annotations

import collections
import re
from pathlib import Path

from rdkit import Chem, RDLogger

from bench.nmrshiftdb2_census import DEFAULT, INDEXED, TAG, records

RDLogger.DisableLog("rdApp.*")

P_ALKYNE_H = Chem.MolFromSmarts("[CX2;H1]#[CX2]")
P_KETONE = Chem.MolFromSmarts("[#6][CX3](=O)[#6]")
P_VINYL = Chem.MolFromSmarts("[CX3;H1,H2;!$(C=O)]=[CX3;!$(C=O)]")  # non-aromatic C=C-H
P_ARH = Chem.MolFromSmarts(
    "[cH;!$(c~[!#6;!#1;R])]"
)  # carbocyclic aromatic C-H (no ring heteroatom neighbour)
P_ENONE_BETA = Chem.MolFromSmarts(
    "[CX3;H1,H2]=[CX3][CX3]=O"
)  # beta carbon of an enone / unsaturated acid or ester


def idx(v: str) -> dict[int, str]:
    return {int(i): s.strip() for i, s in INDEXED.findall(v.strip())}


def _is_carbonyl(c: Chem.Atom) -> bool:
    return any(b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(c).GetSymbol() == "O" for b in c.GetBonds())


def _plain_amine_n(n: Chem.Atom) -> bool:
    """Neutral N with no O/S/aromatic neighbours and no carbonyl on any carbon neighbour (not amide/nitro)."""
    if n.GetSymbol() != "N" or n.GetFormalCharge() != 0:
        return False
    return not any(
        x.GetSymbol() in ("O", "S") or x.GetIsAromatic() or (x.GetSymbol() == "C" and _is_carbonyl(x))
        for x in n.GetNeighbors()
    )


def alpha_category(atom: Chem.Atom, where: str) -> str | None:
    """Pure alpha-carbonyl, pure alpha-amine, or alpha-amino-acid CH category for an sp3 carbon, else None."""
    if (
        atom.GetSymbol() != "C"
        or atom.GetIsAromatic()
        or atom.GetHybridization() != Chem.HybridizationType.SP3
    ):
        return None
    nbs = list(atom.GetNeighbors())
    het = [n for n in nbs if n.GetSymbol() != "C"]
    carbons = [n for n in nbs if n.GetSymbol() == "C"]
    carbonyl = [n for n in carbons if _is_carbonyl(n)]
    plain = [n for n in carbons if n not in carbonyl and not n.GetIsAromatic()]
    plain = [n for n in plain if n.GetHybridization() == Chem.HybridizationType.SP3]
    kind = {3: "CH3", 2: "CH2"}.get(atom.GetTotalNumHs(), "CH")
    if not het and len(carbonyl) == 1 and len(plain) == len(carbons) - 1:
        return f"alpha-carbonyl only, {kind}"
    if len(het) == 1 and _plain_amine_n(het[0]):
        if not carbonyl and len(plain) == len(carbons):
            return f"alpha-amine only, {kind}, {where}"
        acid = len(carbonyl) == 1 and sum(1 for x in carbonyl[0].GetNeighbors() if x.GetSymbol() == "O") == 2
        if acid and kind == "CH":
            return f"alpha-amino acid CH, {where}"
    return None


def beta_class(mol: Chem.Mol, atom: Chem.Atom) -> str | None:
    """Beta class of an sp3 C whose neighbours are all sp3 carbons: halogen/oxygen/nitrogen/nitro/none."""
    if (
        atom.GetSymbol() != "C"
        or atom.GetIsAromatic()
        or atom.GetHybridization() != Chem.HybridizationType.SP3
    ):
        return None
    nbs = list(atom.GetNeighbors())
    if any(n.GetSymbol() != "C" or n.GetHybridization() != Chem.HybridizationType.SP3 for n in nbs):
        return None  # a direct heteroatom, sp2 or sp neighbour: alpha effects, not a pure beta environment
    if atom.IsInRingSize(3):
        return None
    found = set()
    for n in nbs:
        for x in n.GetNeighbors():
            if x.GetIdx() == atom.GetIdx():
                continue
            s = x.GetSymbol()
            if s in ("Cl", "Br", "I"):
                found.add("halogen")
            elif s == "O":
                found.add("oxygen")
            elif s == "N":
                found.add("nitro" if x.GetFormalCharge() > 0 else "nitrogen")
    if len(found) == 1:
        return found.pop()
    return "none" if not found else None


def q(xs: list[float]) -> str:
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return "n=0"
    return (
        f"n={n} min={xs[0]:.2f} p5={xs[int(0.05 * n)]:.2f} median={xs[n // 2]:.2f} "
        f"p95={xs[max(0, int(0.95 * n) - 1)]:.2f} max={xs[-1]:.2f}"
    )


def main(path: Path = DEFAULT) -> None:
    alkyne: list[float] = []
    ket_conj: list[float] = []
    ket_sat: list[float] = []
    beta: dict[tuple[str, int], list[float]] = collections.defaultdict(list)
    vinyl: list[float] = []
    arh: list[float] = []
    alpha: dict[str, list[float]] = collections.defaultdict(list)
    enone_beta: list[float] = []
    for rec in records(path):
        end = rec.find("M  END")
        mol = Chem.MolFromMolBlock(rec[: end + 6])
        if mol is None:
            continue
        tags = {k: v for k, v in TAG.findall(rec[end:])}
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
                for match in mol.GetSubstructMatches(P_ALKYNE_H):
                    alkyne += by_atom.get(match[0], [])
                solvent = idx(tags.get("Solvent", "")).get(int(m[2]), "")
                where = "D2O" if "Deuterium" in solvent else "other solvents"
                for mt in mol.GetSubstructMatches(P_ARH):
                    arh += by_atom.get(mt[0], [])
                for a, shifts in by_atom.items():
                    cat = alpha_category(mol.GetAtomWithIdx(a), where)
                    if cat:
                        alpha[cat] += shifts
                eb = {mt[0] for mt in mol.GetSubstructMatches(P_ENONE_BETA)}
                for a in eb:
                    enone_beta += by_atom.get(a, [])
                for mt in mol.GetSubstructMatches(P_VINYL):
                    if mt[0] not in eb:
                        vinyl += by_atom.get(mt[0], [])
                for a, shifts in by_atom.items():
                    cls = beta_class(mol, mol.GetAtomWithIdx(a))
                    if cls:
                        beta[(cls, mol.GetAtomWithIdx(a).GetTotalNumHs())] += shifts
            else:
                for match in mol.GetSubstructMatches(P_KETONE):
                    c = mol.GetAtomWithIdx(match[1])
                    conj = any(
                        n.GetIsAromatic()
                        or any(
                            b.GetBondTypeAsDouble() == 2 and b.GetOtherAtom(n).GetSymbol() == "C"
                            for b in n.GetBonds()
                        )
                        for n in c.GetNeighbors()
                        if n.GetSymbol() == "C"
                    )
                    (ket_conj if conj else ket_sat).extend(by_atom.get(match[1], []))
    print("# NSDB2 distributions (experimental spectra only)")
    print("terminal alkyne C-H (1H):", q(alkyne))
    print("ketone C=O (13C), conjugated:", q(ket_conj))
    print("ketone C=O (13C), saturated:", q(ket_sat))
    print("aromatic C-H, carbocyclic (1H):", q(arh))
    for k in sorted(alpha):
        print(f"{k}:", q(alpha[k]))
    print("vinyl C-H (1H), not beta to a carbonyl:", q(vinyl))
    print("vinyl C-H (1H), beta carbon of a C=C-C=O:", q(enone_beta))
    print("\nsp3 C-H protons beta to a heteroatom (all neighbours sp3 carbons, not in a 3-ring):")
    names = {3: "CH3", 2: "CH2", 1: "CH"}
    for cls in ("none", "halogen", "oxygen", "nitrogen", "nitro"):
        for n_h in (3, 2, 1):
            print(f"  beta-{cls:8s} {names[n_h]:3s}: {q(beta[(cls, n_h)])}")


if __name__ == "__main__":
    main()
