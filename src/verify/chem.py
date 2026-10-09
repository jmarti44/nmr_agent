"""RDKit helpers for the verifier: parsing, formulas, structural DoU, symmetry classes, 1H environments."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

from src.schemas import parse_formula

RDLogger.DisableLog("rdApp.*")

_PATTERNS: dict[str, Chem.Mol] = {}


def pattern(smarts: str) -> Chem.Mol:
    """Compile (and cache) a SMARTS; raises if it does not compile."""
    p = _PATTERNS.get(smarts)
    if p is None:
        p = Chem.MolFromSmarts(smarts)
        if p is None:
            raise ValueError(f"SMARTS does not compile: {smarts}")
        _PATTERNS[smarts] = p
    return p


def matches(mol: Chem.Mol, smarts: str) -> bool:
    return mol.HasSubstructMatch(pattern(smarts))


@dataclass
class ParsedMol:
    mol: Chem.Mol
    canonical_smiles: str
    inchikey: str
    formula: str
    exact_mass: float
    dou: int
    fragments: int


def parse_smiles(smiles: str) -> tuple[ParsedMol | None, str | None]:
    """Parse and sanitize. Returns (parsed, None) or (None, reason). V01 conditions live here."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "RDKit could not parse or sanitize the SMILES"
    if any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        return None, "wildcard/dummy atom ('*') is not allowed"
    if Chem.GetFormalCharge(mol) != 0:
        return None, f"net formal charge {Chem.GetFormalCharge(mol):+d}; candidates must be neutral"
    return (
        ParsedMol(
            mol=mol,
            canonical_smiles=Chem.MolToSmiles(mol),
            inchikey=Chem.MolToInchiKey(mol),
            formula=rdMolDescriptors.CalcMolFormula(mol),
            exact_mass=float(Descriptors.ExactMolWt(mol)),
            dou=int(round(structural_dou(mol))),
            fragments=len(Chem.GetMolFrags(mol)),
        ),
        None,
    )


def formula_counts(mol: Chem.Mol) -> dict[str, int]:
    return parse_formula(rdMolDescriptors.CalcMolFormula(mol))


def structural_dou(mol: Chem.Mol) -> float:
    """Rings + double bonds + 2 x triple bonds on the Kekulized molecule.

    DEVIATION from SPEC V03 (two): rings are counted as the cyclomatic number, not with CalcNumRings (bridged
    polycyclics), and S/P double bonds to O/N are not counted (below).

    DEVIATION from SPEC V03: S=O / P=O / S=N / P=N double bonds are not counted. The formula-based DoU treats
    S and P like their lower-valent counterparts, so a sulfonamide, sulfoxide or sulfone would otherwise be
    hard-rejected although its formula matches (benzenesulfonamide: structure count 6, formula DoU 4).
    """
    k = Chem.Mol(mol)
    Chem.Kekulize(k, clearAromaticFlags=True)
    # Cyclomatic number (bonds - atoms + fragments). CalcNumRings counts RDKit's symmetrized ring set, which
    # over-counts bridged polycyclics (adamantane: 4 vs the true 3).
    n = float(mol.GetNumBonds() - mol.GetNumAtoms() + len(Chem.GetMolFrags(mol)))
    for b in k.GetBonds():
        bt = b.GetBondType()
        if bt == Chem.BondType.TRIPLE:
            n += 2
        elif bt == Chem.BondType.DOUBLE:
            a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
            hyper = {a1.GetSymbol(), a2.GetSymbol()} & {"S", "P"} and {a1.GetSymbol(), a2.GetSymbol()} & {
                "O",
                "N",
            }
            if not hyper:
                n += 1
    return n


def hill_equal(a: dict[str, int], b: dict[str, int]) -> bool:
    return {k: v for k, v in a.items() if v} == {k: v for k, v in b.items() if v}


def carbon_environment_count(mol: Chem.Mol) -> int:
    """Number of topologically distinct carbons (symmetry classes, ties not broken)."""
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    return len({ranks[a.GetIdx()] for a in mol.GetAtoms() if a.GetSymbol() == "C"})


# --------------------------------------------------------------------------- 1H environments


@dataclass
class HEnv:
    atom_idx: int
    rank: int  # symmetry class of the parent heavy atom
    n_h: int
    keys: tuple[str, ...]  # candidate shift rules that can explain protons in this environment
    exchangeable: bool
    mods: tuple[str, ...]  # distinct direct-substituent rules; with >= 2 the shift is roughly additive


def _is_carbonyl_c(atom: Chem.Atom) -> bool:
    return atom.GetSymbol() == "C" and any(
        b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(atom).GetSymbol() in ("O", "S")
        for b in atom.GetBonds()
    )


def _other_neighbors(atom: Chem.Atom, exclude: Chem.Atom) -> list[Chem.Atom]:
    return [n for n in atom.GetNeighbors() if n.GetIdx() != exclude.GetIdx()]


_BETA_KEY = {3: "h_beta_ch3", 2: "h_beta_ch2", 1: "h_beta_ch"}


def _has_beta_heteroatom(atom: Chem.Atom) -> bool:
    """True if an sp3 carbon neighbour carries Cl/Br/I/O/N (so this carbon is beta to it)."""
    for nb in atom.GetNeighbors():
        if nb.GetSymbol() != "C" or nb.GetIsAromatic() or nb.GetHybridization() != Chem.HybridizationType.SP3:
            continue
        if any(
            x.GetIdx() != atom.GetIdx() and x.GetSymbol() in ("Cl", "Br", "I", "O", "N")
            for x in nb.GetNeighbors()
        ):
            return True
    return False


def _sp3_carbon_keys(atom: Chem.Atom, n_h: int) -> tuple[list[str], tuple[str, ...]]:
    mods: list[str] = []
    in_3ring = atom.IsInRingSize(3)
    for nb in atom.GetNeighbors():
        sym = nb.GetSymbol()
        if sym == "C":
            if nb.GetIsAromatic():
                mods.append("h_benzylic")
            elif _is_carbonyl_c(nb):
                mods.append("h_alpha_carbonyl")
            elif any(b.GetBondType() == Chem.BondType.TRIPLE for b in nb.GetBonds()):
                other = [
                    b.GetOtherAtom(nb) for b in nb.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE
                ][0]
                mods.append("h_unclassified" if other.GetSymbol() == "N" else "h_propargylic")
            elif any(b.GetBondType() == Chem.BondType.DOUBLE for b in nb.GetBonds()):
                other = [
                    b.GetOtherAtom(nb) for b in nb.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
                ][0]
                mods.append("h_allylic" if other.GetSymbol() == "C" else "h_unclassified")
        elif sym == "N":
            if any(_is_carbonyl_c(x) for x in nb.GetNeighbors() if x.GetIdx() != atom.GetIdx()):
                mods.append("h_n_alkyl_amide")
            elif nb.GetFormalCharge() != 0 or any(x.GetSymbol() in ("S", "O") for x in nb.GetNeighbors()):
                mods.append("h_unclassified")  # nitro, sulfonamide, N-oxide
            else:
                mods.append("h_alpha_amine")
        elif sym == "O":
            others = _other_neighbors(nb, atom)
            if any(_is_carbonyl_c(x) for x in others):
                mods += ["h_ester_och"]
            elif any(x.GetIsAromatic() for x in others):
                mods += ["h_aryl_ether"]
            else:
                mods += ["h_alpha_oxygen"]
            if n_h == 3:
                mods.append("h_methoxy")
        elif sym in ("Cl", "Br", "I"):
            mods.append("h_alpha_halide")
        elif sym in ("F", "S", "P", "Si", "B", "Se"):
            mods.append("h_unclassified")
    if in_3ring:
        mods.append("h_unclassified")  # cyclopropyl protons sit below the alkyl range; no verified source
    specific = tuple(sorted({m for m in mods if m not in ("h_unclassified", "h_methoxy")}))
    if mods:
        return sorted(set(mods)), specific
    base = {3: "h_alkyl_ch3", 2: "h_alkyl_ch2"}.get(n_h, "h_alkyl_ch")
    if _has_beta_heteroatom(atom):
        return [base, _BETA_KEY.get(n_h, "h_beta_ch")], ()
    return [base], ()


def _aromatic_keys(atom: Chem.Atom) -> list[str]:
    info = atom.GetOwningMol().GetRingInfo()
    hetero = any(
        any(atom.GetOwningMol().GetAtomWithIdx(i).GetSymbol() != "C" for i in ring)
        for ring in info.AtomRings()
        if atom.GetIdx() in ring
    )
    return ["h_aromatic", "h_heteroaromatic"] if hetero else ["h_aromatic"]


def h_environments(mol: Chem.Mol) -> list[HEnv]:
    """One entry per heavy atom bearing hydrogens, with the shift rules that can explain its protons."""
    molh = Chem.AddHs(mol)
    ranks = list(Chem.CanonicalRankAtoms(molh, breakTies=False))
    envs: list[HEnv] = []
    for atom in mol.GetAtoms():
        n_h = atom.GetTotalNumHs()
        if n_h == 0:
            continue
        sym = atom.GetSymbol()
        keys: list[str]
        exch = False
        mods: tuple[str, ...] = ()
        if sym in ("O", "N", "S"):
            exch = True
            nbs = list(atom.GetNeighbors())
            if sym == "O":
                if any(_is_carbonyl_c(x) for x in nbs):
                    keys = ["h_carboxylic_acid"]
                elif any(x.GetIsAromatic() for x in nbs):
                    keys = ["h_phenol_oh"]
                else:
                    keys = ["h_alcohol_oh"]
            elif sym == "N":
                keys = ["h_amide_nh", "h_amine_nh"]
            else:
                keys = ["h_alcohol_oh"]  # thiol: 1-2 ppm sits inside the wide OH window
        elif sym == "C":
            if atom.GetIsAromatic():
                keys = _aromatic_keys(atom)
            elif atom.GetHybridization() == Chem.HybridizationType.SP3:
                keys, mods = _sp3_carbon_keys(atom, n_h)
            elif atom.GetHybridization() == Chem.HybridizationType.SP2:
                dbl = [
                    b.GetOtherAtom(atom) for b in atom.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE
                ]
                if dbl and dbl[0].GetSymbol() == "O":
                    only_c = all(x.GetSymbol() in ("C", "H") for x in atom.GetNeighbors())
                    keys = ["h_aldehyde"] if only_c else ["h_unclassified"]
                elif dbl and dbl[0].GetSymbol() == "C":
                    keys = ["h_vinyl"]
                else:
                    keys = ["h_unclassified"]
            elif atom.GetHybridization() == Chem.HybridizationType.SP:
                keys = ["h_alkyne_terminal"]
            else:
                keys = ["h_unclassified"]
        else:
            keys = ["h_unclassified"]
        envs.append(HEnv(atom.GetIdx(), ranks[atom.GetIdx()], n_h, tuple(keys), exch, mods))
    return envs
