"""Functional-group SMARTS and the shift evidence each group requires (SPEC §8.4).

Strings only (RDKit compiles them in rules.py), so importing this module stays light.

`Evidence.applies_if` is a SMARTS that must match the candidate before the evidence is demanded. The SPEC
table lists one evidence rule per group, but several groups only produce that signal in some structures:
an aryl ester has no O-CH or O-CH3 proton, a tertiary alcohol has no alpha proton, an aniline has no
alpha C-H. Demanding the signal there would flag true structures (DEVIATION, tested).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.schemas import FunctionalGroup as G


@dataclass(frozen=True)
class Evidence:
    keys: tuple[str, ...]  # any one of these shift rules satisfies the requirement
    applies_if: str | None = None
    hard_in: tuple[str, ...] = ()  # solvents where a missing signal rejects the candidate (needs a formula)
    # Only the aldehyde proton is hard. The COOH proton is listed for only 22% of CDCl3 and 29% of DMSO-d6 acid
    # spectra in NMRShiftDB2, so its absence is not evidence: it warns (weak) and never blocks acceptance.
    unless_exchanged: bool = False  # waived when D2O exchange was performed or the solvent exchanges
    # A missing signal only warns (the candidate can still be accepted). Used for the COOH proton, which databases
    # and chemists usually leave out.
    weak: bool = False


@dataclass(frozen=True)
class GroupSpec:
    group: G
    smarts: tuple[str, ...]  # any pattern matching counts as "present"
    h: tuple[Evidence, ...] = field(default_factory=tuple)
    c: tuple[Evidence, ...] = field(default_factory=tuple)


def _e(*keys: str, **kw) -> Evidence:
    return Evidence(tuple(keys), **kw)


SPECS: dict[G, GroupSpec] = {
    s.group: s
    for s in [
        GroupSpec(G.ALDEHYDE, ("[CX3H1](=O)[#6]",), h=(_e("h_aldehyde", hard_in=("CDCl3", "DMSO-d6")),), c=(_e("c_aldehyde"),)),
        GroupSpec(G.KETONE, ("[#6][CX3](=O)[#6]",), c=(_e("c_ketone"),)),
        GroupSpec(G.CARBOXYLIC_ACID, ("[CX3](=O)[OX2H1]",), h=(_e("h_carboxylic_acid", unless_exchanged=True, weak=True),), c=(_e("c_ester_amide"),)),
        GroupSpec(G.ESTER, ("[CX3](=O)[OX2H0][#6]",), h=(_e("h_ester_och", "h_methoxy", applies_if="[CX3](=O)[OX2][CX4;!H0]"),), c=(_e("c_ester_amide"),)),
        GroupSpec(G.AMIDE, ("[CX3](=O)[NX3]",), h=(_e("h_amide_nh", applies_if="[CX3](=O)[NX3;!H0]", unless_exchanged=True),), c=(_e("c_ester_amide"),)),
        GroupSpec(G.ALCOHOL, ("[CX4][OX2H]",), h=(_e("h_alpha_oxygen", applies_if="[CX4;!H0][OX2H]"),), c=(_e("c_alpha_oxygen", applies_if="[CX4][OX2H]"),)),
        GroupSpec(G.PHENOL, ("c[OX2H]",), h=(_e("h_aromatic"),), c=(_e("c_aromatic_quat"),)),
        GroupSpec(G.ETHER, ("[CX4][OX2][CX4,c]",), h=(_e("h_alpha_oxygen", "h_aryl_ether", applies_if="[CX4;!H0;!r3][OX2][#6]"),), c=(_e("c_alpha_oxygen", applies_if="[CX4][OX2][#6]"),)),
        GroupSpec(G.METHOXY, ("[CH3][OX2]",), h=(_e("h_methoxy"),), c=(_e("c_alpha_oxygen"),)),
        GroupSpec(G.AMINE_PRIMARY, ("[NX3;H2;!$(NC=O)][#6]",), h=(_e("h_alpha_amine", applies_if="[CX4;!H0;!r3][NX3;H2;!$(NC=O)]"),), c=(_e("c_alpha_nitrogen", applies_if="[CX4][NX3;H2;!$(NC=O)]"),)),
        GroupSpec(G.AMINE_SECONDARY, ("[NX3;H1;!$(NC=O)]([#6])[#6]",), h=(_e("h_alpha_amine", applies_if="[CX4;!H0;!r3][NX3;H1;!$(NC=O)]"),), c=(_e("c_alpha_nitrogen", applies_if="[CX4][NX3;H1;!$(NC=O)]"),)),
        GroupSpec(G.AMINE_TERTIARY, ("[NX3;H0;!$(NC=O)]([#6])([#6])[#6]",), h=(_e("h_alpha_amine", applies_if="[CX4;!H0;!r3][NX3;H0;!$(NC=O)]"),), c=(_e("c_alpha_nitrogen", applies_if="[CX4][NX3;H0;!$(NC=O)]"),)),
        GroupSpec(G.AROMATIC_RING, ("a1aaaaa1",), h=(_e("h_aromatic", "h_heteroaromatic", applies_if="[cH]"),), c=(_e("c_sp2_any"),)),
        GroupSpec(G.HETEROAROMATIC_RING, ("[n,o,s]1aaaa1", "[n]1aaaaa1"), h=(_e("h_heteroaromatic", applies_if="[cH,nH]"),), c=(_e("c_sp2_any"),)),
        GroupSpec(G.ALKENE, ("[CX3]=[CX3]",), h=(_e("h_vinyl", applies_if="[CX3;!H0]=[CX3]"),), c=(_e("c_sp2_any"),)),
        GroupSpec(G.ALKYNE_TERMINAL, ("[CX2]#[CX2H1]",), h=(_e("h_alkyne_terminal"),), c=(_e("c_alkyne"),)),
        GroupSpec(G.ALKYNE_INTERNAL, ("[CX2;H0]#[CX2;H0]",), c=(_e("c_alkyne"),)),
        GroupSpec(G.NITRILE, ("[CX2]#[NX1]",), c=(_e("c_nitrile"),)),
        GroupSpec(G.NITRO, ("[NX3](=O)=O", "[NX3+](=O)[O-]")),
        GroupSpec(G.ALKYL_HALIDE, ("[CX4][F,Cl,Br,I]",), h=(_e("h_alpha_halide", applies_if="[CX4;!H0][Cl,Br,I]"),), c=(_e("c_alpha_halide", applies_if="[CX4][Cl,Br]"),)),
        GroupSpec(G.ARYL_HALIDE, ("c[F,Cl,Br,I]",), h=(_e("h_aromatic", "h_heteroaromatic", applies_if="[cH]"),)),
        GroupSpec(G.METHYL, ("[CH3]",), h=(_e("h_alkyl_ch3", "h_benzylic", "h_alpha_carbonyl", "h_methoxy", "h_alpha_amine", "h_allylic", "h_alpha_halide"),)),
        GroupSpec(G.ACETYL, ("[CH3]C(=O)[#6,#7,#8]",), h=(_e("h_alpha_carbonyl"),), c=(_e("c_carbonyl_any"),)),
        GroupSpec(G.SULFONAMIDE, ("S(=O)(=O)N",)),
        GroupSpec(G.THIOETHER, ("[#6][SX2][#6]",)),
    ]
}  # fmt: skip

assert set(SPECS) == set(G), "every FunctionalGroup needs a spec"
