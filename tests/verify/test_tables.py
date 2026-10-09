"""Shift tables, SMARTS table and solvent tagging (SPEC §11.1)."""

import pytest
from rdkit import Chem

from src.schemas import FunctionalGroup, Nucleus, Peak, PeakRole, Solvent
from src.verify.shift_tables import ALL_RULES, C_RULES, H_RULES, SOURCES
from src.verify.smarts import SPECS
from src.verify.solvent_tables import tag_peaks


@pytest.mark.parametrize("rule", ALL_RULES.values(), ids=lambda r: r.key)
def test_every_range_is_sourced_and_ordered(rule):
    assert rule.sources and all(s in SOURCES for s in rule.sources)
    assert rule.lo < rule.hi
    if rule.confirm:
        lo, hi = rule.confirm
        assert rule.lo <= lo < hi <= rule.hi, (
            "the confirmation range must sit inside the explain (union) range"
        )


def test_nuclei_are_consistent_with_key_prefix():
    assert all(r.nucleus == Nucleus.H1 for r in H_RULES.values())
    assert all(r.nucleus == Nucleus.C13 for r in C_RULES.values())
    assert all(k.startswith("h_") for k in H_RULES) and all(k.startswith("c_") for k in C_RULES)


def test_only_thinly_sourced_13c_ranges_are_soft_only():
    assert {k for k, r in C_RULES.items() if r.soft_only} == {
        "c_alpha_nitrogen",
        "c_alpha_halide",
        "c_alkyne",
    }


def test_every_group_has_a_spec_with_valid_keys_and_compiling_smarts():
    assert set(SPECS) == set(FunctionalGroup)
    for g, spec in SPECS.items():
        for s in spec.smarts:
            assert Chem.MolFromSmarts(s) is not None, (g, s)
        for ev in (*spec.h, *spec.c):
            assert all(k in ALL_RULES for k in ev.keys), (g, ev.keys)
            assert ev.applies_if is None or Chem.MolFromSmarts(ev.applies_if) is not None


# group -> (SMILES that must match, SMILES that must not)
EXAMPLES = {
    "aldehyde": ("O=Cc1ccccc1", "CC(=O)C"),
    "ketone": ("CC(=O)c1ccccc1", "O=Cc1ccccc1"),
    "carboxylic_acid": ("CC(=O)Oc1ccccc1C(=O)O", "CC(=O)OC"),
    "ester": ("CCOC(C)=O", "CC(=O)O"),
    "amide": ("CNC(C)=O", "CCN"),
    "alcohol": ("CCO", "c1ccccc1O"),
    "phenol": ("Oc1ccccc1", "CCO"),
    "ether": ("CCOCC", "CCO"),
    "methoxy": ("COc1ccccc1", "CCO"),
    "amine_primary": ("CCN", "CNC"),
    "amine_secondary": ("CNC", "CCN"),
    "amine_tertiary": ("CN(C)C", "CNC"),
    "aromatic_ring": ("c1ccccc1", "C1CCCCC1"),
    "heteroaromatic_ring": ("c1ccncc1", "c1ccccc1"),
    "alkene": ("C=CC", "CCC"),
    "alkyne_terminal": ("C#CCCCC", "CC#CC"),
    "alkyne_internal": ("CC#CC", "C#CC"),
    "nitrile": ("N#Cc1ccccc1", "CCN"),
    "nitro": ("O=[N+]([O-])c1ccccc1", "CCN"),
    "alkyl_halide": ("CCCl", "c1ccccc1Cl"),
    "aryl_halide": ("Clc1ccccc1", "CCCl"),
    "methyl": ("CCC", "c1ccccc1"),
    "acetyl": ("CC(=O)c1ccccc1", "CCC(=O)CC"),
    "sulfonamide": ("NS(=O)(=O)c1ccccc1", "CCN"),
    "thioether": ("CSC", "CS"),
}


def _match(smiles, group):
    mol = Chem.MolFromSmiles(smiles)
    return any(mol.HasSubstructMatch(Chem.MolFromSmarts(s)) for s in SPECS[FunctionalGroup(group)].smarts)


def test_examples_cover_every_group():
    assert set(EXAMPLES) == {g.value for g in FunctionalGroup}


@pytest.mark.parametrize("group", EXAMPLES)
def test_smarts_positive_and_negative_examples(group):
    pos, neg = EXAMPLES[group]
    assert _match(pos, group), f"{group} should match {pos}"
    assert not _match(neg, group), f"{group} should not match {neg}"


# ---- tagging --------------------------------------------------------------------------------------------


def hp(shift, **kw):
    return Peak(id=1, nucleus="1H", shift_ppm=shift, **kw)


def role(shift, solvent, nucleus="1H"):
    p = Peak(id=1, nucleus=nucleus, shift_ppm=shift)
    return tag_peaks([p], solvent)[0].role


def test_residual_chcl3_and_water_in_cdcl3():
    assert role(7.26, Solvent.CDCL3) == PeakRole.SOLVENT
    assert role(1.56, Solvent.CDCL3) == PeakRole.WATER
    assert role(77.0, Solvent.CDCL3, "13C") == PeakRole.SOLVENT


def test_acetone_impurity_is_tagged_but_not_removed():
    peaks = [hp(2.17, integral=3.0, multiplicity="s")]
    out = tag_peaks(peaks, Solvent.CDCL3)
    assert len(out) == 1 and out[0].role == PeakRole.IMPURITY_SUSPECT
    assert out[0].integral == 3.0  # the acetyl-sized singlet stays usable by the verifier


def test_solvents_and_reference():
    assert role(2.50, Solvent.DMSO_D6) == PeakRole.SOLVENT
    assert role(3.33, Solvent.DMSO_D6) == PeakRole.WATER
    assert role(4.80, Solvent.D2O) == PeakRole.WATER
    assert role(0.0, Solvent.CDCL3) == PeakRole.REFERENCE
    assert role(206.7, Solvent.ACETONE_D6, "13C") == PeakRole.SOLVENT


def test_impurities_only_tagged_in_cdcl3_and_analyte_peaks_untouched():
    assert role(2.17, Solvent.DMSO_D6) == PeakRole.ANALYTE
    assert role(2.17, None) == PeakRole.ANALYTE
    assert role(5.5, Solvent.CDCL3) == PeakRole.ANALYTE


def test_existing_roles_are_preserved_and_input_not_mutated():
    p = hp(7.26, role="analyte")
    q = Peak(id=2, nucleus="1H", shift_ppm=1.0, role="reference")
    out = tag_peaks([p, q], Solvent.CDCL3)
    assert out[1].role == PeakRole.REFERENCE
    assert p.role == PeakRole.ANALYTE and out[0].role == PeakRole.SOLVENT
