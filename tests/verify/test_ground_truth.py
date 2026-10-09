"""Invariant (SPEC §11.1): a true structure is never hard-rejected by the verifier."""

import pytest

from src.schemas import Candidate, RuleStatus
from src.verify.verifier import verify_candidate
from tests.verify.cases import CASES


def _verify(case, **kw):
    cand = Candidate(rank=1, smiles=case.smiles, rationale="truth", assignments=[], confidence=0.5)
    return verify_candidate(case.table(), cand, case.context(), use_model_confidence=False, **kw)


@pytest.mark.parametrize("name", list(CASES))
def test_true_structure_is_never_rejected(name):
    v = _verify(CASES[name])
    hard = [r for r in v.rules if r.status == RuleStatus.FAIL and r.severity == "hard"]
    assert not hard, [(r.rule_id, r.message) for r in hard]
    assert v.status != "rejected"


@pytest.mark.parametrize("name", list(CASES))
def test_true_structure_is_accepted(name):
    v = _verify(CASES[name])
    soft = [(r.rule_id, r.message) for r in v.rules if r.status == RuleStatus.FAIL]
    assert v.status == "accepted", soft


def test_terminal_alkyne_range_covers_simple_alkynes():
    """1-hexyne's =C-H is at 1.9 ppm; the 2.3-3.3 range flagged it; experiments span 1.7-3.5."""
    v = _verify(CASES["1_hexyne"])
    assert v.status == "accepted", [(r.rule_id, r.message) for r in v.rules if r.status == RuleStatus.FAIL]
    from src.verify.shift_tables import H_RULES

    r = H_RULES["h_alkyne_terminal"]
    assert (r.lo, r.hi) == (1.7, 3.5) and r.confirm == (2.3, 3.3)


def test_sulfonamide_dou_is_not_a_false_reject():
    """SPEC V03 as written counts S=O: structure 6 vs formula 4 would hard-reject a true sulfonamide."""
    v = _verify(CASES["benzenesulfonamide"])
    v03 = next(r for r in v.rules if r.rule_id.startswith("V03"))
    assert v03.status == RuleStatus.PASS
    assert v.degree_of_unsaturation == 4


def test_aryl_ester_not_flagged_for_missing_alkoxy_proton():
    """Aspirin's ester has no O-CH; V08 must not demand h_ester_och/h_methoxy there."""
    v = _verify(CASES["aspirin"])
    v08 = next(r for r in v.rules if r.rule_id.startswith("V08"))
    assert v08.status == RuleStatus.PASS


def test_tertiary_alcohol_and_aniline_not_demanded_alpha_protons():
    for name in ("tert_butanol", "aniline"):
        v = _verify(CASES[name])
        v08 = next(r for r in v.rules if r.rule_id.startswith("V08"))
        assert v08.status == RuleStatus.PASS, (name, v08.message)


def test_indole_nh_in_dmso_is_explained_via_exchangeable_window():
    v = _verify(CASES["indole_dmso"])
    v09 = next(r for r in v.rules if r.rule_id.startswith("V09"))
    assert v09.status == RuleStatus.PASS


def test_acid_in_d2o_waives_the_acid_proton_requirement():
    v = _verify(CASES["acetic_acid_d2o"])
    assert v.status == "accepted"
