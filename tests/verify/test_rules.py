"""Each rule V01-V11 has a pass and a fail fixture (SPEC §11.1)."""

import pytest

from src.schemas import (
    Candidate,
    EvidenceRef,
    FunctionalGroup,
    FunctionalGroupClaim,
    Nucleus,
    Peak,
    PeakAssignment,
    PeakRole,
    RuleStatus,
    SampleContext,
)
from src.verify.chem import parse_smiles
from src.verify.rules import (
    Ctx,
    v02_formula_match,
    v03_dou_consistency,
    v04_exact_mass,
    v05_proton_accounting,
    v06_c13_environment_count,
    v07_h_environment_count,
    v08_structure_demands_peaks,
    v09_peaks_explained,
    v10_claimed_groups,
)
from src.verify.verifier import verify_candidate, verify_candidates
from tests.verify.cases import CASES

ASPIRIN = CASES["aspirin"]


def ctx(case=ASPIRIN, smiles=None, table=None, context=None, claims=None, hyp_dou=None, assignments=None):
    parsed, why = parse_smiles(smiles or case.smiles)
    assert parsed is not None, why
    cand = Candidate(
        rank=1, smiles=parsed.canonical_smiles, rationale="t", assignments=assignments or [], confidence=0.5
    )
    return Ctx(
        table=table or case.table(),
        context=context or case.context(),
        parsed=parsed,
        candidate=cand,
        claims=claims,
        hypothesis_dou=hyp_dou,
    )


def cand(smiles, rank=1, conf=0.5):
    return Candidate(rank=rank, smiles=smiles, rationale="t", assignments=[], confidence=conf)


# ---- V01 ------------------------------------------------------------------------------------------------


def test_v01_pass_and_fail():
    ok = verify_candidate(ASPIRIN.table(), cand(ASPIRIN.smiles), ASPIRIN.context())
    assert ok.rules[0].rule_id.startswith("V01") and ok.rules[0].status == RuleStatus.PASS
    for bad in ("C1CC", "not a smiles", "*CC", "C[N+](C)(C)C"):
        v = verify_candidate(ASPIRIN.table(), cand(bad), ASPIRIN.context())
        assert v.status == "rejected" and v.rules[0].status == RuleStatus.FAIL, bad
        assert v.canonical_smiles is None and v.verifier_score == 0.0


def test_v01_multiple_fragments_warn():
    v = verify_candidate(ASPIRIN.table(), cand("CC(=O)Oc1ccccc1C(=O)O.O"), ASPIRIN.context())
    assert v.rules[0].status == RuleStatus.WARN


# ---- V02 / V03 / V04 ------------------------------------------------------------------------------------


def test_v02():
    assert v02_formula_match(ctx()).status == RuleStatus.PASS
    assert v02_formula_match(ctx(smiles="CCOC(C)=O")).status == RuleStatus.FAIL  # wrong formula
    no_formula = SampleContext()
    assert v02_formula_match(ctx(context=no_formula)).status == RuleStatus.SKIPPED


def test_v02_failure_is_hard_and_stops_the_ledger():
    v = verify_candidate(ASPIRIN.table(), cand("CCOC(C)=O"), ASPIRIN.context())
    assert v.status == "rejected" and v.verifier_score == 0.0
    assert v.rules[-1].rule_id.startswith("V02")  # nothing after the hard failure


def test_v03_with_formula_and_with_model_dou():
    assert v03_dou_consistency(ctx()).status == RuleStatus.PASS
    no_formula = SampleContext()
    r = v03_dou_consistency(ctx(context=no_formula, hyp_dou=3))  # aspirin is 6
    assert r.status == RuleStatus.FAIL and r.severity == "soft"
    assert v03_dou_consistency(ctx(context=no_formula, hyp_dou=6)).status == RuleStatus.PASS
    assert v03_dou_consistency(ctx(context=no_formula)).status == RuleStatus.SKIPPED


@pytest.mark.parametrize(
    "smiles,dou",
    [("CC(=O)Oc1ccccc1C(=O)O", 6), ("c1ccccc1", 4), ("CCO", 0), ("CCCCl", 0), ("c1ccncc1", 4),
     ("CS(C)=O", 0), ("CS(C)(=O)=O", 0), ("O=[N+]([O-])c1ccccc1", 5), ("CC#N", 2), ("C=C=C", 2),
     # bridged polycyclics: CalcNumRings over-counts these
     ("CC1CCC2CC1C2(C)C", 2), ("C1C2CC3CC1CC(C2)C3", 3), ("C12C3C4C1C5C2C3C45", 5),
     ("CC1(C)C2CC3OC3(C)C1C2", 3), ("C=C1C(O)CC2CC1C2(C)C", 3), ("CC1(C)C2CCC1(C)C(=O)C2", 3),
     ("C1CCC2CCCCC2C1", 2)],
)  # fmt: skip
def test_structural_dou_matches_formula_dou(smiles, dou):
    from src.schemas import degree_of_unsaturation
    from src.verify.chem import formula_counts

    p, _ = parse_smiles(smiles)
    assert p.dou == dou == degree_of_unsaturation(formula_counts(p.mol))


def test_v04():
    assert v04_exact_mass(ctx()).status == RuleStatus.SKIPPED
    ok = SampleContext(molecular_formula="C9H8O4", observed_exact_mass=180.0423)
    assert v04_exact_mass(ctx(context=ok)).status == RuleStatus.PASS
    bad = SampleContext(molecular_formula="C9H8O4", observed_exact_mass=180.0600)
    assert v04_exact_mass(ctx(context=bad)).status == RuleStatus.FAIL


# ---- V05 ------------------------------------------------------------------------------------------------


def test_v05_pass_fail_skipped():
    assert v05_proton_accounting(ctx()).status == RuleStatus.PASS
    t = ASPIRIN.table()
    t.peaks[5].integral = 6.0  # methyl now claims 6 H: sum 9 > 8 + tolerance
    r = v05_proton_accounting(ctx(table=t))
    assert r.status == RuleStatus.FAIL and r.severity == "hard"
    no_int = ASPIRIN.table()
    for p in no_int.peaks:
        p.integral = None
    assert v05_proton_accounting(ctx(table=no_int)).status == RuleStatus.SKIPPED


def test_v05_unobserved_exchangeable_proton_is_allowed():
    t = ASPIRIN.table()
    t.peaks = [p for p in t.peaks if p.id != 1]  # carboxylic OH not observed
    assert v05_proton_accounting(ctx(table=t)).status == RuleStatus.PASS


def test_v05_soft_without_formula_and_assignment_mismatch_warns():
    r = v05_proton_accounting(ctx(context=SampleContext()))
    assert r.severity == "soft"
    assigns = [PeakAssignment(peak_id=6, proton_count=3, environment="CH3")]
    assert v05_proton_accounting(ctx(assignments=assigns)).status == RuleStatus.WARN


def test_v05_counts_impurity_tagged_peak_as_possible_analyte():
    t = CASES["ethyl_acetate"].table()
    t.peaks[1].role = PeakRole.IMPURITY_SUSPECT  # the 2.05 acetyl singlet, tagged as ethyl acetate impurity
    c = CASES["ethyl_acetate"]
    assert v05_proton_accounting(ctx(c, table=t)).status == RuleStatus.PASS


# ---- V06 / V07 ------------------------------------------------------------------------------------------


def test_v06():
    assert v06_c13_environment_count(ctx()).status == RuleStatus.PASS
    few = ASPIRIN.table()
    few.peaks = [p for p in few.peaks if p.nucleus == Nucleus.H1 or p.id <= 8]  # 2 carbons observed
    assert v06_c13_environment_count(ctx(table=few)).status == RuleStatus.FAIL
    no_c = ASPIRIN.table(with_c=False)
    assert v06_c13_environment_count(ctx(table=no_c)).status == RuleStatus.SKIPPED


def test_v06_symmetry_is_respected():
    c = CASES["benzonitrile"]  # 5 distinct carbons out of 7
    assert v06_c13_environment_count(ctx(c)).status == RuleStatus.PASS


def test_v07():
    assert v07_h_environment_count(ctx()).status == RuleStatus.PASS
    t = ASPIRIN.table()
    base = t.peaks[5]
    for i in range(7):  # 5 + 7 non-exchangeable signals vs 5 environments
        t.peaks.append(base.model_copy(update={"id": 100 + i, "shift_ppm": 1.0 + i * 0.1}))
    assert v07_h_environment_count(ctx(table=t)).status == RuleStatus.FAIL


# ---- V08 ------------------------------------------------------------------------------------------------


def test_v08_pass():
    assert v08_structure_demands_peaks(ctx()).status == RuleStatus.PASS


def test_v08_missing_aldehyde_proton_is_a_hard_failure_in_cdcl3_with_formula():
    c = CASES["benzaldehyde"]
    t = c.table()
    t.peaks = [p for p in t.peaks if p.shift_ppm != 10.00]
    r = v08_structure_demands_peaks(ctx(c, table=t))
    assert r.status == RuleStatus.FAIL and r.severity == "hard"
    # Without a formula the same omission is only soft.
    r2 = v08_structure_demands_peaks(ctx(c, table=t, context=SampleContext(solvent="CDCl3")))
    assert r2.status == RuleStatus.FAIL and r2.severity == "soft"


def test_v08_missing_carboxylic_acid_proton_only_warns_and_is_waived_after_d2o_exchange():
    """A listed COOH proton is rare in real peak lists: its absence must not reject or block acceptance."""
    t = ASPIRIN.table()
    t.peaks = [p for p in t.peaks if p.id != 1]
    r = v08_structure_demands_peaks(ctx(table=t))
    assert r.status == RuleStatus.WARN and r.severity == "soft" and "carboxylic_acid" in r.message
    waived = ASPIRIN.context(d2o_exchange_performed=True)
    assert v08_structure_demands_peaks(ctx(table=t, context=waived)).status == RuleStatus.PASS


def test_true_acid_without_its_proton_is_accepted_with_a_warning_in_cdcl3_and_dmso():
    benzoic = CASES["benzoic_acid_no_cooh"]
    for solvent in ("CDCl3", "DMSO-d6"):
        case = type(benzoic)(**{**benzoic.__dict__, "solvent": solvent})
        v = verify_candidate(case.table(), cand(case.smiles), case.context(), use_model_confidence=False)
        v08 = next(r for r in v.rules if r.rule_id.startswith("V08"))
        assert v.status == "accepted" and v08.status == RuleStatus.WARN, (solvent, v.status, v08.message)
        assert v.verifier_score == pytest.approx(0.85)  # one WARN, nothing else


def test_acid_proton_and_carbonyl_carbon_both_missing_is_a_fail():
    """The warning is only for the proton: without the 13C carbonyl either, the acid is unsupported."""
    benzoic = CASES["benzoic_acid_no_cooh"]
    t = benzoic.table()
    t.peaks = [p for p in t.peaks if p.shift_ppm != 172.6]
    v = verify_candidate(t, cand(benzoic.smiles), benzoic.context(), use_model_confidence=False)
    v08 = next(r for r in v.rules if r.rule_id.startswith("V08"))
    assert v08.status == RuleStatus.FAIL and v.status == "flagged"
    assert not any(r.severity == "hard" and r.status == RuleStatus.FAIL for r in v.rules)


def test_v08_missing_13c_carbonyl_is_soft_fail():
    c = CASES["acetophenone"]
    t = c.table()
    t.peaks = [p for p in t.peaks if p.shift_ppm != 198.1]
    r = v08_structure_demands_peaks(ctx(c, table=t))
    assert r.status == RuleStatus.FAIL and r.severity == "soft" and "ketone" in r.message


def test_v08_soft_only_13c_evidence_only_warns():
    """c_alpha_nitrogen rests on thin sourcing (SPEC §8.3): a miss must not count as a soft FAIL."""
    from tests.verify.cases import Case

    ethylamine = Case(
        "ethylamine", "CCN", "C2H7N", "CDCl3",
        [(2.68, "q", 2, False), (1.10, "t", 3, False), (1.20, "br", 2, True)],
        [18.4],  # the C-N carbon (about 36.7) is not listed
    )  # fmt: skip
    r = v08_structure_demands_peaks(ctx(ethylamine))
    assert r.status == RuleStatus.WARN and "c_alpha_nitrogen" in r.message
    ethylamine.c.append(36.7)
    assert v08_structure_demands_peaks(ctx(ethylamine)).status == RuleStatus.PASS


def test_unreferenced_axis_widens_tolerances():
    from tests.verify.cases import Case

    pentane = Case(
        "pentane", "CCCCC", "C5H12", "CDCl3",
        [(1.80, "t", 6, False), (1.30, "m", 6, False)],
        [],
    )  # fmt: skip
    # 1.80 is 0.1 above the CH2 limit (1.7) and beyond the 0.05 slack; an unreferenced axis allows 0.1 more.
    assert v09_peaks_explained(ctx(pentane)).status == RuleStatus.FAIL
    assert (
        v09_peaks_explained(ctx(pentane, table=pentane.table(referencing="uncorrected"))).status
        == RuleStatus.PASS
    )


# ---- V09 ------------------------------------------------------------------------------------------------


def test_v09_pass_and_fail():
    assert v09_peaks_explained(ctx(CASES["ethyl_acetate"])).status == RuleStatus.PASS
    t = CASES["ethyl_acetate"].table()
    t.peaks.append(Peak(id=50, nucleus=Nucleus.H1, shift_ppm=9.5, integral=0.0))
    r = v09_peaks_explained(ctx(CASES["ethyl_acetate"], table=t))
    assert r.status == RuleStatus.FAIL and r.evidence["unexplained_peak_ids"] == [50]


def test_v09_exchangeable_flag_requires_an_exchangeable_proton_in_the_structure():
    t = CASES["ethyl_acetate"].table()
    t.peaks[
        0
    ].exchangeable_hint = True  # the OCH2 quartet labelled exchangeable: nothing in ethyl acetate exchanges
    assert v09_peaks_explained(ctx(CASES["ethyl_acetate"], table=t)).status == RuleStatus.FAIL


def test_v09_warns_when_claims_cite_solvent_peaks():
    c = CASES["ethyl_acetate"]
    t = c.table()
    t.peaks[1].role = PeakRole.SOLVENT
    claim = FunctionalGroupClaim(
        group=FunctionalGroup.ESTER,
        present=True,
        confidence=0.9,
        evidence=EvidenceRef(peak_ids=[2], note="x"),
    )
    r = v09_peaks_explained(ctx(c, table=t, claims=[claim]))
    assert r.status in (RuleStatus.WARN, RuleStatus.FAIL)
    assert r.evidence["tainted_evidence_peak_ids"] == [2]


# ---- V10 ------------------------------------------------------------------------------------------------


def _claim(group, present, ids):
    return FunctionalGroupClaim(
        group=group, present=present, confidence=0.9, evidence=EvidenceRef(peak_ids=ids, note="x")
    )


def test_v10_skipped_without_claims_and_pass_with_good_claims():
    assert v10_claimed_groups(ctx()).status == RuleStatus.SKIPPED
    good = [_claim(FunctionalGroup.CARBOXYLIC_ACID, True, [1]), _claim(FunctionalGroup.ALDEHYDE, False, [])]
    assert v10_claimed_groups(ctx(claims=good)).status == RuleStatus.PASS


def test_v10_fails_for_each_kind_of_inconsistency():
    absent = [_claim(FunctionalGroup.ALDEHYDE, True, [2])]  # claimed present, not in aspirin
    assert v10_claimed_groups(ctx(claims=absent)).status == RuleStatus.FAIL
    present = [_claim(FunctionalGroup.ESTER, False, [])]  # claimed absent, aspirin has an ester
    assert v10_claimed_groups(ctx(claims=present)).status == RuleStatus.FAIL
    wrong_peak = [_claim(FunctionalGroup.CARBOXYLIC_ACID, True, [6])]  # methyl singlet cited for COOH
    assert v10_claimed_groups(ctx(claims=wrong_peak)).status == RuleStatus.FAIL


# ---- V11 + scoring --------------------------------------------------------------------------------------


def test_v11_merges_duplicate_connectivity_and_ranks_by_final_confidence():
    out = verify_candidates(
        ASPIRIN.table(),
        [cand("OC(=O)c1ccccc1OC(C)=O", rank=2, conf=0.9), cand(ASPIRIN.smiles, rank=1, conf=0.8)],
        ASPIRIN.context(),
    )
    assert len(out) == 1 and out[0].candidate.rank == 1
    v11 = out[0].rules[-1]
    assert v11.rule_id.startswith("V11") and "2" in v11.message


def test_scoring_matches_the_spec():
    out = verify_candidates(ASPIRIN.table(), [cand(ASPIRIN.smiles, conf=0.9)], ASPIRIN.context())[0]
    assert out.verifier_score == 1.0 and out.status == "accepted"
    assert out.final_confidence == pytest.approx(0.85 * 1.0 + 0.15 * 0.9, abs=1e-4)  # SPEC example: 0.985
    # One soft FAIL multiplies by 0.6; a critical soft fail (V08) means flagged, not accepted.
    wrong = verify_candidates(ASPIRIN.table(), [cand("COC(=O)c1ccccc1C(=O)O", conf=0.5)], ASPIRIN.context())[
        0
    ]
    assert wrong.status == "flagged" and wrong.verifier_score < 0.6
    rejected = verify_candidates(ASPIRIN.table(), [cand("CCO", conf=1.0)], ASPIRIN.context())[0]
    assert rejected.status == "rejected" and rejected.final_confidence == pytest.approx(0.15)


# ---- functional-group verification ----------------------------------------------------------------------


def test_verify_functional_groups_uses_confirmation_ranges_and_top_candidate():
    from src.verify.verifier import verify_functional_groups

    table = ASPIRIN.table()
    top = verify_candidates(table, [cand(ASPIRIN.smiles)], ASPIRIN.context())[0]
    claims = [
        _claim(
            FunctionalGroup.CARBOXYLIC_ACID, True, [1]
        ),  # 12.95: inside the 10-14 union, outside the 11-12 confirmation range
        _claim(FunctionalGroup.AROMATIC_RING, True, [2, 3]),  # 7.95 and 7.64: inside 6.8-8.0
        _claim(FunctionalGroup.ALDEHYDE, True, [6]),  # methyl singlet cited for an aldehyde
        _claim(FunctionalGroup.NITRO, True, [2]),  # no shift rule exists for nitro
    ]
    out = {v.group: v for v in verify_functional_groups(table, claims, top, ASPIRIN.context())}
    assert (
        out[FunctionalGroup.CARBOXYLIC_ACID].supported_by_shifts == RuleStatus.WARN
    )  # in union, outside intersection
    assert out[FunctionalGroup.AROMATIC_RING].supported_by_shifts == RuleStatus.PASS
    assert out[FunctionalGroup.ALDEHYDE].supported_by_shifts == RuleStatus.FAIL
    assert out[FunctionalGroup.NITRO].supported_by_shifts == RuleStatus.SKIPPED
    assert out[FunctionalGroup.AROMATIC_RING].present_in_top_candidate is True
    assert out[FunctionalGroup.ALDEHYDE].present_in_top_candidate is False


def test_verify_functional_groups_without_an_accepted_top_candidate():
    from src.verify.verifier import verify_functional_groups

    rejected = verify_candidates(ASPIRIN.table(), [cand("CCO")], ASPIRIN.context())[0]
    out = verify_functional_groups(
        ASPIRIN.table(), [_claim(FunctionalGroup.AROMATIC_RING, True, [2])], rejected, ASPIRIN.context()
    )
    assert out[0].present_in_top_candidate is None


# ---- catch-all slots, additive substituents, V07 overlap, status wording, empty InChIKey ----


def _spectrum(shifts, solvent="CDCl3"):
    from src.schemas import PeakTable, Solvent

    peaks = [Peak(id=i + 1, nucleus=Nucleus.H1, shift_ppm=s, integral=1.0) for i, s in enumerate(shifts)]
    return PeakTable(peaks=peaks, solvent=Solvent(solvent))


def test_unclassified_environments_cannot_explain_every_peak():
    """One S or F atom must not explain a spectrum with peaks at 11.5, 10.0, 7.8 and 7.5 ppm."""
    t = _spectrum([11.5, 10.0, 7.8, 7.5])
    for smiles in ("CCCCCCC", "CSCCCCC", "FCCCCCC"):
        r = v09_peaks_explained(ctx(smiles=smiles, table=t, context=SampleContext()))
        assert r.status == RuleStatus.FAIL, smiles
    # CSCCCCC has two unclassified environments (S-CH3 and S-CH2), so exactly two peaks can use the catch-all
    r = v09_peaks_explained(ctx(smiles="CSCCCCC", table=t, context=SampleContext()))
    assert len(r.evidence["unexplained_peak_ids"]) == 2 and len(r.evidence["catch_all_peak_ids"]) == 2
    r = v09_peaks_explained(ctx(smiles="FCCCCCC", table=t, context=SampleContext()))
    assert len(r.evidence["unexplained_peak_ids"]) == 3 and len(r.evidence["catch_all_peak_ids"]) == 1


def test_true_sulfur_and_fluorine_spectra_still_pass_v09():
    for name in ("thioanisole", "1_fluorohexane", "1_2_epoxybutane"):
        c = CASES[name]
        assert v09_peaks_explained(ctx(c)).status == RuleStatus.PASS, name


@pytest.mark.parametrize(
    "name", ["ethyl_bromide", "1_bromobutane", "benzyl_alcohol", "benzylamine", "1_2_epoxybutane"]
)
def test_beta_and_multi_substituent_structures_pass_v08_and_v09(name):
    c = CASES[name]
    for rule in (v08_structure_demands_peaks, v09_peaks_explained):
        r = rule(ctx(c))
        assert r.status == RuleStatus.PASS, (name, r.rule_id, r.message)


def test_additive_range_is_derived_from_the_single_substituent_ranges():
    from src.verify.chem import h_environments
    from src.verify.rules import env_hull

    p, _ = parse_smiles("NCc1ccccc1")
    env = next(e for e in h_environments(p.mol) if e.n_h == 2 and not e.exchangeable and e.mods)
    assert set(env.mods) == {"h_alpha_amine", "h_benzylic"}
    lo, hi = env_hull(env)
    from src.verify.shift_tables import H_RULES

    amine, benzylic = H_RULES["h_alpha_amine"], H_RULES["h_benzylic"]
    assert lo == min(amine.lo, benzylic.lo)
    assert hi == pytest.approx(1.7 + (amine.hi - 1.7) + (benzylic.hi - 1.7))
    single, _ = parse_smiles("CCN")
    assert all(env_hull(e) is None for e in h_environments(single.mol))  # one substituent: its own range only


def test_v07_fewer_signals_than_environments_is_not_penalised():
    for name in ("nonadecane", "1_octanol", "2_methylnaphthalene"):
        r = v07_h_environment_count(ctx(CASES[name]))
        assert r.status == RuleStatus.PASS, (name, r.message)
    r = v07_h_environment_count(ctx(CASES["nonadecane"]))
    assert "fewer signals" in r.message


def test_v07_clearly_too_many_signals_still_fails():
    t = ASPIRIN.table()
    base = t.peaks[5]
    for i in range(4):
        t.peaks.append(base.model_copy(update={"id": 100 + i, "shift_ppm": 1.0 + i * 0.1}))
    assert v07_h_environment_count(ctx(table=t)).status == RuleStatus.FAIL


def test_chalcone_carbonyl_near_190_satisfies_the_ketone_evidence():
    r = v08_structure_demands_peaks(ctx(CASES["chalcone"]))
    assert r.status == RuleStatus.PASS, r.message
    from src.verify.shift_tables import C_RULES

    k = C_RULES["c_ketone"]
    assert (k.lo, k.hi, k.confirm) == (175, 220, (195, 220))


def test_empty_inchikey_never_merges_candidates(monkeypatch):
    from rdkit import Chem

    monkeypatch.setattr(Chem, "MolToInchiKey", lambda mol: "")
    out = verify_candidates(
        ASPIRIN.table(),
        [cand("CCO", rank=1), cand("CCCO", rank=2), cand(ASPIRIN.smiles, rank=3)],
        ASPIRIN.context(),
    )
    assert len(out) == 3 and all(v.inchikey is None for v in out)


def test_accepted_status_is_documented_as_consistency_not_correctness():
    from src.schemas import VerifiedCandidate

    desc = VerifiedCandidate.model_json_schema()["properties"]["status"]["description"]
    assert "consistent with the supplied data" in desc and "NOT mean" in desc
