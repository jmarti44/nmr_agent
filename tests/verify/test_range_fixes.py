"""Rule fixes found on the Phase 5 benchmark (aldehyde, C=N, missing N-H, solvent-tagged peak, V09 ranges)."""

import pytest

from src.schemas import Candidate, Nucleus, Peak, PeakRole, PeakTable, RuleStatus, SampleContext, Solvent
from src.verify.chem import parse_smiles
from src.verify.rules import Ctx, v08_structure_demands_peaks, v09_peaks_explained


def table(solvent, h=(), c=(), roles=None):
    roles = roles or {}
    peaks = [Peak(id=i, nucleus=Nucleus.H1, shift_ppm=s, integral=1.0) for i, s in enumerate(h, start=1)]
    peaks += [Peak(id=100 + i, nucleus=Nucleus.C13, shift_ppm=s) for i, s in enumerate(c, start=1)]
    for p in peaks:
        p.role = roles.get(p.shift_ppm, PeakRole.ANALYTE)
    return PeakTable(peaks=peaks, solvent=Solvent(solvent))


def ctx(smiles, t, formula=None):
    parsed, why = parse_smiles(smiles)
    assert parsed is not None, why
    cand = Candidate(rank=1, smiles=smiles, rationale="t", assignments=[], confidence=0.5)
    return Ctx(
        table=t,
        context=SampleContext(solvent=t.solvent.value, molecular_formula=formula),
        parsed=parsed,
        candidate=cand,
    )


def test_ortho_alkoxy_benzaldehyde_at_10_44_is_not_rejected():
    t = table(
        "CDCl3", [10.44, 7.32, 7.10, 6.93, 3.89, 3.81], [189.0, 156.9, 153.5, 125.0, 123.7, 113.2, 56.1]
    )
    r = v08_structure_demands_peaks(ctx("COc1ccc(OC)c(C=O)c1", t, "C9H10O3"))
    assert r.status == RuleStatus.PASS, r.message


def test_aldehyde_proton_above_the_widened_range_is_still_missing():
    t = table("CDCl3", [11.3, 7.32, 7.10, 6.93, 3.89, 3.81], [189.0, 156.9, 153.5, 56.1])
    r = v08_structure_demands_peaks(ctx("COc1ccc(OC)c(C=O)c1", t, "C9H10O3"))
    assert r.status == RuleStatus.FAIL and r.severity == "hard"


def test_conjugated_aldehyde_carbon_at_185_is_accepted():
    t = table(
        "DMSO-d6", [12.1, 9.95, 8.30, 8.15, 7.5, 7.2], [185.4, 138.7, 137.4, 124.6, 123.9, 118.5, 112.9]
    )
    r = v08_structure_demands_peaks(ctx("O=Cc1c[nH]c2ccccc12", t, "C9H7NO"))
    assert "aldehyde: no 13C" not in r.message and r.status != RuleStatus.FAIL, r.message


def test_aldehyde_carbon_below_the_widened_range_is_still_missing():
    t = table("CDCl3", [9.95, 7.8, 7.5], [175.0, 137.0, 130.0, 129.0])
    r = v08_structure_demands_peaks(ctx("O=Cc1ccccc1", t, "C7H6O"))
    assert r.status == RuleStatus.FAIL and "aldehyde: no 13C" in r.message


def test_thiadiazole_ring_carbons_above_160_count_as_heteroaromatic_evidence():
    t = table("D2O", [2.95, 1.30], [173.2, 168.4, 21.3, 12.8])
    r = v08_structure_demands_peaks(ctx("CCc1nnc(N)s1", t, "C4H7N3S"))
    assert "heteroaromatic_ring: no 13C" not in r.message and r.status != RuleStatus.FAIL, r.message


def test_a_benzene_ring_does_not_accept_a_carbon_only_at_170():
    t = table("CDCl3", [7.3, 2.3], [170.0, 21.0])
    r = v08_structure_demands_peaks(ctx("Cc1ccccc1", t, "C7H8"))
    assert "aromatic_ring: no 13C" in r.message


def test_missing_amide_nh_only_warns():
    t = table("CDCl3", [2.80, 2.00], [171.0, 26.3, 22.9])
    r = v08_structure_demands_peaks(ctx("CC(=O)NC", t, "C3H7NO"))
    assert r.status == RuleStatus.WARN and "amide" in r.message


def test_missing_amide_carbonyl_carbon_still_fails():
    t = table("CDCl3", [2.80, 2.00], [26.3, 22.9])
    r = v08_structure_demands_peaks(ctx("CC(=O)NC", t, "C3H7NO"))
    assert r.status == RuleStatus.FAIL and "c_ester_amide" in r.message


def test_acetyl_methyl_on_the_dmso_line_is_weak_evidence_not_a_failure():
    t = table(
        "DMSO-d6",
        [7.55, 7.45, 6.87, 3.85, 2.50],
        [196.4, 152.0, 147.8, 129.0, 123.5, 115.4, 111.6, 55.7, 26.3],
        roles={2.50: PeakRole.SOLVENT},
    )
    r = v08_structure_demands_peaks(ctx("CC(=O)c1ccc(O)c(OC)c1", t, "C9H10O3"))
    assert r.status == RuleStatus.WARN and "solvent-tagged" in r.message, r.message


def test_acetyl_with_no_peak_anywhere_near_still_fails():
    t = table("DMSO-d6", [7.55, 7.45, 6.87, 3.85], [196.4, 152.0, 147.8, 129.0, 123.5, 115.4, 111.6, 55.7])
    r = v08_structure_demands_peaks(ctx("CC(=O)c1ccc(O)c(OC)c1", t, "C9H10O3"))
    assert r.status == RuleStatus.FAIL


def test_ring_allylic_ch2_at_2_67_is_explained():
    t = table("CDCl3", [5.85, 3.97, 3.82, 3.73, 2.67])
    assert v09_peaks_explained(ctx("COC(=O)C1=CCCN1C(=O)OC", t)).status == RuleStatus.PASS


def test_cyclobutane_ch2_at_2_38_and_1_85_is_explained():
    t = table("DMSO-d6", [12.6, 2.38, 1.85])
    assert v09_peaks_explained(ctx("O=C(O)C1(C(=O)O)CCC1", t)).status == RuleStatus.PASS


@pytest.mark.parametrize("smiles", ["CC1CCC1", "CCCCC"])
def test_an_alkane_does_not_explain_a_vinyl_region_peak(smiles):
    t = table("CDCl3", [4.8, 1.2])
    assert v09_peaks_explained(ctx(smiles, t)).status == RuleStatus.FAIL
