from src.llm.validate import sanitize_hypothesis
from src.processing.normalize import normalize_peaklist
from src.schemas import LLMHypothesis, PeakAssignment
from tests.llm.fake_api import hypothesis
from tests.prompting.helpers import aspirin_request


def table():
    req = aspirin_request()
    return normalize_peaklist(req.peaks, req.context)


def hyp(**over) -> LLMHypothesis:
    return LLMHypothesis.model_validate(hypothesis(**over))


def claim(group, ids):
    return {"group": group, "present": True, "confidence": 0.5, "evidence": {"peak_ids": ids, "note": "n"}}


def test_a_clean_hypothesis_passes_through_without_warnings():
    out, warnings = sanitize_hypothesis(hyp(), table())
    assert warnings == [] and out == hyp()


def test_evidence_ids_missing_from_the_table_are_dropped_and_reported():
    h = hyp(groups=[claim("ester", [6, 99])])
    out, warnings = sanitize_hypothesis(h, table())
    assert out.functional_groups[0].evidence.peak_ids == [6]
    assert any("99" in w and "dropped" in w for w in warnings)


def test_evidence_citing_a_solvent_peak_is_reported_but_kept():
    out, warnings = sanitize_hypothesis(
        hyp(groups=[claim("ester", [6])]), table()
    )  # peak 6 is DMSO-d6 residual
    assert out.functional_groups[0].evidence.peak_ids == [6] or out.functional_groups[0].evidence.peak_ids
    t = table()
    solvent_id = next(p.id for p in t.peaks if p.role.value == "solvent" and p.nucleus.value == "1H")
    out, warnings = sanitize_hypothesis(hyp(groups=[claim("ester", [solvent_id])]), t)
    assert out.functional_groups[0].evidence.peak_ids == [solvent_id]
    assert any("tagged solvent" in w for w in warnings)


def test_confidences_are_clamped_with_a_warning():
    h = hyp(groups=[{**claim("ester", [7]), "confidence": 1.4}])
    h.candidates[0].confidence = -0.2
    out, warnings = sanitize_hypothesis(h, table())
    assert out.functional_groups[0].confidence == 1.0 and out.candidates[0].confidence == 0.0
    assert sum("clamped" in w for w in warnings) == 2


def test_unparsable_inferred_formula_is_discarded():
    out, warnings = sanitize_hypothesis(hyp(formula_inferred="C9H8O4+"), table())
    assert out.formula_inferred is None and any("formula_inferred" in w for w in warnings)
    ok, w2 = sanitize_hypothesis(hyp(formula_inferred="C9H8O4"), table())
    assert ok.formula_inferred == "C9H8O4" and w2 == []


def test_unknown_unexplained_and_assignment_ids_are_reported():
    h = hyp(unexplained_peak_ids=[2, 77])
    h.candidates[0].assignments = [PeakAssignment(peak_id=88, proton_count=1, environment="x")]
    out, warnings = sanitize_hypothesis(h, table())
    assert out.unexplained_peak_ids == [2]
    assert any("88" in w for w in warnings) and any("77" in w for w in warnings)


def test_the_input_hypothesis_is_not_mutated():
    h = hyp(groups=[claim("ester", [7, 99])])
    sanitize_hypothesis(h, table())
    assert h.functional_groups[0].evidence.peak_ids == [7, 99]
