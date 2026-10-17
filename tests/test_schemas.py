import pytest
from pydantic import ValidationError

from src.schemas import (
    AnalysisRequest,
    Candidate,
    FunctionalGroup,
    FunctionalGroupClaim,
    LLMHypothesis,
    Peak,
    PeakListInput,
    SampleContext,
    Solvent,
    degree_of_unsaturation,
    parse_formula,
)


def h(id, shift=7.0, **kw):
    return Peak(id=id, nucleus="1H", shift_ppm=shift, **kw)


@pytest.mark.parametrize(
    "formula,dou", [("C9H8O4", 6), ("C6H6", 4), ("C2H6O", 0), ("C3H7Cl", 0), ("C5H5N", 4), ("C6H7NO2S", 4)]
)
def test_dou(formula, dou):
    assert degree_of_unsaturation(parse_formula(formula)) == dou


@pytest.mark.parametrize("bad", ["", "c9h8o4", "C9H8O4+", "[13C]H4", "H2O", "C9 H8 O4 (x)"])
def test_bad_formulas(bad):
    with pytest.raises(ValueError):
        parse_formula(bad)


def test_formula_whitespace_and_repeats():
    assert parse_formula("C9 H8 O4") == {"C": 9, "H": 8, "O": 4}
    assert parse_formula("CH3CH2OH")["C"] == 2 and parse_formula("CH3CH2OH")["H"] == 6


def test_peak_ranges_per_nucleus():
    with pytest.raises(ValidationError):
        h(1, 17.0)
    with pytest.raises(ValidationError):
        Peak(id=1, nucleus="13C", shift_ppm=245.0)
    assert Peak(id=1, nucleus="13C", shift_ppm=220.0).shift_ppm == 220.0


def test_13c_peaks_cannot_carry_integrals():
    with pytest.raises(ValidationError, match="integrals"):
        Peak(id=1, nucleus="13C", shift_ppm=100.0, integral=1.0)


def test_enums_are_lowercased_but_nucleus_is_exact():
    p = h(1, multiplicity="DD", role="Analyte")
    assert p.multiplicity.value == "dd" and p.role.value == "analyte"
    with pytest.raises(ValidationError):
        Peak(id=1, nucleus="1h", shift_ppm=1.0)


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        h(1, bogus=1)


def test_peaklist_needs_h1_and_unique_ids():
    with pytest.raises(ValidationError, match="1H"):
        PeakListInput(peaks=[Peak(id=1, nucleus="13C", shift_ppm=100.0)])
    with pytest.raises(ValidationError, match="unique"):
        PeakListInput(peaks=[h(1), h(1, 2.0)])


def test_request_cross_field_rules():
    peaks = PeakListInput(peaks=[h(1)])
    assert AnalysisRequest(input_format="peaklist", peaks=peaks).max_candidates == 3
    with pytest.raises(ValidationError, match="requires 'peaks'"):
        AnalysisRequest(input_format="peaklist")
    with pytest.raises(ValidationError, match="only allowed"):
        AnalysisRequest(input_format="jcampdx", peaks=peaks)
    with pytest.raises(ValidationError):
        AnalysisRequest(input_format="peaklist", peaks=peaks, max_candidates=6)


def test_solvent_aliases_and_formula_validation():
    assert SampleContext(solvent="dmso").solvent == Solvent.DMSO_D6
    assert SampleContext(solvent="Chloroform-d").solvent == Solvent.CDCL3
    with pytest.raises(ValidationError):
        SampleContext(molecular_formula="not a formula")
    assert SampleContext(molecular_formula="C9H8O4").formula_counts["O"] == 4


def test_group_claim_lowercases_group():
    c = FunctionalGroupClaim(
        group="Ester", present=True, confidence=0.5, evidence={"peak_ids": [1], "note": "n"}
    )
    assert c.group == FunctionalGroup.ESTER


def _cand(rank):
    return Candidate(rank=rank, smiles="CCO", rationale="r", assignments=[], confidence=0.5)


def test_hypothesis_sorts_candidates_and_bounds_count():
    base = dict(functional_groups=[], reasoning_summary="s")
    hyp = LLMHypothesis(candidates=[_cand(2), _cand(1)], **base)
    assert [c.rank for c in hyp.candidates] == [1, 2]
    with pytest.raises(ValidationError):
        LLMHypothesis(candidates=[], **base)
    with pytest.raises(ValidationError):
        LLMHypothesis(candidates=[_cand(i) for i in range(1, 7)], **base)


def _walk(node, defs, seen, path=()):
    """Return True if a $ref cycle is reachable."""
    if isinstance(node, dict):
        if "$ref" in node:
            name = node["$ref"].split("/")[-1]
            if name in path:
                return True
            return _walk(defs[name], defs, seen, (*path, name))
        return any(_walk(v, defs, seen, path) for v in node.values())
    if isinstance(node, list):
        return any(_walk(v, defs, seen, path) for v in node)
    return False


def test_llm_schema_has_no_recursion_and_few_optionals():
    schema = LLMHypothesis.model_json_schema()
    defs = schema.get("$defs", {})
    assert not _walk(schema, defs, set()), "recursive schemas are not supported by structured outputs"
    optional = 0
    for obj in [schema, *defs.values()]:
        props, required = obj.get("properties", {}), set(obj.get("required", []))
        optional += sum(1 for k in props if k not in required)
    assert optional <= 24


def test_llm_schema_is_accepted_by_the_sdk_transform():
    import anthropic

    out = anthropic.transform_schema(LLMHypothesis)
    assert out["type"] == "object" and out["additionalProperties"] is False


def test_hypothesis_schema_tells_the_model_the_claims_describe_its_first_candidate():
    d = LLMHypothesis.model_json_schema()["properties"]["functional_groups"]["description"]
    assert "FIRST candidate" in d and "rank 1" in d


def test_candidate_ranks_must_be_unique_because_the_claims_describe_the_first():
    base = dict(functional_groups=[], reasoning_summary="s")
    with pytest.raises(ValidationError, match="unique"):
        LLMHypothesis(candidates=[_cand(1), _cand(1)], **base)
    assert [c.rank for c in LLMHypothesis(candidates=[_cand(2), _cand(1)], **base).candidates] == [1, 2]
