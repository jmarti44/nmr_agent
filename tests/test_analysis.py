"""The pipeline against a fake API: normalize, prompt, model, sanitize, verify, repair, response."""

import httpx2
import pytest

from src.analysis import build_repair_report, run_analysis
from src.core.settings import Settings
from src.llm.errors import LLMRefused, LLMUnavailable
from src.prompting.builder import PromptTooLarge
from src.schemas import AnalysisRequest, AnalysisResponse, Candidate
from tests.llm.fake_api import ASPIRIN, ISOMER, FakeAPI, hypothesis, hypothesis_json, message
from tests.prompting.helpers import aspirin_request


def run(api: FakeAPI, request=None, **settings) -> AnalysisResponse:
    return run_analysis(request or aspirin_request(), Settings(**settings), api.client(), "req_test")


def test_accepted_answer_in_one_call():
    api = FakeAPI([message(hypothesis_json(), usage={"input_tokens": 250, "cache_read_input_tokens": 1900})])
    r = run(api)
    assert len(api.requests) == 1 and r.usage.llm_calls == 1
    top = r.candidates[0]
    assert top.status == "accepted" and top.inchikey.startswith("BSYNRYMUTXBXSQ") and top.formula == "C9H8O4"
    assert r.request_id == "req_test" and r.warnings == []
    assert (
        r.versions.model_id == "claude-opus-5-5" and r.versions.prompt_version == "2026.10.3+shifts.2026.10.2"
    )
    assert r.versions.schema_version and r.versions.rdkit and r.versions.nmrglue
    assert [p.id for p in r.peak_table.peaks][:2] == [1, 2] and r.hypothesis_raw.candidates[
        0
    ].smiles == ASPIRIN


def test_usage_and_cost_are_computed_from_the_responses():
    usage = {
        "input_tokens": 250,
        "output_tokens": 1000,
        "cache_read_input_tokens": 1900,
        "cache_creation_input_tokens": 100,
    }
    u = run(FakeAPI([message(hypothesis_json(), usage=usage)])).usage
    assert (u.input_tokens, u.output_tokens) == (250, 1000)
    assert (u.cache_read_input_tokens, u.cache_creation_input_tokens) == (1900, 100)
    assert u.estimated_cost_usd == pytest.approx((250 * 4 + 1000 * 20 + 1900 * 0.2 + 100 * 5) / 1e6, abs=1e-6)


def test_unknown_model_price_is_reported_not_guessed():
    api = FakeAPI([message(hypothesis_json())])
    r = run(api, model="claude-some-future-model")
    assert r.usage.estimated_cost_usd == 0.0 and any("no price known" in w for w in r.warnings)


def test_functional_groups_are_verified_against_the_top_candidate():
    api = FakeAPI([message(hypothesis_json())])
    g = {x.group.value: x for x in run(api).functional_groups}
    assert (
        g["aromatic_ring"].supported_by_shifts.value == "pass"
        and g["aromatic_ring"].present_in_top_candidate is True
    )
    assert g["carboxylic_acid"].present_in_top_candidate is True


def test_timing_breakdown_is_present():
    t = run(FakeAPI([message(hypothesis_json())])).timing_ms
    assert set(t) == {"prompt", "llm", "verify", "total"} and t["total"] >= t["verify"] >= 0


def test_candidates_are_trimmed_to_max_candidates():
    api = FakeAPI([message(hypothesis_json([ASPIRIN, ISOMER]))])
    r = run(api, aspirin_request(max_candidates=1))
    assert len(r.candidates) == 1 and r.candidates[0].status == "accepted"


def test_sanitizer_warnings_reach_the_response():
    groups = [
        {"group": "ester", "present": True, "confidence": 0.5, "evidence": {"peak_ids": [7, 99], "note": "n"}}
    ]
    r = run(FakeAPI([message(hypothesis_json(groups=groups))]))
    assert any("99" in w for w in r.warnings)


# ---- repair round ---------------------------------------------------------------------------------------


def test_repair_round_runs_when_nothing_is_accepted_and_can_rescue_the_answer():
    api = FakeAPI([message(hypothesis_json([ISOMER])), message(hypothesis_json([ASPIRIN]), msg_id="msg_02")])
    r = run(api)
    assert len(api.requests) == 2 and r.usage.llm_calls == 2
    second = api.requests[1]["body"]
    assert (
        second["output_config"]["effort"] == "low"
        and api.requests[0]["body"]["output_config"]["effort"] == "high"
    )
    report = second["messages"][-1]["content"]
    assert "None of your candidates passed verification" in report and ISOMER in report and "V0" in report
    assert r.candidates[0].status == "accepted" and r.candidates[0].candidate.smiles == ASPIRIN
    assert any("repair round was used" in w for w in r.warnings)
    assert r.hypothesis_raw.candidates[0].smiles == ASPIRIN  # the final raw answer


def test_repair_is_skipped_when_a_candidate_is_accepted():
    api = FakeAPI([message(hypothesis_json([ASPIRIN, ISOMER]))])
    run(api)
    assert len(api.requests) == 1


def test_a_failed_repair_keeps_the_first_answer_and_says_so():
    api = FakeAPI([message(hypothesis_json([ISOMER])), message("garbage")])
    r = run(api)
    assert r.candidates[0].candidate.smiles == ISOMER and r.candidates[0].status != "accepted"
    assert any("repair round failed" in w for w in r.warnings) and r.usage.llm_calls == 2


def test_a_refused_repair_is_also_non_fatal():
    refusal = message(
        "", stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber", "explanation": None}
    )
    r = run(FakeAPI([message(hypothesis_json([ISOMER])), refusal]))
    assert r.candidates and any("repair round failed" in w for w in r.warnings)


def test_no_repair_when_the_call_budget_is_used_up():
    api = FakeAPI([message(hypothesis_json([ISOMER]))])
    r = run(api, max_llm_calls=1)
    assert len(api.requests) == 1 and not any("repair" in w for w in r.warnings)


def test_repeating_a_rejected_structure_in_the_repair_round_is_merged_not_duplicated():
    api = FakeAPI([message(hypothesis_json([ISOMER])), message(hypothesis_json([ISOMER]))])
    r = run(api)
    assert len(r.candidates) == 1
    assert any("merged duplicate" in x.message for x in r.candidates[0].rules)


def test_worst_case_is_four_calls():
    bad = message("{}")
    api = FakeAPI([bad, bad, message(hypothesis_json([ISOMER])), message(hypothesis_json([ASPIRIN]))])
    r = run(api)  # two schema corrections, then a valid-but-unaccepted answer, then the repair
    assert r.usage.llm_calls == 4 and len(api.requests) == 4 and r.candidates[0].status == "accepted"


# ---- errors propagate --------------------------------------------------------------------------------------


def test_model_errors_propagate_from_the_first_call():
    with pytest.raises(LLMRefused):
        run(
            FakeAPI(
                [
                    message(
                        "",
                        stop_reason="refusal",
                        stop_details={"type": "refusal", "category": None, "explanation": None},
                    )
                ]
            )
        )
    with pytest.raises(LLMUnavailable):
        run(FakeAPI([httpx2.ConnectError("down")]))


def test_only_peaklist_input_is_supported():
    with pytest.raises(ValueError, match="peaklist"):
        run(FakeAPI([]), AnalysisRequest(input_format="bruker_zip"))


def test_oversized_prompt_is_refused_before_any_model_call(monkeypatch):
    from src.prompting import builder

    monkeypatch.setattr(builder, "MAX_USER_TOKENS_ESTIMATE", 10)
    api = FakeAPI([])
    with pytest.raises(PromptTooLarge):
        run(api)
    assert not api.requests


# ---- repair report ---------------------------------------------------------------------------------------


def test_repair_report_lists_failures_compactly():
    from src.verify.verifier import verify_candidates

    req = aspirin_request()
    from src.processing.normalize import normalize_peaklist

    table = normalize_peaklist(req.peaks, req.context)
    cands = [
        Candidate(rank=1, smiles=ISOMER, rationale="r", assignments=[], confidence=0.5),
        Candidate(rank=2, smiles="C1CC", rationale="r", assignments=[], confidence=0.5),
    ]
    text = build_repair_report(verify_candidates(table, cands, req.context))
    assert text.startswith("None of your candidates passed verification")
    assert (
        "Candidate 1:" in text and "Candidate 2: C1CC -> rejected" in text and "V01_smiles_valid FAIL" in text
    )
    assert text.rstrip().endswith("Return a corrected JSON object only.")
    assert max(len(line) for line in text.splitlines()) < 400


def test_hypothesis_payload_helper_is_a_valid_schema_instance():
    from src.schemas import LLMHypothesis

    assert LLMHypothesis.model_validate(hypothesis()).candidates[0].smiles == ASPIRIN


# ---- claims describe the model's first candidate ----------------------------------------------------------


def _acetyl_absent_and_aromatic():
    """Claims that are true of the isomer and false of aspirin."""
    return [
        {
            "group": "acetyl",
            "present": False,
            "confidence": 0.9,
            "evidence": {"peak_ids": [], "note": "none"},
        },
        {
            "group": "aromatic_ring",
            "present": True,
            "confidence": 0.9,
            "evidence": {"peak_ids": [2, 3], "note": "n"},
        },
    ]


def _rule(v, prefix):
    return next(r for r in v.rules if r.rule_id.startswith(prefix))


def test_a_true_second_candidate_is_not_penalised_by_claims_about_a_wrong_first():
    api = FakeAPI([message(hypothesis_json([ISOMER, ASPIRIN], groups=_acetyl_absent_and_aromatic()))])
    r = run(api)
    by_smiles = {v.candidate.smiles: v for v in r.candidates}
    assert _rule(by_smiles[ASPIRIN], "V10").status.value == "skipped"
    assert by_smiles[ASPIRIN].status == "accepted" and by_smiles[ASPIRIN].verifier_score == 1.0
    assert r.candidates[0].candidate.smiles == ASPIRIN  # the verifier ranks the consistent structure first


def test_functional_groups_are_checked_against_the_candidate_they_describe():
    api = FakeAPI([message(hypothesis_json([ISOMER, ASPIRIN], groups=_acetyl_absent_and_aromatic()))])
    g = {x.group.value: x for x in run(api).functional_groups}
    # 'no acetyl' is true of the isomer (the claimed candidate), although the accepted aspirin has one
    assert g["acetyl"].present_in_top_candidate is False
    assert g["aromatic_ring"].present_in_top_candidate is True


def test_after_a_repair_round_only_the_repair_answers_first_candidate_gets_the_new_claims():
    repair_claims = [
        {
            "group": "aromatic_ring",
            "present": True,
            "confidence": 0.9,
            "evidence": {"peak_ids": [2, 3], "note": "n"},
        }
    ]
    api = FakeAPI([
        message(hypothesis_json([ISOMER], groups=_acetyl_absent_and_aromatic())),
        message(hypothesis_json([ASPIRIN], groups=repair_claims)),
    ])  # fmt: skip
    r = run(api)
    by_smiles = {v.candidate.smiles: v for v in r.candidates}
    assert (
        _rule(by_smiles[ISOMER], "V10").status.value == "skipped"
    )  # first round: not described by the new claims
    assert "claims describe candidate 2" in _rule(by_smiles[ISOMER], "V10").message
    assert _rule(by_smiles[ASPIRIN], "V10").status.value == "pass"
    g = {x.group.value: x for x in r.functional_groups}
    assert list(g) == ["aromatic_ring"] and g["aromatic_ring"].present_in_top_candidate is True


def test_claims_about_a_repair_candidate_that_was_merged_into_an_earlier_one_still_resolve():
    claims = [
        {
            "group": "aromatic_ring",
            "present": True,
            "confidence": 0.9,
            "evidence": {"peak_ids": [2, 3], "note": "n"},
        }
    ]
    api = FakeAPI([message(hypothesis_json([ISOMER])), message(hypothesis_json([ISOMER], groups=claims))])
    r = run(api)
    assert len(r.candidates) == 1  # the repeated structure was merged
    assert r.functional_groups[0].present_in_top_candidate is True  # found through the merge, not lost


def test_response_schema_says_which_candidate_the_claims_and_raw_answer_refer_to():
    s = AnalysisResponse.model_json_schema()
    assert "rank 1" in s["properties"]["functional_groups"]["description"]
    assert "final answer" in s["properties"]["hypothesis_raw"]["description"]
    assert "repair" in s["properties"]["hypothesis_raw"]["description"]
    d = s["$defs"]["VerifiedFunctionalGroup"]["properties"]["present_in_top_candidate"]["description"]
    assert "rank 1" in d and "Not the top-ranked" in d


# ---- the claimed candidate is found from the verifier's marker, not from message text ----------------------


def test_a_repair_candidate_that_repeats_a_first_round_structure_is_checked_against_its_own_claims():
    wrong_claim = [
        {
            "group": "methoxy",
            "present": False,
            "confidence": 0.9,
            "evidence": {"peak_ids": [], "note": "none"},
        }
    ]  # the isomer has a methoxy group, so this contradicts the structure it describes
    api = FakeAPI(
        [message(hypothesis_json([ISOMER])), message(hypothesis_json([ISOMER], groups=wrong_claim))]
    )
    r = run(api)
    assert len(r.candidates) == 1 and r.candidates[0].claims_checked
    v10 = _rule(r.candidates[0], "V10")
    assert v10.status.value == "fail" and "claimed absent but present" in v10.message
    assert (
        r.functional_groups[0].group.value == "methoxy"
        and r.functional_groups[0].present_in_top_candidate is True
    )


def test_claimed_candidate_lookup_does_not_depend_on_the_wording_of_the_merge_message():
    import inspect

    from src.analysis import _claimed_candidate
    from src.processing.normalize import normalize_peaklist
    from src.schemas import Candidate, FunctionalGroup, FunctionalGroupClaim
    from src.verify.verifier import verify_candidates

    req = aspirin_request()
    table = normalize_peaklist(req.peaks, req.context)
    ranked = ((1, ISOMER), (2, ASPIRIN), (3, ISOMER))
    cands = [Candidate(rank=r, smiles=s, rationale="r", assignments=[], confidence=0.5) for r, s in ranked]
    claim = FunctionalGroupClaim.model_validate(
        {"group": FunctionalGroup.AROMATIC_RING, "present": True, "confidence": 0.9,
         "evidence": {"peak_ids": [2, 3], "note": "n"}}
    )  # fmt: skip
    verified = verify_candidates(table, cands, req.context, claims=[claim], claims_rank=3)
    for v in verified:
        for rule in v.rules:
            if rule.rule_id.startswith("V11"):
                rule.message = "completely reworded"
    found = _claimed_candidate(verified)
    assert (
        found is not None and found.candidate.rank == 1
    )  # rank 3 merged into rank 1, which carries the claims
    source = inspect.getsource(_claimed_candidate)
    assert "re." not in source and "regex" not in source
    assert _claimed_candidate([v for v in verified if not v.claims_checked]) is None
