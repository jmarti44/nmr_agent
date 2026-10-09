import pytest

from src import prompting
from src.prompting import builder
from src.schemas import AnalysisRequest
from tests.prompting.helpers import aspirin_request


def test_system_blocks_have_the_cacheable_shape_and_hold_the_static_prompt():
    _, p = prompting.build_prompt_for_request(aspirin_request())
    assert len(p.system_blocks) == 1
    block = p.system_blocks[0]
    assert block["type"] == "text" and block["cache_control"] == {"type": "ephemeral"}
    assert block["text"] == prompting.system_prompt()
    assert p.version == prompting.FULL_VERSION and len(p.system_hash) == 64


def test_changing_max_candidates_changes_only_the_user_message():
    _, a = prompting.build_prompt_for_request(aspirin_request(max_candidates=1))
    _, b = prompting.build_prompt_for_request(aspirin_request(max_candidates=5))
    assert a.system_blocks == b.system_blocks and a.system_hash == b.system_hash
    assert a.user_text != b.user_text
    assert "up to 1 candidate" in a.user_text and "up to 5 candidate" in b.user_text


def test_request_peaks_are_normalized_before_rendering():
    table, p = prompting.build_prompt_for_request(aspirin_request())
    roles = {x.shift_ppm: x.role.value for x in table.peaks}
    assert roles[2.5] == "solvent" and roles[39.5] == "solvent"
    assert "[solvent]" in p.user_text and "16:39.5" in p.user_text


def test_token_estimates_are_reported_and_user_part_is_small():
    _, p = prompting.build_prompt_for_request(aspirin_request())
    assert p.total_tokens_est == p.static_tokens_est + p.user_tokens_est
    assert p.user_tokens_est < 500 < p.static_tokens_est


def test_oversized_user_message_is_refused(monkeypatch):
    monkeypatch.setattr(builder, "MAX_USER_TOKENS_ESTIMATE", 50)
    with pytest.raises(prompting.PromptTooLarge):
        prompting.build_prompt_for_request(aspirin_request())


@pytest.mark.parametrize("n", [0, 6, -1])
def test_max_candidates_bounds(n):
    table, _ = prompting.build_prompt_for_request(aspirin_request())
    with pytest.raises(ValueError, match="between 1 and 5"):
        prompting.build_prompt(table, max_candidates=n)


def test_only_peaklist_requests_are_supported_for_now():
    req = AnalysisRequest(input_format="jcampdx")
    with pytest.raises(ValueError, match="peaklist"):
        prompting.build_prompt_for_request(req)


def test_prompts_are_byte_identical_for_identical_requests():
    _, a = prompting.build_prompt_for_request(aspirin_request())
    _, b = prompting.build_prompt_for_request(aspirin_request())
    assert a == b


def test_aspirin_example_is_consistent_with_its_own_answer():
    """The example spectrum must not argue against the structure it describes (13C bounds, formula, H)."""
    from src.schemas import Candidate
    from src.verify.verifier import verify_candidates

    req = aspirin_request()
    table, _ = prompting.build_prompt_for_request(req)
    aspirin = Candidate(rank=1, smiles="CC(=O)Oc1ccccc1C(=O)O", rationale="x", assignments=[], confidence=0.5)
    v = verify_candidates(table, [aspirin], req.context, use_model_confidence=False)[0]
    assert v.status == "accepted", [(r.rule_id, r.message) for r in v.rules if r.status.value != "pass"]
    v06 = next(r for r in v.rules if r.rule_id.startswith("V06"))
    assert v06.status.value == "pass" and v06.evidence == {"observed": 9, "unique_carbons": 9}


def test_analyte_at_a_solvent_position_without_integrals_is_shown_and_the_prompt_allows_using_it():
    """CDCl3 aromatic H at 7.27 without integrals stays tagged, so the wording must keep it usable."""
    req = AnalysisRequest.model_validate(
        {
            "input_format": "peaklist",
            "context": {"solvent": "CDCl3"},
            "peaks": {
                "peaks": [
                    {"id": 1, "nucleus": "1H", "shift_ppm": 7.27, "multiplicity": "s"},
                    {"id": 2, "nucleus": "1H", "shift_ppm": 2.3, "multiplicity": "s"},
                ]
            },
        }
    )
    table, p = prompting.build_prompt_for_request(req)
    assert {x.shift_ppm: x.role.value for x in table.peaks}[7.27] == "solvent"
    row = [line for line in p.user_text.splitlines() if line.startswith("1 ")][0]
    assert "solvent" in row  # the model still sees the peak
    text = p.system_blocks[0]["text"]
    assert "it can hide a real signal" in text and "aromatic proton at 7.27" in text
    assert "Ignore peaks with role reference." in text  # only reference peaks are simply ignored
