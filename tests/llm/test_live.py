"""Live checks against the real API. Excluded by default (`-m live`); skipped without a key.

These spend real credits (about a cent or two per call) and have NOT been run: the machine this was written on
has no ANTHROPIC_API_KEY. Run with: ANTHROPIC_API_KEY=... uv run pytest -m live
"""

import os

import pytest

from src.analysis import run_analysis
from src.core.settings import Settings
from tests.prompting.helpers import aspirin_request

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY is not set"),
]


@pytest.fixture(scope="module")
def client():
    import anthropic

    return anthropic.Anthropic(max_retries=3)


def test_aspirin_yields_an_accepted_candidate_with_the_right_inchikey(client):
    r = run_analysis(aspirin_request(), Settings(), client)
    top = r.candidates[0]
    assert top.status == "accepted" and top.inchikey.startswith("BSYNRYMUTXBXSQ")


def test_the_static_prompt_is_cached_on_the_second_call(client):
    run_analysis(aspirin_request(), Settings(), client)
    second = run_analysis(aspirin_request(), Settings(), client)
    assert second.usage.cache_read_input_tokens > 0, "the system prompt was not served from the cache"


def test_a_structured_output_request_is_accepted_by_the_api(client):
    """Catches a schema the API's grammar rejects (SPEC §12: 'schema too complex')."""
    r = run_analysis(aspirin_request(max_candidates=1), Settings(), client)
    assert r.hypothesis_raw.candidates
