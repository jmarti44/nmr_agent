"""Replays of real model replies recorded on 2026-10-19 (bench/record_llm_fixtures.py; ids removed).

The hand-written replies elsewhere prove the code handles the documented shapes; these prove it handles what
the API really returned: the extra fields, a real thinking block and real usage.
"""

import json
from pathlib import Path

import pytest

from src.analysis import run_analysis
from src.core.settings import Settings
from src.llm.orchestrator import Conversation
from src.prompting import build_prompt_for_request
from tests.llm.fake_api import FakeAPI
from tests.prompting.helpers import aspirin_request

DIR = Path(__file__).parents[1] / "fixtures/llm"


def load(name: str) -> dict:
    return json.loads((DIR / f"{name}.json").read_text())


def test_fixtures_carry_no_identifiers_or_keys():
    for f in DIR.glob("*.json"):
        text = f.read_text()
        assert "sk-ant" not in text and "req_" not in text
        assert json.loads(text)["id"] == "msg_FIXTURE"


def test_a_real_reply_runs_through_the_whole_pipeline():
    api = FakeAPI([load("aspirin_elucidate")])
    r = run_analysis(aspirin_request(), Settings(), api.client())
    assert r.candidates[0].candidate.smiles == "CC(=O)Oc1ccccc1C(=O)O"
    assert r.candidates[0].status == "accepted" and r.warnings == []
    assert r.usage.llm_calls == 1 and r.usage.cache_read_input_tokens == 5326
    assert r.usage.estimated_cost_usd == pytest.approx(0.0554, abs=0.001)


def test_the_real_thinking_block_is_echoed_unchanged_into_the_follow_up_turn():
    first, followup = load("aspirin_echo_first"), load("aspirin_echo_followup")
    api = FakeAPI([first, followup])
    c = Conversation(build_prompt_for_request(aspirin_request())[1], Settings(), api.client())
    c.elucidate()
    c.repair("None of your candidates passed verification. Return a corrected JSON object only.")
    sent = api.requests[1]["body"]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert [b["type"] for b in sent[1]["content"]] == ["thinking", "text"]
    thinking = next(b for b in first["content"] if b["type"] == "thinking")
    assert sent[1]["content"][0]["signature"] == thinking["signature"]
    assert c.calls[1].output_tokens == followup["usage"]["output_tokens"]
