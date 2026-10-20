"""Record real model replies as test fixtures and run the echo check (needs ANTHROPIC_API_KEY; spends ~$0.2).

    set -a; . ./.env; set +a; uv run python bench/record_llm_fixtures.py

1. A first call on the aspirin example. Run it after more than 5 minutes idle and its usage shows the
   cache-creation (cold) cost.
2. The echo check: a second call whose history is the user turn, the first reply's full content (thinking
   blocks included) and a short follow-up turn, as in a correction or repair. The API must not return a 400.
Replies are written to tests/fixtures/llm/ with ids removed; request bodies and headers are not saved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anthropic

from src.analysis import run_analysis
from src.core.settings import Settings
from src.llm.orchestrator import Conversation
from src.prompting import prepare_request
from src.schemas import AnalysisRequest
from tests.prompting.helpers import ASPIRIN_DMSO

OUT = Path("tests/fixtures/llm")


def scrub(reply: dict) -> dict:
    out = json.loads(json.dumps(reply))
    out["id"] = "msg_FIXTURE"
    return out


def main() -> None:
    captured: list[tuple[dict, dict]] = []

    def hook(response) -> None:
        response.read()
        if response.status_code == 200:
            captured.append((json.loads(response.request.content), response.json()))

    http = anthropic.DefaultHttpxClient(event_hooks={"response": [hook]})
    settings = Settings()
    client = anthropic.Anthropic(max_retries=0, timeout=settings.llm_call_timeout_s, http_client=http)
    OUT.mkdir(parents=True, exist_ok=True)

    r1 = run_analysis(AnalysisRequest.model_validate(ASPIRIN_DMSO), settings, client)
    print("first call usage:", r1.usage.model_dump(), "llm_ms", r1.timing_ms["llm"])
    (OUT / "aspirin_elucidate.json").write_text(json.dumps(scrub(captured[0][1]), indent=2) + "\n")

    # The echo check: elucidate, then send the full reply (thinking blocks included) back with a follow-up.
    n = len(captured)
    _, prompt, _ = prepare_request(AnalysisRequest.model_validate(ASPIRIN_DMSO))
    convo = Conversation(prompt, settings, client)
    convo.elucidate()
    convo.repair(
        "None of your candidates passed verification. The checker's report:\n"
        "Candidate 1: CC(=O)Oc1ccccc1C(=O)O -> flagged\n"
        "  - V06 FAIL: 9 observed 13C signals vs 8 distinct carbons\n"
        "Propose revised candidates that address these problems. Return a corrected JSON object only."
    )
    calls = captured[n:]
    print("echo check: calls", len(calls), "(no 400 means the API accepted the echoed history)")
    echoed = [b["type"] for b in calls[1][0]["messages"][1]["content"]]
    print("block types echoed in the follow-up request:", echoed)
    print("follow-up usage:", calls[1][1]["usage"])
    (OUT / "aspirin_echo_first.json").write_text(json.dumps(scrub(calls[0][1]), indent=2) + "\n")
    (OUT / "aspirin_echo_followup.json").write_text(json.dumps(scrub(calls[1][1]), indent=2) + "\n")


if __name__ == "__main__":
    sys.path.insert(0, ".")
    main()
