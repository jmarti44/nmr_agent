import json

import httpx2
import pytest

from src.core.settings import Settings
from src.llm.errors import LLMRefused, LLMSchemaFailure, LLMUnavailable
from src.llm.orchestrator import FALLBACK_BETA, Conversation
from src.prompting import build_prompt_for_request
from tests.llm.fake_api import FakeAPI, error, hypothesis, hypothesis_json, message
from tests.prompting.helpers import aspirin_request

PROMPT = build_prompt_for_request(aspirin_request())[1]


def convo(api: FakeAPI, sleeps: list[float] | None = None, **settings) -> Conversation:
    """A conversation whose backoff sleeps are recorded (or skipped) instead of waited for."""
    sleep = sleeps.append if sleeps is not None else (lambda s: None)
    return Conversation(PROMPT, Settings(**settings), api.client(), request_id="req_test", sleep=sleep)


def good() -> dict:
    return message(hypothesis_json())


# ---- the request that goes out --------------------------------------------------------------------------


def test_request_shape_for_claude_opus_5_5():
    api = FakeAPI([good()])
    convo(api).elucidate()
    req = api.requests[0]
    body = req["body"]
    assert req["path"] == "/v1/messages" and req["beta"] == FALLBACK_BETA
    assert (
        body["model"] == "claude-opus-5-5" and body["fallbacks"] == "default" and body["max_tokens"] == 16000
    )
    for banned in ("temperature", "top_p", "top_k", "thinking", "tool_choice"):
        assert banned not in body, f"Claude Opus 5.5 rejects or manages {banned}"
    assert body["output_config"]["effort"] == "high"
    fmt = body["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"]["additionalProperties"] is False
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["system"][0]["text"] == PROMPT.system_blocks[0]["text"]
    assert body["messages"] == [{"role": "user", "content": PROMPT.user_text}]


def test_fallbacks_can_be_turned_off_and_then_the_plain_endpoint_is_used():
    api = FakeAPI([good()])
    convo(api, enable_fallbacks=False).elucidate()
    assert api.requests[0]["beta"] is None and "fallbacks" not in api.requests[0]["body"]


def test_success_records_usage_and_returns_the_validated_hypothesis():
    api = FakeAPI([message(hypothesis_json(), usage={"cache_read_input_tokens": 1900, "input_tokens": 250})])
    c = convo(api)
    hyp = c.elucidate()
    assert hyp.candidates[0].smiles == "CC(=O)Oc1ccccc1C(=O)O"
    rec = c.calls[0]
    assert (rec.input_tokens, rec.cache_read_input_tokens, rec.stop_reason, rec.effort) == (
        250,
        1900,
        "end_turn",
        "high",
    )
    assert rec.response_id == "msg_01" and c.calls_left == 3


# ---- correction turns -----------------------------------------------------------------------------------


def test_invalid_output_triggers_a_correction_turn_that_echoes_the_assistant_content_unchanged():
    api = FakeAPI([message('{"candidates": []}'), good()])
    c = convo(api)
    c.elucidate()
    second = api.requests[1]["body"]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assistant_blocks = second[1]["content"]
    assert [b["type"] for b in assistant_blocks] == [
        "thinking",
        "text",
    ]  # thinking block kept: append-only history
    assert assistant_blocks[0]["signature"] == "sig_abc"
    assert second[2]["content"].startswith("Your JSON failed validation:")
    assert second[0] == api.requests[0]["body"]["messages"][0]  # earlier turns are never edited
    assert len(c.calls) == 2 and any("correction" in w for w in c.warnings)


def test_three_invalid_outputs_raise_a_schema_failure_after_exactly_three_calls():
    api = FakeAPI([message("not json"), message("{}"), message('{"x": 1}')])
    c = convo(api)
    with pytest.raises(LLMSchemaFailure):
        c.elucidate()
    assert len(api.requests) == 3


def test_output_cut_off_at_max_tokens_is_retried_with_double_the_budget_then_fails():
    api = FakeAPI([message('{"cand', stop_reason="max_tokens"), message('{"cand', stop_reason="max_tokens")])
    c = convo(api)
    with pytest.raises(LLMUnavailable) as e:
        c.elucidate()
    assert e.value.reason == "output_truncated"
    assert [r["body"]["max_tokens"] for r in api.requests] == [16000, 32000]
    assert api.requests[1]["body"]["messages"] == api.requests[0]["body"]["messages"]  # nothing appended


def test_max_tokens_then_success():
    api = FakeAPI([message("{", stop_reason="max_tokens"), good()])
    assert convo(api).elucidate().candidates
    assert [r["body"]["max_tokens"] for r in api.requests] == [16000, 32000]


# ---- refusals and unexpected stops ----------------------------------------------------------------------


def test_a_refusal_is_not_retried_and_carries_the_category():
    api = FakeAPI(
        [
            message(
                "",
                stop_reason="refusal",
                stop_details={"type": "refusal", "category": "cyber", "explanation": "x"},
            )
        ]
    )
    with pytest.raises(LLMRefused) as e:
        convo(api).elucidate()
    assert e.value.category == "cyber" and len(api.requests) == 1


def test_an_unexpected_stop_reason_is_reported():
    api = FakeAPI([message(hypothesis_json(), stop_reason="pause_turn")])
    with pytest.raises(LLMUnavailable) as e:
        convo(api).elucidate()
    assert e.value.reason == "unexpected_stop_reason"


def test_fallback_that_ran_is_noted():
    it = [
        {"type": "message", "input_tokens": 1, "output_tokens": 1},
        {"type": "fallback_message", "input_tokens": 1, "output_tokens": 1},
    ]
    api = FakeAPI([message(hypothesis_json(), iterations=it, model="claude-opus-4-8")])
    c = convo(api)
    c.elucidate()
    assert c.calls[0].fallback_ran and any("fallback" in w for w in c.warnings)


# ---- transport and API errors --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reply,reason",
    [
        (error(429, "rate_limit_error", "slow down", {"retry-after": "17"}), "rate_limited"),
        (error(529, "overloaded_error", "busy"), "status_529"),
        (error(500, "api_error", "boom"), "status_500"),
        (error(400, "invalid_request_error", "bad"), "status_400"),
        (error(401, "authentication_error", "no"), "status_401"),
        (httpx2.ReadTimeout("slow"), "timeout"),
        (httpx2.ConnectError("down"), "connection"),
    ],
)
def test_sdk_errors_become_llm_unavailable_with_a_reason(reply, reason):
    with pytest.raises(LLMUnavailable) as e:
        convo(FakeAPI([reply]), sdk_max_retries=0).elucidate()  # one attempt: retries have their own tests
    assert e.value.reason == reason


def test_rate_limit_keeps_the_retry_after_header():
    api = FakeAPI([error(429, "rate_limit_error", "slow", {"retry-after": "17"})])
    with pytest.raises(LLMUnavailable) as e:
        convo(api, sdk_max_retries=0).elucidate()
    assert e.value.retry_after == "17"


def test_missing_credentials_are_reported_cleanly(monkeypatch):
    import anthropic

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    api = FakeAPI([good()])
    http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(api.handler))
    with pytest.raises(LLMUnavailable) as e:
        Conversation(PROMPT, Settings(), anthropic.Anthropic(http_client=http)).elucidate()
    assert e.value.reason == "no_credentials"


# ---- budgets ---------------------------------------------------------------------------------------------


def test_call_budget_is_enforced():
    api = FakeAPI([good(), good()])
    c = convo(api, max_llm_calls=1)
    c.elucidate()
    with pytest.raises(LLMUnavailable) as e:
        c.repair("report")
    assert e.value.reason == "call_budget_exhausted" and len(api.requests) == 1


def test_deadline_stops_a_new_call():
    api = FakeAPI([good()])
    c = convo(api, request_timeout_s=0.5)
    with pytest.raises(LLMUnavailable) as e:
        c.elucidate()
    assert e.value.reason == "timeout" and not api.requests


def _record_timeouts(monkeypatch) -> list[float]:
    import anthropic

    seen: list[float] = []
    original = anthropic.Anthropic.with_options

    def spy(self, **kw):
        seen.append(kw["timeout"])
        return original(self, **kw)

    monkeypatch.setattr(anthropic.Anthropic, "with_options", spy)
    return seen


def test_per_call_timeout_is_the_smaller_of_the_call_limit_and_the_time_left(monkeypatch):
    seen = _record_timeouts(monkeypatch)
    convo(FakeAPI([good()]), request_timeout_s=500, llm_call_timeout_s=120).elucidate()
    convo(FakeAPI([good()]), request_timeout_s=60, llm_call_timeout_s=120).elucidate()
    assert seen[0] == 120 and 55 < seen[1] <= 60


# ---- repair turn ------------------------------------------------------------------------------------------


def test_repair_appends_to_history_uses_low_effort_and_returns_a_new_hypothesis():
    api = FakeAPI([good(), message(hypothesis_json([ISOMER_SAFE]))])
    c = convo(api)
    c.elucidate()
    hyp = c.repair("Revise using this report.")
    req = api.requests[1]["body"]
    assert req["output_config"]["effort"] == "low"
    assert [m["role"] for m in req["messages"]] == ["user", "assistant", "user"]
    assert req["messages"][2]["content"] == "Revise using this report."
    assert [b["type"] for b in req["messages"][1]["content"]] == ["thinking", "text"]
    assert hyp.candidates[0].smiles == ISOMER_SAFE and len(c.calls) == 2


def test_repair_requires_a_first_answer():
    with pytest.raises(RuntimeError):
        convo(FakeAPI([])).repair("x")


def test_repair_that_returns_invalid_json_raises_schema_failure():
    api = FakeAPI([good(), message("garbage")])
    c = convo(api)
    c.elucidate()
    with pytest.raises(LLMSchemaFailure):
        c.repair("r")


ISOMER_SAFE = "COC(=O)c1ccccc1C(=O)O"


# ---- retries are done here, inside the deadline (SDK retries are switched off) -------------------------


def test_every_attempt_is_sent_with_sdk_retries_off(monkeypatch):
    import anthropic

    seen: list[dict] = []
    original = anthropic.Anthropic.with_options

    def spy(self, **kw):
        seen.append(kw)
        return original(self, **kw)

    monkeypatch.setattr(anthropic.Anthropic, "with_options", spy)
    convo(FakeAPI([good()])).elucidate()
    assert seen and all(kw["max_retries"] == 0 for kw in seen)


def test_a_rate_limit_is_retried_after_the_advertised_delay_and_then_succeeds():
    api = FakeAPI([error(429, "rate_limit_error", "slow", {"retry-after": "2"}), good()])
    sleeps: list[float] = []
    c = convo(api, sleeps)
    assert c.elucidate().candidates
    assert sleeps == [2.0] and len(api.requests) == 2
    assert len(c.calls) == 1 and c.http_attempts == 2  # a failed attempt is not a model call
    assert len(c.messages) == 1  # nothing was appended to the history


@pytest.mark.parametrize(
    "reply",
    [error(500, "api_error", "boom"), error(529, "overloaded_error", "busy"), httpx2.ConnectError("down")],
)
def test_server_errors_and_connection_errors_are_retried(reply):
    api = FakeAPI([reply, good()])
    sleeps: list[float] = []
    convo(api, sleeps).elucidate()
    assert len(api.requests) == 2 and len(sleeps) == 1 and 0.5 <= sleeps[0] <= 0.75


@pytest.mark.parametrize(
    "status,kind", [(400, "invalid_request_error"), (401, "authentication_error"), (404, "not_found_error")]
)
def test_client_errors_are_not_retried(status, kind):
    api = FakeAPI([error(status, kind, "no"), good()])
    with pytest.raises(LLMUnavailable) as e:
        convo(api).elucidate()
    assert e.value.reason == f"status_{status}" and len(api.requests) == 1


def test_retries_stop_after_the_configured_number_and_back_off_exponentially():
    api = FakeAPI([error(500, "api_error", "boom")] * 4)
    sleeps: list[float] = []
    with pytest.raises(LLMUnavailable) as e:
        convo(api, sleeps, sdk_max_retries=3).elucidate()
    assert e.value.reason == "status_500" and len(api.requests) == 4  # first try + 3 retries
    assert len(sleeps) == 3 and sleeps[0] < sleeps[1] < sleeps[2]


def test_a_retry_that_would_not_fit_in_the_deadline_is_not_attempted():
    api = FakeAPI([error(429, "rate_limit_error", "slow", {"retry-after": "30"}), good()])
    sleeps: list[float] = []
    with pytest.raises(LLMUnavailable) as e:
        convo(api, sleeps, request_timeout_s=10).elucidate()  # waiting 30 s cannot fit in 10 s
    assert e.value.reason == "rate_limited" and e.value.retry_after == "30"
    assert sleeps == [] and len(api.requests) == 1


def test_a_timeout_is_retried_only_if_a_full_length_attempt_still_fits():
    fits = FakeAPI([httpx2.ReadTimeout("slow"), good()])
    convo(fits, llm_call_timeout_s=10, request_timeout_s=100).elucidate()
    assert len(fits.requests) == 2
    tight = FakeAPI([httpx2.ReadTimeout("slow"), good()])
    with pytest.raises(LLMUnavailable) as e:
        convo(tight, llm_call_timeout_s=120, request_timeout_s=60).elucidate()
    assert e.value.reason == "timeout" and len(tight.requests) == 1


def test_the_deadline_holds_with_the_production_client_settings(monkeypatch):
    """Every attempt hangs for the per-call timeout and the client has the production max_retries=3.
    The analysis must still end inside the deadline, after one attempt (scaled-down version of the probe)."""
    import time

    import anthropic

    from src.llm import orchestrator

    monkeypatch.setattr(orchestrator, "_MIN_CALL_SECONDS", 0.2)
    attempts: list[float] = []

    def hang(request: httpx2.Request) -> httpx2.Response:
        attempts.append(time.monotonic())
        time.sleep(0.3)
        raise httpx2.ReadTimeout("read timed out", request=request)

    http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(hang))
    client = anthropic.Anthropic(api_key="sk-ant-test", http_client=http, max_retries=3)  # production value
    c = Conversation(
        PROMPT, Settings(llm_call_timeout_s=0.3, request_timeout_s=0.7, sdk_max_retries=3), client
    )
    started = time.monotonic()
    with pytest.raises(LLMUnavailable) as e:
        c.elucidate()
    assert e.value.reason == "timeout" and len(attempts) == 1
    assert time.monotonic() - started < 0.7 + 0.5


# ---- upstream error text stays out of anything returned ------------------------------------------------


def test_upstream_error_text_is_logged_with_the_request_id_but_not_carried_in_the_exception():
    from structlog.testing import capture_logs

    secret = "Your credit balance is too low: acct_ZX81"
    api = FakeAPI([error(400, "invalid_request_error", secret)])
    with capture_logs() as logs, pytest.raises(LLMUnavailable) as e:
        convo(api).elucidate()
    assert secret not in str(e.value) and secret not in e.value.detail
    assert e.value.reason == "status_400"
    entry = next(x for x in logs if x["event"] == "llm_api_error")
    assert secret in entry["upstream_message"] and entry["request_id"] == "req_test"


# ---- echoing a reply after a mid-output fallback -------------------------------------------------------

FALLBACK = {
    "type": "fallback",
    "from": {"model": "claude-opus-5-5"},
    "to": {"model": "claude-opus-4-8"},
    "trigger": {"type": "refusal", "category": "cyber"},
}


def _blocks_after_fallback(bad_text: str) -> list[dict]:
    return [
        {"type": "thinking", "thinking": "", "signature": "sig_abc"},
        {"type": "text", "text": "partial"},
        FALLBACK,
        {"type": "text", "text": bad_text},
    ]


def test_blocks_before_a_fallback_block_are_not_echoed_on_a_correction_turn():
    api = FakeAPI([message("", blocks=_blocks_after_fallback("{}")), good()])
    convo(api).elucidate()
    echoed = api.requests[1]["body"]["messages"][1]["content"]
    assert [b["type"] for b in echoed] == [
        "text",
        "fallback",
        "text",
    ]  # thinking dropped, text and fallback kept
    assert echoed[2]["text"] == "{}"


def test_the_echo_is_unchanged_when_no_fallback_ran():
    api = FakeAPI([message("{}"), good()])
    convo(api).elucidate()
    assert [b["type"] for b in api.requests[1]["body"]["messages"][1]["content"]] == ["thinking", "text"]


def test_echoable_drops_the_right_blocks():
    from types import SimpleNamespace as B

    from src.llm.orchestrator import _echoable

    content = [
        B(type="thinking"), B(type="redacted_thinking"), B(type="tool_use", id="t1"),
        B(type="server_tool_use", id="s1"), B(type="server_tool_use", id="s2"),
        B(type="web_search_tool_result", tool_use_id="s2"),
        B(type="text", text="a"), B(type="fallback"), B(type="thinking"), B(type="text", text="b"),
    ]  # fmt: skip
    kept = [(b.type, getattr(b, "id", None) or getattr(b, "tool_use_id", None)) for b in _echoable(content)]
    # before the last fallback: thinking/redacted_thinking/tool_use and the unanswered server_tool_use s1 go;
    # the answered s2 pair and the text stay; the thinking block AFTER the fallback stays too
    assert kept == [
        ("server_tool_use", "s2"),
        ("web_search_tool_result", "s2"),
        ("text", None),
        ("fallback", None),
        ("thinking", None),
        ("text", None),
    ]
    plain = [B(type="thinking"), B(type="text", text="x")]
    assert _echoable(plain) is plain


def test_a_reply_with_two_candidates_of_the_same_rank_gets_a_correction_turn():
    h = hypothesis(["CC(=O)Oc1ccccc1C(=O)O", "COC(=O)c1ccccc1C(=O)O"])
    h["candidates"][1]["rank"] = 1  # two candidates ranked 1: the claims would be ambiguous
    api = FakeAPI([message(json.dumps(h)), good()])
    convo(api).elucidate()
    assert len(api.requests) == 2
    correction = api.requests[1]["body"]["messages"][-1]["content"]
    assert "unique" in correction and correction.startswith("Your JSON failed validation")
