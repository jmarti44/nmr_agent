"""One analysis's conversation with the model (SPEC §7.3).

Differences from the SPEC pseudocode, each deliberate:
- The reply is validated here (`LLMHypothesis.model_validate_json`) instead of with `messages.parse`, because
  `parse` discards the response when validation fails and the correction turn must echo the assistant content.
- That content is echoed back whole, thinking blocks included. The SPEC kept only text blocks, which edits the
  earlier turn; on Claude Opus 5.5 a thinking block is only valid in the conversation that produced it.
- Server-side refusal fallbacks are a parameter of the same call (`fallbacks="default"` on the beta endpoint),
  not a separate retry path.
- Retries are done here, not by the SDK: the SDK's own retries would let one call run for
  timeout x (max_retries + 1), past the analysis deadline. Every attempt gets only the time that is left.
- Sampling parameters and forced tool use are never sent (Claude Opus 5.5 rejects them) and `thinking` is left
  unset (always on); depth is set with `output_config.effort`.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import anthropic
import structlog
from pydantic import ValidationError

from src.core.settings import Settings
from src.llm.errors import LLMRefused, LLMSchemaFailure, LLMUnavailable
from src.prompting.builder import Prompt
from src.schemas import LLMHypothesis

FALLBACK_BETA = "server-side-fallback-2026-07-01"
_MAX_VALIDATION_ERROR_CHARS = 1500
_MIN_CALL_SECONDS = 3.0  # do not start a call with less time than this left
_MAX_BACKOFF_S = 8.0
_MAX_RETRY_AFTER_S = 30.0
_RETRYABLE_STATUSES = frozenset({408, 409, 429})  # plus every 5xx
# After a mid-output fallback these blocks must not be echoed back if they precede the last 'fallback' block.
_DROP_BEFORE_FALLBACK = frozenset({"thinking", "redacted_thinking", "tool_use"})

log = structlog.get_logger("nmr_agent.llm")


@dataclass
class CallRecord:
    response_id: str | None
    request_id: str | None
    model: str
    stop_reason: str | None
    effort: str
    max_tokens: int
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    fallback_ran: bool
    duration_s: float = 0.0


def _usage(resp: Any) -> tuple[int, int, int, int]:
    u = resp.usage
    return (
        u.input_tokens or 0,
        u.output_tokens or 0,
        getattr(u, "cache_read_input_tokens", 0) or 0,
        getattr(u, "cache_creation_input_tokens", 0) or 0,
    )


@dataclass(frozen=True)
class _Failure:
    reason: str
    detail: str  # safe to return to callers
    retryable: bool
    retry_after: str | None = None


def _echoable(content: Any) -> Any:
    """The assistant content to send back on a correction or repair turn.

    Normally the whole reply, unchanged. If a fallback ran mid-output (a 'fallback' block is present), the
    API wants the thinking, redacted_thinking and tool_use blocks before the last fallback block left out,
    along with any server_tool_use that has no result. Text blocks are always kept.
    """
    blocks = list(content)
    fallbacks = [i for i, b in enumerate(blocks) if getattr(b, "type", None) == "fallback"]
    if not fallbacks:
        return content
    last = fallbacks[-1]
    answered = {
        getattr(b, "tool_use_id", None)
        for b in blocks
        if str(getattr(b, "type", "")).endswith("_tool_result")
    }
    kept = []
    for i, b in enumerate(blocks):
        kind = getattr(b, "type", None)
        if i < last and kind in _DROP_BEFORE_FALLBACK:
            continue
        if kind == "server_tool_use" and getattr(b, "id", None) not in answered:
            continue
        kept.append(b)
    return kept


class Conversation:
    """History and call budget for one analysis: `elucidate()` first, then optionally `repair()`."""

    def __init__(
        self,
        prompt: Prompt,
        settings: Settings,
        client: anthropic.Anthropic,
        request_id: str | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.request_id = request_id
        self._sleep = sleep
        self.http_attempts = 0  # every request sent, including retries; failed attempts do not count as calls
        self.prompt = prompt
        self.settings = settings
        self.client = client
        self.messages: list[dict[str, Any]] = [{"role": "user", "content": prompt.user_text}]
        self.calls: list[CallRecord] = []
        self.warnings: list[str] = []
        self._deadline = time.monotonic() + settings.request_timeout_s
        self._schema = anthropic.transform_schema(LLMHypothesis)
        self._last_content: Any = None

    # ---- public ------------------------------------------------------------------------------------------

    @property
    def calls_left(self) -> int:
        return self.settings.max_llm_calls - len(self.calls)

    def elucidate(self) -> LLMHypothesis:
        """Primary call plus up to two correction turns; `max_tokens` retries share the attempt budget."""
        s = self.settings
        max_tokens = s.initial_max_tokens
        attempts = 0
        while True:
            attempts += 1
            resp = self._call(s.effort, max_tokens)
            outcome = self._interpret(resp)
            if isinstance(outcome, LLMHypothesis):
                self._last_content = resp.content
                return outcome
            if outcome == "max_tokens":
                if max_tokens < s.max_tokens_cap and attempts < s.max_primary_attempts:
                    max_tokens = min(s.max_tokens_cap, max_tokens * 2)
                    self.warnings.append(f"output hit max_tokens; retried with max_tokens={max_tokens}")
                    continue
                raise LLMUnavailable("output_truncated", f"model output exceeded {max_tokens} tokens")
            if attempts >= s.max_primary_attempts:
                raise LLMSchemaFailure(outcome)
            self._append_turn(resp.content, self._correction_message(outcome))
            self.warnings.append(
                f"model output failed validation (attempt {attempts}); asked for a correction"
            )

    def repair(self, report: str) -> LLMHypothesis:
        """One more turn after verification, with the failure report (SPEC §8.7). May raise LLMError."""
        if self._last_content is None:
            raise RuntimeError("repair() needs a successful elucidate() first")
        self._append_turn(self._last_content, report)
        resp = self._call(self.settings.repair_effort, self.settings.initial_max_tokens)
        outcome = self._interpret(resp)
        if isinstance(outcome, LLMHypothesis):
            self._last_content = resp.content
            return outcome
        if outcome == "max_tokens":
            raise LLMUnavailable("output_truncated", "repair output exceeded max_tokens")
        raise LLMSchemaFailure(outcome)

    # ---- internals ---------------------------------------------------------------------------------------

    def _append_turn(self, assistant_content: Any, user_text: str) -> None:
        # Append-only history: earlier turns are never edited.
        self.messages.append({"role": "assistant", "content": _echoable(assistant_content)})
        self.messages.append({"role": "user", "content": user_text})

    @staticmethod
    def _correction_message(error_text: str) -> str:
        return f"Your JSON failed validation:\n{error_text}\nReturn a corrected JSON object only."

    def _interpret(self, resp: Any) -> LLMHypothesis | str:
        """The parsed hypothesis, the string 'max_tokens', or a validation-error text. Raises on refusal."""
        stop = resp.stop_reason
        if stop == "refusal":
            details = getattr(resp, "stop_details", None)
            raise LLMRefused(getattr(details, "category", None), getattr(details, "explanation", None))
        if stop == "max_tokens":
            return "max_tokens"
        if stop not in ("end_turn", "stop_sequence"):
            raise LLMUnavailable("unexpected_stop_reason", f"stop_reason={stop!r}")
        text = "".join(b.text for b in resp.content if b.type == "text")
        try:
            return LLMHypothesis.model_validate_json(text)
        except ValidationError as ve:
            return str(ve)[:_MAX_VALIDATION_ERROR_CHARS]

    def _call(self, effort: str, max_tokens: int) -> Any:
        if self.calls_left <= 0:
            raise LLMUnavailable(
                "call_budget_exhausted", f"limit of {self.settings.max_llm_calls} model calls"
            )
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": max_tokens,
            "system": self.prompt.system_blocks,
            "messages": self.messages,
            "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": self._schema}},
        }
        failures = 0
        while True:
            remaining = self._deadline - time.monotonic()
            if remaining < _MIN_CALL_SECONDS:
                raise LLMUnavailable("timeout", "the analysis deadline was reached")
            # max_retries=0: retrying is done below, inside the deadline, never by the SDK.
            client = self.client.with_options(
                timeout=min(self.settings.llm_call_timeout_s, remaining), max_retries=0
            )
            started = time.monotonic()
            self.http_attempts += 1
            try:
                if self.settings.enable_fallbacks:
                    resp = client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
                else:
                    resp = client.messages.create(**kwargs)
            except TypeError as e:  # the SDK raises a bare TypeError when no credentials resolve
                if "authentication" in str(e):
                    raise LLMUnavailable(
                        "no_credentials", "no Anthropic API credentials are configured"
                    ) from e
                raise
            except anthropic.APIError as e:
                failure = self._classify(e)
                failures += 1
                self._log_failure(failure, e, failures)
                delay = self._retry_delay(failure, failures)
                time_left = self._deadline - time.monotonic()
                if (
                    failure.retryable
                    and failures <= self.settings.sdk_max_retries
                    and time_left - delay >= _MIN_CALL_SECONDS
                    and (failure.reason != "timeout" or time_left - delay >= self.settings.llm_call_timeout_s)
                ):
                    self._sleep(delay)
                    continue
                raise LLMUnavailable(failure.reason, failure.detail, failure.retry_after) from e
            self._record(resp, effort, max_tokens, time.monotonic() - started)
            return resp

    def _classify(self, e: anthropic.APIError) -> _Failure:
        """Reason, a caller-safe detail, and whether to retry. The upstream message is only logged."""
        if isinstance(e, anthropic.RateLimitError):
            return _Failure(
                "rate_limited",
                "the model API is rate limiting requests",
                True,
                e.response.headers.get("retry-after"),
            )
        if isinstance(e, anthropic.APITimeoutError):
            return _Failure("timeout", "the model call timed out", True)
        if isinstance(e, anthropic.APIConnectionError):
            return _Failure("connection", "could not reach the model API", True)
        if isinstance(e, anthropic.APIStatusError):
            retry = e.status_code in _RETRYABLE_STATUSES or e.status_code >= 500
            return _Failure(f"status_{e.status_code}", "the model API returned an error", retry)
        return _Failure("api_error", "the model API call failed", False)

    def _retry_delay(self, failure: _Failure, failures: int) -> float:
        if failure.retry_after is not None:
            try:
                return min(_MAX_RETRY_AFTER_S, max(0.0, float(failure.retry_after)))
            except ValueError:
                pass
        return min(_MAX_BACKOFF_S, 0.5 * 2 ** (failures - 1)) + random.uniform(0.0, 0.25)

    def _log_failure(self, failure: _Failure, e: anthropic.APIError, failures: int) -> None:
        log.warning(
            "llm_api_error",
            request_id=self.request_id,
            reason=failure.reason,
            attempt=failures,
            upstream_message=getattr(e, "message", str(e)),
            upstream_request_id=getattr(e, "request_id", None),
        )

    def _record(self, resp: Any, effort: str, max_tokens: int, duration_s: float) -> None:
        i, o, cr, cw = _usage(resp)
        iterations = getattr(resp.usage, "iterations", None) or []
        rec = CallRecord(
            response_id=getattr(resp, "id", None),
            request_id=getattr(resp, "_request_id", None),
            model=getattr(resp, "model", self.settings.model),
            stop_reason=resp.stop_reason,
            effort=effort,
            max_tokens=max_tokens,
            input_tokens=i,
            output_tokens=o,
            cache_read_input_tokens=cr,
            cache_creation_input_tokens=cw,
            fallback_ran=any(getattr(e, "type", None) == "fallback_message" for e in iterations),
            duration_s=round(duration_s, 3),
        )
        self.calls.append(rec)
        if rec.fallback_ran:
            self.warnings.append(f"a server-side fallback ran; the answer came from {rec.model}")
        log.info("llm_call", **{k: v for k, v in rec.__dict__.items()})
