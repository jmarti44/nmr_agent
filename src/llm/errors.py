"""Error taxonomy for the model call. The API layer maps these to HTTP errors (SPEC §9.3)."""

from __future__ import annotations


class LLMError(Exception):
    """Base class: something went wrong getting a hypothesis from the model."""


class LLMUnavailable(LLMError):
    """The model could not be reached or did not finish. `reason` is a short machine-readable code."""

    def __init__(self, reason: str, detail: str = "", retry_after: str | None = None):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail or reason
        self.retry_after = retry_after


class LLMRefused(LLMError):
    """The model declined (stop_reason 'refusal'), after server-side fallbacks if enabled."""

    def __init__(self, category: str | None = None, explanation: str | None = None):
        super().__init__(f"refused (category={category})")
        self.category = category
        self.explanation = explanation


class LLMSchemaFailure(LLMError):
    """The model returned output that did not validate after the allowed correction attempts."""
