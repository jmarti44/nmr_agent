"""Assemble the prompt for a request (SPEC §7.2). Builds text only: it never contacts a model.

The prompt has a static, cacheable part (system blocks) and a volatile part (the user message).
`system_blocks` uses the plain-dict shape the Messages API accepts, including the cache marker, but nothing
here imports an SDK or sends anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.processing.normalize import normalize_peaklist
from src.prompting.render import render_context_block, render_peak_table
from src.prompting.templates import FULL_VERSION, estimate_tokens, prompt_hash, system_prompt
from src.schemas import AnalysisRequest, PeakTable, SampleContext

MAX_USER_TOKENS_ESTIMATE = (
    8000  # the spec caps peak lists at 200 entries; this guards against anything larger
)


class PromptTooLarge(ValueError):
    pass


@dataclass(frozen=True)
class Prompt:
    version: str
    system_hash: str
    system_blocks: list[dict]
    user_text: str
    static_tokens_est: int
    user_tokens_est: int

    @property
    def total_tokens_est(self) -> int:
        return self.static_tokens_est + self.user_tokens_est


def build_user_message(table: PeakTable, context: SampleContext, max_candidates: int) -> str:
    return render_peak_table(table, context) + "\n\n" + render_context_block(context, max_candidates)


def build_prompt(table: PeakTable, context: SampleContext | None = None, max_candidates: int = 3) -> Prompt:
    if not 1 <= max_candidates <= 5:
        raise ValueError("max_candidates must be between 1 and 5")
    context = context or SampleContext(solvent=table.solvent)
    user = build_user_message(table, context, max_candidates)
    user_tokens = estimate_tokens(user)
    if user_tokens > MAX_USER_TOKENS_ESTIMATE:
        raise PromptTooLarge(f"user message is about {user_tokens} tokens (limit {MAX_USER_TOKENS_ESTIMATE})")
    system = system_prompt()
    return Prompt(
        version=FULL_VERSION,
        system_hash=prompt_hash(),
        system_blocks=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        user_text=user,
        static_tokens_est=estimate_tokens(system),
        user_tokens_est=user_tokens,
    )


def build_prompt_for_request(request: AnalysisRequest) -> tuple[PeakTable, Prompt]:
    """Peak-list request -> (normalized table, prompt). Raw-spectrum formats arrive in a later phase."""
    if request.input_format != "peaklist" or request.peaks is None:
        raise ValueError("only input_format='peaklist' is supported until raw-data ingestion exists")
    table = normalize_peaklist(request.peaks, request.context)
    return table, build_prompt(table, request.context, request.max_candidates)


def prepare_request(
    request: AnalysisRequest, upload: bytes | None = None
) -> tuple[PeakTable, Prompt, SampleContext]:
    """Any request -> (normalized table, prompt, context to verify with).

    Raw formats are processed first; the context comes back completed with the solvent and spectrometer
    frequency read from the file when the caller did not give them.
    """
    if request.input_format == "peaklist":
        table, prompt = build_prompt_for_request(request)
        return table, prompt, request.context
    if upload is None:
        raise ValueError(f"input_format={request.input_format!r} requires an uploaded file")
    from src.processing.pipeline import process_upload

    processed = process_upload(request.input_format, upload, request.context, request.options)
    table = normalize_peaklist(
        processed.peaks,
        processed.context,
        processed.referencing,
        tag_impurities=processed.tag_impurities,
        tag_reference=processed.tag_reference,
    )
    table = table.model_copy(
        update={
            "noise_sigma": processed.noise_sigma,
            "processing_log": [*processed.log, *table.processing_log],
        }
    )
    return table, build_prompt(table, processed.context, request.max_candidates), processed.context
