"""Peak-list analysis pipeline: normalize, prompt, model, sanitize, verify, optional repair, response.

Business logic only; the HTTP layer lives in src/api. Raw uploads (Bruker zip, JCAMP-DX) are processed first.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

import structlog

from src.core.settings import Settings
from src.core.version import SCHEMA_VERSION, SERVICE_VERSION
from src.llm.errors import LLMError
from src.llm.orchestrator import Conversation
from src.llm.pricing import estimate_cost_usd
from src.llm.validate import sanitize_hypothesis
from src.processing.pipeline import WARNING_PREFIX
from src.prompting import prepare_request
from src.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    LLMHypothesis,
    MemoryCaseRef,
    PeakTable,
    RuleStatus,
    SampleContext,
    Usage,
    VerifiedCandidate,
    Versions,
)

if TYPE_CHECKING:
    from src.memory.runtime import MemoryRuntime

log = structlog.get_logger("nmr_agent.analysis")
_REPORT_RULES_PER_CANDIDATE = 6
_REPORT_MESSAGE_CHARS = 240


class CallMeter:
    """Counts the model calls an analysis made, even if it fails afterwards, so the daily budget is exact."""

    def __init__(self) -> None:
        self.calls = 0


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:16]}"


def build_repair_report(verified: list[VerifiedCandidate]) -> str:
    """Compact text of what failed, sent back to the model once if nothing was accepted (SPEC §8.7)."""
    lines = ["None of your candidates passed verification. The checker's report (rule, status, reason):"]
    for v in verified:
        lines.append(f"Candidate {v.candidate.rank}: {v.candidate.smiles} -> {v.status}")
        bad = [r for r in v.rules if r.status in (RuleStatus.FAIL, RuleStatus.WARN)]
        for r in bad[:_REPORT_RULES_PER_CANDIDATE]:
            lines.append(f"  - {r.rule_id} {r.status.value.upper()}: {r.message[:_REPORT_MESSAGE_CHARS]}")
        if len(bad) > _REPORT_RULES_PER_CANDIDATE:
            lines.append(f"  - ... {len(bad) - _REPORT_RULES_PER_CANDIDATE} more")
    lines.append(
        "Propose revised candidates that address these problems and still follow the hard rules. "
        "Return a corrected JSON object only."
    )
    return "\n".join(lines)


def _verify(
    table: PeakTable, candidates: list[Any], hyp: LLMHypothesis, context: SampleContext, claims_rank: int
) -> list[VerifiedCandidate]:
    """Verify `candidates`; `hyp` gives the claims (about candidate `claims_rank`) and the DoU."""
    from src.verify.verifier import verify_candidates

    return verify_candidates(
        table,
        candidates,
        context,
        claims=hyp.functional_groups,
        hypothesis_dou=hyp.degree_of_unsaturation,
        claims_rank=claims_rank,
    )


def _claimed_candidate(verified: list[VerifiedCandidate]) -> VerifiedCandidate | None:
    """The candidate the model's claims were checked against (it is marked by the verifier)."""
    return next((v for v in verified if v.claims_checked), None)


def verify_after_repair(
    table: PeakTable, context: SampleContext, first: LLMHypothesis, repaired: LLMHypothesis
) -> tuple[list[VerifiedCandidate], LLMHypothesis]:
    """Old and new candidates verified together. The newest claims describe the repair round's first candidate
    only, so first-round candidates are not checked against them. Returns the verified list and the hypothesis
    whose claims and DoU the response reports."""
    extra = _renumber_after(first.candidates, repaired.candidates)
    return _verify(table, [*first.candidates, *extra], repaired, context, extra[0].rank), repaired


def reverify(
    table: PeakTable, context: SampleContext, first: LLMHypothesis, repaired: LLMHypothesis | None = None
) -> list[VerifiedCandidate]:
    """Verify saved model answers with the current verifier, with no model call (used by bench rescoring)."""
    if repaired is None:
        return _verify(table, first.candidates, first, context, first.candidates[0].rank)
    return verify_after_repair(table, context, first, repaired)[0]


def _renumber_after(old: list[Any], new: list[Any]) -> list[Any]:
    """Rank the repair round's candidates after the first round's so both sets can be verified together."""
    start = max((c.rank for c in old), default=0)
    return [c.model_copy(update={"rank": start + i}) for i, c in enumerate(new, start=1)]


def _versions(settings: Settings, prompt_version: str, memory_version: str = "none") -> Versions:
    import nmrglue
    from rdkit import rdBase

    return Versions(
        service=SERVICE_VERSION,
        schema_version=SCHEMA_VERSION,
        prompt_version=prompt_version,
        model_id=settings.model,
        nmrglue=str(nmrglue.__version__),
        rdkit=str(rdBase.rdkitVersion),
        memory_version=memory_version,
    )


def _usage(convo: Conversation, settings: Settings, warnings: list[str]) -> Usage:
    def total(field: str) -> int:
        return sum(getattr(c, field) for c in convo.calls)

    cost = estimate_cost_usd(
        settings.model,
        total("input_tokens"),
        total("output_tokens"),
        total("cache_read_input_tokens"),
        total("cache_creation_input_tokens"),
    )
    if cost is None:
        warnings.append(f"no price known for model {settings.model}; cost reported as 0")
    return Usage(
        llm_calls=len(convo.calls),
        input_tokens=total("input_tokens"),
        output_tokens=total("output_tokens"),
        cache_read_input_tokens=total("cache_read_input_tokens"),
        cache_creation_input_tokens=total("cache_creation_input_tokens"),
        estimated_cost_usd=cost or 0.0,
    )


def run_analysis(
    request: AnalysisRequest,
    settings: Settings,
    client: Any,
    request_id: str | None = None,
    upload: bytes | None = None,
    trace: dict[str, Any] | None = None,
    meter: CallMeter | None = None,
    memory: MemoryRuntime | None = None,
) -> AnalysisResponse:
    """`memory`, if given, supplies worked examples for the prompt only; nothing from it reaches the verifier,
    the scoring or the ranking. `trace`, if given, is filled with what is needed to re-verify offline: the
    table, the context, each round's sanitized answer and the injected case ids. `meter`, if given, ends up
    holding the number of model calls made, whatever the outcome."""
    convo_box: list[Conversation] = []
    try:
        return _run_analysis(request, settings, client, request_id, upload, trace, convo_box, memory)
    finally:
        if meter is not None and convo_box:
            meter.calls = len(convo_box[0].calls)


def _run_analysis(
    request: AnalysisRequest,
    settings: Settings,
    client: Any,
    request_id: str | None,
    upload: bytes | None,
    trace: dict[str, Any] | None,
    convo_box: list[Conversation],
    memory: MemoryRuntime | None = None,
) -> AnalysisResponse:
    from src.verify.verifier import verify_functional_groups

    request_id = request_id or new_request_id()
    started = time.perf_counter()
    table, prompt, context = prepare_request(request, upload, memory)  # ValueError / PromptTooLarge -> 400
    prompt_ms = _ms(time.perf_counter() - started)
    memory_cases = [MemoryCaseRef(case_id=r.case.case_id, similarity=r.similarity) for r in prompt.examples]

    convo = Conversation(prompt, settings, client, request_id=request_id)
    convo_box.append(convo)
    raw = convo.elucidate()
    hypothesis, warnings = sanitize_hypothesis(raw, table)
    first_round = hypothesis
    repaired_round = None
    verify_s = 0.0

    t = time.perf_counter()
    claims_rank = hypothesis.candidates[0].rank  # the claims describe the model's first candidate
    verified = _verify(table, hypothesis.candidates, hypothesis, context, claims_rank)
    verify_s += time.perf_counter() - t

    if not any(v.status == "accepted" for v in verified) and convo.calls_left >= 1:
        try:
            raw2 = convo.repair(build_repair_report(verified))
            hyp2, warn2 = sanitize_hypothesis(raw2, table)
            t = time.perf_counter()
            verified, _ = verify_after_repair(table, context, hypothesis, hyp2)
            verify_s += time.perf_counter() - t
            repaired_round = hyp2
            raw, hypothesis = raw2, hyp2
            warnings += ["no candidate was accepted; one repair round was used", *warn2]
        except LLMError as e:
            warnings.append(
                f"repair round failed ({getattr(e, 'reason', type(e).__name__)}); kept the first answer"
            )
    warnings += convo.warnings
    if trace is not None:
        trace.update(
            table=table.model_dump(mode="json"),
            context=context.model_dump(mode="json"),
            first=first_round.model_dump(mode="json"),
            repaired=repaired_round.model_dump(mode="json") if repaired_round else None,
            verified_untrimmed=[v.model_dump(mode="json") for v in verified],
            memory_cases=[m.model_dump(mode="json") for m in memory_cases],
        )
    warnings += [
        line.removeprefix(WARNING_PREFIX) for line in table.processing_log if line.startswith(WARNING_PREFIX)
    ]

    claimed = _claimed_candidate(verified)  # found before trimming to max_candidates
    verified = verified[: request.max_candidates]
    groups = verify_functional_groups(table, hypothesis.functional_groups, claimed, context)
    usage = _usage(convo, settings, warnings)
    log.info(
        "analysis_done",
        request_id=request_id,
        llm_calls=usage.llm_calls,
        cost_usd=usage.estimated_cost_usd,
        accepted=sum(v.status == "accepted" for v in verified),
    )
    return AnalysisResponse(
        request_id=request_id,
        versions=_versions(settings, prompt.version, memory.memory_version if memory is not None else "none"),
        peak_table=table,
        hypothesis_raw=raw,
        candidates=verified,
        functional_groups=groups,
        warnings=warnings,
        timing_ms={
            "prompt": prompt_ms,
            "llm": _ms(sum(c.duration_s for c in convo.calls)),
            "verify": _ms(verify_s),
            "total": _ms(time.perf_counter() - started),
        },
        usage=usage,
        memory_cases=memory_cases,
    )


def _ms(seconds: float) -> int:
    return int(seconds * 1000)
