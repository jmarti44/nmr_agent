"""Peak-list analysis pipeline: normalize, prompt, model, sanitize, verify, optional repair, response.

Business logic only; the HTTP layer lives in src/api. Raw-spectrum inputs are not supported yet.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import structlog

from src.core.settings import Settings
from src.core.version import SCHEMA_VERSION, SERVICE_VERSION
from src.llm.errors import LLMError
from src.llm.orchestrator import Conversation
from src.llm.pricing import estimate_cost_usd
from src.llm.validate import sanitize_hypothesis
from src.prompting import build_prompt_for_request
from src.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    LLMHypothesis,
    PeakTable,
    RuleStatus,
    SampleContext,
    Usage,
    VerifiedCandidate,
    Versions,
)

log = structlog.get_logger("nmr_agent.analysis")
_REPORT_RULES_PER_CANDIDATE = 6
_REPORT_MESSAGE_CHARS = 240


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


def _renumber_after(old: list[Any], new: list[Any]) -> list[Any]:
    """Rank the repair round's candidates after the first round's so both sets can be verified together."""
    start = max((c.rank for c in old), default=0)
    return [c.model_copy(update={"rank": start + i}) for i, c in enumerate(new, start=1)]


def _versions(settings: Settings, prompt_version: str) -> Versions:
    import nmrglue
    from rdkit import rdBase

    return Versions(
        service=SERVICE_VERSION,
        schema_version=SCHEMA_VERSION,
        prompt_version=prompt_version,
        model_id=settings.model,
        nmrglue=str(nmrglue.__version__),
        rdkit=str(rdBase.rdkitVersion),
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
    request: AnalysisRequest, settings: Settings, client: Any, request_id: str | None = None
) -> AnalysisResponse:
    from src.verify.verifier import verify_functional_groups

    request_id = request_id or new_request_id()
    started = time.perf_counter()
    table, prompt = build_prompt_for_request(request)  # ValueError / PromptTooLarge for unsupported input
    prompt_ms = _ms(time.perf_counter() - started)
    context = request.context

    convo = Conversation(prompt, settings, client, request_id=request_id)
    raw = convo.elucidate()
    hypothesis, warnings = sanitize_hypothesis(raw, table)
    verify_s = 0.0

    t = time.perf_counter()
    claims_rank = hypothesis.candidates[0].rank  # the claims describe the model's first candidate
    verified = _verify(table, hypothesis.candidates, hypothesis, context, claims_rank)
    verify_s += time.perf_counter() - t

    if not any(v.status == "accepted" for v in verified) and convo.calls_left >= 1:
        try:
            raw2 = convo.repair(build_repair_report(verified))
            hyp2, warn2 = sanitize_hypothesis(raw2, table)
            extra = _renumber_after(hypothesis.candidates, hyp2.candidates)
            # Old and new candidates are verified together. The newest claims describe the repair round's
            # first candidate only, so first-round candidates are not checked against them.
            claims_rank = extra[0].rank
            t = time.perf_counter()
            verified = _verify(table, [*hypothesis.candidates, *extra], hyp2, context, claims_rank)
            verify_s += time.perf_counter() - t
            raw, hypothesis = raw2, hyp2
            warnings += ["no candidate was accepted; one repair round was used", *warn2]
        except LLMError as e:
            warnings.append(
                f"repair round failed ({getattr(e, 'reason', type(e).__name__)}); kept the first answer"
            )
    warnings += convo.warnings

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
        versions=_versions(settings, prompt.version),
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
    )


def _ms(seconds: float) -> int:
    return int(seconds * 1000)
