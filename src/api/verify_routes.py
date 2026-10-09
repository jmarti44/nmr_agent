"""Routes that need no LLM: /v1/verify and /v1/reference/shifts. RDKit is imported lazily inside handlers."""

from __future__ import annotations

from fastapi import APIRouter

from src.schemas import Candidate, ErrorResponse, VerifiedCandidate, VerifyRequest

router = APIRouter(prefix="/v1")

_ERRORS = {400: {"model": ErrorResponse}}


@router.post(
    "/verify",
    response_model=list[VerifiedCandidate],
    tags=["verification"],
    responses=_ERRORS,
    summary="Verify candidate structures against a peak table (no LLM)",
    description=(
        "Runs the deterministic rule ledger on each SMILES. `accepted` means consistent with the supplied "
        "data, not proven correct: several isomers can be accepted for the same spectrum."
    ),
)
def verify(req: VerifyRequest) -> list[VerifiedCandidate]:
    from src.verify.verifier import verify_candidates

    candidates = [
        Candidate(
            rank=i, smiles=s.strip(), rationale="user-supplied structure", assignments=[], confidence=0.0
        )
        for i, s in enumerate(req.candidates, start=1)
    ]
    # No model is involved, so final_confidence equals the verifier score.
    return verify_candidates(req.peak_table, candidates, req.context, use_model_confidence=False)


@router.get(
    "/reference/shifts", tags=["reference"], summary="Chemical-shift, SMARTS and solvent tables with sources"
)
def reference_shifts() -> dict:
    from src.verify.shift_tables import (
        BASE_SLACK,
        C_RULES,
        DMSO_EXCHANGEABLE_WINDOW,
        H_RULES,
        SHIFT_TABLE_VERSION,
        SOURCES,
        UNREFERENCED_EXTRA_SLACK,
        rule_as_dict,
    )
    from src.verify.smarts import SPECS
    from src.verify.solvent_tables import table_as_dict

    return {
        "version": SHIFT_TABLE_VERSION,
        "sources": SOURCES,
        "h1_rules": [rule_as_dict(r) for r in H_RULES.values()],
        "c13_rules": [rule_as_dict(r) for r in C_RULES.values()],
        "functional_groups": {
            g.value: {
                "smarts": list(s.smarts),
                "h_evidence": [
                    {"any_of": list(e.keys), "applies_if": e.applies_if, "hard_in": list(e.hard_in)}
                    for e in s.h
                ],
                "c_evidence": [{"any_of": list(e.keys)} for e in s.c],
            }
            for g, s in SPECS.items()
        },
        "solvents_and_impurities": table_as_dict(),
        "tolerances_ppm": {
            "base_slack": {k.value: v for k, v in BASE_SLACK.items()},
            "unreferenced_extra": {k.value: v for k, v in UNREFERENCED_EXTRA_SLACK.items()},
            "dmso_exchangeable_window": list(DMSO_EXCHANGEABLE_WINDOW),
        },
    }
