"""Checks run on a parsed hypothesis before verification (SPEC §7.3 table). No chemistry and no network."""

from __future__ import annotations

from src.schemas import LLMHypothesis, PeakRole, PeakTable, parse_formula


def sanitize_hypothesis(hyp: LLMHypothesis, table: PeakTable) -> tuple[LLMHypothesis, list[str]]:
    """Return a cleaned copy and the warnings describing every change.

    - evidence ids that are not in the table are dropped
    - evidence citing solvent/water/reference/impurity peaks is reported (V09 lowers the score)
    - confidences are clamped to [0, 1]
    - an unparsable `formula_inferred` is discarded
    Duplicate candidates are merged by the verifier (V11, InChIKey first block), not here.
    """
    warnings: list[str] = []
    by_id = {p.id: p for p in table.peaks}
    h = hyp.model_copy(deep=True)

    for claim in h.functional_groups:
        keep = []
        for pid in claim.evidence.peak_ids:
            if pid not in by_id:
                warnings.append(
                    f"{claim.group.value}: evidence peak id {pid} is not in the peak table; dropped"
                )
                continue
            if by_id[pid].role != PeakRole.ANALYTE:
                warnings.append(
                    f"{claim.group.value}: evidence cites peak {pid}, which is tagged {by_id[pid].role.value}"
                )
            keep.append(pid)
        claim.evidence.peak_ids = keep
        claim.confidence = _clamp(claim.confidence, f"{claim.group.value} confidence", warnings)

    for cand in h.candidates:
        cand.confidence = _clamp(cand.confidence, f"candidate {cand.rank} confidence", warnings)
        for a in cand.assignments:
            if a.peak_id not in by_id:
                warnings.append(f"candidate {cand.rank}: assignment cites unknown peak id {a.peak_id}")

    bad_ids = [i for i in h.unexplained_peak_ids if i not in by_id]
    if bad_ids:
        warnings.append(f"unexplained_peak_ids contains ids not in the table: {bad_ids}; dropped")
        h.unexplained_peak_ids = [i for i in h.unexplained_peak_ids if i in by_id]

    if h.formula_inferred is not None:
        try:
            parse_formula(h.formula_inferred)
        except ValueError:
            warnings.append(f"formula_inferred {h.formula_inferred!r} does not parse; ignored")
            h.formula_inferred = None
    return h, warnings


def _clamp(value: float, what: str, warnings: list[str]) -> float:
    if 0.0 <= value <= 1.0:
        return value
    warnings.append(f"{what} {value} is outside [0, 1]; clamped")
    return min(1.0, max(0.0, value))
