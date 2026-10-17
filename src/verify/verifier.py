"""Run the rule ledger on candidates, score them, merge duplicates and re-rank (SPEC §8.6)."""

from __future__ import annotations

from src.schemas import (
    Candidate,
    FunctionalGroupClaim,
    Nucleus,
    PeakRole,
    PeakTable,
    RuleResult,
    RuleStatus,
    SampleContext,
    VerifiedCandidate,
    VerifiedFunctionalGroup,
)
from src.verify.chem import matches, parse_smiles
from src.verify.rules import RULES_AFTER_PARSE, Ctx
from src.verify.shift_tables import ALL_RULES, slack_for
from src.verify.smarts import SPECS

ACCEPT_THRESHOLD = 0.6
SOFT_FAIL_FACTOR = 0.6
WARN_FACTOR = 0.85
MODEL_WEIGHT = 0.15


def score(rules: list[RuleResult]) -> tuple[float, bool]:
    """(verifier_score, hard_failed). Hard failure -> 0; soft FAIL x0.6; WARN x0.85."""
    s = 1.0
    for r in rules:
        if r.status == RuleStatus.FAIL and r.severity == "hard":
            return 0.0, True
        if r.status == RuleStatus.FAIL and r.severity == "soft":
            s *= SOFT_FAIL_FACTOR
        elif r.status == RuleStatus.WARN:
            s *= WARN_FACTOR
    return s, False


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _status(rules: list[RuleResult], vscore: float, hard: bool) -> str:
    if hard:
        return "rejected"
    critical_soft_fail = any(
        r.status == RuleStatus.FAIL and r.rule_id.split("_")[0] in ("V05", "V08") for r in rules
    )
    return "accepted" if vscore >= ACCEPT_THRESHOLD - 1e-9 and not critical_soft_fail else "flagged"


def verify_candidate(
    table: PeakTable,
    cand: Candidate,
    context: SampleContext,
    claims: list[FunctionalGroupClaim] | None = None,
    hypothesis_dou: int | None = None,
    use_model_confidence: bool = True,
    claims_note: str | None = None,
) -> VerifiedCandidate:
    parsed, why = parse_smiles(cand.smiles)
    if parsed is None:
        rules = [
            RuleResult(rule_id="V01_smiles_valid", status=RuleStatus.FAIL, severity="hard", message=why or "")
        ]
        return VerifiedCandidate(
            candidate=cand, rules=rules, verifier_score=0.0, status="rejected",
            final_confidence=_final(0.0, cand, use_model_confidence), claims_checked=claims is not None,
        )  # fmt: skip
    msg = "parsed and sanitized; neutral"
    status = RuleStatus.PASS
    if parsed.fragments > 1:
        status, msg = (
            RuleStatus.WARN,
            f"SMILES contains {parsed.fragments} fragments; a single compound is expected",
        )
    rules = [RuleResult(rule_id="V01_smiles_valid", status=status, severity="hard", message=msg)]

    ctx = Ctx(
        table=table,
        context=context,
        parsed=parsed,
        candidate=cand,
        claims=claims,
        claims_note=claims_note,
        hypothesis_dou=hypothesis_dou,
    )
    for rule in RULES_AFTER_PARSE:
        r = rule(ctx)
        rules.append(r)
        if r.status == RuleStatus.FAIL and r.severity == "hard":
            break  # a hard failure stops further chemistry rules for this candidate
    vscore, hard = score(rules)
    return VerifiedCandidate(
        candidate=cand,
        canonical_smiles=parsed.canonical_smiles,
        inchikey=parsed.inchikey or None,  # MolToInchiKey can return '' for unusual molecules
        formula=parsed.formula,
        exact_mass=round(parsed.exact_mass, 4),
        degree_of_unsaturation=parsed.dou,
        rules=rules,
        verifier_score=round(vscore, 4),
        final_confidence=round(_final(vscore, cand, use_model_confidence), 4),
        claims_checked=claims is not None,
        status=_status(rules, vscore, hard),  # type: ignore[arg-type]
    )


def _final(vscore: float, cand: Candidate, use_model_confidence: bool) -> float:
    if not use_model_confidence:
        return _clamp(vscore)
    return _clamp((1 - MODEL_WEIGHT) * vscore + MODEL_WEIGHT * _clamp(cand.confidence))


def _first_block(smiles: str) -> str | None:
    parsed, _ = parse_smiles(smiles)
    return parsed.inchikey.split("-")[0] if parsed is not None and parsed.inchikey else None


def _claims_target(ordered: list[Candidate], claims_rank: int) -> int | None:
    """Index of the candidate that carries the claims.

    The first candidate ranked `claims_rank`; or, if an earlier candidate has the same connectivity, that
    earlier one, because it is the one V11 keeps when duplicates merge. Exactly one candidate is chosen.
    """
    claimed = next((i for i, c in enumerate(ordered) if c.rank == claims_rank), None)
    if claimed is None:
        return None
    keys = [_first_block(c.smiles) for c in ordered]
    if keys[claimed] is None:
        return claimed
    return keys.index(keys[claimed])


def verify_candidates(
    table: PeakTable,
    candidates: list[Candidate],
    context: SampleContext,
    claims: list[FunctionalGroupClaim] | None = None,
    hypothesis_dou: int | None = None,
    use_model_confidence: bool = True,
    claims_rank: int = 1,
) -> list[VerifiedCandidate]:
    """Verify, merge duplicates (same InChIKey first block, V11) and re-rank by final confidence.

    The model's functional-group claims describe one candidate, the one ranked `claims_rank` (the first). V10
    checks them against that candidate only; for every other candidate V10 is skipped, so claims about one
    structure can never penalise another. If the claimed candidate repeats the connectivity of an earlier one,
    the earlier (kept) candidate is checked against the claims instead of being left unchecked. The candidate
    that carried them is marked `claims_checked`.
    """
    note = f"the model's claims describe candidate {claims_rank}, not this one"
    ordered = sorted(candidates, key=lambda c: c.rank)
    target = _claims_target(ordered, claims_rank) if claims is not None else None
    verified = [
        verify_candidate(
            table,
            c,
            context,
            claims if i == target else None,
            hypothesis_dou,
            use_model_confidence,
            claims_note=note if claims is not None and i != target else None,
        )
        for i, c in enumerate(ordered)
    ]
    kept: list[VerifiedCandidate] = []
    first_block: dict[str, int] = {}
    merged: dict[int, list[int]] = {}
    for v in verified:
        key = v.inchikey.split("-")[0] if v.inchikey else None  # an empty key must never merge candidates
        if key is not None and key in first_block:
            merged.setdefault(first_block[key], []).append(v.candidate.rank)
            continue
        if key is not None:
            first_block[key] = len(kept)
        kept.append(v)
    for i, v in enumerate(kept):
        if v.inchikey is None:
            continue  # unparsable: V11 does not apply
        dupes = merged.get(i, [])
        msg = f"merged duplicate connectivity from rank(s) {dupes}" if dupes else "distinct connectivity"
        v.rules.append(
            RuleResult(
                rule_id="V11_distinct_connectivity", status=RuleStatus.PASS, severity="info", message=msg
            )
        )
    return sorted(kept, key=lambda v: (-v.final_confidence, v.candidate.rank))


def verify_functional_groups(
    table: PeakTable,
    claims: list[FunctionalGroupClaim],
    top: VerifiedCandidate | None,
    context: SampleContext,
) -> list[VerifiedFunctionalGroup]:
    """Per-claim support from shift rules alone (intersection ranges) plus presence in the top candidate."""
    parsed = None
    if top is not None and top.status != "rejected" and top.canonical_smiles:
        parsed, _ = parse_smiles(top.canonical_smiles)
    slack = {n: slack_for(n, table.referencing) for n in Nucleus}
    out: list[VerifiedFunctionalGroup] = []
    for cl in claims:
        spec = SPECS[cl.group]
        keys = [k for ev in (*spec.h, *spec.c) for k in ev.keys]
        peaks = [
            p for i in cl.evidence.peak_ids for p in table.peaks if p.id == i and p.role == PeakRole.ANALYTE
        ]
        if not cl.present or not keys:
            status = RuleStatus.SKIPPED
        else:
            strict = any(
                ALL_RULES[k].confirms(p.shift_ppm, slack[p.nucleus])
                for k in keys
                for p in peaks
                if ALL_RULES[k].nucleus == p.nucleus
            )
            loose = any(
                ALL_RULES[k].contains(p.shift_ppm, slack[p.nucleus])
                for k in keys
                for p in peaks
                if ALL_RULES[k].nucleus == p.nucleus
            )
            status = RuleStatus.PASS if strict else RuleStatus.WARN if loose else RuleStatus.FAIL
        present = None if parsed is None else any(matches(parsed.mol, s) for s in spec.smarts)
        src = ", ".join(sorted({s for k in keys for s in ALL_RULES[k].sources})) or None
        out.append(
            VerifiedFunctionalGroup(
                group=cl.group, claimed_present=cl.present, supported_by_shifts=status,
                present_in_top_candidate=present, source=src,
            )
        )  # fmt: skip
    return out
