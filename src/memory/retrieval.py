"""Retrieval over a MemoryStore (spec §5.2-5.4): scoring, tie-break, one-per-scaffold, k, token budget.

Everything here is deterministic and offline: no model, no network, no module-level mutable state.
Scoring reads only spectrum-side features (see ``features``); ``smiles``, ``inchikey`` and the
scaffold fields are never used for scoring. The scaffold is read only to enforce the
one-case-per-scaffold rule and the offline exclusions.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from src.memory.case import MemoryCase
from src.memory.features import (
    Features,
    case_features,
    dou_closeness,
    element_closeness,
    peak_closeness,
    solvent_match,
)
from src.memory.store import MemoryStore

# Float slack so a score exactly tie_epsilon below the best still counts as "within" it.
_TIE_SLACK = 1e-9


@dataclass(frozen=True)
class RetrievalParams:
    """All retrieval parameters. PROVISIONAL defaults: phase M4 tunes and freezes them."""

    weight_c13: float = 0.35
    weight_h1: float = 0.30
    weight_elements: float = 0.20
    weight_dou: float = 0.15
    solvent_bonus: float = 0.03
    c13_tolerance_ppm: float = 2.0
    h1_tolerance_ppm: float = 0.15
    min_similarity: float = 0.0
    tie_epsilon: float = 0.02
    k: int = 3

    def __post_init__(self) -> None:
        weights = (self.weight_c13, self.weight_h1, self.weight_elements, self.weight_dou)
        if any(w < 0 for w in weights) or self.solvent_bonus < 0:
            raise ValueError("weights and solvent_bonus must be non-negative")
        if self.c13_tolerance_ppm <= 0 or self.h1_tolerance_ppm <= 0:
            raise ValueError("tolerances must be positive")
        if self.k < 0 or self.tie_epsilon < 0:
            raise ValueError("k and tie_epsilon must be non-negative")


@dataclass(frozen=True)
class RetrievalResult:
    """One selected case. ``similarity`` is None for random selection (similarity is ignored)."""

    case: MemoryCase
    similarity: float | None
    closeness: dict[str, float] = field(default_factory=dict)  # available features only
    solvent_matched: bool = False


CostFn = Callable[[MemoryCase], int | float]


def score_case(
    query: Features, case: MemoryCase, params: RetrievalParams
) -> tuple[float, dict[str, float], bool]:
    """Similarity of one case to the query: (similarity, per-feature closeness, solvent matched).

    similarity = sum(weight * closeness) / sum(weight) over the features the query has, plus the
    solvent bonus when both solvents are known and equal, capped at 1.0. 0 if no feature is available.
    """
    cf = case_features(case)
    closeness: dict[str, float] = {}
    weights: dict[str, float] = {}
    if query.c_shifts:
        closeness["c13"] = peak_closeness(query.c_shifts, cf.c_shifts, params.c13_tolerance_ppm)
        weights["c13"] = params.weight_c13
    if query.h_shifts:
        closeness["h1"] = peak_closeness(query.h_shifts, cf.h_shifts, params.h1_tolerance_ppm)
        weights["h1"] = params.weight_h1
    if query.elements is not None and cf.elements is not None:
        closeness["elements"] = element_closeness(query.elements, cf.elements)
        weights["elements"] = params.weight_elements
    if query.dou is not None and cf.dou is not None:
        closeness["dou"] = dou_closeness(query.dou, cf.dou)
        weights["dou"] = params.weight_dou
    if not closeness:
        return 0.0, closeness, False
    matched = solvent_match(query.solvent, cf.solvent)
    total_weight = sum(weights.values())
    base = sum(weights[n] * closeness[n] for n in closeness) / total_weight if total_weight > 0 else 0.0
    bonus = params.solvent_bonus if matched else 0.0
    return min(1.0, base + bonus), closeness, matched


def _allowed(
    cases: Iterable[MemoryCase],
    exclude_case_ids: Iterable[str] | None,
    exclude_scaffolds: Iterable[str] | None,
) -> list[MemoryCase]:
    ids = frozenset(exclude_case_ids or ())
    scaffolds = frozenset(exclude_scaffolds or ())
    return [c for c in cases if c.case_id not in ids and c.murcko_scaffold not in scaffolds]


def _fits(case: MemoryCase, spent: float, budget: float | None, cost_fn: CostFn | None) -> tuple[bool, float]:
    if budget is None:
        return True, 0.0
    if cost_fn is None:
        raise ValueError("token_budget requires a cost_fn")
    cost = float(cost_fn(case))
    return spent + cost <= budget, cost


def retrieve(
    query: Features,
    store: MemoryStore,
    params: RetrievalParams | None = None,
    *,
    token_budget: int | None = None,
    cost_fn: CostFn | None = None,
    exclude_case_ids: Iterable[str] | None = None,
    exclude_scaffolds: Iterable[str] | None = None,
) -> list[RetrievalResult]:
    """Select up to ``params.k`` cases for a query, best first.

    Every case is scored; those below ``min_similarity`` are dropped. Slots are filled one at a time:
    with s* the best remaining score, among remaining cases scoring >= s* - tie_epsilon a ``missed``
    case is preferred (highest score among them), else the highest score; ties break on case_id
    ascending. A case whose Murcko scaffold is already chosen (the empty acyclic scaffold counts as one
    scaffold), or whose ``cost_fn`` estimate does not fit in the remaining ``token_budget``, is skipped
    (never truncated) and the next is considered. With no budget only k limits the count.

    ``exclude_case_ids`` and ``exclude_scaffolds`` are OFFLINE-ONLY (the phase M4 leave-scaffold-out
    check). The service must never pass a query structure's scaffold: it has no query structure, and
    doing so would leak the answer.
    """
    params = params or RetrievalParams()
    scored = []
    for case in _allowed(store.cases, exclude_case_ids, exclude_scaffolds):
        sim, closeness, matched = score_case(query, case, params)
        if sim >= params.min_similarity:
            scored.append(RetrievalResult(case, sim, closeness, matched))
    remaining = sorted(scored, key=lambda r: (-(r.similarity or 0.0), r.case.case_id))

    chosen: list[RetrievalResult] = []
    scaffolds: set[str] = set()
    spent = 0.0
    while len(chosen) < params.k:
        # Cases that can no longer be chosen are skipped before s* is taken, so the tie window is
        # anchored on a case that could actually be selected.
        remaining = [
            r
            for r in remaining
            if r.case.murcko_scaffold not in scaffolds and _fits(r.case, spent, token_budget, cost_fn)[0]
        ]
        if not remaining:
            break
        best = remaining[0].similarity or 0.0
        near = [r for r in remaining if (r.similarity or 0.0) >= best - params.tie_epsilon - _TIE_SLACK]
        missed = [r for r in near if r.case.agent_outcome.status == "missed"]
        pick = (missed or near)[0]  # remaining is ordered by (-similarity, case_id)
        remaining.remove(pick)
        chosen.append(pick)
        scaffolds.add(pick.case.murcko_scaffold)
        spent += _fits(pick.case, spent, token_budget, cost_fn)[1]
    return chosen


def select_random(
    store: MemoryStore,
    seed: int,
    params: RetrievalParams | None = None,
    *,
    token_budget: int | None = None,
    cost_fn: CostFn | None = None,
    exclude_case_ids: Iterable[str] | None = None,
    exclude_scaffolds: Iterable[str] | None = None,
) -> list[RetrievalResult]:
    """Stage-1 control arm (spec §7.3, R200): up to ``params.k`` cases uniformly at random.

    Uses ``random.Random(seed)`` over the cases in case_id order, so it is deterministic for a seed and
    store. Similarity is ignored (reported as None); the one-per-scaffold rule, the budget and the
    exclusions apply as in ``retrieve``. The same offline-only warning on exclusions applies.
    """
    params = params or RetrievalParams()
    pool = sorted(_allowed(store.cases, exclude_case_ids, exclude_scaffolds), key=lambda c: c.case_id)
    random.Random(seed).shuffle(pool)
    chosen: list[RetrievalResult] = []
    scaffolds: set[str] = set()
    spent = 0.0
    for case in pool:
        if len(chosen) >= params.k:
            break
        if case.murcko_scaffold in scaffolds:
            continue
        ok, cost = _fits(case, spent, token_budget, cost_fn)
        if not ok:
            continue
        chosen.append(RetrievalResult(case, None))
        scaffolds.add(case.murcko_scaffold)
        spent += cost
    return chosen
