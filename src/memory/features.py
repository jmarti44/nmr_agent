"""Retrieval features (spec §5.1-5.2): built from a query (peak table + context) or from a stored case.

Only spectrum-side information enters a feature object. Nothing from a structure (SMILES, scaffold,
InChIKey) is ever read here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.memory.case import MemoryCase
from src.schemas import (
    Nucleus,
    PeakTable,
    SampleContext,
    Solvent,
    degree_of_unsaturation,
    parse_formula,
)

# Cost of a pair outside the tolerance. Far above any sum of in-tolerance costs, so the assignment
# first maximises the number of in-tolerance pairs, then minimises their total |delta|.
_FORBIDDEN = 1e6


@dataclass(frozen=True)
class Features:
    """Comparable features of one spectrum. ``None`` means "not available" (formula-derived features)."""

    c_shifts: tuple[float, ...]
    h_shifts: tuple[float, ...]
    elements: dict[str, int] | None
    dou: float | None
    solvent: Solvent | None


def query_features(table: PeakTable, context: SampleContext) -> Features:
    """Features of a query: analyte-role peaks only; formula features only when a formula was given."""
    elements: dict[str, int] | None = None
    dou: float | None = None
    if context.molecular_formula:
        elements = parse_formula(context.molecular_formula)
        dou = degree_of_unsaturation(elements)
    return Features(
        c_shifts=tuple(sorted(p.shift_ppm for p in table.analyte_peaks(Nucleus.C13))),
        h_shifts=tuple(sorted(p.shift_ppm for p in table.analyte_peaks(Nucleus.H1))),
        elements=elements,
        dou=dou,
        solvent=context.solvent if context.solvent is not None else table.solvent,
    )


def case_features(case: MemoryCase) -> Features:
    """Features of a stored case: c_peaks, h_peaks shifts, formula, dou, solvent. No structure fields."""
    return Features(
        c_shifts=tuple(sorted(case.c_peaks)),
        h_shifts=tuple(sorted(p.shift_ppm for p in case.h_peaks)),
        elements=parse_formula(case.formula),
        dou=case.dou,
        solvent=case.solvent,
    )


def peak_closeness(query: tuple[float, ...], case: tuple[float, ...], tolerance: float) -> float:
    """One-to-one matching of two shift lists, in [0, 1].

    Optimal assignment (``linear_sum_assignment``) on cost |dq - dc|; pairs further apart than
    ``tolerance`` cannot match. Each matched pair contributes ``1 - |delta| / tolerance``; the score is
    the sum of contributions divided by the longer list's length. 0 if either list is empty.
    """
    if not query or not case:
        return 0.0
    q = np.asarray(query, dtype=float)
    c = np.asarray(case, dtype=float)
    delta = np.abs(q[:, None] - c[None, :])
    cost = np.where(delta <= tolerance, delta, _FORBIDDEN)
    rows, cols = linear_sum_assignment(cost)
    matched = delta[rows, cols]
    matched = matched[matched <= tolerance]
    total = float(np.sum(1.0 - matched / tolerance))
    return min(1.0, max(0.0, total / max(len(q), len(c))))


def element_closeness(query: dict[str, int], case: dict[str, int]) -> float:
    """1 - sum|q_e - c_e| / sum max(q_e, c_e) over the union of elements present in either formula."""
    elements = set(query) | set(case)
    denom = sum(max(query.get(e, 0), case.get(e, 0)) for e in elements)
    if denom == 0:
        return 1.0
    diff = sum(abs(query.get(e, 0) - case.get(e, 0)) for e in elements)
    return 1.0 - diff / denom


def dou_closeness(query: float, case: float) -> float:
    """1 / (1 + |q - c|)."""
    return 1.0 / (1.0 + abs(query - case))


def solvent_match(query: Solvent | None, case: Solvent | None) -> bool:
    """True only when both solvents are known and equal. ``other`` is not a specific solvent: no match."""
    if query is None or case is None:
        return False
    if query == Solvent.OTHER or case == Solvent.OTHER:
        return False
    return query == case
