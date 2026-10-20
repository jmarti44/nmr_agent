"""Axis referencing for FID-only spectra (SPEC §5.5 step 5): anchor on a residual-solvent line or TMS."""

from __future__ import annotations

from dataclasses import dataclass

from src.schemas import Nucleus, Peak, Solvent
from src.verify.solvent_tables import SOLVENT_TABLE

# SPEC §5.5 allows +-0.3 ppm, but an analyte line within 0.3 ppm of a solvent position is common (DMSO-d6 at
# 2.50 vs a CH3-N or ArCH3 at 2.4), and anchoring on it would shift the whole axis by mistake. Real unapplied
# SR offsets are a few hundredths of a ppm (0.019 on the Wadpressine dataset), so 1H looks within +-0.1.
SEARCH_WINDOW_PPM = {Nucleus.H1: 0.1, Nucleus.C13: 1.0}
# The anchor must be tall: a small peak near 0 or at a solvent position is often an impurity.
MIN_ANCHOR_REL_HEIGHT = 0.02
# If TMS and the solvent line disagree by more than this after referencing, trust the solvent (TMS-position
# lines are often silyl groups or grease in a sample that has no TMS).
TMS_DISAGREEMENT_PPM = 0.02
LINE_TOL_HZ = 3.0

# 13C solvent lines: expected ppm -> (J_CD in Hz, number of deuterons). Carbon bearing n deuterons is a
# (2n+1)-line multiplet with binomial heights. A line with n = 0 is a singlet (carbonyl, nitrile).
_C13_MULTIPLETS: dict[Solvent, dict[float, tuple[float, int]]] = {
    Solvent.CDCL3: {77.2: (32.0, 1)},
    Solvent.DMSO_D6: {39.5: (21.0, 3)},
    Solvent.CD3OD: {49.2: (21.4, 3)},
    Solvent.ACETONE_D6: {206.7: (0.0, 0), 29.9: (19.5, 3)},
    Solvent.C6D6: {128.4: (24.3, 1)},
    Solvent.CD3CN: {118.7: (0.0, 0), 1.4: (21.0, 3)},
}


@dataclass
class Anchor:
    name: str
    expected_ppm: float
    observed_ppm: float
    note: str | None = None
    zero_line_rejected: bool = False  # a near-zero line was seen and shown not to be TMS

    @property
    def shift(self) -> float:
        return self.expected_ppm - self.observed_ppm


def _pattern(j_hz: float, n_d: int, mhz: float) -> list[tuple[float, float]]:
    """(offset in ppm, relative height) of the 2n+1 lines of a carbon with n deuterons."""
    if n_d == 0 or j_hz == 0.0 or mhz <= 0:
        return [(0.0, 1.0)]
    # Each spin-1 deuteron splits a line into three equal ones; the weights are the trinomial coefficients.
    weights = [1.0]
    for _ in range(n_d):
        weights = [
            sum(weights[x] for x in (k - 2, k - 1, k) if 0 <= x < len(weights))
            for k in range(len(weights) + 2)
        ]
    mid = len(weights) // 2
    return [((i - mid) * j_hz / mhz, w) for i, w in enumerate(weights)]


def _centre_of_multiplet(
    lines: list[Peak], expected_ppm: float, j_hz: float, n_d: int, mhz: float
) -> tuple[float, Peak] | None:
    """Observed centre of a solvent multiplet near `expected_ppm`: the line position whose expected pattern
    (centre and satellites) collects the most height. Using the tallest single line would pick an outer line
    of a 1:1:1 triplet about as often as the middle one."""
    if mhz <= 0:  # no frequency, so no J in ppm: centre of the strong lines in reach
        strong = [p for p in lines if (p.height or 0.0) >= 0.5 * max(q.height or 0.0 for q in lines)]
        centre = (min(p.shift_ppm for p in strong) + max(p.shift_ppm for p in strong)) / 2
        return centre, max(strong, key=lambda p: p.height or 0.0)
    pattern = _pattern(j_hz, n_d, mhz)
    tol = LINE_TOL_HZ / mhz
    w_centre = max(w for _, w in pattern)
    best: tuple[float, Peak] | None = None
    best_score = (0.0, 0.0)
    for cand in lines:
        # Treat `cand` as the centre line: the pattern then predicts every other line's height. Score the
        # fraction of the predicted height that is really there; ties go to the taller candidate.
        scale = (cand.height or 0.0) / w_centre
        explained = expected = 0.0
        for off, w in pattern:
            seen = max(
                (p.height or 0.0 for p in lines if abs(p.shift_ppm - (cand.shift_ppm + off)) <= tol),
                default=0.0,
            )
            explained += min(seen, w * scale)
            expected += w * scale
        score = (explained / expected if expected else 0.0, cand.height or 0.0)
        if score > best_score:
            best, best_score = (cand.shift_ppm, cand), score
    return best


def _solvent_anchors(
    mine: list[Peak], nucleus: Nucleus, solvent: Solvent, top: float, mhz: float
) -> list[Anchor]:
    """Candidate solvent anchors, best first. For 1H every tall line within the window is a candidate (the
    tallest first); for 13C each solvent line contributes the centre of its multiplet."""
    entry = SOLVENT_TABLE.get(solvent)
    if not entry:
        return []
    window = SEARCH_WINDOW_PPM[nucleus]
    out: list[Anchor] = []
    name = f"{solvent.value} residual"
    if nucleus == Nucleus.H1:
        for exp_ppm in entry["h"]:
            near = [
                p
                for p in mine
                if abs(p.shift_ppm - exp_ppm) <= window and (p.height or 0.0) >= MIN_ANCHOR_REL_HEIGHT * top
            ]
            out += [
                Anchor(name, exp_ppm, p.shift_ppm) for p in sorted(near, key=lambda p: -(p.height or 0.0))
            ]
        return out
    for exp_ppm in entry["c"]:
        j_hz, n_d = _C13_MULTIPLETS.get(solvent, {}).get(exp_ppm, (0.0, 0))
        # Outer lines can be further than `window` from the centre, so gather generously.
        reach = window + (3 * j_hz / mhz if mhz > 0 else 0.0)
        near = [
            p
            for p in mine
            if abs(p.shift_ppm - exp_ppm) <= reach and (p.height or 0.0) >= MIN_ANCHOR_REL_HEIGHT * top
        ]
        found = _centre_of_multiplet(near, exp_ppm, j_hz, n_d, mhz) if near else None
        if found and abs(found[0] - exp_ppm) <= window:
            out.append(Anchor(name, exp_ppm, found[0]))
    return out


def _tms_anchors(mine: list[Peak], top: float) -> list[Anchor]:
    """Every tall line near 0 ppm as a TMS candidate, tallest first."""
    window = SEARCH_WINDOW_PPM[Nucleus.H1]
    near = [
        p for p in mine if abs(p.shift_ppm) <= window and (p.height or 0.0) >= MIN_ANCHOR_REL_HEIGHT * top
    ]
    return [Anchor("TMS", 0.0, p.shift_ppm) for p in sorted(near, key=lambda p: -(p.height or 0.0))]


def find_anchor(
    peaks: list[Peak], nucleus: Nucleus, solvent: Solvent | None, mhz: float = 0.0
) -> Anchor | None:
    """The reference line, or None. `mhz` is the observe frequency of `nucleus` (needed for 13C multiplets).

    A known solvent line decides which line is the reference. TMS alone is used when there is no solvent
    line. If a TMS-position line agrees with the solvent line (within 0.02 ppm after referencing), TMS is
    returned because it is exact where the solvent table is only good to about 0.01 ppm. If it disagrees,
    the solvent wins and the TMS-position line stays analyte (a silyl group or grease).
    """
    mine = [p for p in peaks if p.nucleus == nucleus]
    if not mine:
        return None
    top = max(p.height or 0.0 for p in mine)
    candidates = _solvent_anchors(mine, nucleus, solvent, top, mhz) if solvent else []
    tms_all = _tms_anchors(mine, top) if nucleus == Nucleus.H1 else []
    if not candidates:
        return tms_all[0] if tms_all else None
    if tms_all:
        # A solvent candidate that puts a near-zero line at 0 confirms both; use that TMS line (exact).
        for tms in tms_all:
            if any(abs(tms.observed_ppm + c.shift) <= TMS_DISAGREEMENT_PPM for c in candidates):
                return tms
        tms, solv = tms_all[0], candidates[0]
        gap = tms.observed_ppm + solv.shift
        solv.zero_line_rejected = True
        solv.note = (
            f"line at {tms.observed_ppm:g} ppm would sit at {gap:+.3f} ppm, not 0, once referenced on "
            "the solvent; treated as analyte (silyl group or grease?), not as TMS"
        )
        return solv
    return candidates[0]
