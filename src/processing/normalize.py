"""Peak-list normalization (SPEC §3.2 step 4, §5.5): tag, order, number and scale a supplied peak list.

Pure functions, no chemistry toolkit and no network. The result is the PeakTable that both the prompt builder
and the verifier read, so the model and the checker always see the same numbers.
"""

from __future__ import annotations

import itertools

from src.schemas import (
    Multiplicity,
    Nucleus,
    Peak,
    PeakListInput,
    PeakRole,
    PeakTable,
    SampleContext,
    Solvent,
    degree_of_unsaturation,
)
from src.verify.solvent_tables import tag_peaks

# Xi(13C) / Xi(1H) = 25.145020 % (IUPAC), the usual ratio between the 13C and 1H observe frequencies.
C13_TO_H1_RATIO = 0.25145020
# A broad 1H signal downfield of this is almost always COOH / phenol / NH; aldehydes are sharp.
BROAD_EXCHANGEABLE_MIN_PPM = 9.5
_EXCHANGING_SOLVENTS = (Solvent.D2O, Solvent.CD3OD)
_MAX_LOGGED_ID_CHANGES = 10
# A residual-solvent or water signal is a small fraction of a proton on the analyte's scale, so only a tagged
# peak at least this large can be a hidden analyte signal and is considered for restoration or logged.
RESIDUAL_MAX_PROTONS = 0.5
_MAX_RESTORE_CANDIDATES = 4  # subsets are tried exhaustively, so keep the search small


def _formula_h(context: SampleContext) -> int | None:
    counts = context.formula_counts
    return counts.get("H", 0) if counts else None


def formula_summary(context: SampleContext) -> tuple[int, float] | None:
    """(H count, degree of unsaturation) of the supplied formula, if any."""
    counts = context.formula_counts
    return (counts.get("H", 0), degree_of_unsaturation(counts)) if counts else None


def _mark_broad_downfield_as_exchangeable(peaks: list[Peak], log: list[str]) -> list[Peak]:
    out = []
    for p in peaks:
        if (
            p.nucleus == Nucleus.H1
            and p.role == PeakRole.ANALYTE
            and p.multiplicity == Multiplicity.BR
            and p.shift_ppm >= BROAD_EXCHANGEABLE_MIN_PPM
            and not p.exchangeable_hint
        ):
            p = p.model_copy(update={"exchangeable_hint": True})
            limit = f"{BROAD_EXCHANGEABLE_MIN_PPM:g}"
            log.append(f"1H {p.shift_ppm:g} ppm is broad and downfield of {limit}: marked exchangeable")
        out.append(p)
    return out


def _count_like(values: list[float], tol: float = 0.15) -> bool:
    """True if every value is within tol of a whole or half number, i.e. it looks like a proton count."""
    return all(abs(v - round(v * 2) / 2) <= tol for v in values)


def _formula_window(context: SampleContext) -> tuple[float, float] | None:
    """Range the analyte 1H integral sum may take for the formula: H minus the O/N/S atoms that could carry an
    unlisted exchangeable proton, up to H plus a tolerance."""
    h = _formula_h(context)
    if h is None:
        return None
    counts = context.formula_counts or {}
    hetero = sum(counts.get(e, 0) for e in ("O", "N", "S"))
    tol = max(0.5, 0.1 * h)
    return h - hetero - tol, h + tol


def _fits_formula(values: list[float], context: SampleContext) -> bool:
    """Do these integrals look like counts that add up to the formula's H (allowing unlisted OH/NH)?"""
    window = _formula_window(context)
    return window is not None and _count_like(values) and window[0] <= sum(values) <= window[1]


def _scale(peaks: list[Peak], factor: float, rounding: float | None = None) -> list[Peak]:
    out = []
    for p in peaks:
        if p.nucleus == Nucleus.H1 and p.role == PeakRole.ANALYTE and p.integral is not None:
            value = p.integral * factor
            if rounding:
                value = round(value / rounding) * rounding
            p = p.model_copy(update={"integral": round(value, 3)})
        out.append(p)
    return out


def normalize_integrals(
    peaks: list[Peak], context: SampleContext, log: list[str]
) -> tuple[list[Peak], str, float]:
    """Scale analyte 1H integrals to proton counts; returns (peaks, mode, factor applied).

    mode is 'formula_H', 'smallest_peak' or 'none'. Only analyte 1H peaks are scaled; solvent, water,
    reference and impurity-tagged peaks keep what was supplied.

    With a formula: integrals are kept only if they already look like proton counts (each within 0.15 of a
    whole or half number) and their sum lies between H minus the number of O/N/S atoms (each could carry an
    unlisted exchangeable proton) and H plus a tolerance. Anything else is treated as arbitrary units and
    scaled so the sum equals H, which assumes every proton is listed. Without a formula the smallest
    multiplet is set to 1 and values are rounded to the nearest 0.5.
    """
    analyte = [p for p in peaks if p.nucleus == Nucleus.H1 and p.role == PeakRole.ANALYTE]
    if not analyte:
        return peaks, "none", 1.0
    if any(p.integral is None for p in analyte):
        log.append("integrals not normalised: not every analyte 1H peak has one")
        return peaks, "none", 1.0
    total = sum(p.integral for p in analyte)  # type: ignore[misc]
    if total <= 0:
        log.append("integrals not normalised: they sum to zero")
        return peaks, "none", 1.0

    h = _formula_h(context)
    if h is not None:
        lo, hi = _formula_window(context)  # type: ignore[misc]
        values = [p.integral for p in analyte]  # type: ignore[misc]
        in_window = lo <= total <= hi
        if in_window and _count_like(values):
            log.append(
                f"integrals sum to {total:g} and look like proton counts, consistent with H = {h}: kept"
            )
            return peaks, "formula_H", 1.0
        if in_window:
            log.append(
                "integral sum fits H but values are not near whole or half numbers: "
                "treated as arbitrary units"
            )
        factor = h / total
        exchanging = context.d2o_exchange_performed or context.solvent in _EXCHANGING_SOLVENTS
        log.append(
            f"integrals summed to {total:g}; scaled by {factor:.4g} so the analyte 1H total equals H = {h}"
        )
        if exchanging:
            log.append(
                "D2O/exchange: exchangeable protons may be missing, so scaled integrals can be inflated"
            )
        return _scale(peaks, factor), "formula_H", factor

    smallest = min(p.integral for p in analyte if p.integral > 0)  # type: ignore[type-var]
    factor = 1.0 if 0.9 <= smallest <= 1.1 else 1.0 / smallest
    log.append(
        f"no formula: smallest multiplet ({smallest:g}) set to 1 (factor {factor:.4g}); rounded to 0.5"
    )
    return _scale(peaks, factor, rounding=0.5), "smallest_peak", factor


def _restore_by_formula_fit(peaks: list[Peak], context: SampleContext, log: list[str]) -> list[Peak]:
    """The only rule that gives a solvent- or water-tagged peak back to the analyte.

    Applies when the integrals are counts and a formula is known. If the analyte integrals alone already fit
    the formula, nothing is restored (the tag stays; the prompt lets the model use the peak if it needs it).
    Otherwise the smallest set of tagged peaks, each integrating to at least RESIDUAL_MAX_PROTONS, that makes
    them fit is restored with its integrals as supplied: a phenyl multiplet at 7.27 in CDCl3, or a CH2 at the
    water position. HOD in D2O or CD3OD is never restored, since that peak is mostly solvent. If no set fits,
    nothing changes here and normalization falls back to rescaling.
    """
    if _formula_window(context) is None:
        return peaks
    analyte = [p for p in peaks if p.nucleus == Nucleus.H1 and p.role == PeakRole.ANALYTE]
    if not analyte or any(p.integral is None for p in analyte):
        return peaks
    base = [p.integral for p in analyte]  # type: ignore[misc]
    if _fits_formula(base, context):
        return peaks
    never_water = context.solvent in _EXCHANGING_SOLVENTS
    candidates = [
        i
        for i, p in enumerate(peaks)
        if p.nucleus == Nucleus.H1
        and p.integral is not None
        and p.integral >= RESIDUAL_MAX_PROTONS
        and (p.role == PeakRole.SOLVENT or (p.role == PeakRole.WATER and not never_water))
    ][:_MAX_RESTORE_CANDIDATES]
    h = _formula_h(context) or 0
    for size in range(1, len(candidates) + 1):
        fitting = [
            subset
            for subset in itertools.combinations(candidates, size)
            if _fits_formula(base + [peaks[i].integral for i in subset], context)  # type: ignore[misc]
        ]
        if fitting:
            best = min(fitting, key=lambda sub: abs(sum(base) + sum(peaks[i].integral for i in sub) - h))  # type: ignore[misc]
            out = list(peaks)
            for i in best:
                q = out[i]
                log.append(
                    f"1H {q.shift_ppm:g} ppm is at the {q.role.value} position, but with it the integrals "
                    f"fit H = {h} and without it they do not: restored as analyte"
                )
                out[i] = q.model_copy(update={"role": PeakRole.ANALYTE, "label": None})
            return out
    return peaks


def _log_unresolved_solvent_positions(peaks: list[Peak], factor: float, log: list[str]) -> None:
    """Say so when a large solvent- or water-tagged peak stays tagged: it may hide an analyte signal."""
    for p in peaks:
        if (
            p.role in (PeakRole.SOLVENT, PeakRole.WATER)
            and p.nucleus == Nucleus.H1
            and p.integral is not None
        ):
            if p.integral * factor >= RESIDUAL_MAX_PROTONS:
                log.append(
                    f"1H {p.shift_ppm:g} ppm is tagged {p.role.value} but would be "
                    f"{p.integral * factor:.2f} H; left tagged (it may overlap an analyte signal)"
                )


def _number_peaks(peaks: list[Peak], log: list[str]) -> list[Peak]:
    """Order 1H then 13C, each downfield to upfield, and number 1..N."""
    ordered = sorted(peaks, key=lambda p: (p.nucleus != Nucleus.H1, -p.shift_ppm, p.id))
    changed = [(p.id, i) for i, p in enumerate(ordered, start=1) if p.id != i]
    if changed:
        shown = ", ".join(f"{a}->{b}" for a, b in changed[:_MAX_LOGGED_ID_CHANGES])
        more = (
            f" (+{len(changed) - _MAX_LOGGED_ID_CHANGES} more)"
            if len(changed) > _MAX_LOGGED_ID_CHANGES
            else ""
        )
        log.append(f"peak ids renumbered downfield to upfield, 1H before 13C: {shown}{more}")
    return [p.model_copy(update={"id": i}) for i, p in enumerate(ordered, start=1)]


def normalize_peaklist(
    peaks: PeakListInput | list[Peak],
    context: SampleContext | None = None,
    referencing: str = "as_supplied",
    tag_impurities: bool = False,
) -> PeakTable:
    """Supplied peak list -> normalized, solvent-tagged PeakTable. Nothing is dropped, only tagged.

    Common-impurity tagging is off by default: chemists curate peak lists, and on real CDCl3 spectra the
    impurity table mislabels about 30% of genuine 1H peaks. It is meant for the raw-spectrum path, where
    impurity signals are real and small.
    """
    context = context or SampleContext()
    plist = peaks.peaks if isinstance(peaks, PeakListInput) else list(peaks)
    log: list[str] = []

    tagged = tag_peaks(plist, context.solvent, impurities=tag_impurities)
    for old, new in zip(plist, tagged, strict=True):
        if new.role != old.role:
            log.append(f"{old.nucleus.value} {old.shift_ppm:g} ppm tagged {new.role.value}: {new.label}")
    tagged = _mark_broad_downfield_as_exchangeable(tagged, log)
    # Decide from the supplied integrals which tagged peaks are really analyte, then normalize once.
    tagged = _restore_by_formula_fit(list(tagged), context, log)
    tagged, mode, factor = normalize_integrals(tagged, context, log)
    if mode != "none":
        _log_unresolved_solvent_positions(tagged, factor, log)
    if not any(p.nucleus == Nucleus.H1 and p.role == PeakRole.ANALYTE for p in tagged):
        log.append(
            "no analyte 1H peaks remain after tagging: "
            "every 1H peak is at a solvent, water or reference position"
        )
    numbered = _number_peaks(tagged, log)

    freqs: dict[str, float] = {}
    if context.spectrometer_mhz:
        freqs = {"1H": context.spectrometer_mhz, "13C": round(context.spectrometer_mhz * C13_TO_H1_RATIO, 2)}
    return PeakTable(
        peaks=numbered,
        nucleus_frequencies_mhz=freqs,
        solvent=context.solvent,
        referencing=referencing,
        integral_normalization=mode,
        processing_log=log,
    )
