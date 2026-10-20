"""Processed spectrum -> peak list (SPEC §5.5): noise, peak picking, multiplets, integrals.

Pure NumPy/SciPy functions. Positions are in ppm on a descending axis; widths and couplings in Hz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_widths

from src.schemas import Multiplicity, Nucleus, Peak

MIN_FWHM_HZ = 0.3  # narrower lines are spikes or noise
MAX_J_GAP_HZ = 16.0  # adjacent lines further apart than this are separate signals
MAX_LINE_HEIGHT_RATIO = 6.0  # ...as are neighbours whose heights differ more than this
J_TOL_HZ = 0.6  # equal spacings within this are one coupling constant
_NOISE_WINDOWS = {"1H": ((11.5, 10.5), (15.0, 12.0)), "13C": ((230.0, 215.0),)}
_MAX_PEAKS = 200  # the PeakListInput limit
# Integration limits: this many FWHM either side of the outermost line, clipped at the neighbour.
# A Lorentzian line has 1/x^2 tails: +-8 FWHM holds about 96% of its area.
_INTEGRATION_PAD_FWHM = 8.0
_BROAD_FWHM_HZ = 12.0  # a lone line this wide is reported as broad


@dataclass
class RawPeak:
    index: int
    ppm: float
    height: float
    fwhm_hz: float


def noise_sigma(spec: np.ndarray, ppm: np.ndarray, nucleus: str) -> float:
    """1.4826 x MAD in a signal-free window; the whole-spectrum MAD when no such window is on the axis."""
    for hi, lo in _NOISE_WINDOWS[nucleus]:
        mask = (ppm <= hi) & (ppm >= lo)
        if mask.sum() >= 64:
            seg = spec[mask]
            break
    else:
        seg = spec
    sigma = 1.4826 * float(np.median(np.abs(seg - np.median(seg))))
    if not sigma > 0:
        raise ValueError("the spectrum has no measurable noise (flat or synthetic data)")
    return sigma


def pick_peaks(
    spec: np.ndarray, ppm: np.ndarray, hz_per_pt: float, sigma: float, k_sigma: float
) -> list[RawPeak]:
    thr = k_sigma * sigma
    idx, _ = find_peaks(spec, height=thr, prominence=thr)
    if idx.size == 0:
        return []
    widths = peak_widths(spec, idx, rel_height=0.5)[0] * hz_per_pt
    return [
        RawPeak(int(i), float(ppm[i]), float(spec[i]), float(w))
        for i, w in zip(idx, widths, strict=True)
        if w >= MIN_FWHM_HZ
    ]


def group_lines(peaks: list[RawPeak], sfo_mhz: float) -> list[list[RawPeak]]:
    """Chain neighbouring lines into candidate multiplets.

    Two adjacent lines belong together when they are within MAX_J_GAP_HZ and of comparable height. Lorentzian
    tails keep the signal above any noise-based level, so contiguity above the threshold cannot be used.
    """
    groups: list[list[RawPeak]] = []
    for p in sorted(peaks, key=lambda r: r.index):
        prev = groups[-1][-1] if groups else None
        if prev is not None:
            gap_hz = abs(p.ppm - prev.ppm) * sfo_mhz
            ratio = max(p.height, prev.height) / min(p.height, prev.height)
            if gap_hz <= MAX_J_GAP_HZ and ratio <= MAX_LINE_HEIGHT_RATIO:
                groups[-1].append(p)
                continue
        groups.append([p])
    return groups


def classify_multiplet(positions_hz: list[float]) -> tuple[Multiplicity, list[float]]:
    """Multiplicity and J list from sub-peak positions (Hz, ascending). Equal spacings -> d/t/q, else m."""
    n = len(positions_hz)
    if n == 1:
        return Multiplicity.S, []
    gaps = np.diff(sorted(positions_hz))
    equal = bool(np.ptp(gaps) <= J_TOL_HZ)
    j = [round(float(np.mean(gaps)), 1)] if equal else []
    if equal and n in (2, 3, 4):
        return {2: Multiplicity.D, 3: Multiplicity.T, 4: Multiplicity.Q}[n], j
    if n == 4 and abs(gaps[0] - gaps[2]) <= J_TOL_HZ:  # J1, J2, J1: doublet of doublets
        return Multiplicity.DD, sorted((round(float(gaps[0]), 1), round(float(gaps[1]), 1)), reverse=True)
    return Multiplicity.M, []


def peaks_from_spectrum(
    spec: np.ndarray, ppm: np.ndarray, sfo_mhz: float, nucleus: str, k_sigma: float = 6.0
) -> tuple[list[Peak], float, list[str]]:
    """Pick, group and (for 1H) integrate -> (peaks, noise sigma, notes for the processing log).

    Integrals are in arbitrary area units, to be normalized later. When more than 200 signals are found,
    the tallest 200 are kept and a note says how many were left out.
    """
    ppm = np.asarray(ppm, float)
    hz_per_pt = float(abs(ppm[0] - ppm[1])) * sfo_mhz
    sigma = noise_sigma(spec, ppm, nucleus)
    raw = pick_peaks(spec, ppm, hz_per_pt, sigma, k_sigma)
    notes: list[str] = []
    if not raw:
        return [], sigma, notes
    nuc = Nucleus(nucleus)
    if nuc == Nucleus.C13:
        if len(raw) > _MAX_PEAKS:
            notes.append(f"13C: {len(raw)} lines found; kept the tallest {_MAX_PEAKS}")
        raw = sorted(raw, key=lambda r: -r.height)[:_MAX_PEAKS]
        top = max(r.height for r in raw)
        peaks = [
            Peak(id=i, nucleus=nuc, shift_ppm=round(r.ppm, 3), multiplicity=Multiplicity.S,
                 height=round(r.height / top, 4), fwhm_hz=round(r.fwhm_hz, 2))
            for i, r in enumerate(sorted(raw, key=lambda r: -r.ppm), start=1)
        ]  # fmt: skip
        return peaks, sigma, notes

    groups = group_lines(raw, sfo_mhz)
    # Integration limits per group: pad the outer lines, never crossing the neighbouring group.
    bounds: list[tuple[int, int]] = []
    for g in groups:
        pad = int(round(_INTEGRATION_PAD_FWHM * max(r.fwhm_hz for r in g) / hz_per_pt))
        bounds.append((max(0, g[0].index - pad), min(spec.size - 1, g[-1].index + pad)))
    for k in range(len(groups) - 1):
        lo_b, hi_b = bounds[k][1], bounds[k + 1][0]
        if lo_b >= hi_b:
            mid = (groups[k][-1].index + groups[k + 1][0].index) // 2
            bounds[k] = (bounds[k][0], mid)
            bounds[k + 1] = (mid + 1, bounds[k + 1][1])
    step = abs(float(ppm[0] - ppm[1]))
    out: list[Peak] = []
    for g, (a, b) in zip(groups, bounds, strict=True):
        positions_hz = [-r.ppm * sfo_mhz for r in g]  # ascending Hz = downfield first
        mult, js = classify_multiplet(positions_hz)
        centre = float(np.mean([r.ppm for r in g])) if mult != Multiplicity.M else (g[0].ppm + g[-1].ppm) / 2
        fwhm = max(r.fwhm_hz for r in g)
        if mult == Multiplicity.S and fwhm >= _BROAD_FWHM_HZ:
            mult = Multiplicity.BR
        # Not clipped sample by sample (that biases noise upward); a negative total is a baseline artifact.
        area = max(0.0, float(np.sum(spec[a : b + 1]) * step))
        out.append(
            Peak(id=len(out) + 1, nucleus=nuc, shift_ppm=round(centre, 3), multiplicity=mult, coupling_hz=js,
                 integral=area, height=max(r.height for r in g), fwhm_hz=round(fwhm, 2))
        )  # fmt: skip
    if len(out) > _MAX_PEAKS:
        notes.append(f"1H: {len(out)} signals found; kept the tallest {_MAX_PEAKS}")
        out = sorted(out, key=lambda p: -(p.height or 0.0))[:_MAX_PEAKS]
    ordered = sorted(out, key=lambda p: -p.shift_ppm)
    return [p.model_copy(update={"id": i}) for i, p in enumerate(ordered, start=1)], sigma, notes
