"""Synthetic processed spectra and JCAMP-DX files for ingestion tests."""

from __future__ import annotations

import numpy as np

MHZ = 400.0


def lorentz(ppm: np.ndarray, centre: float, area: float, fwhm_hz: float = 1.0) -> np.ndarray:
    """Lorentzian line with the given area (in ppm x intensity) and FWHM in Hz."""
    g = fwhm_hz / MHZ / 2
    return area * g / np.pi / ((ppm - centre) ** 2 + g**2)


def multiplet(ppm: np.ndarray, centre: float, area: float, j_hz: float, n: int, fwhm_hz: float = 1.0):
    """First-order multiplet with n lines and binomial-free equal intensities; total area is `area`."""
    offsets = (np.arange(n) - (n - 1) / 2) * j_hz / MHZ
    return sum(lorentz(ppm, centre + o, area / n, fwhm_hz) for o in offsets)


def h1_spectrum(noise: float = 0.5, seed: int = 3, offset: float = 0.0, npts: int = 32768):
    """-> (spec, ppm). Descending ppm 14..-1 with TMS, a triplet, a doublet, a singlet and CDCl3."""
    ppm = np.linspace(14.0, -1.0, npts)
    spec = (
        lorentz(ppm, 0.0 + offset, 1.0)  # TMS (reference)
        + multiplet(ppm, 1.25 + offset, 3.0, 7.0, 3)  # CH3 triplet
        + multiplet(ppm, 4.10 + offset, 2.0, 7.0, 4)  # CH2 quartet
        + multiplet(ppm, 6.90 + offset, 2.0, 8.5, 2)  # aromatic doublet
        + lorentz(ppm, 8.30 + offset, 1.0)  # small singlet
        + lorentz(ppm, 7.26 + offset, 6.0, 1.2)  # CHCl3 residual
    )
    spec = spec * 4000.0
    rng = np.random.default_rng(seed)
    return spec + rng.normal(0, noise, ppm.size), ppm


def jcamp_text(
    spec: np.ndarray, ppm: np.ndarray, nucleus: str = "1H", units: str = "PPM", solvent="CDCl3"
) -> str:
    x = ppm * MHZ if units == "HZ" else ppm
    lines = [
        "##TITLE=synthetic",
        "##JCAMP-DX=5.00",
        "##DATA TYPE=NMR SPECTRUM",
        f"##.OBSERVE FREQUENCY={MHZ}",
        f"##.OBSERVE NUCLEUS=^{nucleus}",
        f"##.SOLVENT NAME={solvent}",
        f"##XUNITS={units}",
        "##YUNITS=ARBITRARY UNITS",
        "##XFACTOR=1",
        "##YFACTOR=1",
        f"##FIRSTX={x[0]:.6f}",
        f"##LASTX={x[-1]:.6f}",
        f"##NPOINTS={spec.size}",
        f"##FIRSTY={spec[0]:.6f}",
        "##XYDATA=(X++(Y..Y))",
    ]
    for i in range(0, spec.size, 8):
        lines.append(f"{x[i]:.6f} " + " ".join(f"{v:.4f}" for v in spec[i : i + 8]))
    lines.append("##END=")
    return "\n".join(lines) + "\n"


def peaktable_text(pairs: list[tuple[float, float]], units: str = "PPM") -> str:
    body = "\n".join(f"{x}, {y}" for x, y in pairs)
    return (
        f"##TITLE=peaks\n##JCAMP-DX=5.00\n##DATA TYPE=NMR PEAK TABLE\n##.OBSERVE FREQUENCY={MHZ}\n"
        f"##.OBSERVE NUCLEUS=^1H\n##XUNITS={units}\n##PEAK TABLE=(XY..XY)\n{body}\n##END=\n"
    )
