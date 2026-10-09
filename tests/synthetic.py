"""Synthetic Bruker experiment builder shared by processing tests."""

from __future__ import annotations

from pathlib import Path

import nmrglue as ng
import numpy as np

BF1_MHZ = 400.13
SW_PPM = 20.0
CARRIER_PPM = 6.0
GRPDLY = 67.98
# (shift ppm, relative integral)
PEAKS: list[tuple[float, float]] = [
    (0.000, 0.5),
    (2.351, 2.0),
    (7.260, 3.0),
    (7.301, 3.0),
    (7.651, 1.0),
    (9.801, 1.0),
]


def write_synthetic_bruker(exp_dir: Path, n: int = 16384, noise: float = 0.01, seed: int = 1) -> Path:
    """Write a 1H Bruker experiment (fid + acqus) with a digital-filter group delay baked in."""
    sw_h = SW_PPM * BF1_MHZ
    o1 = CARRIER_PPM * BF1_MHZ
    t = np.arange(n) / sw_h
    fid = sum(a * np.exp(2j * np.pi * (d - CARRIER_PPM) * BF1_MHZ * t) * np.exp(-t / 0.5) for d, a in PEAKS)
    # Emulate the DSP group delay as a circular (fractional) shift, which remove_digital_filter undoes.
    k = np.fft.fftfreq(n) * n
    fid = np.fft.ifft(np.fft.fft(fid) * np.exp(-2j * np.pi * k * GRPDLY / n))
    rng = np.random.default_rng(seed)
    fid = fid + (rng.normal(size=n) + 1j * rng.normal(size=n)) * noise

    acqus = {
        "BYTORDA": 1, "DTYPA": 2, "AQ_mod": 3,  # AQ_mod=3 (DQD) makes nmrglue read the FID as complex
        "DECIM": 24, "DSPFVS": 20, "GRPDLY": GRPDLY,
        "SW_h": sw_h, "SW": SW_PPM, "SFO1": BF1_MHZ + o1 / 1e6, "BF1": BF1_MHZ, "O1": o1,
        "TD": 2 * n, "NUC1": "1H", "SOLVENT": "CDCl3", "NS": 16, "PARMODE": 0, "FW": 0,
        "_comments": ["$$ synthetic"], "_coreheader": ["##TITLE= synthetic", "##JCAMPDX= 5.0"],
    }  # fmt: skip
    exp_dir.mkdir(parents=True, exist_ok=True)
    ng.bruker.write(str(exp_dir), {"acqus": acqus}, fid, overwrite=True, write_prog=False)
    return exp_dir
