"""Raw Bruker FID -> processed 1H spectrum. Pure functions over NumPy arrays; no network."""

from __future__ import annotations

import nmrglue as ng
import numpy as np

# proc_bl.med takes its window (mw) and smoothing (sf) in points, so they are defined in Hz here and
# converted using the spectrum resolution. 15 Hz is wider than a multiplet; 3.5 Hz (SPEC §5.2's mw=24 at
# 0.147 Hz/pt) subtracted real signal and left negative lobes (review R0-1).
BASELINE_MW_HZ = 15.0
BASELINE_SF_HZ = 2.4


def process_bruker_fid(
    exp_dir: str,
    lb_hz: float = 0.3,
    baseline_mw_hz: float = BASELINE_MW_HZ,
    baseline_sf_hz: float = BASELINE_SF_HZ,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Raw Bruker experiment directory -> (spectrum_real, ppm_axis, meta). ppm axis is descending."""
    dic, fid = ng.bruker.read(exp_dir)
    fid = ng.bruker.remove_digital_filter(dic, fid)

    sw_hz = float(dic["acqus"]["SW_h"])
    # nmrglue's em() takes lb as a fraction of the sweep width, not Hz.
    fid = ng.proc_base.em(fid, lb=lb_hz / sw_hz)

    fid = ng.proc_base.zf_size(fid, max(32768, 2 * fid.shape[-1]))
    fid = ng.proc_base.zf_auto(fid)
    spec = ng.proc_base.fft(fid)
    spec, (p0, p1) = ng.proc_autophase.autops(spec, "acme", return_phases=True, disp=False)

    spec = ng.proc_base.di(spec)
    # Bruker data are mirrored after fft; without rev every peak appears reflected about the carrier.
    spec = ng.proc_base.rev(spec)
    hz_per_pt = sw_hz / spec.size
    mw = max(1, round(baseline_mw_hz / hz_per_pt))
    sf = max(1, round(baseline_sf_hz / hz_per_pt))
    spec = ng.proc_bl.med(spec, mw=mw, sf=sf, sigma=5.0)

    udic = ng.bruker.guess_udic(dic, spec)
    uc = ng.fileiobase.uc_from_udic(udic)
    ppm = uc.ppm_scale()

    meta = {
        "sfo1_mhz": float(dic["acqus"]["SFO1"]),
        "sw_hz": sw_hz,
        "nucleus": dic["acqus"].get("NUC1", "1H"),
        "solvent": dic["acqus"].get("SOLVENT"),
        "ns": dic["acqus"].get("NS"),
        "p0_deg": float(p0),
        "p1_deg": float(p1),
        "referenced": "procs" in dic,
    }
    return spec, ppm, meta
