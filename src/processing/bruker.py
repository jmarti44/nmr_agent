"""Raw Bruker FID -> processed 1H spectrum. Pure functions over NumPy arrays; no network."""

from __future__ import annotations

import warnings
from pathlib import Path

import nmrglue as ng
import numpy as np

from src.ingest.errors import IngestError
from src.ingest.safety import MAX_FID_POINTS, check_data_file_sizes

# proc_bl.med takes its window (mw) and smoothing (sf) in points, so they are defined in Hz here and
# converted using the spectrum resolution. 15 Hz is wider than a multiplet; 3.5 Hz (SPEC §5.2's mw=24 at
# 0.147 Hz/pt) subtracted real signal and left negative lobes.
BASELINE_MW_HZ = 15.0
BASELINE_SF_HZ = 2.4


def process_bruker_fid(
    exp_dir: str,
    lb_hz: float = 0.3,
    baseline_mw_hz: float = BASELINE_MW_HZ,
    baseline_sf_hz: float = BASELINE_SF_HZ,
    autophase: str = "acme",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Raw Bruker experiment directory -> (spectrum_real, ppm_axis, meta). ppm axis is descending."""
    check_data_file_sizes(Path(exp_dir))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dic, fid = ng.bruker.read(exp_dir)
            fid = ng.bruker.remove_digital_filter(dic, fid)
    except Exception as e:  # nmrglue raises assorted errors on truncated or odd files
        raise IngestError(f"could not read the Bruker FID: {type(e).__name__}") from e
    if fid.ndim != 1:
        raise IngestError("the experiment is multidimensional; only 1D data is supported")
    if fid.size < 64 or fid.size > MAX_FID_POINTS or not np.all(np.isfinite(fid)):
        raise IngestError(f"the FID has an unusable number of points ({fid.size})")

    sw_hz = float(dic["acqus"]["SW_h"])
    # nmrglue's em() takes lb as a fraction of the sweep width, not Hz.
    fid = ng.proc_base.em(fid, lb=lb_hz / sw_hz)

    fid = ng.proc_base.zf_size(fid, max(32768, 2 * fid.shape[-1]))
    fid = ng.proc_base.zf_auto(fid)
    spec = ng.proc_base.fft(fid)
    spec, (p0, p1), phase_note = _autophase(spec, autophase)

    spec = ng.proc_base.di(spec)
    # Bruker data are mirrored after fft; without rev every peak appears reflected about the carrier.
    spec = ng.proc_base.rev(spec)
    hz_per_pt = sw_hz / spec.size
    mw = max(1, round(baseline_mw_hz / hz_per_pt))
    sf = max(1, round(baseline_sf_hz / hz_per_pt))
    spec = ng.proc_bl.med(spec, mw=mw, sf=sf, sigma=5.0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "referencing was not corrected for sr": handled by auto-referencing
        udic = ng.bruker.guess_udic(dic, spec)
    ppm = ng.fileiobase.uc_from_udic(udic).ppm_scale()

    meta = {
        "sfo1_mhz": float(dic["acqus"]["SFO1"]),
        "sw_hz": sw_hz,
        "nucleus": dic["acqus"].get("NUC1", "1H"),
        "solvent": dic["acqus"].get("SOLVENT"),
        "ns": dic["acqus"].get("NS"),
        "p0_deg": float(p0),
        "p1_deg": float(p1),
        "phase_note": phase_note,
        "referenced": "procs" in dic,
    }
    return spec, ppm, meta


def _autophase(spec: np.ndarray, method: str) -> tuple[np.ndarray, tuple[float, float], str | None]:
    """Phase with `method`; fall back to peak_minima, then to no correction, when a fit fails or is NaN."""
    order = [method] + [m for m in ("acme", "peak_minima") if m != method]
    for m in order:
        try:
            out, (p0, p1) = ng.proc_autophase.autops(spec, m, return_phases=True, disp=False)
        except Exception:  # noqa: BLE001 - the optimizer can fail on pathological spectra
            continue
        if np.all(np.isfinite(out)) and np.all(np.isfinite([p0, p1])):
            return out, (p0, p1), None if m == method else f"{method} failed; phased with {m}"
    return spec, (0.0, 0.0), "autophase failed; spectrum left unphased"


def read_bruker_pdata(exp_dir: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """TopSpin-processed spectrum (pdata/N/1r + procs): already phased, baseline-corrected and referenced."""
    exp = Path(exp_dir)
    check_data_file_sizes(exp)
    pdirs = sorted(d for d in (exp / "pdata").glob("*") if (d / "1r").is_file() and (d / "procs").is_file())
    if not pdirs:
        raise IngestError("no processed spectrum (pdata/N/1r with procs) in the experiment")
    pdir = next((d for d in pdirs if d.name == "1"), pdirs[0])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dic, data = ng.bruker.read_pdata(str(pdir), scale_data=True)
            udic = ng.bruker.guess_udic(dic, data)
            uc = ng.fileiobase.uc_from_udic(udic)
    except Exception as e:
        raise IngestError(f"could not read the processed Bruker spectrum: {type(e).__name__}") from e
    spec = np.asarray(data, float)
    if spec.ndim != 1 or spec.size < 64 or not np.all(np.isfinite(spec)):
        raise IngestError("the processed spectrum is not a usable 1D array")
    acqus = _read_acqus(exp)
    meta = {
        "sfo1_mhz": float(dic["procs"].get("SF") or acqus.get("SFO1") or 0.0),
        "sw_hz": float(acqus.get("SW_h", 0.0)),
        "nucleus": acqus.get("NUC1", udic[0].get("label", "1H")),
        "solvent": acqus.get("SOLVENT"),
        "ns": acqus.get("NS"),
        "referenced": True,
    }
    return spec, uc.ppm_scale(), meta


def _read_acqus(exp: Path) -> dict:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ng.bruker.read_jcamp(str(exp / "acqus"))
    except Exception:  # noqa: BLE001
        return {}
