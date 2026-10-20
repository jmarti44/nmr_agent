"""Uploaded raw data -> supplied-style peak list (SPEC §5): the glue between ingest, processing and normalize.

The result is a `ProcessedInput`; `normalize_peaklist` then treats it like any other peak list, so the prompt
and the verifier see one table format whatever the input was.
"""

from __future__ import annotations

import statistics
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import structlog

from src.ingest.errors import IngestError
from src.ingest.jcamp import JcampSpectrum, read_jcamp
from src.ingest.safety import check_upload_size, extract_zip, find_experiments
from src.processing.bruker import process_bruker_fid, read_bruker_pdata
from src.processing.peaks import peaks_from_spectrum
from src.processing.referencing import find_anchor
from src.schemas import Nucleus, Peak, PeakRole, ProcessingOptions, SampleContext, Solvent
from src.verify.solvent_tables import classify_peak

log = structlog.get_logger("nmr_agent.pipeline")

# An impurity-position peak counts as an impurity only if it is this much smaller than the typical multiplet.
IMPURITY_MAX_REL_AREA = 0.25
C13_MIN_LB_HZ = 1.0
# Log lines that start with this are also returned as response warnings by /v1/analyze.
WARNING_PREFIX = "warning: "
UNVALIDATED_13C_FID = (
    f"{WARNING_PREFIX}13C peaks come from a raw FID; this path is not yet validated on a plain 13C "
    "experiment (only the processed-spectrum path is), so check the 13C list"
)
_SOLVENT_NAMES = {
    "cdcl3": Solvent.CDCL3,
    "dmso": Solvent.DMSO_D6,
    "dmso-d6": Solvent.DMSO_D6,
    "d2o": Solvent.D2O,
    "meod": Solvent.CD3OD,
    "cd3od": Solvent.CD3OD,
    "acetone": Solvent.ACETONE_D6,
    "acetone-d6": Solvent.ACETONE_D6,
    "c6d6": Solvent.C6D6,
    "cd3cn": Solvent.CD3CN,
}


@dataclass
class ProcessedInput:
    peaks: list[Peak]
    context: SampleContext
    referencing: str = "as_supplied"
    noise_sigma: dict[str, float] = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    tag_impurities: bool = False
    tag_reference: bool = True  # False when referencing showed the near-zero line is not TMS


def _solvent_from_header(name: str | None) -> Solvent | None:
    return _SOLVENT_NAMES.get((name or "").strip().lower())


def _nucleus_of(meta_nuc: str) -> str:
    n = str(meta_nuc).replace("^", "").strip().upper()
    if n in ("1H", "H1"):
        return "1H"
    if n in ("13C", "C13"):
        return "13C"
    raise IngestError(f"unsupported nucleus {meta_nuc!r}; only 1H and 13C are supported")


def _apply_shift(peaks: list[Peak], delta: float) -> list[Peak]:
    return [p.model_copy(update={"shift_ppm": round(p.shift_ppm + delta, 3)}) for p in peaks]


def _gate_impurities(peaks: list[Peak], solvent: Solvent | None, log: list[str]) -> list[Peak]:
    """Tag CDCl3 impurity positions, but only for 1H peaks much smaller than the typical multiplet.

    13C has no integrals to gate on, so 13C impurity positions stay analyte."""
    areas = [p.integral for p in peaks if p.nucleus == Nucleus.H1 and p.integral]
    typical = statistics.median(areas) if areas else None
    out = []
    for p in peaks:
        role, label = classify_peak(p, solvent, impurities=True)
        if role != PeakRole.IMPURITY_SUSPECT:
            out.append(p)
            continue
        small = p.nucleus == Nucleus.H1 and typical and (p.integral or 0.0) < IMPURITY_MAX_REL_AREA * typical
        if small:
            p = p.model_copy(update={"role": role, "label": label})
            log.append(f"1H {p.shift_ppm:g} ppm is small and at {label}: tagged impurity")
        out.append(p)
    return out


def _reference(
    peaks: list[Peak], nucleus: Nucleus, solvent: Solvent | None, mhz: float, log: list[str]
) -> tuple[list[Peak], str, bool]:
    """-> (shifted peaks, referencing label, whether the near-zero line was shown not to be TMS)."""
    anchor = find_anchor(peaks, nucleus, solvent, mhz)
    if anchor is None:
        log.append(f"{nucleus.value}: no TMS or residual-solvent line to reference on; axis left uncorrected")
        return peaks, "uncorrected", False
    log.append(
        f"{nucleus.value}: axis shifted by {anchor.shift:+.3f} ppm so {anchor.name} "
        f"({anchor.observed_ppm:g} -> {anchor.expected_ppm:g})"
    )
    if anchor.note:
        log.append(f"{nucleus.value}: {anchor.note}")
    return _apply_shift(peaks, anchor.shift), f"auto:{anchor.name}", anchor.zero_line_rejected


def _zero_line_rejected(
    peaks: list[Peak], nucleus: str, solvent: Solvent | None, mhz: float, log: list[str]
) -> bool:
    """For an already-referenced 1H spectrum: is the line near 0 ppm contradicted by the solvent line?"""
    if nucleus != "1H":
        return False
    anchor = find_anchor(peaks, Nucleus.H1, solvent, mhz)
    if anchor is None or not anchor.zero_line_rejected:
        return False
    log.append(f"1H: {anchor.note}")
    return True


def _spectrum_peaks(
    spec: np.ndarray, ppm: np.ndarray, mhz: float, nucleus: str, options: ProcessingOptions
) -> tuple[list[Peak], float, list[str]]:
    return peaks_from_spectrum(spec, ppm, mhz, nucleus, k_sigma=options.k_sigma)


def _process_experiment(exp: Path, options: ProcessingOptions) -> tuple[str, list[Peak], dict, list[str]]:
    """One Bruker experiment dir -> (nucleus, peaks, meta, log). Peaks are unreferenced for FID-only data."""
    log: list[str] = []
    has_pdata = any((d / "1r").is_file() for d in (exp / "pdata").glob("*"))
    if has_pdata and (options.prefer_pdata or not (exp / "fid").is_file()):
        spec, ppm, meta = read_bruker_pdata(str(exp))
        log.append(f"{exp.name}: used the TopSpin-processed spectrum (pdata)")
    else:
        nuc0 = _nucleus_of(_peek_nucleus(exp))
        lb = max(options.lb_hz, C13_MIN_LB_HZ) if nuc0 == "13C" else options.lb_hz
        spec, ppm, meta = process_bruker_fid(str(exp), lb_hz=lb, autophase=options.autophase)
        log.append(
            f"{exp.name}: processed the FID (lb {lb:g} Hz, p0 {meta['p0_deg']:.1f}, p1 {meta['p1_deg']:.1f})"
        )
        if meta.get("phase_note"):
            log.append(f"{exp.name}: {meta['phase_note']}")
    nucleus = _nucleus_of(meta["nucleus"])
    if nucleus == "13C" and "p0_deg" in meta:  # only the FID path reports its phase
        log.append(UNVALIDATED_13C_FID)
    peaks, sigma, notes = _spectrum_peaks(spec, ppm, meta["sfo1_mhz"], nucleus, options)
    meta["sigma"] = sigma
    log += [f"{exp.name}: {n}" for n in notes]
    return nucleus, peaks, meta, log


def _peek_nucleus(exp: Path) -> str:
    import nmrglue as ng

    try:
        return str(ng.bruker.read_jcamp(str(exp / "acqus")).get("NUC1", "1H"))
    except Exception as e:  # noqa: BLE001
        raise IngestError("could not read acqus") from e


def process_bruker_zip(data: bytes, context: SampleContext, options: ProcessingOptions) -> ProcessedInput:
    check_upload_size(data)
    with tempfile.TemporaryDirectory(prefix="nmr_") as tmp:
        root = Path(tmp)
        extract_zip(data, root)
        exps = find_experiments(root)
        if not exps:
            raise IngestError("no Bruker experiment (a directory with fid and acqus) found in the zip")
        if len(exps) > 2:
            names = ", ".join(str(e.relative_to(root)) for e in exps[:5])
            raise IngestError(
                f"the zip holds {len(exps)} experiments ({names}...); send one 1H and at most one 13C"
            )
        results = [_process_experiment(e, options) for e in exps]
    seen = [n for n, *_ in results]
    if len(set(seen)) != len(seen):
        raise IngestError(f"the zip holds two {seen[0]} experiments; send one per nucleus")
    return _combine(results, context)


def _combine(
    results: list[tuple[str, list[Peak], dict, list[str]]], context: SampleContext
) -> ProcessedInput:
    log: list[str] = []
    solvent = context.solvent
    h_meta = next((m for n, _, m, _ in results if n == "1H"), None)
    if solvent is None:
        found = next((s for _, _, m, _ in results if (s := _solvent_from_header(m.get("solvent")))), None)
        if found:
            solvent = found
            log.append(f"solvent {found.value} taken from the acquisition parameters")
    mhz = context.spectrometer_mhz or (h_meta["sfo1_mhz"] if h_meta else None)
    ctx = context.model_copy(update={"solvent": solvent, "spectrometer_mhz": mhz})
    peaks: list[Peak] = []
    refs: list[str] = []
    sigmas: dict[str, float] = {}
    tag_reference = True
    for nucleus, nuc_peaks, meta, exp_log in results:
        log += exp_log
        sigmas[nucleus] = meta["sigma"]
        if meta.get("referenced"):
            refs.append("procs")
            # Already referenced: do not move the axis, but still check whether a near-zero line is TMS.
            tag_reference &= not _zero_line_rejected(nuc_peaks, nucleus, solvent, meta["sfo1_mhz"], log)
        else:
            nuc_peaks, ref, rejected = _reference(nuc_peaks, Nucleus(nucleus), solvent, meta["sfo1_mhz"], log)
            refs.append(ref)
            tag_reference &= not (rejected and nucleus == "1H")
        peaks += nuc_peaks
    if not any(p.nucleus == Nucleus.H1 for p in peaks):
        raise IngestError("no 1H peaks were found; a 1H experiment is required")
    if "uncorrected" in refs:
        referencing = "uncorrected"
    elif all(r == "procs" for r in refs):
        referencing = "procs"
    else:
        referencing = next(r for r in refs if r.startswith("auto:"))
    peaks = _gate_impurities(peaks, solvent, log)
    return ProcessedInput(peaks, ctx, referencing, sigmas, log, tag_reference=tag_reference)


def process_jcamp(path: Path, context: SampleContext, options: ProcessingOptions) -> ProcessedInput:
    jd: JcampSpectrum = read_jcamp(path)
    log: list[str] = []
    solvent = context.solvent or _solvent_from_header(jd.solvent)
    mhz = context.spectrometer_mhz or (jd.sfo1_mhz if jd.nucleus == "1H" else None)
    ctx = context.model_copy(update={"solvent": solvent, "spectrometer_mhz": mhz})
    if jd.kind == "peaktable":
        nuc = Nucleus(jd.nucleus)
        if nuc != Nucleus.H1:
            raise IngestError("the JCAMP-DX peak table is not 1H; a 1H peak table is required")
        top = max(y for _, y in jd.peaks) or 1.0
        tallest = sorted(jd.peaks, key=lambda xy: -xy[1])[:200]
        if len(jd.peaks) > 200:
            log.append(f"JCAMP-DX peak table has {len(jd.peaks)} entries; kept the tallest 200")
        peaks = [
            Peak(id=i, nucleus=nuc, shift_ppm=round(x, 3), height=y / top)
            for i, (x, y) in enumerate(sorted(tallest, reverse=True), start=1)
        ]
        log.append(
            f"JCAMP-DX PEAK TABLE with {len(jd.peaks)} entries read; multiplicity and integrals unknown"
        )
        return ProcessedInput(peaks, ctx, "as_supplied", {}, log)
    sfo = mhz or jd.sfo1_mhz
    if not sfo:
        raise IngestError("the JCAMP-DX file has no .OBSERVE FREQUENCY; pass context.spectrometer_mhz")
    peaks, sigma, notes = _spectrum_peaks(jd.y, jd.ppm, sfo, jd.nucleus, options)
    log += notes
    peaks, ref, rejected = _reference(peaks, Nucleus(jd.nucleus), solvent, sfo, log)
    if not any(p.nucleus == Nucleus.H1 for p in peaks):
        raise IngestError("no 1H peaks were found in the JCAMP-DX spectrum")
    peaks = _gate_impurities(peaks, solvent, log)
    return ProcessedInput(
        peaks, ctx, ref, {jd.nucleus: sigma}, log, tag_reference=not (rejected and jd.nucleus == "1H")
    )


def process_upload(
    input_format: str, data: bytes, context: SampleContext, options: ProcessingOptions
) -> ProcessedInput:
    """Upload -> ProcessedInput. Anything unexpected from parsing an upload becomes an IngestError (a 400)."""
    try:
        return _process_upload(input_format, data, context, options)
    except ValueError:  # includes IngestError: the message is already safe to show
        raise
    except Exception as e:
        log.exception("upload_processing_failed", input_format=input_format, error=type(e).__name__)
        raise IngestError(
            "the upload could not be processed (unexpected error while reading the data)"
        ) from e


def _process_upload(
    input_format: str, data: bytes, context: SampleContext, options: ProcessingOptions
) -> ProcessedInput:
    check_upload_size(data)
    if input_format == "bruker_zip":
        return process_bruker_zip(data, context, options)
    if input_format == "jcampdx":
        with tempfile.TemporaryDirectory(prefix="nmr_") as tmp:
            path = Path(tmp) / "upload.jdx"
            path.write_bytes(data)
            return process_jcamp(path, context, options)
    raise IngestError(f"cannot process input_format {input_format!r}")
