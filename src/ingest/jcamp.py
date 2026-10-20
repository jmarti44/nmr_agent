"""JCAMP-DX 1D NMR reader (SPEC §5.3): full spectra via nmrglue, PEAK TABLE blocks via a small parser."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import nmrglue as ng
import numpy as np

from src.ingest.errors import IngestError

_LDR = re.compile(r"^##\s*([^=]+?)\s*=\s*(.*)$")
_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
MAX_POINTS = 2**19  # same reasoning as MAX_FID_POINTS in safety.py; real spectra are 16k-128k points
# JCAMP-DX compresses numbers: a DUP token (S-Z, s plus digits) repeats the previous value that many times.
_DUP = re.compile(r"[S-Zs](\d*)")
_DUP_FIRST = {**{c: str(i) for i, c in enumerate("STUVWXYZ", start=1)}, "s": "9"}
# Anything that can start a number: a sign, a compression letter, or a digit after whitespace.
_VALUE_START = re.compile(r"[-+@A-Za-z%]|(?<=\s)\d")


@dataclass
class JcampSpectrum:
    """Either a spectrum (`y`, `ppm`) or a peak table (`peaks` as (ppm, intensity) pairs)."""

    kind: str  # "spectrum" | "peaktable"
    sfo1_mhz: float | None = None
    nucleus: str = "1H"
    solvent: str | None = None
    y: np.ndarray | None = None
    ppm: np.ndarray | None = None
    peaks: list[tuple[float, float]] = field(default_factory=list)


def _headers(text: str) -> dict[str, str]:
    """First value of each labelled data record, with labels upper-cased and stripped of spaces/dashes."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        m = _LDR.match(line)
        if m:
            key = re.sub(r"[\s_\-/]", "", m.group(1)).upper()
            out.setdefault(key, m.group(2).strip())
    return out


def _nucleus(h: dict[str, str]) -> str:
    raw = h.get(".OBSERVENUCLEUS", "").replace("^", "").replace("<", "").replace(">", "").strip().upper()
    if raw in ("13C", "C13", "C-13"):
        return "13C"
    if raw in ("1H", "H1", "H-1", "PROTON", ""):
        return "1H"
    raise IngestError(f"unsupported nucleus {raw!r}; only 1H and 13C are supported")


def _observe_mhz(h: dict[str, str]) -> float | None:
    m = _NUM.search(h.get(".OBSERVEFREQUENCY", ""))
    return float(m.group()) if m else None


def parse_peak_table(text: str) -> list[tuple[float, float]]:
    """Pairs from a `##PEAK TABLE=(XY..XY)` block, e.g. `7.26, 100; 2.35, 40`."""
    pairs: list[tuple[float, float]] = []
    inside = False
    for line in text.splitlines():
        if line.startswith("##"):
            inside = re.sub(r"[\s_\-]", "", line.split("=")[0]).upper() == "##PEAKTABLE"
            continue
        if not inside:
            continue
        for group in line.split("$$")[0].split(";"):
            nums = [float(n) for n in _NUM.findall(group)]
            if len(nums) % 2:
                raise IngestError(f"PEAK TABLE entry is not an (x, y) pair: {group.strip()!r}")
            pairs.extend(zip(nums[::2], nums[1::2], strict=True))
    return pairs


def check_decoded_size(text: str, header: dict[str, str]) -> None:
    """Refuse a file that would decode to more than MAX_POINTS values, before nmrglue decodes it.

    nmrglue expands DUP runs into Python lists, so a few KB of `@T999999` repeats can need gigabytes. The
    estimate over-counts (every sign or letter counts as a value), which is safe.
    """
    declared = _NUM.search(header.get("NPOINTS", ""))
    if declared and float(declared.group()) > MAX_POINTS:
        raise IngestError(f"the JCAMP-DX file declares {declared.group()} points; the limit is {MAX_POINTS}")
    total = 0
    for line in text.splitlines():
        if line.startswith("##") or not line.strip():
            continue
        line = line.split("$$")[0]
        total += len(_VALUE_START.findall(line))
        for m in _DUP.finditer(line):
            total += int(_DUP_FIRST[m.group()[0]] + m.group(1)) - 1
        if total > MAX_POINTS:
            raise IngestError(f"the JCAMP-DX file would expand to more than {MAX_POINTS} points")


def read_jcamp(path: str | Path) -> JcampSpectrum:
    path = Path(path)
    text = path.read_text(encoding="latin-1", errors="replace")
    h = _headers(text)
    check_decoded_size(text, h)
    if "PEAKTABLE" in h and "XYDATA" not in h and "NTUPLES" not in h:
        peaks = parse_peak_table(text)
        if not peaks:
            raise IngestError("the JCAMP-DX PEAK TABLE block has no readable (x, y) pairs")
        units = h.get("XUNITS", "PPM").strip().upper()
        mhz = _observe_mhz(h)
        if units == "HZ":
            if not mhz:
                raise IngestError("JCAMP-DX peak table is in Hz without .OBSERVE FREQUENCY; cannot get ppm")
            peaks = [(x / mhz, y) for x, y in peaks]
        return JcampSpectrum("peaktable", mhz, _nucleus(h), h.get(".SOLVENTNAME"), peaks=peaks)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            dic, data = ng.jcampdx.read(str(path))
        except Exception as e:  # nmrglue raises assorted errors on malformed files
            raise IngestError(f"could not read the JCAMP-DX file: {type(e).__name__}") from e
    if data is None:
        raise IngestError("the JCAMP-DX file has no XYDATA, NTUPLES or PEAK TABLE data")
    y = np.asarray(data[0] if isinstance(data, list) else data, dtype=float).ravel()
    if y.size < 16 or y.size > MAX_POINTS or not np.all(np.isfinite(y)):
        raise IngestError(f"the JCAMP-DX spectrum has an unusable number of points ({y.size})")
    try:
        first, last = float(dic["FIRSTX"][0]), float(dic["LASTX"][0])
    except (KeyError, IndexError, ValueError) as e:
        raise IngestError("the JCAMP-DX file lacks FIRSTX/LASTX") from e
    units = str(dic.get("XUNITS", ["PPM"])[0]).strip().upper()
    mhz = _observe_mhz(h)
    x = np.linspace(first, last, y.size)
    if units == "HZ":
        if not mhz:
            raise IngestError("JCAMP-DX is in Hz without .OBSERVE FREQUENCY; cannot convert to ppm")
        x = x / mhz
    elif units != "PPM":
        raise IngestError(f"unsupported JCAMP-DX XUNITS {units!r}")
    if x[0] < x[-1]:  # descending ppm: left is downfield
        x, y = x[::-1], y[::-1]
    return JcampSpectrum("spectrum", mhz, _nucleus(h), h.get(".SOLVENTNAME"), y=y, ppm=x)
