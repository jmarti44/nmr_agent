"""Spot-check of the "predicted spectrum" rule (review R0-2 follow-up).

For each sampled spectrum (bench/nmrshiftdb2_spotcheck_sample.txt, produced by
`nmrshiftdb2_census.py --spotcheck 20`) compare the shifts in the SD file with the "Meas. Shift" column of the
saved NMRShiftDB2 record page (bench/data/nmrshiftdb2/pages/<id>.html). A really measured spectrum must have
its shifts in that column; a predicted one would not.

Page text is untrusted: it is only parsed with regexes and never executed.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
SD = HERE / "data/nmrshiftdb2/nmrshiftdb2withsignals.sd"
PAGES = HERE / "data/nmrshiftdb2/pages"
SAMPLE = HERE / "nmrshiftdb2_spotcheck_sample.txt"


def sd_shifts(rid: str, nuc: str, idx: int) -> list[float] | None:
    buf: list[str] = []
    with SD.open(errors="replace", newline=None) as fh:
        for line in fh:
            if line.startswith("$$$$"):
                rec = "".join(buf)
                buf = []
                if re.search(rf"> <nmrshiftdb2 ID>\n{rid}\n", rec):
                    m = re.search(rf"> <Spectrum {nuc} {idx}>\n(.*)\n", rec)
                    return [float(s.split(";")[0]) for s in m[1].split("|") if s.strip()] if m else None
            else:
                buf.append(line)
    return None


def page_blocks(rid: str) -> list[tuple[str, str]]:
    t = (PAGES / f"{rid}.html").read_text(errors="replace")
    t = re.sub(r"<script.*?</script>|<style.*?</style>", "", t, flags=re.S)
    t = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t)))
    parts = re.split(r"Type: ", t)[1:]
    return [(p.split(" ", 1)[0], p.split("Spectral Data")[0]) for p in parts]


def measured_values(nuc: str, block: str) -> set[float]:
    """First numeric column of a block headed 'Meas. Shift'."""
    if "Meas. Shift" not in block.split("Atom Mult.(coupling const.)")[1][:40]:
        return set()
    if nuc == "13C":
        return {float(m[1]) for m in re.finditer(r"\d+ [SDTQ] (-?\d+\.\d+)", block)}
    return {float(m[1]) for m in re.finditer(r"\d+-H (-?\d+\.\d+)", block)}


def main() -> int:
    rows = [ln.split("\t") for ln in SAMPLE.read_text().splitlines() if ln.startswith("nmrshiftdb2 ID")]
    ok = 0
    for r in rows:
        rid, nuc, idx = r[0].split()[-1], r[1].split()[0], int(r[1].split()[-1])
        sd = sd_shifts(rid, nuc, idx)
        blocks = [b for n, b in page_blocks(rid) if n == nuc]
        meas = set().union(*(measured_values(nuc, b) for b in blocks)) if blocks else set()
        if not sd or not meas:
            print(f"{rid:>9} {nuc:>3} #{idx}: NO DATA (sd={bool(sd)}, page_measured={bool(meas)})")
            continue
        hit = sum(any(abs(v - m) < 0.006 for m in meas) for v in sd)
        frac = hit / len(sd)
        ok += frac >= 0.8
        print(f"{rid:>9} {nuc:>3} #{idx}: {hit}/{len(sd)} SD shifts found in page 'Meas. Shift' column -> "
              f"{'EXPERIMENTAL' if frac >= 0.8 else 'MISMATCH'}")  # fmt: skip
    print(f"\n{ok}/{len(rows)} confirmed experimental")
    return 0


if __name__ == "__main__":
    sys.exit(main())
