"""How often does solvent/impurity tagging mislabel genuine analyte peaks?

NMRShiftDB2 peak lists contain analyte signals only, so on its CDCl3 spectra every peak tagged solvent, water,
reference or impurity_suspect is a false tag. Reported twice: as peak-list input is normalized (impurity
tagging off) and with the impurity table on, which is what the raw-spectrum path would use.
Usage: uv run python -m bench.nmrshiftdb2_tagging_rate
"""

from __future__ import annotations

import collections
import re

from pydantic import ValidationError

from bench.nmrshiftdb2_census import DEFAULT, INDEXED, TAG, records
from src.schemas import Nucleus, Peak, PeakRole, Solvent
from src.verify.solvent_tables import tag_peaks

NUCLEI = ("1H", "13C")


def idx(v: str) -> dict[int, str]:
    return {int(i): s.strip() for i, s in INDEXED.findall(v.strip())}


def measure(impurities: bool) -> None:
    by_role = {n: collections.Counter() for n in NUCLEI}
    spectra = {n: collections.Counter() for n in NUCLEI}
    false_tags = {n: collections.Counter() for n in NUCLEI}
    skipped = collections.Counter()
    for rec in records(DEFAULT):
        tags = {k: v for k, v in TAG.findall(rec[rec.find("M  END") :])}
        solvents = idx(tags.get("Solvent", ""))
        for k, v in tags.items():
            m = re.fullmatch(r"Spectrum (1H|13C) (\d+)", k)
            if not m or "Chloroform" not in solvents.get(int(m[2]), ""):
                continue
            nuc = m[1]
            values = sorted(
                {float(e.split(";")[0]) for e in v.strip().strip("|").split("|") if e.split(";")[0]}
            )
            peaks: list[Peak] = []
            for shift in values:
                try:
                    peaks.append(Peak(id=len(peaks) + 1, nucleus=Nucleus(nuc), shift_ppm=shift))
                except ValidationError:
                    skipped[nuc] += 1  # outside the schema's plausible range
            if not peaks:
                continue
            tagged = tag_peaks(peaks, Solvent.CDCL3, impurities=impurities)
            spectra[nuc]["spectra"] += 1
            spectra[nuc]["with_false_tag"] += any(p.role != PeakRole.ANALYTE for p in tagged)
            for p in tagged:
                by_role[nuc][p.role.value] += 1
                if p.role != PeakRole.ANALYTE:
                    false_tags[nuc][f"{p.shift_ppm:g} ({p.label})"] += 1
    for nuc in NUCLEI:
        total = sum(by_role[nuc].values())
        false = total - by_role[nuc]["analyte"]
        n, bad = spectra[nuc]["spectra"], spectra[nuc]["with_false_tag"]
        print(f"{nuc} in CDCl3: {n} spectra, {total} peaks ({skipped[nuc]} skipped, out of range)")
        print(f"   mis-tagged peaks: {false} ({100 * false / total:.1f}%)")
        print(f"   spectra with at least one: {bad} ({100 * bad / n:.1f}%)")
        print("   by role:", dict(by_role[nuc]))
        print("   most common false tags:", false_tags[nuc].most_common(6))


def main() -> None:
    print("== peak-list input (impurity tagging off) ==")
    measure(impurities=False)
    print("\n== impurity tagging on (raw-spectrum path) ==")
    measure(impurities=True)


if __name__ == "__main__":
    main()
