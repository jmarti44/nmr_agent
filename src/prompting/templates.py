"""The versioned system prompt (SPEC §7.2). Static text plus reference tables generated from the verifier's own tables.

Everything here is a pure function of the code: the same input gives byte-identical text, which is what makes the
prompt cacheable and its version meaningful. Any edit to the wording, or to the shift/SMARTS tables it embeds, must
bump PROMPT_VERSION (and SHIFT_TABLE_VERSION for table changes); a test enforces this with a content hash.

DEVIATIONS from the SPEC text, each deliberate:
- Rule 2: solvent, water and impurity_suspect peaks are NOT simply ignored. SPEC §8.5 says a tagged impurity peak
  stays usable, and tagging by position mislabels real analyte peaks (measured on NMRShiftDB2: about 4% of 1H peaks
  fall on a solvent or water position). They may be used only when the structure needs them, with a stated reason.
  Impurity tagging is off for peak-list input, so impurity_suspect only appears on the raw-spectrum path.
- Rule 4: carbon-count bounds match what rule V06 actually checks (observed - 1 .. observed + 2 distinct carbons);
  the SPEC wording had the two bounds swapped relative to V06.
- The number of candidates is not in the system prompt (it is in the user message), so the cached prefix is the
  same for every request.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache

from src.verify.shift_tables import (
    C_RULES,
    DMSO_EXCHANGEABLE_WINDOW,
    H_RULES,
    SHIFT_TABLE_VERSION,
    ShiftRule,
)
from src.verify.smarts import SPECS

PROMPT_VERSION = "2026.10.3"
FULL_VERSION = f"{PROMPT_VERSION}+shifts.{SHIFT_TABLE_VERSION}"

_HEAD = """\
You are an expert small-molecule NMR spectroscopist. You will receive a 1D NMR peak table (1H required, 13C
optional), an optional HRMS molecular formula, the solvent, and optional sample details. Your task: infer the
functional groups and propose candidate structures as SMILES. The request states how many candidates to propose.
Your answer is checked by software that parses every SMILES and tests it against the peak table, so errors are
caught; propose only structures you can justify from the data.

Hard rules:
1. Use only peaks present in the table. Never invent, split, or merge peaks. Cite peak ids (the numbers in the
   first column for 1H, and the id before the colon for 13C) for every claim.
2. Ignore peaks with role reference. A peak with role solvent or water sits at a residual-solvent or water
   position; it is usually not analyte, but it can hide a real signal (for example an aromatic proton at 7.27
   in CDCl3, or an OH or CH2 near the water position). Use such a peak as evidence only if the structure needs
   it and nothing else explains it, and say so. Peaks with role impurity_suspect matched a common impurity
   (for example acetone or toluene) but may also be genuine analyte signals; treat them the same way.
3. If a molecular formula is given, every candidate MUST have exactly that formula and the corresponding degree
   of unsaturation. The analyte 1H integrals must account for the formula's H count; OH, NH and COOH protons
   are often unlisted, so a total below H by up to the number of such protons is acceptable. Proton counts in
   your peak assignments should add up to the H count minus any exchangeable protons you did not assign.
4. Count 13C signals. The number of distinct carbon environments in a candidate (symmetry-equivalent carbons
   count once) should be between (observed signals - 1) and (observed signals + 2).
5. Write SMILES that parse: balanced ring closures, correct valences, no wildcards, no charges, no stereo unless
   you are certain.
6. Rank candidates best first. Candidates must differ in connectivity, not only in stereochemistry.
7. Prefer the simplest structure consistent with all the evidence, and list the peaks you cannot explain.
8. Every functional group in a candidate needs supporting peaks inside the reference ranges below; a group whose
   expected signal is missing will be flagged. The functional_groups list in your answer describes your first
   (rank 1) candidate only and is checked against that candidate alone. Use only the functional-group names
   defined below.
9. Multiplicity gives neighbours (n+1). Typical couplings: ortho-aromatic 7-9 Hz, meta about 2 Hz, vicinal
   alkyl about 7 Hz, cis alkene 10-12 Hz, trans alkene 14-18 Hz. Treat these as guidance, not law. "m" and
   unknown multiplicity are not evidence for or against a structure.
"""

_OUTPUT = """\
Output: respond only with the JSON object that matches the provided schema. Do not add any other text.
"""


def _range_line(r: ShiftRule) -> str:
    notes = []
    if r.confirm:
        notes.append(f"core {r.confirm[0]:g}-{r.confirm[1]:g}")
    if r.exchangeable:
        notes.append("exchangeable, broad and variable")
    if r.soft_only:
        notes.append("approximate")
    suffix = f"  ({'; '.join(notes)})" if notes else ""
    return f"  {r.key:<19} {r.description:<54} {r.lo:g}-{r.hi:g}{suffix}"


@lru_cache(maxsize=1)
def reference_block() -> str:
    """Shift ranges, generated from the same tables the verifier uses."""
    lo, hi = DMSO_EXCHANGEABLE_WINDOW
    h = [_range_line(r) for r in H_RULES.values() if r.key != "h_unclassified"]
    c = [_range_line(r) for r in C_RULES.values()]
    return (
        "Reference chemical-shift ranges (ppm, TMS = 0). Ranges are inclusive and a little tolerant.\n"
        "1H:\n" + "\n".join(h) + "\n"
        f"  In DMSO-d6 any broad 1H signal between {lo:g} and {hi:g} ppm may be an exchangeable proton.\n"
        "13C:\n" + "\n".join(c)
    )


@lru_cache(maxsize=1)
def functional_group_block() -> str:
    """Names the model must use for functional groups, each with the SMARTS the checker applies."""
    lines = [f"  {g.value:<20} {' or '.join(s.smarts)}" for g, s in SPECS.items()]
    return "Functional-group names and their SMARTS definitions (what the checker will test):\n" + "\n".join(
        lines
    )


@lru_cache(maxsize=1)
def system_prompt() -> str:
    return "\n".join([_HEAD, functional_group_block(), "", reference_block(), "", _OUTPUT])


def prompt_hash() -> str:
    return hashlib.sha256(system_prompt().encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    """Rough token estimate (about 4 characters per token). An estimate only: no tokenizer or API is used."""
    return -(-len(text) // 4)
