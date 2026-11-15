"""Render stored cases as the worked-examples block of the user message (spec §5.4).

Only stored, structured fields are rendered. ``approval`` (including ``approval.note``), provenance and every
other free-text field of a case never reach the prompt, and no explanation of why a proposal was wrong is
generated. Pure functions of the case: no randomness, no clock, no model.
"""

from __future__ import annotations

from collections.abc import Sequence

from src.memory.case import MemoryCase
from src.prompting.templates import estimate_tokens

EXAMPLES_HEADER = (
    "Worked examples: other molecules, shown only for their reasoning pattern. They are not the sample to "
    "analyze, and none of their structures is the answer."
)
EXAMPLES_FOOTER = "End of worked examples. The sample to analyze follows."


def _fmt_dou(dou: float) -> str:
    return f"{dou:g}"


def _h_line(case: MemoryCase) -> str:
    items = []
    for p in sorted(case.h_peaks, key=lambda p: p.shift_ppm):
        items.append(f"{p.shift_ppm:.2f}" + (f" ({p.integral:.1f}H)" if p.integral is not None else ""))
    return "1H shifts (ppm): " + (", ".join(items) if items else "none")


def _c_line(case: MemoryCase) -> str:
    return "13C shifts (ppm): " + (", ".join(f"{s:.1f}" for s in sorted(case.c_peaks)) or "none")


def render_case(case: MemoryCase) -> str:
    """One case as text. A ``missed`` case with a stored proposal shows its first proposed SMILES, labelled
    as an earlier wrong proposal, before the correct structure; every other case is a plain worked example."""
    solvent = f", solvent {case.solvent.value}" if case.solvent else ""
    lines = [
        f"Example molecule: formula {case.formula}, DoU {_fmt_dou(case.dou)}{solvent}",
        _h_line(case),
        _c_line(case),
    ]
    outcome = case.agent_outcome
    if outcome.status == "missed" and outcome.proposed_smiles:
        lines.append(f"Earlier wrong proposal for this spectrum: {outcome.proposed_smiles[0]}")
        lines.append(f"Correct structure: {case.smiles}")
    else:
        lines.append(f"Structure: {case.smiles}")
    if case.assignments:
        lines.append("Database assignments:")
        for a in sorted(case.assignments, key=lambda a: (a.nucleus.value, a.shift_ppm)):
            lines.append(f"- {a.nucleus.value} {a.shift_ppm:.2f} ppm: {a.environment}")
    return "\n".join(lines)


def render_examples(cases: Sequence[MemoryCase]) -> str:
    """The whole examples block, best case first; the empty string when there are no cases."""
    if not cases:
        return ""
    return "\n\n".join([EXAMPLES_HEADER, *(render_case(c) for c in cases), EXAMPLES_FOOTER])


def case_cost(case: MemoryCase) -> int:
    """Estimated tokens of the rendered case: the cost function for the retrieval token budget."""
    return estimate_tokens(render_case(case))
