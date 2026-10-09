"""Render a PeakTable as the compact text block a chemist would paste into a chat (SPEC §5.6).

Numbers are rounded (1H shifts 2 decimals, 13C 1 decimal, integrals 1 decimal) so the text, and therefore the
prompt, is stable for prompt caching. Pure function of the table and context: no randomness, no clock.
"""

from __future__ import annotations

from src.processing.normalize import formula_summary
from src.schemas import Nucleus, Peak, PeakRole, PeakTable, SampleContext


def _mult(p: Peak) -> str:
    return "-" if p.multiplicity.value == "unknown" else p.multiplicity.value


def _j(p: Peak) -> str:
    return ", ".join(f"{j:.1f}" for j in p.coupling_hz) if p.coupling_hz else "-"


def _integral(p: Peak) -> str:
    return "-" if p.integral is None or p.role in (PeakRole.SOLVENT, PeakRole.WATER) else f"{p.integral:.1f}"


def _role(p: Peak) -> str:
    note = []
    if p.label and p.role != PeakRole.ANALYTE:
        note.append(p.label)
    if p.exchangeable_hint:
        note.append("exchangeable?")
    return p.role.value + (f"   ({'; '.join(note)})" if note else "")


def _header(table: PeakTable, context: SampleContext) -> str:
    parts = ["Nucleus: 1H"]
    mhz = table.nucleus_frequencies_mhz.get("1H")
    if mhz:
        parts.append(f"{mhz:g} MHz")
    solvent = table.solvent or context.solvent
    parts.append(f"solvent {solvent.value}" if solvent else "solvent not given")
    parts.append(f"referencing: {table.referencing}")
    return ", ".join(parts)


def _formula_line(context: SampleContext) -> str:
    summary = formula_summary(context)
    if summary is None:
        return "Formula: not supplied"
    h, dou = summary
    return f"Formula (HRMS): {context.molecular_formula}   (H = {h}, DoU = {dou:g})"


def render_peak_table(table: PeakTable, context: SampleContext | None = None) -> str:
    context = context or SampleContext(solvent=table.solvent)
    h_peaks = [p for p in table.peaks if p.nucleus == Nucleus.H1]
    c_peaks = [p for p in table.peaks if p.nucleus == Nucleus.C13]
    lines = [_header(table, context), _formula_line(context)]
    lines.append(f"{'#':<3} {'shift(ppm)':<11} {'mult':<6} {'J(Hz)':<12} {'integral':<9} role")
    for p in h_peaks:
        lines.append(f"{p.id:<3} {p.shift_ppm:<11.2f} {_mult(p):<6} {_j(p):<12} {_integral(p):<9} {_role(p)}")
    if c_peaks:
        mhz = table.nucleus_frequencies_mhz.get("13C")
        head = f"13C ({mhz:g} MHz)" if mhz else "13C"
        items = [
            f"{p.id}:{p.shift_ppm:.1f}" + ("" if p.role == PeakRole.ANALYTE else f"[{p.role.value}]")
            for p in c_peaks
        ]
        lines.append(f"{head} signals as id:shift(ppm): " + ", ".join(items))
    return "\n".join(lines)


def render_context_block(context: SampleContext, max_candidates: int) -> str:
    """Sample details not in the table header, then the request size (the volatile part of the prompt)."""
    items = []
    if context.observed_exact_mass is not None:
        items.append(f"- Observed monoisotopic mass: {context.observed_exact_mass:.4f} Da")
    if context.starting_material_smiles:
        items.append(f"- Starting material (SMILES): {context.starting_material_smiles}")
    if context.d2o_exchange_performed:
        items.append("- D2O exchange was performed: exchangeable protons are not expected in the 1H data")
    if context.notes:
        items.append(f"- Notes from the chemist: {context.notes}")
    block = ("Sample details:\n" + "\n".join(items) + "\n\n") if items else ""
    return (
        block + f"Propose up to {max_candidates} candidate structure(s) (1 to {max_candidates}), best first."
    )
