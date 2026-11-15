"""The pending case (spec §6): a ``MemoryCase`` without its approval, and the step that approves it.

A pending case is proposed by ``bench/memory.py propose`` and waits in ``memory/pending/`` until a person
approves it. ``finalize`` is the only code that turns one into a ``MemoryCase``: it adds the approval fields
and validates the result with the same model the loader uses.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict

from src.memory.case import (
    AgentOutcome,
    Approval,
    Assignment,
    CaseId,
    HPeak,
    MemoryCase,
    Provenance,
)
from src.memory.store import _chemistry_problems
from src.schemas import Solvent


class PendingCase(BaseModel):
    """Every ``MemoryCase`` field except ``approval``; same names, types and field order."""

    model_config = ConfigDict(extra="forbid")

    case_id: CaseId
    schema_version: Literal["1"]
    provenance: Provenance
    formula: str
    dou: float
    carbon_count: int
    heavy_atoms: int
    solvent: Solvent | None
    h_peaks: list[HPeak]
    c_peaks: list[float]
    smiles: str
    inchikey: str
    murcko_scaffold: str
    generic_scaffold: str
    assignments: list[Assignment] | None
    agent_outcome: AgentOutcome


def finalize(
    pending: PendingCase, approved_by: str, date_added: dt.date, note: str | None = None
) -> MemoryCase:
    """The pending case plus approval fields, validated as a ``MemoryCase`` (raises ValidationError)."""
    approval = Approval(approved_by=approved_by, date_added=date_added, note=note)
    return MemoryCase.model_validate({**pending.model_dump(), "approval": approval.model_dump()})


def placeholder_case(pending: PendingCase) -> MemoryCase:
    """A ``MemoryCase`` view of a pending case, for checks written for approved
    cases (leakage rules, chemistry).

    The approval is a placeholder and must never be written to a store.
    """
    return finalize(pending, "pending-check", dt.date(1970, 1, 1))


def chemistry_problems(pending: PendingCase) -> list[str]:
    """What the loader's chemistry rule would report for this case (empty when consistent)."""
    return _chemistry_problems(placeholder_case(pending))


def dump_case(case: PendingCase | MemoryCase) -> str:
    """The exact text written to a case file (the SHA-256 in the manifest is over its UTF-8 bytes)."""
    return json.dumps(case.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
