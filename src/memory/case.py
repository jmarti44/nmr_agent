"""The stored case (spec §4.2): one approved, human-reviewed record with the agent's own outcome on it."""

from __future__ import annotations

import datetime as dt
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from src.schemas import Nucleus, Solvent

CASE_SCHEMA_VERSION = "1"

CaseId = Annotated[str, StringConstraints(pattern=r"^mem-\d{6}$")]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Provenance(_Strict):
    source: str = Field(min_length=1, description="'nmrshiftdb2' in v1; later 'user-confirmed'")
    source_record_id: str = Field(min_length=1)
    spectrum_ids: dict[str, str | int] = Field(description="which 1H and 13C spectra of the record")
    truth_basis: str = Field(min_length=1)
    extracted_by: str = Field(min_length=1, description="script name and git commit")
    proposed_on: dt.date


class Approval(_Strict):
    approved_by: str = Field(min_length=1)
    date_added: dt.date
    note: str | None = Field(None, description="never rendered into a prompt")


class AgentOutcome(_Strict):
    collected_on: dt.date
    model_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    settings_hash: str = Field(min_length=1)
    status: Literal["solved", "missed", "no_answer"]
    proposed_smiles: list[str]

    @model_validator(mode="after")
    def _no_answer_has_no_candidates(self) -> AgentOutcome:
        if self.status == "no_answer" and self.proposed_smiles:
            raise ValueError("proposed_smiles must be empty when status is 'no_answer'")
        return self


class HPeak(_Strict):
    shift_ppm: float
    integral: float | None = None


class Assignment(_Strict):
    nucleus: Nucleus
    shift_ppm: float
    environment: str


class MemoryCase(_Strict):
    case_id: CaseId
    schema_version: Literal["1"] = Field(description="must equal CASE_SCHEMA_VERSION")

    provenance: Provenance
    approval: Approval

    # Query-side features: only what exists at proposal time.
    formula: str = Field(min_length=1)
    dou: float
    carbon_count: int
    heavy_atoms: int
    solvent: Solvent | None
    h_peaks: list[HPeak]
    c_peaks: list[float]

    # Answer: never used for retrieval.
    smiles: str = Field(min_length=1)
    inchikey: str = Field(min_length=1)
    murcko_scaffold: str = Field(description="canonical SMILES; empty string for an acyclic molecule")
    generic_scaffold: str = Field(description="canonical SMILES of the generic scaffold; '' if acyclic")
    assignments: list[Assignment] | None

    agent_outcome: AgentOutcome
