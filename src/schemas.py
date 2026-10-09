"""All request, pipeline, LLM, verifier and response models (SPEC §6).

Imports nothing from RDKit so the API and prompt layers can use it without native dependencies.
Deviations from the SPEC text are marked `DEVIATION`.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.core.version import SCHEMA_VERSION  # noqa: F401  (re-exported)

# --------------------------------------------------------------------------- enums


class Nucleus(StrEnum):
    H1 = "1H"
    C13 = "13C"


class Multiplicity(StrEnum):
    """Observed splitting. 'br' = broad (exchangeable or unresolved); 'unknown' when not reported."""

    S = "s"
    D = "d"
    T = "t"
    Q = "q"
    QUINT = "quint"
    SEXT = "sext"
    SEPT = "sept"
    DD = "dd"
    DT = "dt"
    TD = "td"
    DDD = "ddd"
    M = "m"
    BR = "br"
    UNKNOWN = "unknown"


class PeakRole(StrEnum):
    ANALYTE = "analyte"
    SOLVENT = "solvent"  # residual solvent signal (e.g. CHCl3 at 7.26)
    WATER = "water"  # H2O/HOD in the given solvent
    IMPURITY_SUSPECT = "impurity_suspect"  # matched a common-impurity table entry; not removed
    REFERENCE = "reference"  # TMS / DSS


class Solvent(StrEnum):
    CDCL3 = "CDCl3"
    DMSO_D6 = "DMSO-d6"
    D2O = "D2O"
    CD3OD = "CD3OD"
    ACETONE_D6 = "acetone-d6"
    C6D6 = "C6D6"
    CD3CN = "CD3CN"
    OTHER = "other"


class FunctionalGroup(StrEnum):
    """Controlled vocabulary. Each value maps to a SMARTS pattern and shift rules in src/verify."""

    ALDEHYDE = "aldehyde"
    KETONE = "ketone"
    CARBOXYLIC_ACID = "carboxylic_acid"
    ESTER = "ester"
    AMIDE = "amide"
    ALCOHOL = "alcohol"
    PHENOL = "phenol"
    ETHER = "ether"
    METHOXY = "methoxy"
    AMINE_PRIMARY = "amine_primary"
    AMINE_SECONDARY = "amine_secondary"
    AMINE_TERTIARY = "amine_tertiary"
    AROMATIC_RING = "aromatic_ring"
    HETEROAROMATIC_RING = "heteroaromatic_ring"
    ALKENE = "alkene"
    ALKYNE_TERMINAL = "alkyne_terminal"
    ALKYNE_INTERNAL = "alkyne_internal"
    NITRILE = "nitrile"
    NITRO = "nitro"
    ALKYL_HALIDE = "alkyl_halide"
    ARYL_HALIDE = "aryl_halide"
    METHYL = "methyl"
    ACETYL = "acetyl"
    SULFONAMIDE = "sulfonamide"
    THIOETHER = "thioether"


def _lower(v: Any) -> Any:
    return v.lower() if isinstance(v, str) else v


# --------------------------------------------------------------------------- input models


class Peak(BaseModel):
    """One resolved signal (a multiplet for 1H, a line for 13C)."""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(
        ..., ge=1, description="1-based id used for evidence references; assigned downfield→upfield"
    )
    nucleus: Nucleus
    shift_ppm: float = Field(..., ge=-5.0, le=250.0)
    multiplicity: Multiplicity = Multiplicity.UNKNOWN
    coupling_hz: list[float] = Field(
        default_factory=list, description="J values in Hz, largest first; empty if unresolved"
    )
    integral: float | None = Field(
        None, ge=0.0, description="Normalized proton count (1H only). None for 13C or when unavailable"
    )
    height: float | None = Field(
        None, description="Peak height in arbitrary units; 13C relative intensity only"
    )
    fwhm_hz: float | None = Field(None, ge=0.0)
    role: PeakRole = PeakRole.ANALYTE
    exchangeable_hint: bool = Field(
        False, description="True if broad and in an OH/NH/COOH region, or if it disappears on D2O shake"
    )
    label: str | None = Field(None, max_length=60, description="Free text such as 'CHCl3 residual'")

    @field_validator("multiplicity", "role", mode="before")
    @classmethod
    def _norm(cls, v: Any) -> Any:
        return _lower(v)

    @model_validator(mode="after")
    def _ranges(self) -> Peak:
        lo, hi = (-2.0, 16.0) if self.nucleus == Nucleus.H1 else (-10.0, 240.0)
        if not (lo <= self.shift_ppm <= hi):
            raise ValueError(
                f"{self.nucleus.value} shift {self.shift_ppm} ppm outside plausible range [{lo}, {hi}]"
            )
        if self.nucleus == Nucleus.C13 and self.integral is not None:
            raise ValueError(
                "13C peaks must not carry integrals (not quantitative under standard acquisition)"
            )
        return self


_FORMULA_RE = re.compile(r"^([A-Z][a-z]?\d*)+$")
_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> dict[str, int]:
    """'C9H8O4' -> {'C': 9, 'H': 8, 'O': 4}. Raises ValueError on charge/isotope/bracket notation."""
    f = formula.replace(" ", "")
    if not _FORMULA_RE.match(f):
        raise ValueError(
            f"Unsupported molecular formula syntax: {formula!r} (no charges, isotopes or brackets)"
        )
    counts: dict[str, int] = {}
    for el, n in _ELEMENT_RE.findall(f):
        counts[el] = counts.get(el, 0) + (int(n) if n else 1)
    if "C" not in counts:
        raise ValueError("Formula must contain carbon")
    return counts


def degree_of_unsaturation(counts: dict[str, int]) -> float:
    """DoU = C + 1 + N/2 - H/2 - X/2 (X = F, Cl, Br, I). Si counts as C, P as N; O and S do not contribute."""
    c = counts.get("C", 0) + counts.get("Si", 0)
    n = counts.get("N", 0) + counts.get("P", 0)
    h = counts.get("H", 0)
    x = sum(counts.get(k, 0) for k in ("F", "Cl", "Br", "I"))
    return c + 1 + n / 2 - h / 2 - x / 2


class SampleContext(BaseModel):
    """Everything the chemist knows besides the spectrum. All optional; more context → better candidates."""

    model_config = ConfigDict(extra="forbid")

    molecular_formula: str | None = Field(
        None, examples=["C9H8O4"], description="Neutral formula from HRMS, Hill order preferred"
    )
    observed_exact_mass: float | None = Field(
        None, gt=0, description="Monoisotopic neutral mass in Da (not m/z) if available"
    )
    solvent: Solvent | None = None
    spectrometer_mhz: float | None = Field(
        None, gt=10, lt=1500, description="1H observe frequency, e.g. 400.13"
    )
    starting_material_smiles: str | None = Field(
        None, max_length=500, description="Reaction starting material; improves inverse elucidation"
    )
    d2o_exchange_performed: bool = False
    notes: str | None = Field(None, max_length=500)

    @field_validator("molecular_formula")
    @classmethod
    def _formula(cls, v: str | None) -> str | None:
        if v is not None:
            parse_formula(v)
        return v

    @field_validator("solvent", mode="before")
    @classmethod
    def _solv(cls, v: Any) -> Any:
        aliases = {
            "cdcl3": "CDCl3",
            "chloroform-d": "CDCl3",
            "dmso": "DMSO-d6",
            "dmso-d6": "DMSO-d6",
            "d2o": "D2O",
            "cd3od": "CD3OD",
            "methanol-d4": "CD3OD",
            "acetone-d6": "acetone-d6",
            "c6d6": "C6D6",
            "benzene-d6": "C6D6",
            "cd3cn": "CD3CN",
            "acetonitrile-d3": "CD3CN",
        }
        return aliases.get(v.lower(), v) if isinstance(v, str) else v

    @property
    def formula_counts(self) -> dict[str, int] | None:
        return parse_formula(self.molecular_formula) if self.molecular_formula else None


class ProcessingOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lb_hz: float = Field(0.3, ge=0.0, le=5.0, description="Exponential line broadening in Hz")
    k_sigma: float = Field(6.0, ge=3.0, le=20.0, description="Peak threshold as multiple of MAD noise sigma")
    autophase: Literal["acme", "peak_minima"] = "acme"
    prefer_pdata: bool = Field(True, description="Use TopSpin-processed 1r when present in a Bruker zip")


class PeakListInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    peaks: list[Peak] = Field(..., min_length=1, max_length=200)

    @model_validator(mode="after")
    def _has_h1_and_unique_ids(self) -> PeakListInput:
        if not any(p.nucleus == Nucleus.H1 for p in self.peaks):
            raise ValueError("At least one 1H peak is required")
        ids = [p.id for p in self.peaks]
        if len(ids) != len(set(ids)):
            raise ValueError("Peak ids must be unique")
        return self


class AnalysisRequest(BaseModel):
    """JSON body for /v1/analyze (peaklist), or the 'metadata' part of the multipart form."""

    model_config = ConfigDict(extra="forbid")
    input_format: Literal["peaklist", "jcampdx", "bruker_zip"]
    peaks: PeakListInput | None = None
    context: SampleContext = Field(default_factory=SampleContext)
    options: ProcessingOptions = Field(default_factory=ProcessingOptions)
    max_candidates: int = Field(3, ge=1, le=5)

    @model_validator(mode="after")
    def _peaks_required(self) -> AnalysisRequest:
        if self.input_format == "peaklist" and self.peaks is None:
            raise ValueError("input_format='peaklist' requires 'peaks'")
        if self.input_format != "peaklist" and self.peaks is not None:
            raise ValueError("'peaks' is only allowed with input_format='peaklist'")
        return self


# --------------------------------------------------------------------------- pipeline models


class PeakTable(BaseModel):
    """Normalized, solvent-tagged signal list shared by the prompt builder and the verifier."""

    model_config = ConfigDict(extra="forbid")
    peaks: list[Peak]
    nucleus_frequencies_mhz: dict[str, float] = Field(
        default_factory=dict, description="e.g. {'1H': 400.13, '13C': 100.62}"
    )
    solvent: Solvent | None = None
    referencing: str = Field(
        "as_supplied", description="'as_supplied' | 'procs' | 'auto:<anchor>' | 'uncorrected'"
    )
    noise_sigma: dict[str, float] = Field(default_factory=dict)
    integral_normalization: str = Field("none", description="'formula_H' | 'smallest_peak' | 'none'")
    processing_log: list[str] = Field(default_factory=list)

    def analyte_peaks(self, nucleus: Nucleus) -> list[Peak]:
        return [p for p in self.peaks if p.nucleus == nucleus and p.role == PeakRole.ANALYTE]


# --------------------------------------------------------------------------- LLM-facing models


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    peak_ids: list[int] = Field(
        ..., description="Ids from the peak table that support this claim; must not be empty"
    )
    note: str = Field(..., description="One sentence tying the peaks to the claim, ≤ 200 chars")


class FunctionalGroupClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    group: FunctionalGroup
    present: bool
    confidence: float = Field(..., description="0.0–1.0, the model's own belief")
    evidence: EvidenceRef

    @field_validator("group", mode="before")
    @classmethod
    def _g(cls, v: Any) -> Any:
        return _lower(v)


class PeakAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    peak_id: int
    proton_count: int = Field(
        ..., description="Protons this signal accounts for in the candidate (0 for 13C)"
    )
    environment: str = Field(..., description="Short label, e.g. 'ArH ortho to COOH' or 'OCH3'")


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rank: int
    smiles: str = Field(
        ..., description="Valid, neutral, fully specified SMILES; no wildcards or stereo unless certain"
    )
    name: str | None = Field(None, description="IUPAC or common name if known")
    rationale: str = Field(..., description="Why this structure fits; cite peak ids; ≤ 600 chars")
    assignments: list[PeakAssignment]
    confidence: float = Field(..., description="0.0–1.0, the model's own belief")


class LLMHypothesis(BaseModel):
    """The structured output requested from Claude; validated by the SDK, then re-checked by the verifier."""

    model_config = ConfigDict(extra="forbid")
    formula_inferred: str | None = Field(
        None, description="Only when no formula was supplied; otherwise echo it"
    )
    degree_of_unsaturation: int | None = None
    functional_groups: list[FunctionalGroupClaim]
    candidates: list[Candidate] = Field(
        ..., description="Ranked best-first, 1–5 entries, distinct connectivity"
    )
    unexplained_peak_ids: list[int] = Field(default_factory=list)
    reasoning_summary: str = Field(
        ..., description="≤ 1200 chars; must not introduce peaks absent from the table"
    )

    @field_validator("candidates")
    @classmethod
    def _n(cls, v: list[Candidate]) -> list[Candidate]:
        if not 1 <= len(v) <= 5:
            raise ValueError("1–5 candidates required")
        return sorted(v, key=lambda c: c.rank)


# --------------------------------------------------------------------------- verifier and response models


class RuleStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIPPED = "skipped"


class RuleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rule_id: str = Field(..., examples=["V03_dou_consistency"])
    status: RuleStatus
    severity: Literal["hard", "soft", "info"]
    message: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    source: str | None = Field(None, description="Citation for the chemistry rule, by name")


class VerifiedCandidate(BaseModel):
    """DEVIATION: structure-derived fields are optional; an unparsable SMILES (V01 fail) has none."""

    model_config = ConfigDict(extra="forbid")
    candidate: Candidate
    canonical_smiles: str | None = None
    inchikey: str | None = None
    formula: str | None = None
    exact_mass: float | None = None
    degree_of_unsaturation: int | None = None
    rules: list[RuleResult]
    verifier_score: float = Field(..., description="0–1 from rule outcomes only")
    final_confidence: float = Field(..., description="0–1 blend of verifier score and model confidence")
    status: Literal["accepted", "flagged", "rejected"] = Field(
        ...,
        description=(
            "accepted = consistent with the supplied data (no hard failure, score >= 0.6, no failed "
            "proton-accounting or required-evidence rule). It does NOT mean the structure is correct: "
            "isomers can also be accepted. flagged = at least one soft check failed. "
            "rejected = a hard check failed (invalid SMILES, formula or integral mismatch)."
        ),
    )


class VerifiedFunctionalGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    group: FunctionalGroup
    claimed_present: bool
    supported_by_shifts: RuleStatus
    present_in_top_candidate: bool | None
    source: str | None = None


class Versions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    service: str
    schema_version: str
    prompt_version: str
    model_id: str
    nmrglue: str
    rdkit: str


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    llm_calls: int
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    estimated_cost_usd: float


class AnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str
    versions: Versions
    peak_table: PeakTable
    hypothesis_raw: LLMHypothesis
    candidates: list[VerifiedCandidate]
    functional_groups: list[VerifiedFunctionalGroup]
    warnings: list[str]
    timing_ms: dict[str, int]
    usage: Usage


class JobStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    job_id: str
    state: Literal["queued", "running", "done", "error"]
    result: AnalysisResponse | None = None
    error: str | None = None


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    error_code: str = Field(
        ..., examples=["INVALID_INPUT", "LLM_UNAVAILABLE", "LLM_REFUSED", "RATE_LIMITED", "TIMEOUT"]
    )
    detail: str
    request_id: str


class VerifyRequest(BaseModel):
    """Body of POST /v1/verify (SPEC §9.1): verifier only, no LLM."""

    model_config = ConfigDict(extra="forbid")
    peak_table: PeakTable
    candidates: list[str] = Field(..., min_length=1, max_length=5, description="SMILES strings, best first")
    context: SampleContext = Field(default_factory=SampleContext)

    @field_validator("candidates")
    @classmethod
    def _smiles_len(cls, v: list[str]) -> list[str]:
        if any(not s.strip() or len(s) > 500 for s in v):
            raise ValueError("each SMILES must be 1-500 characters")
        return v

    @model_validator(mode="after")
    def _has_evidence(self) -> VerifyRequest:
        # With no analyte 1H peak every rule is skipped and a candidate would score 1.0 on nothing.
        if not self.peak_table.analyte_peaks(Nucleus.H1):
            raise ValueError("peak_table needs at least one analyte 1H peak to verify against")
        return self
