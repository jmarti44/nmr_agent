"""Store manifest, ``memory_version`` and the loader (spec §4.1-4.2).

The loader reads only ``<root>/cases/*.json`` and ``<root>/manifest.json``; ``pending/`` and ``rejected/``
are never opened.

Two kinds of failure, kept apart on purpose:

* A *case* fault (rules a-f in :func:`load_store`) raises :class:`CaseRejectedError` in strict mode
  (benchmark) and is skipped, logged and recorded in ``MemoryStore.rejections`` in lenient mode (service).
* A *store* fault (the manifest is missing, unparseable, invalid, lists a case id twice, or names an unknown
  store schema version; or ``cases/`` is missing while the manifest lists cases) raises
  :class:`ManifestError` in **both** modes: without a manifest no case can be trusted. Contract for the
  service caller (phase M5): catch ``ManifestError`` and run with memory disabled.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import structlog
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.memory.case import MemoryCase
from src.schemas import degree_of_unsaturation, parse_formula

RDLogger.DisableLog("rdApp.*")

log = structlog.get_logger("nmr_agent.memory")

STORE_SCHEMA_VERSION = "1"
EMPTY_MEMORY_VERSION = "none"
MANIFEST_NAME = "manifest.json"
CASES_DIR = "cases"


# --------------------------------------------------------------------------- errors


class ManifestError(Exception):
    """The store as a whole is unusable (manifest missing or invalid). Raised in strict and lenient mode."""


class CaseRejectedError(Exception):
    """Strict mode: a case file was rejected. ``file`` and ``reason`` name the file and why."""

    def __init__(self, file: str, reason: str):
        super().__init__(f"{file}: {reason}")
        self.file = file
        self.reason = reason


# --------------------------------------------------------------------------- manifest

_Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: Annotated[str, StringConstraints(pattern=r"^mem-\d{6}$")]
    sha256: _Sha256 = Field(description="hex SHA-256 of the raw bytes of cases/<case_id>.json")


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    store_schema_version: str
    cases: list[ManifestEntry]


def memory_version(cases: list[tuple[str, str]] | list[ManifestEntry]) -> str:
    """First 12 hex chars of SHA-256 over ``"<case_id>:<sha256>\\n"`` lines sorted by case id.

    "none" for an empty store.

    Accepts ManifestEntry objects or (case_id, sha256) pairs.
    """
    pairs = sorted((c.case_id, c.sha256) if isinstance(c, ManifestEntry) else c for c in cases)
    if not pairs:
        return EMPTY_MEMORY_VERSION
    text = "".join(f"{cid}:{sha}\n" for cid, sha in pairs)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest(root: Path) -> Manifest:
    path = root / MANIFEST_NAME
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise ManifestError(f"{path}: cannot read manifest ({e.strerror or e})") from e
    try:
        manifest = Manifest.model_validate_json(raw)
    except ValidationError as e:
        raise ManifestError(f"{path}: invalid manifest: {_short(e)}") from e
    if manifest.store_schema_version != STORE_SCHEMA_VERSION:
        raise ManifestError(
            f"{path}: store_schema_version {manifest.store_schema_version!r} not supported "
            f"(expected {STORE_SCHEMA_VERSION!r})"
        )
    ids = [c.case_id for c in manifest.cases]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ManifestError(f"{path}: duplicate case ids in manifest: {', '.join(dupes)}")
    return manifest


def write_manifest(root: Path, entries: list[ManifestEntry]) -> None:
    """Write ``manifest.json`` (entries sorted by case id). Used by tests and later by the approve command."""
    body = {
        "store_schema_version": STORE_SCHEMA_VERSION,
        "cases": [
            {"case_id": e.case_id, "sha256": e.sha256} for e in sorted(entries, key=lambda e: e.case_id)
        ],
    }
    (root / MANIFEST_NAME).write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- outcome config / store


class OutcomeConfig(BaseModel):
    """The frozen evaluation configuration every case's ``agent_outcome`` must have been collected under."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    model_id: str
    prompt_version: str
    settings_hash: str


@dataclass(frozen=True)
class Rejection:
    file: str
    reason: str


@dataclass(frozen=True)
class MemoryStore:
    cases: tuple[MemoryCase, ...]  # accepted cases, in case_id order
    memory_version: str  # over the accepted cases only
    rejections: tuple[Rejection, ...] = field(default=())  # lenient mode; always empty in strict mode


# --------------------------------------------------------------------------- derived chemistry


def _short(e: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}" for err in e.errors()
    )


def _scaffolds(mol: Chem.Mol) -> tuple[str, str]:
    """(Murcko, generic Murcko) canonical SMILES; '' for an acyclic molecule."""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return "", ""
    generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    return Chem.MolToSmiles(scaffold), Chem.MolToSmiles(generic)


def chemistry_fields(smiles: str) -> dict[str, object] | None:
    """Derived fields RDKit computes from a SMILES (None if unparseable); the loader's source of truth."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    murcko, generic = _scaffolds(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    counts = parse_formula(formula)
    return {
        "smiles": Chem.MolToSmiles(mol),
        "inchikey": Chem.MolToInchiKey(mol),
        "formula": formula,
        "murcko_scaffold": murcko,
        "generic_scaffold": generic,
        "dou": degree_of_unsaturation(counts),
        "carbon_count": counts.get("C", 0),
        "heavy_atoms": sum(n for el, n in counts.items() if el != "H"),
    }


def _chemistry_problems(case: MemoryCase) -> list[str]:
    computed = chemistry_fields(case.smiles)
    if computed is None:
        return [f"smiles {case.smiles!r} cannot be parsed by RDKit"]
    problems = [
        f"{name} stored {getattr(case, name)!r} != recomputed {value!r}"
        for name, value in computed.items()
        if name not in ("dou", "carbon_count", "heavy_atoms") and getattr(case, name) != value
    ]
    # dou / carbon_count / heavy_atoms are checked against the *stored* formula, which the loop above has
    # already matched to the structure (or reported).
    try:
        counts = parse_formula(case.formula)
    except ValueError as e:
        return [*problems, f"formula {case.formula!r} is not parseable: {e}"]
    from_formula = {
        "dou": degree_of_unsaturation(counts),
        "carbon_count": counts.get("C", 0),
        "heavy_atoms": sum(n for el, n in counts.items() if el != "H"),
    }
    problems += [
        f"{name} stored {getattr(case, name)!r} != from formula {value!r}"
        for name, value in from_formula.items()
        if getattr(case, name) != value
    ]
    return problems


def _outcome_problems(case: MemoryCase, expected: OutcomeConfig) -> list[str]:
    out = case.agent_outcome
    return [
        f"agent_outcome.{name} {getattr(out, name)!r} != expected {getattr(expected, name)!r}"
        for name in ("model_id", "prompt_version", "settings_hash")
        if getattr(out, name) != getattr(expected, name)
    ]


# --------------------------------------------------------------------------- loader


def load_store(
    root: Path, *, strict: bool, expected_outcome_config: OutcomeConfig | None = None
) -> MemoryStore:
    """Load ``root/cases/*.json`` against ``root/manifest.json``.

    A case file is rejected when:

    a. its JSON fails ``MemoryCase`` validation (approval fields are required);
    b. its file name is not ``<case_id>.json``;
    c. it is missing from the manifest, or the manifest lists a case with no file;
    d. its SHA-256 differs from the manifest;
    e. recomputing from the stored smiles with RDKit gives a different canonical SMILES, InChIKey, molecular
       formula, Murcko scaffold or generic scaffold, or ``dou``, ``carbon_count`` or ``heavy_atoms`` differ
       from the formula;
    f. ``expected_outcome_config`` is given and ``agent_outcome`` differs in model_id, prompt_version or
       settings_hash.

    ``strict=True``: the first rejection raises :class:`CaseRejectedError` naming file and reason.
    ``strict=False``: the case is skipped, a structlog error is emitted, and the rejection is returned in
    ``MemoryStore.rejections``.

    Raises :class:`ManifestError` (both modes) when the store itself is unusable; the service caller
    (phase M5) catches it and runs with memory disabled.
    """
    manifest = load_manifest(root)
    listed = {e.case_id: e.sha256 for e in manifest.cases}
    cases_dir = root / CASES_DIR
    if cases_dir.is_dir():
        files = sorted(p for p in cases_dir.glob("*.json") if p.is_file())
    elif listed:
        raise ManifestError(f"{cases_dir}: directory missing but the manifest lists {len(listed)} case(s)")
    else:
        files = []

    accepted: list[tuple[MemoryCase, str]] = []
    rejections: list[Rejection] = []

    def reject(file: str, reason: str) -> None:
        if strict:
            raise CaseRejectedError(file, reason)
        rejections.append(Rejection(file, reason))
        log.error("memory_case_rejected", file=file, reason=reason)

    present_names = {p.name for p in files}
    for path in files:
        rel = f"{CASES_DIR}/{path.name}"
        raw = path.read_bytes()
        try:
            case = MemoryCase.model_validate_json(raw)
        except ValidationError as e:
            reject(rel, f"fails MemoryCase validation: {_short(e)}")
            continue
        if path.name != f"{case.case_id}.json":
            reject(rel, f"file name does not match case_id {case.case_id!r} (expected {case.case_id}.json)")
            continue
        if case.case_id not in listed:
            reject(rel, f"case {case.case_id} is missing from the manifest")
            continue
        digest = sha256_hex(raw)
        if digest != listed[case.case_id]:
            reject(rel, f"sha256 {digest} differs from manifest {listed[case.case_id]}")
            continue
        problems = _chemistry_problems(case)
        if problems:
            reject(rel, "recomputed chemistry differs: " + "; ".join(problems))
            continue
        if expected_outcome_config is not None:
            problems = _outcome_problems(case, expected_outcome_config)
            if problems:
                reject(rel, "agent_outcome does not match the frozen configuration: " + "; ".join(problems))
                continue
        accepted.append((case, digest))

    for case_id in sorted(listed):
        if f"{case_id}.json" not in present_names:
            reject(f"{CASES_DIR}/{case_id}.json", f"manifest lists {case_id} but the file does not exist")

    accepted.sort(key=lambda pair: pair[0].case_id)
    return MemoryStore(
        cases=tuple(c for c, _ in accepted),
        memory_version=memory_version([(c.case_id, d) for c, d in accepted]),
        rejections=tuple(rejections),
    )
