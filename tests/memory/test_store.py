"""Loader tests. Every store is built in a temporary directory, never under the repository's memory/."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest
import structlog
from pydantic import ValidationError

from src.memory import (
    CASE_SCHEMA_VERSION,
    CaseRejectedError,
    ManifestError,
    MemoryCase,
    OutcomeConfig,
    load_store,
    memory_version,
)

# Hand-checked values (not produced by the code under test).
ETHYL_BENZOATE = {
    "smiles": "CCOC(=O)c1ccccc1",
    "inchikey": "MTZQAGJQAFMTAQ-UHFFFAOYSA-N",
    "formula": "C9H10O2",
    "dou": 5.0,
    "carbon_count": 9,
    "heavy_atoms": 11,
    "murcko_scaffold": "c1ccccc1",
    "generic_scaffold": "C1CCCCC1",
}
ETHYL_ACETATE = {
    "smiles": "CCOC(C)=O",
    "inchikey": "XEKOWRVHYACXOJ-UHFFFAOYSA-N",
    "formula": "C4H8O2",
    "dou": 1.0,
    "carbon_count": 4,
    "heavy_atoms": 6,
    "murcko_scaffold": "",
    "generic_scaffold": "",
}
CONFIG = OutcomeConfig(model_id="claude-opus-5-5", prompt_version="v9", settings_hash="abc123")


def make_case(case_id: str = "mem-000001", chem: dict | None = None, **overrides) -> dict:
    case = {
        "case_id": case_id,
        "schema_version": CASE_SCHEMA_VERSION,
        "provenance": {
            "source": "nmrshiftdb2",
            "source_record_id": "12345",
            "spectrum_ids": {"1H": 1, "13C": 2},
            "truth_basis": "database-assigned structure",
            "extracted_by": "propose.py@deadbeef",
            "proposed_on": "2026-11-14",
        },
        "approval": {"approved_by": "J", "date_added": "2026-11-15", "note": None},
        "solvent": "CDCl3",
        "h_peaks": [{"shift_ppm": 1.4, "integral": 3.0}, {"shift_ppm": 4.4, "integral": None}],
        "c_peaks": [14.3, 61.0],
        "assignments": [{"nucleus": "1H", "shift_ppm": 1.4, "environment": "CH3"}],
        "agent_outcome": {
            "collected_on": "2026-11-16",
            "model_id": CONFIG.model_id,
            "prompt_version": CONFIG.prompt_version,
            "settings_hash": CONFIG.settings_hash,
            "status": "solved",
            "proposed_smiles": ["CCOC(=O)c1ccccc1"],
        },
        **(chem or ETHYL_BENZOATE),
    }
    case.update(overrides)
    return case


def dump(case: dict) -> bytes:
    return (json.dumps(case, indent=2) + "\n").encode()


def write_store(root: Path, files: dict[str, bytes], manifest: dict[str, str] | None = None) -> None:
    """files: name -> bytes under cases/. manifest: case_id -> sha256 (default: hash every file by stem)."""
    (root / "cases").mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (root / "cases" / name).write_bytes(data)
    if manifest is None:
        manifest = {Path(n).stem: hashlib.sha256(d).hexdigest() for n, d in files.items()}
    body = {
        "store_schema_version": "1",
        "cases": [{"case_id": k, "sha256": v} for k, v in manifest.items()],
    }
    (root / "manifest.json").write_text(json.dumps(body))


def one_case_store(root: Path, case: dict, *, name: str | None = None) -> None:
    data = dump(case)
    name = name or f"{case['case_id']}.json"
    write_store(root, {name: data}, {case["case_id"]: hashlib.sha256(data).hexdigest()})


# --------------------------------------------------------------------------- rejection scenarios
# Each builds a store that violates exactly one rule. (rule, builder, expected substring of the reason)


def _a_missing_approval(root: Path) -> None:
    case = make_case()
    del case["approval"]
    one_case_store(root, case)


def _a_extra_field(root: Path) -> None:
    one_case_store(root, make_case(surprise=1))


def _a_bad_id(root: Path) -> None:
    # the manifest cannot list an id that is itself invalid, so it is empty here
    write_store(root, {"mem-12.json": dump(make_case(case_id="mem-12"))}, {})


def _a_no_answer_with_candidates(root: Path) -> None:
    case = make_case()
    case["agent_outcome"].update(status="no_answer", proposed_smiles=["CCO"])
    one_case_store(root, case)


def _b_wrong_name(root: Path) -> None:
    case = make_case()
    data = dump(case)
    write_store(root, {"mem-000002.json": data}, {"mem-000001": hashlib.sha256(data).hexdigest()})


def _c_not_in_manifest(root: Path) -> None:
    write_store(root, {"mem-000001.json": dump(make_case())}, {})


def _c_manifest_without_file(root: Path) -> None:
    data = dump(make_case())
    write_store(root, {}, {"mem-000001": hashlib.sha256(data).hexdigest()})


def _d_hash_differs(root: Path) -> None:
    write_store(root, {"mem-000001.json": dump(make_case())}, {"mem-000001": "0" * 64})


def _e_smiles_not_canonical(root: Path) -> None:
    one_case_store(root, make_case(chem={**ETHYL_BENZOATE, "smiles": "c1ccccc1C(=O)OCC"}))


def _e_inchikey(root: Path) -> None:
    one_case_store(root, make_case(chem={**ETHYL_BENZOATE, "inchikey": "AAAAAAAAAAAAAA-UHFFFAOYSA-N"}))


def _e_formula(root: Path) -> None:
    one_case_store(root, make_case(chem={**ETHYL_BENZOATE, "formula": "C9H10O3"}))


def _e_murcko(root: Path) -> None:
    one_case_store(root, make_case(chem={**ETHYL_BENZOATE, "murcko_scaffold": ""}))


def _e_murcko_acyclic_not_empty(root: Path) -> None:
    one_case_store(
        root, make_case(chem={**ETHYL_ACETATE, "murcko_scaffold": "c1ccccc1", "generic_scaffold": ""})
    )


def _e_generic(root: Path) -> None:
    one_case_store(root, make_case(chem={**ETHYL_BENZOATE, "generic_scaffold": "c1ccccc1"}))


def _e_dou(root: Path) -> None:
    one_case_store(root, make_case(chem={**ETHYL_BENZOATE, "dou": 4.0}))


def _e_carbon_count(root: Path) -> None:
    one_case_store(root, make_case(chem={**ETHYL_BENZOATE, "carbon_count": 8}))


def _e_heavy_atoms(root: Path) -> None:
    one_case_store(root, make_case(chem={**ETHYL_BENZOATE, "heavy_atoms": 12}))


def _e_unparseable_smiles(root: Path) -> None:
    one_case_store(root, make_case(chem={**ETHYL_BENZOATE, "smiles": "not a smiles"}))


def _f_model_id(root: Path) -> None:
    case = make_case()
    case["agent_outcome"]["model_id"] = "other-model"
    one_case_store(root, case)


def _f_prompt_version(root: Path) -> None:
    case = make_case()
    case["agent_outcome"]["prompt_version"] = "v1"
    one_case_store(root, case)


def _f_settings_hash(root: Path) -> None:
    case = make_case()
    case["agent_outcome"]["settings_hash"] = "zzz"
    one_case_store(root, case)


SCENARIOS = [
    ("a", _a_missing_approval, "approval"),
    ("a", _a_extra_field, "surprise"),
    ("a", _a_bad_id, "case_id"),
    ("a", _a_no_answer_with_candidates, "proposed_smiles"),
    ("b", _b_wrong_name, "file name"),
    ("c", _c_not_in_manifest, "missing from the manifest"),
    ("c", _c_manifest_without_file, "file does not exist"),
    ("d", _d_hash_differs, "sha256"),
    ("e", _e_smiles_not_canonical, "smiles stored"),
    ("e", _e_inchikey, "inchikey"),
    ("e", _e_formula, "formula stored"),
    ("e", _e_murcko, "murcko_scaffold"),
    ("e", _e_murcko_acyclic_not_empty, "murcko_scaffold"),
    ("e", _e_generic, "generic_scaffold"),
    ("e", _e_dou, "dou"),
    ("e", _e_carbon_count, "carbon_count"),
    ("e", _e_heavy_atoms, "heavy_atoms"),
    ("e", _e_unparseable_smiles, "cannot be parsed"),
    ("f", _f_model_id, "model_id"),
    ("f", _f_prompt_version, "prompt_version"),
    ("f", _f_settings_hash, "settings_hash"),
]
IDS = [f"{rule}-{fn.__name__.lstrip('_')}" for rule, fn, _ in SCENARIOS]


@pytest.mark.parametrize(("rule", "build", "needle"), SCENARIOS, ids=IDS)
def test_strict_mode_raises_naming_file_and_reason(tmp_path, rule, build, needle):
    build(tmp_path)
    with pytest.raises(CaseRejectedError) as exc:
        load_store(tmp_path, strict=True, expected_outcome_config=CONFIG)
    assert exc.value.file.startswith("cases/mem-")
    assert exc.value.file.endswith(".json")
    assert needle in exc.value.reason
    assert exc.value.file in str(exc.value)


@pytest.mark.parametrize(("rule", "build", "needle"), SCENARIOS, ids=IDS)
def test_lenient_mode_skips_logs_and_reports(tmp_path, rule, build, needle):
    build(tmp_path)
    with structlog.testing.capture_logs() as logs:
        store = load_store(tmp_path, strict=False, expected_outcome_config=CONFIG)
    assert store.cases == ()
    assert store.memory_version == "none"
    # a misnamed file (rule b) also leaves its manifest entry without a file, so it can yield two rejections
    assert 1 <= len(store.rejections) <= 2
    rej = store.rejections[0]
    assert needle in rej.reason
    errors = [e for e in logs if e["log_level"] == "error"]
    assert [(e["file"], e["reason"]) for e in errors] == [(r.file, r.reason) for r in store.rejections]


def test_expected_outcome_config_is_required(tmp_path):
    one_case_store(tmp_path, make_case())
    for strict in (True, False):
        with pytest.raises(TypeError):
            load_store(tmp_path, strict=strict)  # type: ignore[call-arg]


def test_rule_f_not_applied_when_config_is_explicitly_none(tmp_path):
    case = make_case()
    case["agent_outcome"]["model_id"] = "anything"
    one_case_store(tmp_path, case)
    assert len(load_store(tmp_path, strict=True, expected_outcome_config=None).cases) == 1


def test_lenient_keeps_good_cases_and_rejects_only_bad(tmp_path):
    good1 = dump(make_case("mem-000001"))
    good2 = dump(make_case("mem-000003", chem=ETHYL_ACETATE))
    bad = dump(make_case("mem-000002", chem={**ETHYL_BENZOATE, "dou": 4.0}))
    write_store(tmp_path, {"mem-000001.json": good1, "mem-000002.json": bad, "mem-000003.json": good2})
    store = load_store(tmp_path, strict=False, expected_outcome_config=CONFIG)
    assert [c.case_id for c in store.cases] == ["mem-000001", "mem-000003"]
    assert [r.file for r in store.rejections] == ["cases/mem-000002.json"]
    # version covers accepted cases only
    expected = memory_version(
        [
            ("mem-000001", hashlib.sha256(good1).hexdigest()),
            ("mem-000003", hashlib.sha256(good2).hexdigest()),
        ]
    )
    assert store.memory_version == expected


# --------------------------------------------------------------------------- valid stores


def test_valid_store_loads_in_case_id_order_with_expected_version(tmp_path):
    d1 = dump(make_case("mem-000002", chem=ETHYL_ACETATE))
    d2 = dump(make_case("mem-000001"))
    write_store(tmp_path, {"mem-000002.json": d1, "mem-000001.json": d2})
    s1, s2 = hashlib.sha256(d1).hexdigest(), hashlib.sha256(d2).hexdigest()
    text = f"mem-000001:{s2}\nmem-000002:{s1}\n"
    expected = hashlib.sha256(text.encode()).hexdigest()[:12]

    for strict in (True, False):
        store = load_store(tmp_path, strict=strict, expected_outcome_config=CONFIG)
        assert [c.case_id for c in store.cases] == ["mem-000001", "mem-000002"]
        assert store.memory_version == expected
        assert len(store.memory_version) == 12
        assert store.rejections == ()


def test_memory_version_is_order_independent_and_hash_sensitive():
    a, b = ("mem-000001", "a" * 64), ("mem-000002", "b" * 64)
    assert memory_version([a, b]) == memory_version([b, a])
    assert memory_version([a, b]) != memory_version([a, ("mem-000002", "c" * 64)])


def test_empty_store_version_is_none(tmp_path):
    write_store(tmp_path, {}, {})
    for strict in (True, False):
        store = load_store(tmp_path, strict=strict, expected_outcome_config=CONFIG)
        assert store.cases == ()
        assert store.memory_version == "none"
    assert memory_version([]) == "none"


def test_empty_store_without_cases_dir(tmp_path):
    write_store(tmp_path, {}, {})
    (tmp_path / "cases").rmdir()
    assert load_store(tmp_path, strict=True, expected_outcome_config=CONFIG).memory_version == "none"


def test_pending_and_rejected_directories_are_ignored(tmp_path):
    one_case_store(tmp_path, make_case())
    for sub in ("pending", "rejected"):
        (tmp_path / sub).mkdir()
        # a broken file and a valid-looking unlisted file: neither may be read or raise
        (tmp_path / sub / "mem-000009.json").write_text("{ not json")
        (tmp_path / sub / "mem-000010.json").write_bytes(dump(make_case("mem-000010")))
    baseline = tmp_path.parent / (tmp_path.name + "-baseline")
    baseline.mkdir()
    one_case_store(baseline, make_case())
    for strict in (True, False):
        store = load_store(tmp_path, strict=strict, expected_outcome_config=CONFIG)
        assert [c.case_id for c in store.cases] == ["mem-000001"]
        assert store.rejections == ()
        assert (
            store.memory_version
            == load_store(baseline, strict=True, expected_outcome_config=CONFIG).memory_version
        )


def test_only_json_files_in_cases_are_read(tmp_path):
    one_case_store(tmp_path, make_case())
    (tmp_path / "cases" / "README.md").write_text("notes")
    (tmp_path / "cases" / "sub").mkdir()
    assert len(load_store(tmp_path, strict=True, expected_outcome_config=CONFIG).cases) == 1


# --------------------------------------------------------------------------- store-level faults


def test_missing_manifest_raises_manifest_error_in_both_modes(tmp_path):
    (tmp_path / "cases").mkdir()
    for strict in (True, False):
        with pytest.raises(ManifestError):
            load_store(tmp_path, strict=strict, expected_outcome_config=CONFIG)


@pytest.mark.parametrize(
    "body",
    [
        "{ not json",
        json.dumps({"store_schema_version": "1"}),
        json.dumps({"store_schema_version": "99", "cases": []}),
        json.dumps({"store_schema_version": "1", "cases": [], "extra": 1}),
        json.dumps({"store_schema_version": "1", "cases": [{"case_id": "bad", "sha256": "0" * 64}]}),
        json.dumps({"store_schema_version": "1", "cases": [{"case_id": "mem-000001", "sha256": "xyz"}]}),
        json.dumps(
            {
                "store_schema_version": "1",
                "cases": [{"case_id": "mem-000001", "sha256": "0" * 64}] * 2,
            }
        ),
    ],
    ids=["unparseable", "no-cases-key", "unknown-version", "extra-key", "bad-id", "bad-hash", "duplicate-id"],
)
def test_invalid_manifest_raises_manifest_error_in_both_modes(tmp_path, body):
    (tmp_path / "manifest.json").write_text(body)
    for strict in (True, False):
        with pytest.raises(ManifestError):
            load_store(tmp_path, strict=strict, expected_outcome_config=CONFIG)


def test_missing_cases_dir_with_listed_cases_raises_manifest_error(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"store_schema_version": "1", "cases": [{"case_id": "mem-000001", "sha256": "0" * 64}]})
    )
    with pytest.raises(ManifestError):
        load_store(tmp_path, strict=False, expected_outcome_config=CONFIG)


# --------------------------------------------------------------------------- MemoryCase model


def test_valid_case_model_roundtrip_with_nested_models():
    case = MemoryCase.model_validate(make_case())
    assert case.h_peaks[1].integral is None
    assert case.approval.date_added.isoformat() == "2026-11-15"
    assert case.solvent == "CDCl3"
    assert MemoryCase.model_validate_json(case.model_dump_json()) == case


def test_unreported_solvent_and_null_assignments_are_valid():
    MemoryCase.model_validate(make_case(solvent=None, assignments=None))


def test_no_answer_with_empty_candidates_is_valid():
    case = make_case()
    case["agent_outcome"].update(status="no_answer", proposed_smiles=[])
    MemoryCase.model_validate(case)


@pytest.mark.parametrize("bad", ["mem-12345", "mem-1234567", "MEM-000001", "mem-00000a", "x mem-000001"])
def test_case_id_pattern(bad):
    with pytest.raises(ValidationError):
        MemoryCase.model_validate(make_case(case_id=bad))


def test_unknown_schema_version_and_status_rejected():
    with pytest.raises(ValidationError):
        MemoryCase.model_validate(make_case(schema_version="2"))
    case = make_case()
    case["agent_outcome"]["status"] = "maybe"
    with pytest.raises(ValidationError):
        MemoryCase.model_validate(case)


@pytest.mark.parametrize("path", [("approval",), ("approval", "approved_by"), ("approval", "date_added")])
def test_missing_approval_fields_fail(path):
    case = copy.deepcopy(make_case())
    target = case
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]
    with pytest.raises(ValidationError):
        MemoryCase.model_validate(case)


def test_extra_fields_forbidden_in_nested_models():
    case = make_case()
    case["provenance"]["extra"] = 1
    with pytest.raises(ValidationError):
        MemoryCase.model_validate(case)
    case = make_case()
    case["h_peaks"][0]["extra"] = 1
    with pytest.raises(ValidationError):
        MemoryCase.model_validate(case)


def test_unknown_solvent_rejected():
    with pytest.raises(ValidationError):
        MemoryCase.model_validate(make_case(solvent="toluene-d8"))


# --------------------------------------------------------------------------- RM1-1: unreadable files


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read mode-000 files")
def test_unreadable_file_is_a_rejection_strict(tmp_path):
    one_case_store(tmp_path, make_case())
    path = tmp_path / "cases" / "mem-000001.json"
    path.chmod(0o000)
    try:
        with pytest.raises(CaseRejectedError) as exc:
            load_store(tmp_path, strict=True, expected_outcome_config=CONFIG)
    finally:
        path.chmod(0o644)
    assert exc.value.file == "cases/mem-000001.json"
    assert "cannot read file" in exc.value.reason


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read mode-000 files")
def test_unreadable_file_is_skipped_lenient_and_other_case_loads(tmp_path):
    good, locked = dump(make_case("mem-000001")), dump(make_case("mem-000002", chem=ETHYL_ACETATE))
    write_store(tmp_path, {"mem-000001.json": good, "mem-000002.json": locked})
    path = tmp_path / "cases" / "mem-000002.json"
    path.chmod(0o000)
    try:
        with structlog.testing.capture_logs() as logs:
            store = load_store(tmp_path, strict=False, expected_outcome_config=CONFIG)
    finally:
        path.chmod(0o644)
    assert [c.case_id for c in store.cases] == ["mem-000001"]
    assert [r.file for r in store.rejections] == ["cases/mem-000002.json"]
    assert "cannot read file" in store.rejections[0].reason
    assert [e["file"] for e in logs if e["log_level"] == "error"] == ["cases/mem-000002.json"]


@pytest.mark.parametrize("strict", [True, False])
def test_oserror_on_read_is_a_rejection_even_as_root(tmp_path, monkeypatch, strict):
    """Same rule without relying on file modes: the read itself raises."""
    one_case_store(tmp_path, make_case())
    real = Path.read_bytes

    def flaky(self):
        if self.name == "mem-000001.json":
            raise PermissionError(13, "Permission denied")
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", flaky)
    if strict:
        with pytest.raises(CaseRejectedError, match="cannot read file: Permission denied"):
            load_store(tmp_path, strict=True, expected_outcome_config=CONFIG)
    else:
        store = load_store(tmp_path, strict=False, expected_outcome_config=CONFIG)
        assert store.cases == ()
        assert "Permission denied" in store.rejections[0].reason


# --------------------------------------------------------------------------- RM1-3: not a regular file


def _symlinked_store(root: Path, outside: Path) -> None:
    data = dump(make_case())
    target = outside / "elsewhere.json"
    target.write_bytes(data)
    (root / "cases").mkdir()
    os.symlink(target, root / "cases" / "mem-000001.json")
    write_store(root, {}, {"mem-000001": hashlib.sha256(data).hexdigest()})


def test_symlinked_case_file_rejected_strict(tmp_path):
    root, outside = tmp_path / "store", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    _symlinked_store(root, outside)
    with pytest.raises(CaseRejectedError, match="symlink") as exc:
        load_store(root, strict=True, expected_outcome_config=CONFIG)
    assert exc.value.file == "cases/mem-000001.json"


def test_symlinked_case_file_rejected_lenient_with_single_rejection(tmp_path):
    root, outside = tmp_path / "store", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    _symlinked_store(root, outside)
    store = load_store(root, strict=False, expected_outcome_config=CONFIG)
    assert store.cases == ()
    assert [r.file for r in store.rejections] == ["cases/mem-000001.json"]
    assert "symlink" in store.rejections[0].reason


@pytest.mark.parametrize("strict", [True, False])
def test_dangling_symlink_and_directory_named_json_are_rejected(tmp_path, strict):
    one_case_store(tmp_path, make_case())
    os.symlink(tmp_path / "nowhere.json", tmp_path / "cases" / "mem-000002.json")
    (tmp_path / "cases" / "mem-000003.json").mkdir()
    if strict:
        with pytest.raises(CaseRejectedError):
            load_store(tmp_path, strict=True, expected_outcome_config=CONFIG)
        return
    store = load_store(tmp_path, strict=False, expected_outcome_config=CONFIG)
    assert [c.case_id for c in store.cases] == ["mem-000001"]
    assert [r.file for r in store.rejections] == ["cases/mem-000002.json", "cases/mem-000003.json"]
