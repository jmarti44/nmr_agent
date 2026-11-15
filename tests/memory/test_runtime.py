"""Outcome configuration and the memory runtime. Stores live in temporary directories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.core.settings import Settings
from src.memory import ManifestError, MemoryRuntime, ParamsError, RetrievalParams, outcome_config
from src.memory.runtime import (
    EXAMPLES_TOKEN_BUDGET,
    OUTCOME_SETTINGS_FIELDS,
    load_params,
    params_hash,
    settings_hash,
)
from src.memory.store import CaseRejectedError
from src.prompting.templates import FULL_VERSION
from src.schemas import Nucleus, Peak, PeakTable, SampleContext
from tests.memory.test_store import ETHYL_ACETATE, ETHYL_BENZOATE, dump, make_case, write_store

FROZEN = {
    "weight_c13": 0.5,
    "weight_h1": 0.3,
    "weight_elements": 0.2,
    "weight_dou": 0.0,
    "solvent_bonus": 0.03,
    "c13_tolerance_ppm": 2.0,
    "h1_tolerance_ppm": 0.15,
    "tie_epsilon": 0.02,
    "k": 3,
    "min_similarity": 0.0,
}
# Hash recorded in bench/memory/retrieval_params.json: pins params_hash to the benchmark's definition.
FROZEN_SHA = "bf933bfc891372472f6826e3daa80ad987af96a0ce10d18169e0f885067ac951"


def write_params(root: Path, parameters: dict = FROZEN, sha: str | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    doc = {"description": "test", "parameters": parameters, "sha256": sha or params_hash(parameters)}
    (root / "retrieval_params.json").write_text(json.dumps(doc))


def case_dict(n: int, settings: Settings, chem: dict, **overrides) -> dict:
    cfg = outcome_config(settings)
    case = make_case(f"mem-{n:06d}", chem)
    case["agent_outcome"].update(
        model_id=cfg.model_id, prompt_version=cfg.prompt_version, settings_hash=cfg.settings_hash
    )
    case.update(overrides)
    return case


def build_store(root: Path, settings: Settings, cases: list[dict]) -> Path:
    write_store(root, {f"{c['case_id']}.json": dump(c) for c in cases})
    write_params(root)
    return root


def three_cases(settings: Settings) -> list[dict]:
    third = {**ETHYL_BENZOATE, "smiles": "CCOC(=O)c1ccccc1", "murcko_scaffold": "c1ccccc1"}
    return [
        case_dict(1, settings, ETHYL_BENZOATE),
        case_dict(2, settings, ETHYL_ACETATE),
        case_dict(3, settings, third, murcko_scaffold="c1ccccc1"),  # same scaffold as case 1
    ]


def query() -> tuple[PeakTable, SampleContext]:
    peaks = [Peak(id=i, nucleus=Nucleus.C13, shift_ppm=s) for i, s in enumerate((14.3, 61.0, 130.0), 1)]
    return PeakTable(peaks=peaks), SampleContext(molecular_formula="C9H10O2")


# --------------------------------------------------------------------------- outcome_config


def test_outcome_config_values_and_stability():
    s = Settings()
    cfg = outcome_config(s)
    assert cfg.model_id == s.model and cfg.prompt_version == FULL_VERSION
    assert cfg == outcome_config(Settings()) and len(cfg.settings_hash) == 64


def test_settings_hash_is_the_sha256_of_canonical_json_of_the_listed_fields():
    s = Settings()
    canonical = json.dumps(
        {f: getattr(s, f) for f in OUTCOME_SETTINGS_FIELDS}, sort_keys=True, separators=(",", ":")
    )
    assert settings_hash(s) == hashlib.sha256(canonical.encode()).hexdigest()
    assert set(OUTCOME_SETTINGS_FIELDS) == {
        "model", "effort", "repair_effort", "enable_fallbacks", "max_llm_calls", "max_primary_attempts",
        "initial_max_tokens", "max_tokens_cap", "llm_call_timeout_s", "request_timeout_s",
    }  # fmt: skip


@pytest.mark.parametrize(
    "change",
    [
        {"model": "claude-other"},
        {"effort": "low"},
        {"repair_effort": "high"},
        {"enable_fallbacks": False},
        {"max_llm_calls": 3},
        {"max_primary_attempts": 2},
        {"initial_max_tokens": 8000},
        {"max_tokens_cap": 20000},
        {"llm_call_timeout_s": 60.0},
        {"request_timeout_s": 90.0},
    ],
)
def test_outcome_config_changes_when_a_listed_setting_changes(change):
    assert outcome_config(Settings(**change)) != outcome_config(Settings())


def test_unlisted_settings_do_not_change_the_config(tmp_path):
    base = outcome_config(Settings())
    assert outcome_config(Settings(daily_llm_budget=5, per_ip_per_hour=2, memory_dir=tmp_path)) == base


# --------------------------------------------------------------------------- parameter file


def test_frozen_hash_matches_the_benchmark_definition():
    assert params_hash(FROZEN) == FROZEN_SHA


def test_repository_frozen_file_loads_with_its_recorded_hash():
    params = load_params(Path(__file__).parents[2] / "bench" / "memory" / "retrieval_params.json")
    assert params == RetrievalParams(**FROZEN)


def test_params_with_wrong_sha256_are_refused(tmp_path):
    write_params(tmp_path, sha="0" * 64)
    with pytest.raises(ParamsError, match="does not match"):
        load_params(tmp_path / "retrieval_params.json")


def test_params_edited_after_hashing_are_refused(tmp_path):
    write_params(tmp_path)
    path = tmp_path / "retrieval_params.json"
    doc = json.loads(path.read_text())
    doc["parameters"]["k"] = 5
    path.write_text(json.dumps(doc))
    with pytest.raises(ParamsError):
        load_params(path)


@pytest.mark.parametrize("content", ["not json", "[]", '{"sha256": "x"}'])
def test_unreadable_or_malformed_params_are_refused(tmp_path, content):
    (tmp_path / "retrieval_params.json").write_text(content)
    with pytest.raises(ParamsError):
        load_params(tmp_path / "retrieval_params.json")
    with pytest.raises(ParamsError):
        load_params(tmp_path / "missing.json")


def test_runtime_refuses_a_bad_param_file(tmp_path):
    s = Settings()
    build_store(tmp_path, s, three_cases(s))
    write_params(tmp_path, sha="f" * 64)
    with pytest.raises(ParamsError):
        MemoryRuntime.from_directory(tmp_path, s, strict=True)


# --------------------------------------------------------------------------- runtime


def test_retrieved_mode_returns_best_cases_one_per_scaffold(tmp_path):
    s = Settings()
    rt = MemoryRuntime.from_directory(build_store(tmp_path, s, three_cases(s)), s, strict=True)
    assert rt.mode == "retrieved" and rt.memory_version != "none"
    results = rt.examples_for(*query())
    ids = [r.case.case_id for r in results]
    assert ids[0] in {"mem-000001", "mem-000003"} and "mem-000002" in ids
    assert len({r.case.murcko_scaffold for r in results}) == len(results) == 2
    assert all(r.similarity is not None for r in results)


def test_random_mode_is_deterministic_for_a_seed_and_has_no_similarity(tmp_path):
    s = Settings()
    root = build_store(tmp_path, s, three_cases(s))
    a = MemoryRuntime.from_directory(root, s, strict=True, mode="random", seed=7)
    b = MemoryRuntime.from_directory(root, s, strict=True, mode="random", seed=7)
    ids = lambda rt: [r.case.case_id for r in rt.examples_for(*query())]  # noqa: E731
    assert ids(a) == ids(b) and all(r.similarity is None for r in a.examples_for(*query()))
    assert {tuple(ids(a.with_seed(n))) for n in range(20)} != {tuple(ids(a))}  # the seed matters


def test_random_mode_needs_a_seed(tmp_path):
    s = Settings()
    with pytest.raises(ValueError, match="seed"):
        MemoryRuntime.from_directory(build_store(tmp_path, s, three_cases(s)), s, strict=True, mode="random")


def test_budget_is_the_examples_cap():
    assert EXAMPLES_TOKEN_BUDGET == 2000


def test_a_case_collected_under_another_configuration_is_rejected_strict_and_skipped_lenient(tmp_path):
    s = Settings()
    cases = three_cases(s)
    cases[1]["agent_outcome"]["settings_hash"] = "other"
    root = build_store(tmp_path, s, cases)
    with pytest.raises(CaseRejectedError, match="settings_hash"):
        MemoryRuntime.from_directory(root, s, strict=True)
    rt = MemoryRuntime.from_directory(root, s, strict=False)
    assert [c.case_id for c in rt.store.cases] == ["mem-000001", "mem-000003"]
    assert [r.file for r in rt.store.rejections] == ["cases/mem-000002.json"]


def test_changing_a_listed_setting_invalidates_every_collected_case(tmp_path):
    s = Settings()
    root = build_store(tmp_path, s, three_cases(s))
    rt = MemoryRuntime.from_directory(root, Settings(effort="low"), strict=False)
    assert rt.store.cases == () and rt.memory_version == "none" and len(rt.store.rejections) == 3
    assert rt.examples_for(*query()) == []


def test_missing_manifest_raises_manifest_error(tmp_path):
    write_params(tmp_path)
    with pytest.raises(ManifestError):
        MemoryRuntime.from_directory(tmp_path, Settings(), strict=False)
