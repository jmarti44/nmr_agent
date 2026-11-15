"""CI test over the committed bench/memory/ selections (spec §7.1, §7.2, §9). Reads JSON only: no SD file."""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

from src.memory.leakage import (
    BENZENE_SCAFFOLD,
    rule1_structure_key,
    rule2_scaffold,
    rule3_eval_scaffolds,
    rule4_similarity,
    scaffold_of,
    structure_key,
)

BENCH = Path(__file__).resolve().parents[2] / "bench"
OLD_FILES = [BENCH / d / "selection.json" for d in ("results", "results_main", "results_unreported")]


def _load(name: str, key: str) -> list[dict]:
    return json.loads((BENCH / "memory" / name).read_text())[key]


@pytest.fixture(scope="module")
def mem_eval() -> list[dict]:
    return _load("mem_eval.json", "records")


@pytest.fixture(scope="module")
def entries() -> list[dict]:
    return _load("memory_order.json", "entries")


@pytest.fixture(scope="module")
def old_sets() -> list[dict]:
    return [r for f in OLD_FILES for r in json.loads(f.read_text())["records"]]


def stratum(heavy: int) -> str:
    return ">20" if heavy > 20 else "11-20" if heavy >= 11 else "<=10"


# ---- MEM-EVAL


def test_mem_eval_size_and_bins(mem_eval):
    assert len(mem_eval) == 100
    assert len({r["id"] for r in mem_eval}) == 100
    assert collections.Counter(stratum(r["heavy_atoms"]) for r in mem_eval) == {"11-20": 50, ">20": 50}


def test_mem_eval_keeps_the_benchmark_record_fields(mem_eval):
    for r in mem_eval:
        assert {
            "id",
            "smiles",
            "inchikey",
            "formula",
            "solvent",
            "h_spec",
            "c_spec",
            "heavy_atoms",
            "scaffold",
        } <= set(r)
        assert structure_key(r["smiles"]) == r["inchikey"].split("-")[0]
        assert scaffold_of(r["smiles"]) == r["scaffold"]


def test_mem_eval_at_most_two_per_scaffold(mem_eval):
    assert max(collections.Counter(scaffold_of(r["smiles"]) for r in mem_eval).values()) <= 2


def test_mem_eval_has_no_acyclic_or_benzene_molecule(mem_eval):
    assert rule3_eval_scaffolds(mem_eval) == []
    assert {scaffold_of(r["smiles"]) for r in mem_eval}.isdisjoint({"", BENZENE_SCAFFOLD})


def test_mem_eval_shares_no_structure_or_scaffold_with_the_old_sets(mem_eval, old_sets):
    assert len(old_sets) == 90
    old_keys = {structure_key(r["smiles"]) for r in old_sets}
    old_scaffolds = {scaffold_of(r["smiles"]) for r in old_sets} - {""}
    assert {structure_key(r["smiles"]) for r in mem_eval}.isdisjoint(old_keys)
    assert {scaffold_of(r["smiles"]) for r in mem_eval}.isdisjoint(old_scaffolds)


# ---- memory order


@pytest.mark.parametrize("scope", ["first 200", "all candidates"])
def test_leakage_rules_hold_on_the_selections(scope, mem_eval, entries, old_sets):
    items = entries[:200] if scope == "first 200" else entries
    assert len(items) == (200 if scope == "first 200" else len(entries))
    old_keys = {structure_key(r["smiles"]) for r in old_sets}
    assert rule1_structure_key(items, mem_eval, old_keys) == []
    assert rule2_scaffold(items, mem_eval) == []
    assert rule4_similarity(items, mem_eval) == []


def test_candidates_are_unique_and_positions_are_consecutive(entries):
    assert [e["position"] for e in entries] == list(range(1, len(entries) + 1))
    assert len({e["id"] for e in entries}) == len(entries)
    assert len({structure_key(e["smiles"]) for e in entries}) == len(entries)


@pytest.mark.parametrize("n", [25, 50, 100, 200])
def test_prefix_is_40_40_20(n, entries):
    c = collections.Counter(stratum(e["heavy_atoms"]) for e in entries[:n])
    assert (c[">20"], c["11-20"], c["<=10"]) == (n * 2 // 5, n * 2 // 5, n // 5)
    assert all(e["stratum"] == stratum(e["heavy_atoms"]) for e in entries[:n])


# ---- split


def test_split_covers_every_candidate_once(entries):
    assign = _load("split.json", "assignments")
    assert sorted(a["id"] for a in assign) == sorted(e["id"] for e in entries)
    assert {a["half"] for a in assign} == {"tuning", "check"}


def test_split_keeps_scaffold_groups_whole_except_acyclic_and_benzene(entries):
    half = {a["id"]: a["half"] for a in _load("split.json", "assignments")}
    halves = collections.defaultdict(set)
    for e in entries:
        halves[scaffold_of(e["smiles"])].add(half[e["id"]])
    split_up = {s for s, h in halves.items() if len(h) > 1}
    assert split_up <= {"", BENZENE_SCAFFOLD}
    assert split_up == {"", BENZENE_SCAFFOLD}  # both big groups really are divided between the halves


# ---- the check command on synthetic stores


def test_check_command_store_path(tmp_path, mem_eval, capsys):
    from bench.memory import check
    from tests.memory.test_store import make_case, one_case_store

    assert check(None) == 0

    clean = tmp_path / "clean"
    one_case_store(clean, make_case())
    assert check(str(clean)) == 0

    leaking = tmp_path / "leaking"
    case = make_case()
    case["agent_outcome"].update(status="missed", proposed_smiles=[mem_eval[0]["smiles"]])
    one_case_store(leaking, case)
    capsys.readouterr()
    assert check(str(leaking)) == 1
    assert "rule 5 proposed SMILES: FAIL (1)" in capsys.readouterr().out
