"""Leakage rules 1-5 (spec §9) on synthetic molecules: each rule has a violating and a clean case."""

from __future__ import annotations

from src.memory import MemoryCase, check_injected
from src.memory.leakage import (
    rule1_structure_key,
    rule2_scaffold,
    rule3_eval_scaffolds,
    rule4_similarity,
    rule5_proposed,
    scaffold_of,
    structure_key,
    tanimoto,
)
from src.memory.store import chemistry_fields

# Hand-chosen molecules. NAPH_OME is the eval molecule; its scaffold is naphthalene.
NAPH_OME = "COc1ccc2ccccc2c1"
NAPH_OME_REORDERED = "c1ccc2cc(OC)ccc2c1"  # same structure, different SMILES text
NAPH_ET = "CCc1ccc2ccccc2c1"  # same scaffold, different structure, Tanimoto 0.44 to NAPH_OME
TOLUENE = "Cc1ccccc1"  # benzene scaffold
HEXANAMIDE = "CCCCCC(=O)NCC"  # acyclic
C12_ACID = "CCCCCCCCCCCC(=O)O"  # acyclic homologues: different structure, Tanimoto 1.0 to each other
C13_ACID = "CCCCCCCCCCCCC(=O)O"
CYCLOHEXYL_ESTER = "CN(C)CCOC(=O)C1CCCCC1"  # unrelated ring system


def item(smiles: str, position: int = 1) -> dict:
    return {"position": position, "id": f"x{position}", "smiles": smiles}


def case(smiles: str, proposed: list[str] | None = None, number: int = 1) -> MemoryCase:
    chem = chemistry_fields(smiles)
    assert chem is not None
    return MemoryCase.model_validate(
        {
            "case_id": f"mem-{number:06d}",
            "schema_version": "1",
            "provenance": {
                "source": "nmrshiftdb2",
                "source_record_id": "1",
                "spectrum_ids": {"1H": 1, "13C": 2},
                "truth_basis": "database-assigned structure",
                "extracted_by": "test",
                "proposed_on": "2026-11-14",
            },
            "approval": {"approved_by": "J", "date_added": "2026-11-15", "note": None},
            "solvent": None,
            "h_peaks": [],
            "c_peaks": [1.0],
            "assignments": None,
            "agent_outcome": {
                "collected_on": "2026-11-16",
                "model_id": "m",
                "prompt_version": "v",
                "settings_hash": "h",
                "status": "missed" if proposed else "solved",
                "proposed_smiles": proposed or [],
            },
            **chem,
        }
    )


EVAL = [item(NAPH_OME, 900)]


def test_helpers_agree_with_hand_values():
    assert scaffold_of(NAPH_OME) == "c1ccc2ccccc2c1"
    assert scaffold_of(HEXANAMIDE) == ""
    assert scaffold_of("not a smiles") is None
    assert structure_key(NAPH_OME) == structure_key(NAPH_OME_REORDERED)
    assert tanimoto(NAPH_OME, NAPH_ET) < 0.70 <= tanimoto(C12_ACID, C13_ACID)


# ---- rule 1


def test_rule1_flags_same_structure_key_even_when_the_smiles_text_differs():
    v = rule1_structure_key([item(NAPH_OME_REORDERED)], EVAL)
    assert [x.rule for x in v] == [1]


def test_rule1_flags_an_old_set_structure():
    old = {structure_key(HEXANAMIDE)}
    assert [x.rule for x in rule1_structure_key([item(HEXANAMIDE)], EVAL, old)] == [1]


def test_rule1_clean():
    assert rule1_structure_key([item(NAPH_ET), item(TOLUENE, 2)], EVAL, {structure_key(HEXANAMIDE)}) == []


def test_rule1_accepts_memory_cases():
    assert len(rule1_structure_key([case(NAPH_OME_REORDERED)], EVAL)) == 1
    assert rule1_structure_key([case(TOLUENE)], EVAL) == []


# ---- rule 2


def test_rule2_flags_a_shared_scaffold():
    v = rule2_scaffold([item(NAPH_ET)], EVAL)
    assert [x.rule for x in v] == [2]


def test_rule2_recomputes_the_scaffold_from_the_smiles_not_the_stored_field():
    lying = {**item(NAPH_ET), "scaffold": "C1CCCCC1"}  # the stored field claims a different scaffold
    assert len(rule2_scaffold([lying], EVAL)) == 1


def test_rule2_clean_and_acyclic_never_overlaps():
    assert rule2_scaffold([item(CYCLOHEXYL_ESTER), item(TOLUENE, 2)], EVAL) == []
    assert rule2_scaffold([item(HEXANAMIDE)], [item(C12_ACID)]) == []


def test_rule2_accepts_memory_cases():
    assert len(rule2_scaffold([case(NAPH_ET)], EVAL)) == 1
    assert rule2_scaffold([case(CYCLOHEXYL_ESTER)], EVAL) == []


# ---- rule 3


def test_rule3_flags_acyclic_and_benzene_eval_molecules():
    v = rule3_eval_scaffolds([item(HEXANAMIDE, 1), item(TOLUENE, 2), item(NAPH_OME, 3)])
    assert [(x.rule, x.item) for x in v] == [(3, "x1@1"), (3, "x2@2")]


def test_rule3_clean():
    assert rule3_eval_scaffolds([item(NAPH_OME), item(CYCLOHEXYL_ESTER, 2)]) == []


# ---- rule 4


def test_rule4_flags_tanimoto_at_or_above_0_70():
    v = rule4_similarity([item(C13_ACID)], [item(C12_ACID, 900)])
    assert [x.rule for x in v] == [4]


def test_rule4_clean_below_the_limit():
    assert rule4_similarity([item(NAPH_ET)], EVAL) == []  # 0.44
    assert rule4_similarity([case(CYCLOHEXYL_ESTER)], EVAL) == []


def test_rule4_limit_is_inclusive():
    assert len(rule4_similarity([item(NAPH_ET)], EVAL, limit=tanimoto(NAPH_OME, NAPH_ET))) == 1


def test_rule4_accepts_memory_cases():
    assert len(rule4_similarity([case(C13_ACID)], [item(C12_ACID)])) == 1


# ---- rule 5


def test_rule5_flags_a_proposed_smiles_that_is_an_eval_molecule():
    v = rule5_proposed([case(TOLUENE, proposed=[CYCLOHEXYL_ESTER, NAPH_OME_REORDERED])], EVAL)
    assert [x.rule for x in v] == [5]
    assert "MEM-EVAL" in v[0].detail


def test_rule5_flags_a_proposed_smiles_with_an_eval_scaffold():
    assert len(rule5_proposed([case(TOLUENE, proposed=[NAPH_ET])], EVAL)) == 1


def test_rule5_clean():
    assert (
        rule5_proposed([case(TOLUENE, proposed=[CYCLOHEXYL_ESTER, HEXANAMIDE, "not a smiles"])], EVAL) == []
    )
    assert rule5_proposed([case(TOLUENE)], EVAL) == []  # solved, nothing proposed


# ---- one eval molecule against injected cases (rule 6 reuse)


def test_check_injected_reports_rules_1_2_4_for_one_eval_molecule():
    cases = [case(NAPH_OME_REORDERED, number=1), case(CYCLOHEXYL_ESTER, number=2)]
    v = check_injected(item(NAPH_OME, 900), cases)
    assert sorted({x.rule for x in v}) == [1, 2, 4]
    assert {x.item for x in v} == {"mem-000001"}
    assert check_injected(item(NAPH_OME, 900), [cases[1]]) == []
