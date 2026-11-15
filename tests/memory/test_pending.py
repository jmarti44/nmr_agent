"""The pending case and the approval transition (spec §6). No store, no model."""

from __future__ import annotations

import datetime as dt
import json

import pytest
from pydantic import ValidationError

from src.memory import MemoryCase, PendingCase, finalize
from src.memory.pending import chemistry_problems, dump_case, placeholder_case
from tests.memory.test_store import ETHYL_BENZOATE, make_case


def pending_dict(**overrides) -> dict:
    case = make_case(**overrides)
    del case["approval"]
    return case


def test_pending_has_every_memory_case_field_except_approval_in_the_same_order():
    expected = [n for n in MemoryCase.model_fields if n != "approval"]
    assert list(PendingCase.model_fields) == expected
    for name in expected:
        assert PendingCase.model_fields[name].annotation == MemoryCase.model_fields[name].annotation, name


def test_pending_refuses_an_approval_and_unknown_fields():
    with pytest.raises(ValidationError):
        PendingCase.model_validate(make_case())  # still has "approval"
    with pytest.raises(ValidationError):
        PendingCase.model_validate(pending_dict(surprise=1))


def test_finalize_adds_the_approval_and_validates_as_a_memory_case():
    p = PendingCase.model_validate(pending_dict())
    case = finalize(p, "A. Person", dt.date(2026, 11, 20))
    assert isinstance(case, MemoryCase)
    assert case.approval.approved_by == "A. Person"
    assert case.approval.date_added == dt.date(2026, 11, 20) and case.approval.note is None
    assert case.model_dump(exclude={"approval"}) == p.model_dump()


def test_finalize_refuses_an_empty_approver():
    p = PendingCase.model_validate(pending_dict())
    with pytest.raises(ValidationError):
        finalize(p, "", dt.date(2026, 11, 20))


def test_dump_case_is_stable_and_puts_the_approval_after_the_provenance():
    case = finalize(PendingCase.model_validate(pending_dict()), "A", dt.date(2026, 11, 20))
    text = dump_case(case)
    assert text.endswith("\n") and text == dump_case(case)
    keys = list(json.loads(text))
    assert keys.index("approval") == keys.index("provenance") + 1
    assert MemoryCase.model_validate_json(text) == case


def test_chemistry_problems_are_the_loaders_rules():
    assert chemistry_problems(PendingCase.model_validate(pending_dict())) == []
    wrong = PendingCase.model_validate(pending_dict(dou=4.0))
    assert any("dou" in p for p in chemistry_problems(wrong))
    wrong = PendingCase.model_validate(pending_dict(murcko_scaffold="C1CCCCC1"))
    assert any("murcko_scaffold" in p for p in chemistry_problems(wrong))


def test_the_placeholder_view_is_not_an_approval():
    p = PendingCase.model_validate(pending_dict(**{"smiles": ETHYL_BENZOATE["smiles"]}))
    assert placeholder_case(p).approval.approved_by == "pending-check"
