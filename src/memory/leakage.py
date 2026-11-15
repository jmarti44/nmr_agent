"""Leakage rules between memory and the held-out set (spec §9, rules 1-5).

Every rule is a pure function returning its list of violations (empty when clean). Memory items may be
``memory_order.json`` entries (dicts with at least ``smiles``) or :class:`MemoryCase` objects. Structure
keys, scaffolds and fingerprints are always recomputed from the SMILES; stored values are not trusted.
Rule 6 (benchmark time) reuses rules 1, 2 and 4 through :func:`check_injected`.

An empty scaffold (acyclic molecule) is never counted as an overlap in rules 2 and 5: two acyclic molecules
share no ring system. MEM-EVAL must contain none, which rule 3 enforces.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from src.memory.case import MemoryCase
from src.memory.store import chemistry_fields

RDLogger.DisableLog("rdApp.*")

SIMILARITY_LIMIT = 0.70  # rule 4: a Tanimoto at or above this is a violation
BENZENE_SCAFFOLD = "c1ccccc1"
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

Item = Mapping[str, Any] | MemoryCase


@dataclass(frozen=True)
class Violation:
    rule: int
    item: str  # which memory item (or eval molecule for rule 3)
    detail: str

    def __str__(self) -> str:
        return f"rule {self.rule}: {self.item}: {self.detail}"


@dataclass(frozen=True)
class Profile:
    key: str  # InChIKey connectivity block
    scaffold: str  # canonical Murcko scaffold; '' for acyclic
    fingerprint: Any


@cache
def profile(smiles: str) -> Profile | None:
    """Structure key, scaffold and Morgan fingerprint of a SMILES; None if RDKit cannot parse it."""
    fields = chemistry_fields(smiles)
    if fields is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Profile(
        key=str(fields["inchikey"]).split("-")[0],
        scaffold=str(fields["murcko_scaffold"]),
        fingerprint=_MORGAN.GetFingerprint(mol),
    )


def structure_key(smiles: str) -> str | None:
    p = profile(smiles)
    return p.key if p else None


def scaffold_of(smiles: str) -> str | None:
    p = profile(smiles)
    return p.scaffold if p else None


def tanimoto(a: str, b: str) -> float | None:
    pa, pb = profile(a), profile(b)
    if pa is None or pb is None:
        return None
    return DataStructs.TanimotoSimilarity(pa.fingerprint, pb.fingerprint)


def _get(item: Item, name: str) -> Any:
    return getattr(item, name) if isinstance(item, MemoryCase) else item.get(name)


def _smiles(item: Item) -> str:
    return str(_get(item, "smiles"))


def label(item: Item) -> str:
    """A short name for an item in a violation message."""
    if isinstance(item, MemoryCase):
        return item.case_id
    if "case_id" in item:
        return str(item["case_id"])
    pos = item.get("position")
    return f"{item.get('id', '?')}" + (f"@{pos}" if pos is not None else "")


def _unparseable(rule: int, item: Item) -> Violation:
    return Violation(rule, label(item), f"smiles {_smiles(item)!r} cannot be parsed, so it cannot be cleared")


def rule1_structure_key(
    items: Iterable[Item], mem_eval: Sequence[Item], old_keys: Iterable[str] = ()
) -> list[Violation]:
    """Rule 1: no memory item shares an InChIKey connectivity block with MEM-EVAL or an old-set molecule."""
    eval_keys: dict[str, str] = {}
    for m in mem_eval:
        k = structure_key(_smiles(m))
        if k is not None:
            eval_keys.setdefault(k, label(m))
    old = set(old_keys)
    out: list[Violation] = []
    for it in items:
        k = structure_key(_smiles(it))
        if k is None:
            out.append(_unparseable(1, it))
        elif k in eval_keys:
            out.append(Violation(1, label(it), f"same structure key {k} as MEM-EVAL molecule {eval_keys[k]}"))
        elif k in old:
            out.append(Violation(1, label(it), f"structure key {k} belongs to an old-set molecule"))
    return out


def rule2_scaffold(items: Iterable[Item], mem_eval: Sequence[Item]) -> list[Violation]:
    """Rule 2: no memory item shares a canonical Murcko scaffold with a MEM-EVAL molecule."""
    eval_scaffolds: dict[str, str] = {}
    for m in mem_eval:
        s = scaffold_of(_smiles(m))
        if s:
            eval_scaffolds.setdefault(s, label(m))
    out: list[Violation] = []
    for it in items:
        s = scaffold_of(_smiles(it))
        if s is None:
            out.append(_unparseable(2, it))
        elif s and s in eval_scaffolds:
            out.append(
                Violation(2, label(it), f"scaffold {s} shared with MEM-EVAL molecule {eval_scaffolds[s]}")
            )
    return out


def rule3_eval_scaffolds(mem_eval: Iterable[Item]) -> list[Violation]:
    """Rule 3: MEM-EVAL contains no acyclic molecule and no bare-benzene-scaffold molecule."""
    out: list[Violation] = []
    for m in mem_eval:
        s = scaffold_of(_smiles(m))
        if s is None:
            out.append(_unparseable(3, m))
        elif s == "":
            out.append(Violation(3, label(m), "acyclic (empty scaffold)"))
        elif s == BENZENE_SCAFFOLD:
            out.append(Violation(3, label(m), "benzene scaffold"))
    return out


def rule4_similarity(
    items: Iterable[Item], mem_eval: Sequence[Item], limit: float = SIMILARITY_LIMIT
) -> list[Violation]:
    """Rule 4: no memory item has Morgan (r=2, 2048 bits) Tanimoto >= ``limit`` to a MEM-EVAL molecule."""
    eval_fps = [(label(m), p.fingerprint) for m in mem_eval if (p := profile(_smiles(m))) is not None]
    out: list[Violation] = []
    for it in items:
        p = profile(_smiles(it))
        if p is None:
            out.append(_unparseable(4, it))
            continue
        if not eval_fps:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(p.fingerprint, [fp for _, fp in eval_fps])
        best = max(range(len(sims)), key=sims.__getitem__)
        if sims[best] >= limit:
            out.append(
                Violation(
                    4, label(it), f"Tanimoto {sims[best]:.3f} >= {limit:.2f} to MEM-EVAL {eval_fps[best][0]}"
                )
            )
    return out


def rule5_proposed(cases: Iterable[MemoryCase], mem_eval: Sequence[Item]) -> list[Violation]:
    """Rule 5: no ``agent_outcome.proposed_smiles`` matches a MEM-EVAL molecule by structure key or scaffold.

    A proposal RDKit cannot parse cannot show the model a structure, so it is skipped.
    """
    by_key: dict[str, str] = {}
    by_scaffold: dict[str, str] = {}
    for m in mem_eval:
        p = profile(_smiles(m))
        if p is None:
            continue
        by_key.setdefault(p.key, label(m))
        if p.scaffold:
            by_scaffold.setdefault(p.scaffold, label(m))
    out: list[Violation] = []
    for case in cases:
        for proposed in case.agent_outcome.proposed_smiles:
            p = profile(proposed)
            if p is None:
                continue
            if p.key in by_key:
                out.append(
                    Violation(5, case.case_id, f"proposed {proposed!r} is MEM-EVAL molecule {by_key[p.key]}")
                )
            elif p.scaffold and p.scaffold in by_scaffold:
                out.append(
                    Violation(
                        5,
                        case.case_id,
                        f"proposed {proposed!r} has scaffold {p.scaffold} of "
                        f"MEM-EVAL molecule {by_scaffold[p.scaffold]}",
                    )
                )
    return out


def check_injected(
    eval_molecule: Item, cases: Sequence[Item], old_keys: Iterable[str] = ()
) -> list[Violation]:
    """Rules 1, 2 and 4 for one eval molecule against the cases injected for it (rule 6, phase M7)."""
    return [
        *rule1_structure_key(cases, [eval_molecule], old_keys),
        *rule2_scaffold(cases, [eval_molecule]),
        *rule4_similarity(cases, [eval_molecule]),
    ]
