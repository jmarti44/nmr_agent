"""Retrieval tests. Cases and stores are built in memory; no repository memory/ directory, no network."""

from __future__ import annotations

import random
import time

import pytest

from src.memory import (
    MemoryCase,
    MemoryStore,
    RetrievalParams,
    case_features,
    query_features,
    retrieve,
    score_case,
    select_random,
)
from src.memory.features import (
    Features,
    dou_closeness,
    element_closeness,
    peak_closeness,
    solvent_match,
)
from src.schemas import Nucleus, Peak, PeakRole, PeakTable, SampleContext, Solvent

P = RetrievalParams()


def make_case(
    n: int = 1,
    *,
    c=(20.0, 60.0, 130.0),
    h=(1.0, 3.5, 7.2),
    formula="C9H10O2",
    dou=5.0,
    solvent: str | None = "CDCl3",
    status="solved",
    scaffold="c1ccccc1",
    smiles="CCOC(=O)c1ccccc1",
    inchikey="MTZQAGJQAFMTAQ-UHFFFAOYSA-N",
) -> MemoryCase:
    return MemoryCase.model_validate(
        {
            "case_id": f"mem-{n:06d}",
            "schema_version": "1",
            "provenance": {
                "source": "nmrshiftdb2",
                "source_record_id": str(n),
                "spectrum_ids": {"1H": 1, "13C": 2},
                "truth_basis": "database-assigned structure",
                "extracted_by": "test",
                "proposed_on": "2026-11-14",
            },
            "approval": {"approved_by": "J", "date_added": "2026-11-15", "note": None},
            "formula": formula,
            "dou": dou,
            "carbon_count": 9,
            "heavy_atoms": 11,
            "solvent": solvent,
            "h_peaks": [{"shift_ppm": x, "integral": None} for x in h],
            "c_peaks": list(c),
            "smiles": smiles,
            "inchikey": inchikey,
            "murcko_scaffold": scaffold,
            "generic_scaffold": "",
            "assignments": None,
            "agent_outcome": {
                "collected_on": "2026-11-16",
                "model_id": "m",
                "prompt_version": "v",
                "settings_hash": "h",
                "status": status,
                "proposed_smiles": [] if status == "no_answer" else ["C"],
            },
        }
    )


def make_store(*cases: MemoryCase) -> MemoryStore:
    return MemoryStore(cases=tuple(sorted(cases, key=lambda c: c.case_id)), memory_version="x" * 12)


def feats(c=(), h=(), elements=None, dou=None, solvent=None) -> Features:
    return Features(tuple(sorted(c)), tuple(sorted(h)), elements, dou, solvent)


def table(c=(), h=(), solvent=None, roles=None) -> PeakTable:
    peaks, i = [], 1
    for nuc, shifts in ((Nucleus.C13, c), (Nucleus.H1, h)):
        for s in shifts:
            peaks.append(Peak(id=i, nucleus=nuc, shift_ppm=s))
            i += 1
    for role, nuc, s in roles or ():
        peaks.append(Peak(id=i, nucleus=nuc, shift_ppm=s, role=role))
        i += 1
    return PeakTable(peaks=peaks, solvent=solvent)


# --------------------------------------------------------------------------- query features


def test_query_features_analyte_only_and_formula():
    t = table(
        c=(130.0, 20.0),
        h=(7.2, 1.0),
        roles=[(PeakRole.SOLVENT, Nucleus.H1, 7.26), (PeakRole.SOLVENT, Nucleus.C13, 77.0)],
    )
    f = query_features(t, SampleContext(molecular_formula="C9H10O2"))
    assert f.c_shifts == (20.0, 130.0)
    assert f.h_shifts == (1.0, 7.2)
    assert f.elements == {"C": 9, "H": 10, "O": 2}
    assert f.dou == 5.0


def test_query_features_no_formula_and_solvent_precedence():
    f = query_features(table(c=(1.0,), solvent=Solvent.D2O), SampleContext())
    assert f.elements is None and f.dou is None
    assert f.solvent == Solvent.D2O
    f = query_features(table(solvent=Solvent.D2O), SampleContext(solvent=Solvent.CDCL3))
    assert f.solvent == Solvent.CDCL3


def test_case_features_from_case():
    f = case_features(make_case(c=(60.0, 20.0), h=(3.5, 1.0)))
    assert f.c_shifts == (20.0, 60.0) and f.h_shifts == (1.0, 3.5)
    assert f.elements == {"C": 9, "H": 10, "O": 2} and f.dou == 5.0 and f.solvent == Solvent.CDCL3


# --------------------------------------------------------------------------- peak matching


def test_peak_exact_match_is_one():
    assert peak_closeness((20.0, 60.0, 130.0), (20.0, 60.0, 130.0), 2.0) == pytest.approx(1.0)


@pytest.mark.parametrize("tol", [2.0, 0.15])
def test_peak_just_inside_and_just_outside_tolerance(tol):
    inside = peak_closeness((50.0,), (50.0 + tol * 0.99,), tol)
    assert inside == pytest.approx(0.01, abs=1e-9)
    assert peak_closeness((50.0,), (50.0 + tol * 1.01,), tol) == 0.0
    # closeness falls linearly inside the tolerance
    assert peak_closeness((50.0,), (50.0 + tol / 2,), tol) == pytest.approx(0.5)


def test_peak_matching_is_one_to_one():
    # one query peak cannot match both case peaks: 1 pair / max(1, 2)
    assert peak_closeness((50.0,), (50.0, 50.0), 2.0) == pytest.approx(0.5)
    # two query peaks near one case peak: only one matches
    assert peak_closeness((50.0, 50.1), (50.0,), 2.0) == pytest.approx(1.0 / 2)


def test_peak_matching_is_optimal_not_greedy():
    # greedy nearest-first would pair 10.0 with 11.0 (d=1.0) and leave 12.0 unmatched-ish;
    # the optimum pairs 10.0-9.0 and 12.0-11.0... here: q=(10,12), c=(9,11) -> both within tol 2
    assert peak_closeness((10.0, 12.0), (9.0, 11.0), 2.0) == pytest.approx(0.5)
    # q=(0, 1.9), c=(1.0, 2.9) tol 2: optimal pairs 0-1.0 and 1.9-2.9 (both 1.0 away) -> 2 matches
    assert peak_closeness((0.0, 1.9), (1.0, 2.9), 2.0) == pytest.approx((0.5 + 0.5) / 2)


def test_peak_unequal_lengths_and_empty():
    assert peak_closeness((10.0, 20.0), (10.0, 20.0, 30.0, 40.0), 2.0) == pytest.approx(0.5)
    assert peak_closeness((), (10.0,), 2.0) == 0.0
    assert peak_closeness((10.0,), (), 2.0) == 0.0


# --------------------------------------------------------------------------- other closeness


def test_element_closeness():
    assert element_closeness({"C": 9, "H": 10, "O": 2}, {"C": 9, "H": 10, "O": 2}) == 1.0
    # union includes N: diff = |9-9| + |10-12| + |2-0| + |0-1| = 5 ; denom = 9+12+2+1 = 24
    got = element_closeness({"C": 9, "H": 10, "O": 2}, {"C": 9, "H": 12, "N": 1})
    assert got == pytest.approx(1 - 5 / 24)
    assert element_closeness({"C": 1}, {"C": 1, "Cl": 3}) == pytest.approx(1 - 3 / 4)


def test_dou_closeness():
    assert dou_closeness(5.0, 5.0) == 1.0
    assert dou_closeness(5.0, 7.0) == pytest.approx(1 / 3)
    assert dou_closeness(4.5, 5.0) == pytest.approx(1 / 1.5)


# --------------------------------------------------------------------------- similarity


def test_similarity_full_match_with_all_features():
    sim, closeness, _ = score_case(
        feats((20.0, 60.0, 130.0), (1.0, 3.5, 7.2), {"C": 9, "H": 10, "O": 2}, 5.0), make_case(), P
    )
    assert sim == pytest.approx(1.0)
    assert set(closeness) == {"c13", "h1", "elements", "dou"}


def test_similarity_weight_renormalisation():
    case = make_case()
    # no formula: elements and DoU dropped; 13C exact (1.0), 1H closeness 0 (far)
    sim, closeness, _ = score_case(feats((20.0, 60.0, 130.0), (9.0, 9.5, 9.9)), case, P)
    assert set(closeness) == {"c13", "h1"}
    assert sim == pytest.approx(0.35 * 1.0 / (0.35 + 0.30))
    # no 13C in the query: 13C dropped
    sim, closeness, _ = score_case(feats((), (1.0, 3.5, 7.2), {"C": 9, "H": 10, "O": 2}, 5.0), case, P)
    assert "c13" not in closeness
    assert sim == pytest.approx(1.0)
    # no 1H in the query
    _, closeness, _ = score_case(feats((20.0,), ()), case, P)
    assert set(closeness) == {"c13"}


def test_similarity_zero_when_no_feature_available():
    sim, closeness, matched = score_case(feats(solvent=Solvent.CDCL3), make_case(), P)
    assert (sim, closeness, matched) == (0.0, {}, False)  # the solvent bonus alone never scores


def test_case_with_empty_peak_list_scores_zero_for_that_feature():
    _, closeness, _ = score_case(feats((20.0,), (1.0,)), make_case(c=(), h=()), P)
    assert closeness == {"c13": 0.0, "h1": 0.0}


def test_solvent_bonus_and_neutrality():
    q = dict(c=(20.0, 60.0, 130.0), h=(9.0,))  # imperfect so the cap does not hide the bonus
    base = score_case(feats(**q), make_case(solvent=None), P)[0]
    same = score_case(feats(**q, solvent=Solvent.CDCL3), make_case(solvent="CDCl3"), P)[0]
    diff = score_case(feats(**q, solvent=Solvent.D2O), make_case(solvent="CDCl3"), P)[0]
    unknown_q = score_case(feats(**q), make_case(solvent="CDCl3"), P)[0]
    unknown_c = score_case(feats(**q, solvent=Solvent.CDCL3), make_case(solvent=None), P)[0]
    assert same == pytest.approx(base + 0.03)
    assert diff == base == unknown_q == unknown_c


def test_solvent_bonus_is_capped_at_one():
    sim, _, matched = score_case(feats((20.0,), solvent=Solvent.CDCL3), make_case(c=(20.0,)), P)
    assert matched and sim == 1.0


def test_solvent_other_is_not_a_match():
    assert not solvent_match(Solvent.OTHER, Solvent.OTHER)
    assert solvent_match(Solvent.D2O, Solvent.D2O)
    assert not solvent_match(None, None)


def test_params_are_frozen_and_validated():
    with pytest.raises(AttributeError):
        P.k = 5  # type: ignore[misc]
    with pytest.raises(ValueError):
        RetrievalParams(weight_c13=-1)
    custom = RetrievalParams(k=1)
    store = make_store(make_case(1), make_case(2, scaffold="C1CC1"))
    assert len(retrieve(feats((20.0,)), store, custom)) == 1


# --------------------------------------------------------------------------- selection

Q = feats((20.0, 60.0, 130.0))  # 13C only, so a case's score is its 13C closeness


def case_with_score(n, score, **kw):
    """A case whose 13C closeness against Q is exactly `score` (one of three peaks shifted)."""
    # peaks 20 and 60 match exactly; 130 shifted by d gives contribution 1 - d/2
    d = 2.0 * (1.0 - (3 * score - 2.0))
    return make_case(n, c=(20.0, 60.0, 130.0 + d), scaffold=kw.pop("scaffold", f"s{n}"), **kw)


def test_tie_break_prefers_missed_within_epsilon():
    solved = case_with_score(1, 0.95, status="solved")
    missed = case_with_score(2, 0.94, status="missed")  # 0.01 below
    out = retrieve(Q, make_store(solved, missed), RetrievalParams(k=2))
    assert [r.case.case_id for r in out] == ["mem-000002", "mem-000001"]


def test_tie_break_not_applied_outside_epsilon():
    solved = case_with_score(1, 0.95, status="solved")
    missed = case_with_score(2, 0.90, status="missed")  # 0.05 below
    out = retrieve(Q, make_store(solved, missed), RetrievalParams(k=2))
    assert [r.case.case_id for r in out] == ["mem-000001", "mem-000002"]


def test_tie_break_highest_score_among_missed_then_rest():
    a = case_with_score(1, 0.95, status="solved")
    b = case_with_score(2, 0.94, status="missed")
    c = case_with_score(3, 0.945, status="missed")
    out = retrieve(Q, make_store(a, b, c), RetrievalParams(k=3))
    assert [r.case.case_id for r in out] == ["mem-000003", "mem-000002", "mem-000001"]


def test_equal_scores_order_by_case_id():
    cases = [make_case(n, scaffold=f"s{n}") for n in (3, 1, 2)]
    out = retrieve(Q, make_store(*cases), RetrievalParams(k=3))
    assert [r.case.case_id for r in out] == ["mem-000001", "mem-000002", "mem-000003"]
    # also among equal missed cases
    cases = [make_case(n, scaffold=f"s{n}", status="missed") for n in (2, 1)]
    out = retrieve(Q, make_store(*cases), RetrievalParams(k=2))
    assert [r.case.case_id for r in out] == ["mem-000001", "mem-000002"]


def test_one_per_scaffold_including_empty_scaffold():
    cases = [
        make_case(1, scaffold=""),
        make_case(2, scaffold=""),  # acyclic: same scaffold as 1
        make_case(3, scaffold="c1ccccc1"),
        make_case(4, scaffold="c1ccccc1"),
        make_case(5, scaffold="C1CC1", c=(20.0, 60.0, 131.0)),
    ]
    out = retrieve(Q, make_store(*cases), RetrievalParams(k=5))
    assert [r.case.case_id for r in out] == ["mem-000001", "mem-000003", "mem-000005"]


def test_scaffold_dup_does_not_anchor_tie_window():
    # best score is a duplicate scaffold; the missed case is within epsilon of the next eligible, not of it
    a = case_with_score(1, 0.99, scaffold="X")
    b = case_with_score(2, 0.98, scaffold="X")
    c = case_with_score(3, 0.97, scaffold="Y", status="missed")
    out = retrieve(Q, make_store(a, b, c), RetrievalParams(k=2))
    assert [r.case.case_id for r in out] == ["mem-000003", "mem-000001"]  # c within 0.02 of a; b ineligible


def test_k_limits_and_k_zero():
    cases = [make_case(n, scaffold=f"s{n}") for n in range(1, 6)]
    assert len(retrieve(Q, make_store(*cases))) == 3
    assert len(retrieve(Q, make_store(*cases), RetrievalParams(k=5))) == 5
    assert retrieve(Q, make_store(*cases), RetrievalParams(k=0)) == []
    assert retrieve(Q, make_store(), P) == []


def test_min_similarity_drops_cases():
    good = case_with_score(1, 0.9)
    bad = make_case(2, c=(90.0, 91.0, 92.0), scaffold="b")
    out = retrieve(Q, make_store(good, bad), RetrievalParams(min_similarity=0.5))
    assert [r.case.case_id for r in out] == ["mem-000001"]
    assert len(retrieve(Q, make_store(good, bad), RetrievalParams(min_similarity=0.0))) == 2
    assert retrieve(Q, make_store(bad), RetrievalParams(min_similarity=0.5)) == []


def test_results_carry_similarity_and_closeness():
    (r,) = retrieve(Q, make_store(make_case(1)), RetrievalParams(k=1))
    assert r.similarity == pytest.approx(1.0) and r.closeness == {"c13": pytest.approx(1.0)}


# --------------------------------------------------------------------------- budget


def test_budget_skips_non_fitting_and_adds_smaller_later_case():
    costs = {"mem-000001": 400, "mem-000002": 900, "mem-000003": 300}
    cases = [
        make_case(1, scaffold="a"),
        make_case(2, scaffold="b", c=(20.0, 60.0, 130.5)),
        make_case(3, scaffold="c", c=(20.0, 60.0, 131.0)),
    ]
    out = retrieve(
        Q,
        make_store(*cases),
        RetrievalParams(k=3),
        token_budget=800,
        cost_fn=lambda c: costs[c.case_id],
    )
    # 1 (400) fits; 2 (900) does not -> skipped; 3 (300) fits: total 700
    assert [r.case.case_id for r in out] == ["mem-000001", "mem-000003"]


def test_budget_boundary_inclusive_and_no_budget_ignores_cost():
    cases = [make_case(n, scaffold=f"s{n}") for n in (1, 2)]
    out = retrieve(Q, make_store(*cases), RetrievalParams(k=2), token_budget=200, cost_fn=lambda c: 100)
    assert len(out) == 2
    out = retrieve(Q, make_store(*cases), RetrievalParams(k=2), token_budget=199, cost_fn=lambda c: 100)
    assert len(out) == 1
    out = retrieve(Q, make_store(*cases), RetrievalParams(k=2), cost_fn=lambda c: 10**9)
    assert len(out) == 2
    with pytest.raises(ValueError):
        retrieve(Q, make_store(*cases), P, token_budget=10)


# --------------------------------------------------------------------------- exclusions, random


def test_exclusions():
    cases = [make_case(1, scaffold="a"), make_case(2, scaffold="b"), make_case(3, scaffold="c")]
    store = make_store(*cases)
    out = retrieve(Q, store, P, exclude_case_ids=["mem-000001"])
    assert [r.case.case_id for r in out] == ["mem-000002", "mem-000003"]
    out = retrieve(Q, store, P, exclude_scaffolds=["b"])
    assert [r.case.case_id for r in out] == ["mem-000001", "mem-000003"]
    out = select_random(store, 0, exclude_scaffolds=["a", "b"], exclude_case_ids=["mem-000003"])
    assert out == []


def test_random_selection_deterministic_and_seed_dependent():
    store = make_store(*[make_case(n, scaffold=f"s{n}") for n in range(1, 41)])
    ids = lambda seed: [r.case.case_id for r in select_random(store, seed)]  # noqa: E731
    assert ids(7) == ids(7)
    assert len(ids(7)) == 3
    assert any(ids(s) != ids(7) for s in range(8, 14))
    assert all(r.similarity is None for r in select_random(store, 7))


def test_random_selection_rules_and_independence_from_store_order():
    cases = [make_case(n, scaffold="same" if n < 20 else f"s{n}") for n in range(1, 30)]
    out = select_random(make_store(*cases), 3, RetrievalParams(k=5))
    assert len({r.case.murcko_scaffold for r in out}) == len(out) == 5
    # budget: nothing fits -> empty; unit cost with budget 2 -> 2 cases
    assert select_random(make_store(*cases), 3, token_budget=5, cost_fn=lambda c: 10) == []
    assert len(select_random(make_store(*cases), 3, token_budget=2, cost_fn=lambda c: 1)) == 2
    shuffled = list(cases)
    random.Random(1).shuffle(shuffled)
    store2 = MemoryStore(cases=tuple(shuffled), memory_version="x" * 12)
    assert [r.case.case_id for r in select_random(store2, 3)] == [
        r.case.case_id for r in select_random(make_store(*cases), 3)
    ]


# --------------------------------------------------------------------------- no structure leakage


def test_scoring_ignores_smiles_inchikey_and_scaffold():
    a = make_case(1, smiles="CCOC(=O)c1ccccc1", inchikey="AAAAAAAAAAAAAA-UHFFFAOYSA-N", scaffold="c1ccccc1")
    b = make_case(1, smiles="CCCCCC", inchikey="BBBBBBBBBBBBBB-UHFFFAOYSA-N", scaffold="")
    q = feats((20.0, 60.0, 131.0), (1.0, 3.4), {"C": 9, "H": 10, "O": 2}, 5.0, Solvent.CDCL3)
    assert score_case(q, a, P) == score_case(q, b, P)
    assert case_features(a) == case_features(b)


# --------------------------------------------------------------------------- timing


def test_timing_5000_cases(capsys):
    rng = random.Random(0)
    cases = [
        make_case(
            n,
            c=[rng.uniform(10, 200) for _ in range(20)],
            h=[rng.uniform(0.5, 9) for _ in range(15)],
            scaffold=f"s{n % 400}",
            status=rng.choice(["solved", "missed"]),
        )
        for n in range(1, 5001)
    ]
    store = make_store(*cases)
    q = feats(
        [rng.uniform(10, 200) for _ in range(20)],
        [rng.uniform(0.5, 9) for _ in range(15)],
        {"C": 9, "H": 10, "O": 2},
        5.0,
        Solvent.CDCL3,
    )
    t0 = time.perf_counter()
    out = retrieve(q, store, P)
    elapsed = time.perf_counter() - t0
    with capsys.disabled():
        print(f"\n[timing] one query over 5000 cases: {elapsed:.3f}s")
    assert len(out) == 3
