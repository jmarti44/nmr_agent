"""Tests for the offline go/no-go (bench/memory_gonogo.py, phase M4): small hand-made examples. The
fast-path equivalence test reads the committed outputs and the NMRShiftDB2 export; skipped without them."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from bench import memory_gonogo as g
from src.memory.features import Features
from src.memory.retrieval import RetrievalParams, retrieve

ROOT = Path(__file__).resolve().parents[2]
SD = ROOT / "bench" / "data" / "nmrshiftdb2" / "nmrshiftdb2withsignals.sd"


# --------------------------------------------------------------------------- the stratified draw


def _strata() -> dict[str, list[str]]:
    return {
        ">20": [f"{1000 + i}" for i in range(150)],
        "11-20": [f"{2000 + i}" for i in range(300)],
        "<=10": [f"{3000 + i}" for i in range(100)],
    }


def test_sample_sizes_and_strata():
    s = _strata()
    ids = g.draw_sample(s, 20261117)
    assert len(ids) == len(set(ids)) == 200
    for stratum, quota in g.SAMPLE_QUOTA.items():
        assert sum(i in set(s[stratum]) for i in ids) == quota
    assert g.SAMPLE_QUOTA == {">20": 80, "11-20": 80, "<=10": 40}


def test_sample_is_deterministic_and_independent_of_input_order():
    s = _strata()
    shuffled = {k: list(reversed(v)) for k, v in s.items()}
    assert g.draw_sample(s, 20261117) == g.draw_sample(s, 20261117) == g.draw_sample(shuffled, 20261117)
    assert g.draw_sample(s, 20261117) != g.draw_sample(s, 20261118)


def test_seeds_are_the_fixed_ones():
    assert g.SAMPLE_SEEDS["tuning"] == (20261117, 20261118, 20261119, 20261120, 20261121)
    assert g.SAMPLE_SEEDS["check"] == (20261122, 20261123, 20261124, 20261125, 20261126)


def test_committed_samples_match_a_redraw():
    path = ROOT / "bench" / "memory" / "gonogo_samples.json"
    if not path.exists():
        pytest.skip("gonogo_samples.json is not committed yet")
    by_id, halves = g.read_selection()
    excluded = g.read_excluded()
    drawn = g.draw_all_samples(by_id, halves, excluded)
    committed = json.loads(path.read_text())["samples"]
    assert [(r["half"], r["index"], r["seed"], r["ids"]) for r in committed] == [
        (h, s["index"], s["seed"], s["ids"]) for h in g.HALVES for s in drawn[h]
    ]
    for r in committed:
        assert all(halves[i] == r["half"] and i not in excluded for i in r["ids"])
        counts = {k: sum(by_id[i]["stratum"] == k for i in r["ids"]) for k in g.SAMPLE_QUOTA}
        assert counts == g.SAMPLE_QUOTA


# ------------------------------------------------------------------ molecules the service cannot take


def _record(h_spec: str) -> dict:
    return {
        "id": "1",
        "smiles": "CCO",
        "formula": "C2H6O",
        "solvent": "CDCl3",
        "h_spec": h_spec,
        "c_spec": "18.0;0.0;0|58.0;0.0;1|",
    }


def test_request_error_accepts_a_valid_record():
    assert g.request_error(_record("1.2;0.0;1|3.6;0.0;2|")) is None


@pytest.mark.parametrize(
    ("h_spec", "expected"),
    [("", "At least one 1H peak"), ("1.2;0.0;1|16.5;0.0;2|", "outside plausible range")],
)
def test_request_error_reports_what_the_service_rejects(h_spec, expected):
    err = g.request_error(_record(h_spec))
    assert err is not None and expected in err


def test_excluded_ids_never_enter_a_sample():
    s = _strata()
    banned = frozenset(s[">20"][:70] + s["<=10"][:55])
    by_id = {i: {"stratum": k} for k, ids in s.items() for i in ids}
    halves = {i: "tuning" for i in by_id} | {f"9{n}": "check" for n in range(1000)}
    by_id |= {f"9{n}": {"stratum": ("<=10", "11-20", ">20")[n % 3]} for n in range(1000)}
    drawn = g.draw_all_samples(by_id, halves, banned)
    for half in g.HALVES:
        for sample in drawn[half]:
            assert len(sample["ids"]) == 200 and not banned & set(sample["ids"])


# --------------------------------------------------------------------------- pool construction


def _case(n: int, smiles: str, features: Features | None = None):
    f = features or Features((20.0, 60.0), (1.0, 3.5), {"C": 4, "H": 8, "O": 1}, 1.0, None)
    entry = {"id": str(n), "smiles": smiles, "heavy_atoms": 6}
    return g.make_case(f"mem-{n:06d}", entry, f, [], "C4H8O")


def test_pool_excludes_the_query_and_its_scaffold():
    q = _case(1, "c1ccccc1CC")  # benzene scaffold
    same_scaffold = _case(2, "c1ccccc1O")
    other = _case(3, "C1CCCCC1C")
    acyclic = _case(4, "CCCO")
    pool = g.build_pool([q, same_scaffold, other, acyclic], q)
    assert [c.case_id for c in pool.cases] == ["mem-000003", "mem-000004"]


def test_pool_for_an_acyclic_query_drops_every_acyclic_case():
    q = _case(1, "CCCC")
    pool = g.build_pool([q, _case(2, "CCO"), _case(3, "C1CC1")], q)
    assert [c.case_id for c in pool.cases] == ["mem-000003"]


def test_random_seed_is_the_documented_hash():
    assert g.pair_seed("123", 0) == int(hashlib.sha256(b"123:0").hexdigest()[:8], 16)
    assert g.pair_seed("123", 0) != g.pair_seed("123", 1) != g.pair_seed("124", 1)


# --------------------------------------------------------------------------- scores


def test_best_similarity():
    assert g.best_similarity("CCO", []) == 0.0
    assert g.best_similarity("CCO", ["CCO"]) == pytest.approx(1.0)
    low = g.best_similarity("CCO", ["c1ccccc1"])
    both = g.best_similarity("CCO", ["c1ccccc1", "CCO"])
    assert 0.0 <= low < 1.0
    assert both == pytest.approx(1.0)


def test_fragment_groups_and_coverage():
    truth = g.fragment_groups("COC(=O)c1ccccc1")  # methyl ester of benzoic acid
    assert {"ester", "methoxy", "aromatic_ring", "methyl"} <= truth
    ester_only = g.fragment_groups("CCOC(C)=O")  # ethyl acetate
    benzene = g.fragment_groups("c1ccccc1")
    assert "ester" in ester_only and "aromatic_ring" in benzene
    assert g.fragment_coverage(truth, []) == 0.0
    only_ester = g.fragment_coverage(truth, [ester_only])
    both = g.fragment_coverage(truth, [ester_only, benzene])
    assert only_ester == len(truth & ester_only) / len(truth)
    assert both == len(truth & (ester_only | benzene)) / len(truth)
    assert both > only_ester
    assert g.fragment_coverage(truth, [truth]) == 1.0


def test_coverage_is_none_without_a_group():
    assert g.fragment_groups("O=C=O") == frozenset()  # CO2 has none of the verifier's groups
    assert g.fragment_coverage(frozenset(), [frozenset({"ester"})]) is None


def _pair(sim_r, sim_n, cov_r, cov_n, stratum=">20", q="1", i=0):
    return {
        "query_id": q,
        "stratum": stratum,
        "sample_index": i,
        "best_similarity_retrieved": sim_r,
        "best_similarity_random": sim_n,
        "coverage_retrieved": cov_r,
        "coverage_random": cov_n,
        "win": sim_r > sim_n,
    }


def test_summary_counts_ties_as_losses_and_excludes_empty_truth_from_coverage():
    pairs = [
        _pair(0.6, 0.2, 1.0, 0.5, q="1"),
        _pair(0.3, 0.3, 0.5, 0.5, q="2"),  # tie: not a win
        _pair(0.4, 0.5, None, None, q="3"),  # truth without groups
    ]
    s = g.summarize(pairs)
    assert s["pairs"] == 3 and s["queries"] == 3 and s["cov_excluded"] == 1
    assert s["win_rate"] == pytest.approx(1 / 3)
    assert s["sim_diff"] == pytest.approx((0.6 + 0.3 + 0.4) / 3 - (0.2 + 0.3 + 0.5) / 3)
    assert s["cov_retrieved"] == pytest.approx(0.75) and s["cov_random"] == pytest.approx(0.5)


# --------------------------------------------------------------------------- the verdict


def test_thresholds_are_the_spec_ones():
    assert (g.GO_SIMILARITY, g.GO_COVERAGE, g.GO_WIN_RATE) == (0.10, 0.15, 0.60)


def test_verdict_at_the_thresholds_is_go():
    go, lines = g.verdict(0.10, 0.15, 0.60)
    assert go and len(lines) == 3 and all("PASS" in line for line in lines)


@pytest.mark.parametrize(
    "args",
    [(0.0999, 0.15, 0.60), (0.10, 0.1499, 0.60), (0.10, 0.15, 0.5999), (0.0, 0.0, 0.0), (-0.2, 0.5, 0.9)],
)
def test_verdict_below_any_threshold_is_no_go(args):
    go, lines = g.verdict(*args)
    assert not go and any("FAIL" in line for line in lines)


def test_verdict_above_all_thresholds_is_go():
    assert g.verdict(0.3, 0.4, 0.9)[0]


def test_verdict_with_no_data_is_no_go():
    assert not g.verdict(math.nan, 0.5, 0.9)[0]


def test_report_ends_with_the_verdict_line():
    pairs = [_pair(0.9, 0.1, 1.0, 0.0, stratum=s) for s in (">20", "11-20", "<=10")]
    params = {"sha256": "x", "parameters": dict.fromkeys(g.WEIGHT_NAMES, 0.25)}
    text = g.report_text(pairs, pairs, params, "abc")
    assert text.rstrip().splitlines()[-1] == "VERDICT: GO"
    bad = g.report_text([_pair(0.1, 0.1, 0.0, 0.0)], pairs, params, "abc")
    assert bad.rstrip().splitlines()[-1] == "VERDICT: NO-GO"


# --------------------------------------------------------------------------- tuning grid


def test_grid_is_every_composition_of_ten_tenths():
    grid = g.grid_weights()
    assert len(grid) == len(set(grid)) == 286
    assert all(abs(sum(w) - 1.0) < 1e-12 and min(w) >= 0 for w in grid)
    assert (0.3, 0.3, 0.2, 0.2) in grid and (1.0, 0.0, 0.0, 0.0) in grid


def test_pick_best_breaks_ties_by_distance_to_defaults_then_lexicographically():
    near, far = (0.4, 0.3, 0.2, 0.1), (0.2, 0.3, 0.3, 0.2)  # distances 0.10 and 0.30 from the defaults
    other = (0.5, 0.5, 0.0, 0.0)
    assert g.pick_best([far, near, other], [0.5, 0.5, 0.4])[0] == 1
    assert g.pick_best([far, near, other], [0.5, 0.5, 0.9])[0] == 2  # a higher objective wins outright
    # equal objective and equal distance (0.10 each, up to float noise): lexicographic order decides
    a, b = (0.3, 0.3, 0.2, 0.2), (0.4, 0.3, 0.2, 0.1)
    assert g.pick_best([b, a], [0.5, 0.5])[0] == 1
    assert g.pick_best([a, b], [0.5, 0.5])[0] == 0


# --------------------------------------------------------------------------- fast path == retrieve()


@pytest.mark.skipif(
    not SD.exists(), reason="bench/data (NMRShiftDB2 export) is absent; the test needs the spectra"
)
def test_fast_path_selects_the_same_cases_as_retrieve():
    params_file = ROOT / "bench" / "memory" / "retrieval_params.json"
    samples_file = ROOT / "bench" / "memory" / "gonogo_samples.json"
    if not (params_file.exists() and samples_file.exists()):
        pytest.skip("retrieval_params.json / gonogo_samples.json are not committed yet")
    frozen = json.loads(params_file.read_text())["parameters"]
    by_id, halves = g.read_selection()
    samples = g.read_samples()["tuning"]
    mols = g.load_molecules(by_id, halves, g.read_excluded())
    queries = sorted((m for m in mols if mols[m].half == "tuning"), key=g.id_order)[:40]  # 40 x 5 = 200 pairs
    fs = g.FastScorer(mols, queries, samples, workers=1)
    weight_sets = [
        tuple(frozen[n] for n in g.WEIGHT_NAMES),
        g.DEFAULT_WEIGHTS,
        (0.5, 0.5, 0.0, 0.0),
        (0.0, 0.0, 0.5, 0.5),
        (0.1, 0.1, 0.1, 0.7),
    ]
    compared = 0
    for weights in weight_sets:
        params = g.params_from(weights)
        for qi, qid in enumerate(queries):
            for si, ids in enumerate(samples):
                pool = g.build_pool([mols[i].case for i in ids], mols[qid].case)
                real = [r.case.case_id for r in retrieve(mols[qid].features, pool, params)]
                fast = fs.select(qi, si, np.array([weights]), params.solvent_bonus)[0]
                assert [fs.col_mol[j].case_id for j in fast] == real, (weights, qid, si)
                compared += 1
    assert compared >= 200 * len(weight_sets)


def test_fast_scorer_arithmetic_matches_score_case_on_a_toy_store():
    """No data needed: the numpy recombination equals retrieve() on hand-made cases."""
    smiles = ["CCO", "CCCO", "CCN", "c1ccccc1C", "c1ccccc1O", "C1CCCCC1", "CC(=O)C", "COC", "CCCC", "CC(C)O"]
    rng = np.random.default_rng(0)
    cases = []
    for n, s in enumerate(smiles, start=1):
        feats = Features(
            tuple(sorted(float(x) for x in rng.uniform(10, 150, 3))),
            tuple(sorted(float(x) for x in rng.uniform(0.5, 8, 3))),
            {"C": int(rng.integers(2, 8)), "H": int(rng.integers(4, 14)), "O": int(rng.integers(0, 3))},
            float(rng.integers(0, 4)),
            None,
        )
        cases.append((s, feats, _case(n, s, feats)))
    mols = {}
    for s, feats, case in cases:
        prof = g.profile(s)
        mols[case.case_id[4:].lstrip("0")] = g.Molecule(
            id=case.case_id[4:].lstrip("0"),
            pool="named",
            smiles=s,
            stratum="11-20",
            scaffold=case.murcko_scaffold,
            half="tuning",
            case_id=case.case_id,
            features=feats,
            case=case,
            groups=g.fragment_groups(s),
            fingerprint=prof.fingerprint,
        )
    ids = list(mols)
    fs = g.FastScorer(mols, ids, [ids], workers=1)
    for weights in [(0.35, 0.30, 0.20, 0.15), (0.0, 1.0, 0.0, 0.0), (0.25, 0.25, 0.25, 0.25)]:
        params = RetrievalParams(
            weight_c13=weights[0], weight_h1=weights[1], weight_elements=weights[2], weight_dou=weights[3]
        )
        for qi, qid in enumerate(ids):
            pool = g.build_pool([mols[i].case for i in ids], mols[qid].case)
            real = [r.case.case_id for r in retrieve(mols[qid].features, pool, params)]
            fast = fs.select(qi, 0, np.array([weights]), params.solvent_bonus)[0]
            assert [fs.col_mol[j].case_id for j in fast] == real
