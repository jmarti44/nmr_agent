# ruff: noqa: E501
"""bench/memory.py compare on synthetic result files. The expected values are worked out by hand in each test."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scipy.stats import wilcoxon

import bench.memory as bm
from tests.llm.fake_api import ASPIRIN, ISOMER

N = 10  # molecules in a synthetic MEM-EVAL file
IDS = [f"m{i}" for i in range(N)]
X = bm.tanimoto(ISOMER, ASPIRIN)  # a partial similarity, 0 < X < 1
NEAR = "CC(=O)Oc1ccccc1C(=O)OC"  # aspirin methyl ester, Tanimoto 0.667 to aspirin
TRUTH_KEY = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"  # aspirin's InChIKey

ARMS = bm.ARMS


class Eval:
    """A temporary eval directory and the MEM-EVAL file that lists its molecules (all with aspirin as truth)."""

    def __init__(self, tmp: Path, ids=IDS):
        self.dir = tmp / "eval"
        self.file = tmp / "mem_eval.json"
        rec = {"smiles": ASPIRIN, "inchikey": TRUTH_KEY, "solvent": "CDCl3", "heavy_atoms": 25}
        small = {**rec, "id": "small", "heavy_atoms": 10}
        self.file.write_text(json.dumps({"records": [{**rec, "id": i} for i in ids] + [small]}))

    def put(
        self,
        arm: str,
        rid: str,
        kind: str = "truth",
        *,
        m=None,
        cost=1.0,
        ms=1000,
        reason="output_truncated",
        **over,
    ):
        """kind: truth (first candidate is the truth), isomer, near, bad (unparseable), none (no_answer)."""
        smiles = {"truth": ASPIRIN, "isomer": ISOMER, "bad": "not a smiles", "near": NEAR}.get(kind)
        doc = {
            "arm": arm, "id": rid, "truth": ASPIRIN, "truth_inchikey": TRUTH_KEY,
            "status": "no_answer" if kind == "none" else ("solved" if kind == "truth" else "missed"),
            "candidates": [] if smiles is None else [{"smiles": smiles, "model_rank": 1, "status": "ok"}],
            "trace": {"first": None, "repaired": None, "memory_cases": []},
            "injected": [] if m is None else [{"case_id": "mem-1", "similarity": 0.5, "tanimoto_to_truth": m}],
            "cost_usd": cost, "timing_ms": {} if kind == "none" else {"total": ms},
        }  # fmt: skip
        if kind == "none":
            doc.update(error="LLMUnavailable: x", error_reason=reason)
        doc.update(over)
        (self.dir / arm).mkdir(parents=True, exist_ok=True)
        (self.dir / arm / f"{rid}.json").write_text(json.dumps(doc))

    def fill(self, kinds: dict[str, list[str]], m=None, **kw) -> None:
        """kinds[arm][k] is the kind of molecule k's result; ``m`` (a list, for M200L) its m_i."""
        for arm in ARMS:
            for k, rid in enumerate(IDS):
                self.put(arm, rid, kinds[arm][k], m=(m[k] if arm == "M200L" and m else None), **kw)

    def compare(self, **kw) -> dict:
        return bm.compare(self.dir, self.file, **kw)


@pytest.fixture
def ev(tmp_path) -> Eval:
    return Eval(tmp_path)


def kinds(*, ones: int, n: int = N, tail: str = "none") -> list[str]:
    """The first ``ones`` molecules get the true structure, the rest ``tail`` (similarity 0 for none/bad)."""
    return ["truth"] * ones + [tail] * (n - ones)


def cond(report: dict, text: str) -> dict:
    (c,) = [c for c in report["verdict"]["conditions"] if c["condition"].startswith(text)]
    return c


def holding(report: dict, role: str) -> list[str]:
    return [c["condition"] for c in report["verdict"]["conditions"] if c["role"] == role and c["holds"]]


# --------------------------------------------------------------------------- the measures


def test_similarity_is_the_tanimoto_of_the_first_candidate_and_zero_without_one(ev):
    ev.put("M0-a", "m0", "truth")
    ev.put("M0-a", "m1", "isomer")
    ev.put("M0-a", "m2", "none")
    ev.put("M0-a", "m3", "bad")  # unparseable first candidate
    ev.put("M0-a", "m4", "truth", candidates=[])  # answered but no candidate at all
    rep = ev.compare(write=False)
    per = rep["per_molecule"]["M0-a"]
    assert per["m0"]["similarity"] == 1.0 and per["m1"]["similarity"] == pytest.approx(X) and 0 < X < 1
    assert (
        per["m2"]["similarity"] == 0.0 and per["m3"]["similarity"] == 0.0 and per["m4"]["similarity"] == 0.0
    )
    assert rep["arms"]["M0-a"]["S"] == pytest.approx((1.0 + X) / 5)  # S is over the molecules with a result
    assert per["m3"]["top1"] is False and per["m2"]["no_answer"] is True


def test_the_first_candidate_is_the_services_first_not_the_model_rank_one(ev):
    cands = [
        {"smiles": ISOMER, "model_rank": 2, "status": "ok"},
        {"smiles": ASPIRIN, "model_rank": 1, "status": "ok"},
    ]
    ev.put("M0-a", "m0", "truth", candidates=cands)
    m = ev.compare(write=False)["per_molecule"]["M0-a"]["m0"]
    assert m["similarity"] == pytest.approx(X) and m["top1"] is False and m["top3"] is True
    assert m["own_top1"] is True  # the model's own order puts aspirin first


def test_secondary_measures(ev):
    ev.put("M0-a", "m0", "truth", cost=1.0, ms=2000)
    ev.put("M0-a", "m1", "isomer", cost=3.0, ms=4000, candidates=[
        {"smiles": ISOMER, "model_rank": 1}, {"smiles": "CCO", "model_rank": 2}, {"smiles": ASPIRIN, "model_rank": 3},
    ])  # fmt: skip
    ev.put("M0-a", "m2", "isomer", trace={"first": {"candidates": [{"smiles": ASPIRIN}]}, "repaired": None})
    ev.put("M0-a", "m3", "none", reason="timeout", cost=0)
    ev.put("M0-a", "m4", "none", reason="output_truncated", cost=0)
    a = ev.compare(write=False)["arms"]["M0-a"]
    assert a["n"] == 5 and a["top1"] == 1 and a["top3"] == 2  # m0; m0 and m1 (third candidate)
    assert a["proposed_any_round"] == 3  # m0, m1 (final list), m2 (only in the first round of the trace)
    assert a["own_order_top1"] == 1  # m0 (m1's rank-1 is the isomer)
    assert a["no_answer"] == 2 and a["timeouts"] == 1  # only the reason "timeout" is a timeout
    assert a["mean_time_s"] == pytest.approx((2 + 4 + 1) / 3)  # no_answer has no timing
    assert a["mean_cost_usd"] == pytest.approx((1 + 3 + 1 + 0 + 0) / 5)


def test_examples_and_m_i_are_read_from_the_injected_list(ev):
    ev.put("M200L", "m0", "truth", injected=[
        {"case_id": "a", "similarity": 0.4, "tanimoto_to_truth": 0.2},
        {"case_id": "b", "similarity": 0.3, "tanimoto_to_truth": 0.7},
        {"case_id": "c", "similarity": 0.2, "tanimoto_to_truth": 0.1},
    ])  # fmt: skip
    ev.put("M200L", "m1", "truth", injected=[])
    rep = ev.compare(write=False)
    assert rep["m_i_M200L"] == {"m0": 0.7, "m1": None} and rep["arms"]["M200L"]["mean_examples"] == 1.5


# --------------------------------------------------------------------------- PARTIAL


def test_an_empty_directory_is_partial_with_no_verdict(ev, capsys):
    rep = ev.compare()
    assert (
        rep["complete"] is False
        and rep["verdict"] is None
        and rep["results"] == 0
        and rep["expected"] == 4 * N
    )
    text = capsys.readouterr().out
    assert text.startswith("PARTIAL") and "VERDICT" not in text and "0 of 40" in text
    assert json.loads((ev.dir / "compare.json").read_text())["verdict"] is None
    assert (ev.dir / "compare.txt").read_text() == text


def test_a_missing_result_gives_partial_measures_and_no_verdict(ev, capsys):
    ev.fill({a: kinds(ones=N) for a in ARMS})
    (ev.dir / "R200L" / "m7.json").unlink()
    rep = ev.compare()
    assert rep["results"] == 4 * N - 1 and rep["complete"] is False and rep["verdict"] is None
    assert rep["arms"]["R200L"]["n"] == N - 1 and rep["arms"]["M200L"]["n"] == N
    assert "PARTIAL" in capsys.readouterr().out


def test_only_the_large_molecules_of_the_eval_file_count(ev):
    ev.put("M0-a", "small", "truth")  # not one of the molecules
    assert ev.compare(write=False)["results"] == 0


# --------------------------------------------------------------------------- the numbers of the verdict


def test_s_delta_n_and_wilcoxon_against_scipy(ev):
    # M0-a: truth on m0-m1 (S 0.2); M0-b: on m1-m2 (S 0.2); M200L: m0-m7 (S 0.8); R200L: m0-m2 (S 0.3)
    k = {
        "M0-a": ["truth", "truth"] + ["none"] * 8,
        "M0-b": ["none", "truth", "truth"] + ["none"] * 7,
        "M200L": kinds(ones=8),
        "R200L": kinds(ones=3),
    }
    ev.fill(k, m=[0.1] * N)
    v = ev.compare(write=False)["verdict"]
    s = {a: 0.2 if a.startswith("M0") else 0.8 if a == "M200L" else 0.3 for a in ARMS}
    assert v["delta"] == pytest.approx(s["M200L"] - 0.2) == pytest.approx(0.6)
    assert v["N"] == pytest.approx(0.0)
    d = [1.0 - 0.5, 1.0 - 1.0, 1.0 - 0.5] + [1.0] * 5 + [0.0, 0.0]  # m0, m1, m2, m3-m7, m8, m9
    assert [v["d"][i] for i in IDS] == pytest.approx(d)
    expected = wilcoxon(d, alternative="two-sided", zero_method="wilcox", method="auto").pvalue
    assert v["wilcoxon_p"] == pytest.approx(expected) and v["nonzero_d"] == 7
    assert v["S_M200L_minus_S_R200L"] == pytest.approx(0.5)


def test_no_nonzero_difference_gives_no_p_and_the_condition_fails(ev):
    ev.fill({a: kinds(ones=4) for a in ARMS}, m=[0.1] * N)
    rep = ev.compare(write=False)
    assert rep["verdict"]["wilcoxon_p"] is None and rep["verdict"]["nonzero_d"] == 0
    assert cond(rep, "Wilcoxon")["holds"] is False


# --------------------------------------------------------------------------- the verdict branches


def supported_scenario(ev: Eval, m: list[float] | None = None, r200l: int = 3, m200l_ones: int = 10) -> dict:
    k = {
        "M0-a": kinds(ones=2),
        "M0-b": kinds(ones=2),  # same S: N = 0
        "M200L": kinds(ones=m200l_ones),
        "R200L": kinds(ones=r200l),
    }
    ev.fill(k, m=m or [0.2] * N)
    return ev.compare(write=False)


def test_supported_when_every_condition_holds(ev, capsys):
    rep = supported_scenario(ev)
    v = rep["verdict"]
    # S: M0 0.2, M200L 1.0, R200L 0.3; Δ = 0.8, N = 0; d = +1 on 8 molecules, 0 on 2: p = 2/2^8
    assert v["verdict"] == "H-large supported"
    assert (
        v["delta"] == pytest.approx(0.8) and v["wilcoxon_p"] == pytest.approx(2 / 256) and v["nonzero_d"] == 8
    )
    assert len(holding(rep, "supports")) == 4 and holding(rep, "falsifies") == [] and v["harm"] == []
    assert v["delta_low_m"] == pytest.approx(0.8) and v["n_low_m"] == 10
    out = capsys.readouterr().out
    assert "VERDICT: H-large supported" in out and "[HOLDS]" in out and "harm: none" in out


def test_falsified_by_a_delta_of_at_most_0_02(ev):
    ev.fill({a: kinds(ones=4) for a in ARMS}, m=[0.1] * N)
    rep = ev.compare(write=False)
    assert rep["verdict"]["verdict"] == "H-large falsified" and cond(rep, "Δ ≤ +0.02")["holds"]
    assert cond(rep, "S(R200L) ≥ S(M200L)")["holds"]  # equal means R200L is not worse


def test_falsified_by_delta_not_above_n(ev):
    # M0-a S 0.2, M0-b S 0.6 (N 0.4, mean 0.4); M200L 0.7 (Δ 0.3 <= N); R200L lower
    ev.fill(
        {"M0-a": kinds(ones=2), "M0-b": kinds(ones=6), "M200L": kinds(ones=7), "R200L": kinds(ones=3)},
        m=[0.1] * N,
    )
    rep = ev.compare(write=False)
    assert holding(rep, "falsifies") == ["Δ ≤ N"]
    assert rep["verdict"]["verdict"] == "H-large falsified"
    assert rep["verdict"]["delta"] == pytest.approx(0.3) and rep["verdict"]["N"] == pytest.approx(0.4)


def test_falsified_when_random_is_at_least_as_good(ev):
    rep = supported_scenario(ev, r200l=10)  # R200L S 1.0 = M200L S
    assert holding(rep, "falsifies") == ["S(R200L) ≥ S(M200L)"]
    assert rep["verdict"]["verdict"] == "H-large falsified"


def test_falsified_when_the_gain_is_confined_to_near_duplicates_and_that_beats_support(ev):
    # M200L gains on m2-m9 (S 1.0 vs M0 0.2) but those molecules carry a case at Tanimoto 0.6 to the truth;
    # on the two molecules with m_i < 0.5 (m0, m1: both arms right) the gain is 0.
    m = [0.1, 0.1] + [0.6] * 8
    rep = supported_scenario(ev, m=m)
    v = rep["verdict"]
    assert v["n_low_m"] == 2 and v["delta_low_m"] == pytest.approx(0.0)
    assert holding(rep, "supports") == [c["condition"] for c in v["conditions"] if c["role"] == "supports"]
    assert holding(rep, "falsifies") == ["gain confined to molecules with m_i ≥ 0.5"]
    assert v["verdict"] == "H-large falsified"  # every supporting condition holds, and falsification wins


def test_m_i_of_exactly_0_5_counts_as_a_near_duplicate(ev):
    rep = supported_scenario(ev, m=[0.5] * N)
    assert rep["verdict"]["n_low_m"] == 0 and rep["verdict"]["verdict"] == "H-large falsified"


def test_the_gain_counts_as_confined_when_no_molecule_has_m_i_below_0_5(ev):
    rep = supported_scenario(ev, m=[0.9] * N)
    c = cond(rep, "gain confined")
    assert c["holds"] is True and "n/a" in c["value"] and rep["verdict"]["delta_low_m"] is None
    assert rep["verdict"]["verdict"] == "H-large falsified"


def test_the_gain_is_not_confined_when_it_also_appears_where_m_i_is_low(ev):
    m = [0.1] * 5 + [0.6] * 5
    rep = supported_scenario(ev, m=m)  # d = +1 on m2-m9: low-m molecules m0-m4 gain on m2-m4
    assert cond(rep, "gain confined")["holds"] is False
    assert rep["verdict"]["delta_low_m"] == pytest.approx(3 / 5)
    assert rep["verdict"]["verdict"] == "H-large supported"


def test_inconclusive_when_nothing_falsifies_and_something_supporting_fails(ev):
    # M200L right on m0-m2 only (S 0.3) against M0 0.0 -> Δ 0.3, N 0, three positive d: p = 2/8 = 0.25
    ev.fill(
        {"M0-a": kinds(ones=0), "M0-b": kinds(ones=0), "M200L": kinds(ones=3), "R200L": kinds(ones=0)},
        m=[0.1] * N,
    )
    rep = ev.compare(write=False)
    assert rep["verdict"]["wilcoxon_p"] == pytest.approx(0.25)
    assert holding(rep, "falsifies") == [] and rep["verdict"]["verdict"] == "inconclusive"
    assert [
        c["condition"] for c in rep["verdict"]["conditions"] if c["role"] == "supports" and not c["holds"]
    ] == ["Wilcoxon two-sided p < 0.05"]


def test_inconclusive_when_the_gap_to_random_is_positive_but_below_0_03(ev):
    # M200L: nine right and one near miss (0.667); R200L: nine right and one miss at X (0.484): gap 0.0183
    k = {
        "M0-a": kinds(ones=2),
        "M0-b": kinds(ones=2),
        "M200L": kinds(ones=9, tail="near"),
        "R200L": kinds(ones=9, tail="isomer"),
    }
    ev.fill(k, m=[0.1] * N)
    rep = ev.compare(write=False)
    near = bm.tanimoto("CC(=O)Oc1ccccc1C(=O)OC", ASPIRIN)
    assert rep["verdict"]["S_M200L_minus_S_R200L"] == pytest.approx((near - X) / 10)
    assert 0 < rep["verdict"]["S_M200L_minus_S_R200L"] < 0.03
    assert holding(rep, "falsifies") == [] and rep["verdict"]["verdict"] == "inconclusive"
    assert not cond(rep, "S(M200L) − S(R200L)")["holds"]


def test_a_delta_that_is_0_05_on_paper_meets_the_threshold_despite_float_error(ev):
    # M0 means (0.4 + 0.5) / 2 = 0.45 and M200L 0.5: 0.5 - 0.45 is 0.04999999999999999 in floating point
    ev.fill(
        {"M0-a": kinds(ones=4), "M0-b": kinds(ones=5), "M200L": kinds(ones=5), "R200L": kinds(ones=4)},
        m=[0.1] * N,
    )
    rep = ev.compare(write=False)
    assert rep["verdict"]["delta"] < 0.05 and cond(rep, "Δ ≥ +0.05")["holds"]
    assert rep["verdict"]["verdict"] == "H-large falsified" and cond(rep, "Δ ≤ N")["holds"]  # N is 0.1


# --------------------------------------------------------------------------- harm


def test_timeout_harm_counts_only_timeouts_and_is_reported_with_the_verdict(ev):
    # M200L has 5 timeouts, the M0 arms 1 and 0 (mean 0.5): 5 - 0.5 > 3 -> harm. A non-timeout no_answer is not counted.
    iso = {"tail": "isomer"}
    k = {
        "M0-a": kinds(ones=2, **iso),
        "M0-b": kinds(ones=2, **iso),
        "M200L": kinds(ones=5, **iso),
        "R200L": kinds(ones=2, **iso),
    }
    ev.fill(k, m=[0.1] * N)
    for rid in IDS[5:]:
        ev.put("M200L", rid, "none", reason="timeout", m=0.1)
    ev.put("M0-a", "m9", "none", reason="timeout")
    ev.put("M0-b", "m9", "none", reason="call_budget_exhausted")
    rep = ev.compare(write=False)
    assert rep["arms"]["M200L"]["timeouts"] == 5 and rep["arms"]["M0-a"]["timeouts"] == 1
    assert rep["arms"]["M0-b"]["timeouts"] == 0 and rep["arms"]["M0-b"]["no_answer"] == 1
    assert [h["condition"] for h in rep["verdict"]["harm"]] == ["timeouts: M200L − mean(M0) > 3"]
    assert rep["verdict"]["verdict"]  # harm does not replace the verdict


def test_three_more_timeouts_than_the_m0_mean_is_not_harm(ev):
    k = {a: kinds(ones=2) for a in ARMS}
    ev.fill(k, m=[0.1] * N)
    for rid in IDS[7:]:
        ev.put("M200L", rid, "none", reason="timeout", m=0.1)  # 3 timeouts against 0
    assert ev.compare(write=False)["verdict"]["harm"] == []


def test_cost_harm_is_a_rise_of_more_than_30_percent(ev):
    ev.fill({a: kinds(ones=2) for a in ARMS}, m=[0.1] * N, cost=1.0)
    for rid in IDS:  # M200L at 1.3 per molecule against 1.0: +30% exactly (0.3000000000000001 in floats)
        ev.put("M200L", rid, "truth", cost=1.3, m=0.1)
    assert ev.compare(write=False)["verdict"]["harm"] == []
    for rid in IDS:
        ev.put("M200L", rid, "truth", cost=1.31, m=0.1)
    harm = ev.compare(write=False)["verdict"]["harm"]
    assert [h["condition"] for h in harm] == ["cost per molecule rises by more than 30%"]


def test_the_text_lists_every_condition_with_its_value(ev):
    rep = supported_scenario(ev)
    text = bm.compare_text(rep)
    for c in rep["verdict"]["conditions"]:
        assert c["condition"] in text and c["value"] in text
    assert "Wilcoxon" in text and "non-zero d_i = 8" in text


def test_compare_writes_the_json_and_the_summary(ev):
    supported_scenario(ev)
    rep = ev.compare()
    on_disk = json.loads((ev.dir / "compare.json").read_text())
    assert on_disk["verdict"]["verdict"] == rep["verdict"]["verdict"] == "H-large supported"
    assert (ev.dir / "compare.txt").read_text().rstrip().splitlines()[-1].startswith("harm:")
    assert bm.eval_spent(ev.dir) == pytest.approx(
        40.0
    )  # compare.json and compare.txt add nothing to the spend
