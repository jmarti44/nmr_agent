"""bench/memory.py eval: the four H-large arms. Fake API only; every file is in a temporary directory."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import bench.memory as bm
from src.core.settings import Settings
from src.memory import MemoryRuntime, outcome_config
from src.memory.render import EXAMPLES_HEADER, case_cost
from src.memory.retrieval import select_random
from src.memory.runtime import EXAMPLES_TOKEN_BUDGET
from src.memory.store import chemistry_fields
from tests.llm.fake_api import ASPIRIN, ISOMER, FakeAPI, error, hypothesis_json, message
from tests.memory.test_collect import TIMEOUT, make_record, reader
from tests.memory.test_runtime import build_store, case_dict

NO_RETRY = Settings(sdk_max_retries=0)
TEREPHTHALATE = "COC(=O)c1ccc(C(=O)O)cc1"  # a third C9H8O4 isomer
STORE_SMILES = ["CC(=O)OC1CCCCC1", "CCC(=O)OC1CCCC1", "CC(=O)OC1CCCCCC1"]  # three scaffolds, none an eval one
OK = lambda: message(hypothesis_json([ASPIRIN]))  # noqa: E731 - a reply the verifier accepts


class Env:
    """A 3-case store, a MEM-EVAL file with 3 large and 1 small molecule, and the SD records behind them."""

    def __init__(self, tmp: Path, large=(("a1", ASPIRIN), ("a2", ISOMER), ("a3", TEREPHTHALATE))):
        self.tmp = tmp
        self.store = tmp / "memory"
        chems = [chemistry_fields(s) for s in STORE_SMILES]
        cases = [case_dict(i, Settings(), c) for i, c in enumerate(chems, 1)]
        for c, chem in zip(cases, chems, strict=True):
            c["agent_outcome"]["proposed_smiles"] = [chem["smiles"]]
        build_store(self.store, Settings(), cases)
        molecules = [*large, ("small", ASPIRIN)]
        self.recs = {rid: make_record(rid, smi, aspirin_spectra=True) for rid, smi in molecules}
        self.rows = [
            {
                "id": rid, "smiles": r.smiles, "inchikey": r.inchikey, "formula": r.formula,
                "solvent": "CDCl3", "h_spec": r.h_spec, "c_spec": r.c_spec,
                "heavy_atoms": 15 if rid == "small" else 25, "scaffold": "x", "generic_scaffold": "y",
            }
            for rid, r in self.recs.items()
        ]  # fmt: skip
        self.eval = tmp / "mem_eval.json"
        self.eval.write_text(json.dumps({"records": self.rows}))
        self.out = tmp / "eval"

    def run(self, api: FakeAPI | None = None, **kw):
        api = api or FakeAPI([])
        args = dict(
            eval_file=self.eval, store=self.store, out_dir=self.out, collect_dir=self.tmp / "collect",
            read_records=reader(self.recs), client_factory=lambda s: api.client(), settings=NO_RETRY,
            cases_expected=3, old_keys=set(),
        )  # fmt: skip
        return bm.eval_run(**(args | kw))

    def result(self, arm: str, rid: str) -> dict:
        return json.loads((self.out / arm / f"{rid}.json").read_text())

    def files(self) -> list[str]:
        """The result files (run.json is not one)."""
        found = (
            sorted(str(p.relative_to(self.out)) for p in self.out.rglob("*.json"))
            if self.out.exists()
            else []
        )
        return [f for f in found if f != "run.json"]


@pytest.fixture
def env(tmp_path) -> Env:
    return Env(tmp_path)


def user_text(request: dict) -> str:
    content = request["body"]["messages"][0]["content"]
    return content if isinstance(content, str) else "".join(b.get("text", "") for b in content)


# --------------------------------------------------------------------------- order, arms, examples


def test_every_arm_runs_once_per_molecule_in_the_seeded_order(env):
    api = FakeAPI([OK() for _ in range(12)])
    out = env.run(api)
    expected = [(i, a) for i in ("a1", "a2", "a3") for a in bm.arm_order(i)]
    assert out["ran"] == expected and len(api.requests) == 12
    for rid in ("a1", "a2", "a3"):
        assert sorted(a for i, a in out["ran"] if i == rid) == sorted(bm.ARMS)
        for arm in bm.ARMS:
            assert env.result(arm, rid)["arm"] == arm
    assert len(env.files()) == 12


def test_the_order_seed_is_the_spec_rule_and_is_not_the_same_for_every_molecule():
    for rid in ("a1", "60001975"):
        seed = int(hashlib.sha256(f"{rid}:order".encode()).hexdigest()[:8], 16)
        assert bm.order_seed(rid) == seed
        import random

        shuffled = list(bm.ARMS)
        random.Random(seed).shuffle(shuffled)
        assert bm.arm_order(rid) == shuffled
    assert len({tuple(bm.arm_order(f"id{i}")) for i in range(40)}) > 1


def test_an_arm_subset_keeps_the_seeded_relative_order():
    full = bm.arm_order("60001975")
    assert bm.arm_order("60001975", ("R200L", "M0-a")) == [a for a in full if a in ("R200L", "M0-a")]


def test_only_the_memory_arms_send_examples(env):
    api = FakeAPI([OK() for _ in range(4)])
    out = env.run(api, ids=["a1"])
    sent = {arm: user_text(r) for (_, arm), r in zip(out["ran"], api.requests, strict=True)}
    assert EXAMPLES_HEADER not in sent["M0-a"] and EXAMPLES_HEADER not in sent["M0-b"]
    assert EXAMPLES_HEADER in sent["M200L"] and EXAMPLES_HEADER in sent["R200L"]
    for arm in ("M0-a", "M0-b"):
        r = env.result(arm, "a1")
        assert r["injected"] == [] and r["memory_version"] is None and r["trace"]["memory_cases"] == []
    for arm in bm.MEMORY_ARMS:
        r = env.result(arm, "a1")
        assert len(r["injected"]) >= 1 and r["memory_version"]
        assert [i["case_id"] for i in r["injected"]] == [m["case_id"] for m in r["trace"]["memory_cases"]]
        assert all(0 <= i["tanimoto_to_truth"] <= 1 for i in r["injected"])


def test_the_four_arms_of_a_molecule_send_the_same_request_apart_from_the_examples(env):
    api = FakeAPI([OK() for _ in range(4)])
    out = env.run(api, ids=["a2"])
    sent = {arm: user_text(r) for (_, arm), r in zip(out["ran"], api.requests, strict=True)}
    assert sent["M0-a"] == sent["M0-b"]
    for arm in bm.MEMORY_ARMS:
        assert sent[arm].endswith("\n\n" + sent["M0-a"]) and sent[arm].startswith(EXAMPLES_HEADER)
    hashes = {env.result(a, "a2")["request_sha256"] for a in bm.ARMS}
    assert hashes == {hashlib.sha256(sent["M0-a"].encode()).hexdigest()}
    prompts = {a: env.result(a, "a2")["prompt_sha256"] for a in bm.ARMS}
    assert prompts[("M0-a")] == prompts["M0-b"] != prompts["M200L"]
    assert prompts["M200L"] == hashlib.sha256(sent["M200L"].encode()).hexdigest()


def test_random_arm_uses_the_spec_seed_and_the_same_cases_as_select_random(env):
    api = FakeAPI([OK() for _ in range(4)])
    env.run(api, ids=["a1"])
    assert bm.random_seed("a1") == int(hashlib.sha256(b"a1:0").hexdigest()[:8], 16)
    runtime = MemoryRuntime.from_directory(env.store, NO_RETRY, strict=True)
    direct = select_random(
        runtime.store,
        bm.random_seed("a1"),
        runtime.params,
        token_budget=EXAMPLES_TOKEN_BUDGET,
        cost_fn=case_cost,
    )
    got = [i["case_id"] for i in env.result("R200L", "a1")["injected"]]
    assert got == [r.case.case_id for r in direct] and got
    assert all(i["similarity"] is None for i in env.result("R200L", "a1")["injected"])
    assert all(i["similarity"] is not None for i in env.result("M200L", "a1")["injected"])


def test_the_result_file_carries_every_field(env):
    api = FakeAPI([OK() for _ in range(4)])
    env.run(api, ids=["a1"])
    r = env.result("M200L", "a1")
    assert {
        "arm", "id", "truth", "truth_inchikey", "heavy_atoms", "request_sha256", "outcome_config", "status",
        "candidates", "trace", "injected", "cost_usd", "llm_calls", "timing_ms", "warnings", "memory_version",
    } <= set(r)  # fmt: skip
    assert r["truth"] == env.recs["a1"].smiles and r["heavy_atoms"] == 25 and r["status"] == "solved"
    assert r["outcome_config"] == outcome_config(NO_RETRY).model_dump()
    assert r["candidates"][0]["smiles"] == ASPIRIN and r["cost_usd"] > 0 and r["llm_calls"] == 1


def test_missed_and_solved_follow_the_collect_rule(env):
    api = FakeAPI([OK() for _ in range(4)])
    env.run(api, ids=["a2"])  # the truth is the isomer, the fake model answers aspirin
    assert {env.result(a, "a2")["status"] for a in bm.ARMS} == {"missed"}


def test_only_the_large_mem_eval_molecules_are_used(env):
    assert [r["id"] for r in bm.eval_rows(env.eval)] == ["a1", "a2", "a3"]
    api = FakeAPI([OK() for _ in range(12)])
    env.run(api)
    assert not [f for f in env.files() if "small" in f]
    with pytest.raises(SystemExit, match="not among"):
        env.run(FakeAPI([]), ids=["small"])


def test_a_memory_molecule_id_is_refused(env):
    with pytest.raises(SystemExit, match="not among"):
        env.run(FakeAPI([]), ids=["id1"])


# --------------------------------------------------------------------------- resume, run.json, store checks


def test_resume_skips_the_analyses_that_have_a_result(env):
    first = FakeAPI([OK() for _ in range(2)])
    out1 = env.run(first, ids=["a1"], arms=("M0-a", "M200L"))
    assert len(out1["ran"]) == 2
    second = FakeAPI([OK() for _ in range(2)])  # a1's other two arms only: any further call would raise
    out2 = env.run(second, ids=["a1"])
    assert sorted(a for _, a in out2["ran"]) == ["M0-b", "R200L"] and len(second.requests) == 2
    third = FakeAPI([])
    assert env.run(third, ids=["a1"])["ran"] == [] and third.requests == []


def test_run_json_records_the_store_and_the_configuration_and_is_written_once(env):
    env.run(FakeAPI([OK()]), ids=["a1"], arms=("M0-a",))
    run = json.loads((env.out / "run.json").read_text())
    assert run["manifest_sha256"] == hashlib.sha256((env.store / "manifest.json").read_bytes()).hexdigest()
    assert run["memory_version"] and run["cases"] == 3
    assert (
        run["retrieval_params_sha256"]
        == json.loads((env.store / "retrieval_params.json").read_text())["sha256"]
    )
    assert run["outcome_config"]["model_id"] == NO_RETRY.model and run["molecule_ids"] == ["a1", "a2", "a3"]
    before = (env.out / "run.json").read_bytes()
    env.run(FakeAPI([OK()]), ids=["a1"], arms=("M0-b",))
    assert (env.out / "run.json").read_bytes() == before


@pytest.mark.parametrize(
    "key", ["manifest_sha256", "memory_version", "retrieval_params_sha256", "outcome_config"]
)
def test_a_run_json_that_differs_refuses_the_resume(env, key):
    env.run(FakeAPI([OK()]), ids=["a1"], arms=("M0-a",))
    run = json.loads((env.out / "run.json").read_text())
    run[key] = "changed" if key != "outcome_config" else {"model_id": "x"}
    (env.out / "run.json").write_text(json.dumps(run))
    api = FakeAPI([OK()])
    with pytest.raises(SystemExit, match="does not match this run"):
        env.run(api, ids=["a1"], arms=("M0-b",))
    assert api.requests == [] and not (env.out / "M0-b" / "a1.json").exists()


def test_a_changed_store_refuses_the_resume(env):
    env.run(FakeAPI([OK()]), ids=["a1"], arms=("M0-a",))
    case = case_dict(4, Settings(), chemistry_fields("CC(=O)OC1CCCCCCC1"))
    case["agent_outcome"]["proposed_smiles"] = ["CC(=O)OC1CCCCCCC1"]
    import tests.memory.test_store as ts

    data = ts.dump(case)
    (env.store / "cases" / "mem-000004.json").write_bytes(data)
    manifest = json.loads((env.store / "manifest.json").read_text())
    manifest["cases"].append({"case_id": "mem-000004", "sha256": hashlib.sha256(data).hexdigest()})
    (env.store / "manifest.json").write_text(json.dumps(manifest))
    api = FakeAPI([OK()])
    with pytest.raises(SystemExit, match="does not match this run"):
        env.run(api, ids=["a1"], arms=("M0-b",), cases_expected=4)
    assert api.requests == []


def test_results_without_a_run_json_refuse(env):
    env.run(FakeAPI([OK()]), ids=["a1"], arms=("M0-a",))
    (env.out / "run.json").unlink()
    with pytest.raises(SystemExit, match="missing"):
        env.run(FakeAPI([OK()]), ids=["a1"], arms=("M0-b",))


def test_a_store_that_does_not_hold_200_cases_is_refused(env):
    api = FakeAPI([])
    with pytest.raises(SystemExit, match="holds 3 cases, not 200"):
        env.run(api, cases_expected=200, dry_run=True)
    with pytest.raises(SystemExit, match="holds 3 cases, not 200"):
        env.run(api, cases_expected=200)
    assert api.requests == [] and env.files() == []


def test_a_store_that_breaks_a_leakage_rule_is_refused(env):
    rows = json.loads(env.eval.read_text())["records"]
    clash = chemistry_fields(STORE_SMILES[0])
    rows[0].update(
        smiles=clash["smiles"], inchikey=clash["inchikey"]
    )  # an eval molecule that is a store case
    env.eval.write_text(json.dumps({"records": rows}))
    with pytest.raises(SystemExit, match="leakage"):
        env.run(FakeAPI([]))
    assert env.files() == []


def test_a_store_that_does_not_load_strictly_is_refused(env):
    (env.store / "cases" / "mem-000001.json").write_text("{}")  # no longer the hashed bytes
    with pytest.raises(SystemExit, match="cannot be loaded strictly"):
        env.run(FakeAPI([]))


def test_a_stale_result_stops_the_run_before_any_call(env):
    env.run(FakeAPI([OK()]), ids=["a1"], arms=("M0-a",))
    f = env.out / "M0-a" / "a1.json"
    doc = json.loads(f.read_text())
    doc["request_sha256"] = "0" * 64
    f.write_text(json.dumps(doc))
    api = FakeAPI([OK()])
    with pytest.raises(SystemExit, match="do not match"):
        env.run(api, ids=["a1"])
    assert api.requests == []
    del doc["request_sha256"]
    f.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="do not match"):
        env.run(api, ids=["a1"])
    assert api.requests == []


def test_a_request_that_cannot_be_built_stops_the_run_before_any_call(env):
    del env.recs["a2"]
    api = FakeAPI([])
    with pytest.raises(SystemExit, match="cannot be built"):
        env.run(api)
    assert api.requests == [] and env.files() == []


# --------------------------------------------------------------------------- errors, cap, dry run


def test_an_account_error_stops_with_nothing_saved_for_that_analysis(env):
    credit = error(400, "invalid_request_error", "Your credit balance is too low to access the Anthropic API")
    api = FakeAPI([OK(), credit])
    with pytest.raises(SystemExit, match="STOPPED at"):
        env.run(api, ids=["a1"])
    saved = env.files()
    assert len(saved) == 1  # the first analysis only: the one that failed left no file
    assert all("no_answer" != json.loads((env.out / f).read_text())["status"] for f in saved)
    resumed = FakeAPI([OK(), OK(), OK()])
    out = env.run(resumed, ids=["a1"])
    assert len(out["ran"]) == 3  # the three analyses without a file, the failed one among them


@pytest.mark.parametrize("status", [401, 403, 429, 500, 529])
def test_other_http_failures_also_stop_the_run(env, status):
    api = FakeAPI([error(status, "api_error", "x")])
    with pytest.raises(SystemExit, match="STOPPED at"):
        env.run(api, ids=["a1"])
    assert env.files() == []


def test_a_timeout_is_saved_as_no_answer(env):
    api = FakeAPI([TIMEOUT, OK(), OK(), OK()])
    out = env.run(api, ids=["a1"])
    first_arm = out["ran"][0][1]
    r = env.result(first_arm, "a1")
    assert r["status"] == "no_answer" and r["candidates"] == [] and r["error_reason"] == "timeout"
    assert r["error"] and r["cost_usd"] == 0 and r["injected"] is not None
    assert len(out["ran"]) == 4


def test_the_spend_cap_stops_before_the_next_analysis(env):
    api = FakeAPI([OK() for _ in range(12)])
    out = env.run(api, cap=1e-9)
    assert len(out["ran"]) == 1 and out["stopped"] == "cap" and len(api.requests) == 1
    again = env.run(FakeAPI([]), cap=1e-9)
    assert again["ran"] == [] and again["stopped"] == "cap"
    assert bm.eval_spent(env.out) == out["spent"] > 0


def test_the_spend_counts_every_result_file_under_the_eval_directory(env):
    env.run(FakeAPI([OK() for _ in range(4)]), ids=["a1"])
    total = sum(env.result(a, "a1")["cost_usd"] for a in bm.ARMS)
    assert bm.eval_spent(env.out) == pytest.approx(total)
    out = env.run(FakeAPI([]), cap=total)  # already at the cap: nothing more is sent
    assert out["ran"] == [] and out["stopped"] == "cap"


def test_dry_run_calls_nothing_writes_nothing_and_reports(env, capsys):
    def refuse(settings):
        raise AssertionError("a client was created in a dry run")

    rep = env.run(client_factory=refuse, dry_run=True)
    assert env.files() == [] and not env.out.exists()
    assert rep["analyses"] == 12 and rep["buildable"] == 12 and rep["failures"] == {}
    assert set(rep["input_tokens_est"]) == set(bm.ARMS)
    assert (
        rep["input_tokens_est"]["M200L"] > rep["input_tokens_est"]["M0-a"] == rep["input_tokens_est"]["M0-b"]
    )
    assert set(rep["examples_per_molecule"]) == set(bm.MEMORY_ARMS)
    assert all(len(v) == 3 and min(v) >= 1 for v in rep["examples_per_molecule"].values())
    text = capsys.readouterr().out
    assert "12 analyses, 12 buildable, 0 failed" in text and "no model call" in text
    assert "examples injected per molecule" in text and "estimated input tokens" in text
    assert "no priced collect results" in text


def test_dry_run_reports_a_build_failure_and_an_estimate(env, capsys):
    (env.tmp / "collect").mkdir()
    large = env.tmp / "memory_large.json"
    rows = [
        {"index": 1, "stratum": ">20", "position": 1, "id": "m1", "pool": "named", "smiles": ASPIRIN,
         "inchikey": "x", "heavy_atoms": 25, "scaffold": "", "solvent": "CDCl3"}
    ]  # fmt: skip
    rec = make_record("m1", ASPIRIN, aspirin_spectra=True)
    rows[0].update(smiles=rec.smiles, inchikey=rec.inchikey)
    large.write_text(json.dumps({"entries": rows, "replacement_order": {">20": [], "11-20": [], "<=10": []}}))
    (env.tmp / "collect" / "m1.json").write_text(json.dumps({"id": "m1", "stratum": ">20", "cost_usd": 0.1}))
    env.recs["m1"] = rec
    del env.recs["a3"]
    rep = env.run(dry_run=True, large_file=large)
    assert rep["buildable"] == 8 and set(rep["failures"]) == {f"{a}/a3" for a in bm.ARMS}
    est = rep["estimate"]
    assert est["priced_collect_results"] == 1 and est["mean_cost_per_molecule"] == pytest.approx(0.1)
    assert est["method_1_total"] == pytest.approx(0.1 * 12)
    assert est["method_2_total"] > 0
    text = capsys.readouterr().out
    assert "FAILS M0-a a3" in text and "method 1" in text and "method 2" in text


def test_main_wires_the_subcommands(monkeypatch):
    calls = []
    monkeypatch.setattr(bm, "eval_run", lambda *a: calls.append(("eval", a)))
    monkeypatch.setattr(bm, "compare", lambda: calls.append(("compare",)))
    monkeypatch.setattr(
        "sys.argv", ["memory.py", "eval", "--dry-run", "--arms", "M0-a", "R200L", "--cap", "5"]
    )
    bm.main()
    monkeypatch.setattr("sys.argv", ["memory.py", "compare"])
    bm.main()
    assert calls[0] == ("eval", (("M0-a", "R200L"), None, 5.0, True)) and calls[1] == ("compare",)


def test_dry_run_reports_a_stale_result(env, capsys):
    env.run(FakeAPI([OK()]), ids=["a1"], arms=("M0-a",))
    f = env.out / "M0-a" / "a1.json"
    doc = json.loads(f.read_text())
    doc["request_sha256"] = "0" * 64
    f.write_text(json.dumps(doc))
    rep = env.run(dry_run=True)
    assert list(rep["stale"]) == ["M0-a/a1"] and "STALE RESULT M0-a a1" in capsys.readouterr().out
