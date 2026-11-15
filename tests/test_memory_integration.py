"""Memory in the proposal prompt: byte-identical when off, examples in the user message only, size limits,
the response fields, and no path from memory to the verifier. Fake API only; stores in temporary dirs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import analysis
from src.api import analyze_routes
from src.api.analyze_routes import get_llm_client, get_memory
from src.api.main import app
from src.core.settings import Settings, get_settings
from src.memory import MemoryRuntime, RetrievalParams, RetrievalResult, render_examples
from src.memory.store import MemoryStore
from src.prompting import FULL_VERSION, builder, prepare_request, system_prompt
from src.prompting.builder import PromptTooLarge, build_prompt, build_user_message
from src.prompting.render import render_context_block, render_peak_table
from src.schemas import SampleContext
from tests.llm.fake_api import ISOMER, FakeAPI, hypothesis_json, message
from tests.memory.test_retrieval import make_case
from tests.memory.test_runtime import build_store, case_dict, three_cases, write_params
from tests.memory.test_store import ETHYL_ACETATE, ETHYL_BENZOATE
from tests.prompting.helpers import ASPIRIN_DMSO, aspirin_request

SETTINGS = Settings()
RULE_FRAGMENT = "Never copy a structure from them."


def aspirin_table():
    from src.processing.normalize import normalize_peaklist

    req = aspirin_request()
    return normalize_peaklist(req.peaks, req.context), req.context


def old_user_message(table, context, n: int) -> str:
    """The user message exactly as it was composed before memory existed."""
    return render_peak_table(table, context) + "\n\n" + render_context_block(context, n)


def results(*cases) -> list[RetrievalResult]:
    return [RetrievalResult(c, 0.9 - 0.1 * i) for i, c in enumerate(cases)]


def synthetic_cases(n: int = 3):
    return [make_case(i + 1, scaffold=f"scaffold{i}") for i in range(n)]


def runtime(tmp_path: Path, settings: Settings = SETTINGS, **kw) -> MemoryRuntime:
    root = build_store(tmp_path, settings, three_cases(settings))
    return MemoryRuntime.from_directory(root, settings, strict=True, **kw)


# --------------------------------------------------------------------------- prompt


def test_user_message_without_examples_is_byte_identical_to_before():
    table, context = aspirin_table()
    assert build_user_message(table, context, 3) == old_user_message(table, context, 3)
    assert build_user_message(table, context, 3, "") == old_user_message(table, context, 3)
    assert build_prompt(table, context, 3).user_text == old_user_message(table, context, 3)
    assert build_prompt(table, context, 3, examples=[]).examples == ()


def test_request_path_without_memory_is_byte_identical():
    table, prompt, context = prepare_request(aspirin_request())
    assert prompt.user_text == old_user_message(table, context, 3) and prompt.examples == ()
    assert prepare_request(aspirin_request(), None, None)[1].user_text == prompt.user_text


def test_memory_with_no_match_leaves_the_message_unchanged(tmp_path):
    empty = MemoryRuntime(MemoryStore(cases=(), memory_version="none"), RetrievalParams())
    _, plain, _ = prepare_request(aspirin_request())
    _, with_empty, _ = prepare_request(aspirin_request(), None, empty)
    assert with_empty.user_text == plain.user_text and with_empty.examples == ()


def test_examples_precede_the_table_and_the_context_block(tmp_path):
    rt = runtime(tmp_path)
    table, prompt, context = prepare_request(aspirin_request(), None, rt)
    block = render_examples([r.case for r in prompt.examples])
    assert prompt.examples and prompt.user_text == block + "\n\n" + old_user_message(table, context, 3)


def test_system_block_is_identical_with_and_without_examples(tmp_path):
    _, plain, _ = prepare_request(aspirin_request())
    _, injected, _ = prepare_request(aspirin_request(), None, runtime(tmp_path))
    assert injected.examples and plain.system_blocks == injected.system_blocks
    assert plain.system_hash == injected.system_hash and plain.version == injected.version == FULL_VERSION
    assert plain.system_blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert "Worked examples" not in plain.system_blocks[0]["text"] + injected.system_blocks[0]["text"]


def test_the_new_rule_is_present_whether_or_not_examples_are_injected(tmp_path):
    _, plain, _ = prepare_request(aspirin_request())
    _, injected, _ = prepare_request(aspirin_request(), None, runtime(tmp_path))
    for prompt in (plain, injected):
        text = " ".join(prompt.system_blocks[0]["text"].split())  # the rule wraps across lines
        assert RULE_FRAGMENT in text and "reasoning pattern" in text and "earlier wrong proposal" in text
        assert "must fit the present peak table and formula" in text
    assert RULE_FRAGMENT in " ".join(system_prompt().split())


def test_prompt_version_is_bumped():
    assert FULL_VERSION.startswith("2026.11.1+")


# --------------------------------------------------------------------------- size


def sizes(table, context, cases):
    tokens = [builder.estimate_tokens(old_user_message(table, context, 3))]
    for n in range(1, len(cases) + 1):
        text = build_user_message(table, context, 3, render_examples(cases[:n]))
        tokens.append(builder.estimate_tokens(text))
    return tokens  # tokens[n] = size with the first n examples


def test_examples_are_dropped_lowest_ranked_first_until_the_message_fits(monkeypatch):
    table, context = aspirin_table()
    cases = synthetic_cases(3)
    t = sizes(table, context, cases)
    assert t[0] < t[1] < t[2] < t[3]
    for limit, kept in ((t[3], 3), (t[3] - 1, 2), (t[2], 2), (t[2] - 1, 1), (t[1], 1), (t[1] - 1, 0)):
        monkeypatch.setattr(builder, "MAX_USER_TOKENS_ESTIMATE", limit)
        prompt = build_prompt(table, context, 3, results(*cases))
        assert [r.case.case_id for r in prompt.examples] == [c.case_id for c in cases[:kept]], limit
        assert prompt.user_tokens_est <= limit
        if kept == 0:
            assert prompt.user_text == old_user_message(table, context, 3)


def test_a_message_too_large_without_examples_still_raises(monkeypatch):
    table, context = aspirin_table()
    base = sizes(table, context, [])[0]
    monkeypatch.setattr(builder, "MAX_USER_TOKENS_ESTIMATE", base - 1)
    with pytest.raises(PromptTooLarge):
        build_prompt(table, context, 3)
    with pytest.raises(PromptTooLarge):  # examples never change this outcome
        build_prompt(table, context, 3, results(*synthetic_cases(3)))


def test_the_examples_block_respects_its_own_cap(tmp_path):
    rt = runtime(tmp_path)
    table, context = aspirin_table()
    block = render_examples([r.case for r in rt.examples_for(table, context)])
    assert builder.estimate_tokens(block) <= 2000 + 100  # cases within 2,000; header and footer are extra


# --------------------------------------------------------------------------- end to end


def run(api, memory=None, trace=None):
    return analysis.run_analysis(
        aspirin_request(), SETTINGS, api.client(), "req_test", None, trace, None, memory
    )


def test_memory_off_adds_only_the_new_fields():
    r = run(FakeAPI([message(hypothesis_json())]))
    assert r.versions.memory_version == "none" and r.memory_cases == []
    assert r.versions.schema_version == "1.1.0" and r.versions.prompt_version == FULL_VERSION
    assert "memory_cases" in json.loads(r.model_dump_json())


def test_analysis_with_memory_injects_examples_and_reports_them(tmp_path):
    rt = runtime(tmp_path)
    api, trace = FakeAPI([message(hypothesis_json())]), {}
    r = run(api, rt, trace)
    sent = api.requests[0]["body"]["messages"][0]["content"]
    assert sent.startswith("Worked examples:") and "Example molecule: formula C9H10O2" in sent
    assert r.versions.memory_version == rt.memory_version != "none"
    ids = [m.case_id for m in r.memory_cases]
    assert ids and set(ids) <= {c.case_id for c in rt.store.cases}
    assert all(m.similarity is not None and 0 <= m.similarity <= 1 for m in r.memory_cases)
    assert trace["memory_cases"] == [m.model_dump(mode="json") for m in r.memory_cases]


def test_random_mode_reports_null_similarity(tmp_path):
    r = run(FakeAPI([message(hypothesis_json())]), runtime(tmp_path, mode="random", seed=3))
    assert r.memory_cases and all(m.similarity is None for m in r.memory_cases)


def test_system_block_sent_is_identical_with_and_without_memory(tmp_path):
    a, b = FakeAPI([message(hypothesis_json())]), FakeAPI([message(hypothesis_json())])
    run(a)
    run(b, runtime(tmp_path))
    assert a.requests[0]["body"]["system"] == b.requests[0]["body"]["system"]


def spied_verification(monkeypatch):
    calls = []
    real_verify, real_groups = (
        analysis._verify,
        __import__("src.verify.verifier", fromlist=["x"]).verify_functional_groups,
    )

    def verify(table, candidates, hyp, context, claims_rank):
        calls.append(("verify", table.model_dump(mode="json"), [c.model_dump() for c in candidates],
                      hyp.model_dump(mode="json"), context.model_dump(mode="json"), claims_rank))  # fmt: skip
        return real_verify(table, candidates, hyp, context, claims_rank)

    monkeypatch.setattr(analysis, "_verify", verify)
    return calls, real_groups


@pytest.mark.parametrize("answer", [hypothesis_json(), hypothesis_json([ISOMER])])
def test_verification_is_identical_with_and_without_memory(tmp_path, monkeypatch, answer):
    calls, _ = spied_verification(monkeypatch)
    off = run(FakeAPI([message(answer), message(answer)]))
    off_calls, calls[:] = list(calls), []
    on = run(FakeAPI([message(answer), message(answer)]), runtime(tmp_path))
    assert on.memory_cases and calls == off_calls  # the verifier saw exactly the same inputs

    def dump(r):
        return r.model_dump(
            mode="json", exclude={"request_id", "timing_ms", "usage", "memory_cases", "versions"}
        )

    assert dump(on) == dump(off)
    assert [(c.status, c.final_confidence) for c in on.candidates] == [
        (c.status, c.final_confidence) for c in off.candidates
    ]


def test_no_verifier_module_imports_memory():
    root = Path(__file__).parents[1] / "src" / "verify"
    for path in root.rglob("*.py"):
        assert "src.memory" not in path.read_text(), path


def test_upload_branch_also_retrieves(tmp_path):
    from tests.processing.test_pipeline import zip_dir
    from tests.synthetic import write_synthetic_bruker

    upload = zip_dir(write_synthetic_bruker(tmp_path / "bruker"))
    from src.schemas import AnalysisRequest

    req = AnalysisRequest.model_validate({"input_format": "bruker_zip", "context": {}})
    rt = runtime(tmp_path / "store")
    seen = []
    real = MemoryRuntime.examples_for
    MemoryRuntime.examples_for = lambda self, t, c: seen.append((t, c)) or real(self, t, c)  # type: ignore[method-assign]
    try:
        table, prompt, context = prepare_request(req, upload, rt)
    finally:
        MemoryRuntime.examples_for = real  # type: ignore[method-assign]
    assert len(seen) == 1 and seen[0][0] is table and seen[0][1] is context
    assert prompt.user_text.endswith(
        render_peak_table(table, context) + "\n\n" + render_context_block(context, 3)
    )


# --------------------------------------------------------------------------- the route


@pytest.fixture
def fresh_memory_cache():
    get_settings.cache_clear()
    analyze_routes._memory.cache_clear()
    yield
    get_settings.cache_clear()
    analyze_routes._memory.cache_clear()
    app.dependency_overrides.clear()


def post(api):
    app.dependency_overrides[get_llm_client] = lambda: api.client()
    return TestClient(app).post("/v1/analyze", json=ASPIRIN_DMSO)


def test_get_memory_is_none_when_unset(fresh_memory_cache, monkeypatch):
    monkeypatch.delenv("NMR_AGENT_MEMORY_DIR", raising=False)
    assert get_memory() is None


def test_get_memory_loads_a_valid_store_once(fresh_memory_cache, monkeypatch, tmp_path):
    build_store(tmp_path, SETTINGS, three_cases(SETTINGS))
    monkeypatch.setenv("NMR_AGENT_MEMORY_DIR", str(tmp_path))
    rt = get_memory()
    assert rt is not None and rt.mode == "retrieved" and len(rt.store.cases) == 3 and get_memory() is rt
    api = FakeAPI([message(hypothesis_json())])
    body = post(api).json()
    assert body["versions"]["memory_version"] == rt.memory_version
    assert body["memory_cases"] and api.requests[0]["body"]["messages"][0]["content"].startswith("Worked")


def test_get_memory_is_lenient_about_bad_cases(fresh_memory_cache, monkeypatch, tmp_path):
    cases = three_cases(SETTINGS)
    cases[0]["agent_outcome"]["prompt_version"] = "old"
    build_store(tmp_path, SETTINGS, cases)
    monkeypatch.setenv("NMR_AGENT_MEMORY_DIR", str(tmp_path))
    rt = get_memory()
    assert rt is not None and [c.case_id for c in rt.store.cases] == ["mem-000002", "mem-000003"]


def break_missing_dir(root: Path) -> Path:
    return root / "nope"


def break_no_manifest(root: Path) -> Path:
    root.mkdir()
    write_params(root)
    return root


def break_bad_manifest(root: Path) -> Path:
    build_store(root, SETTINGS, three_cases(SETTINGS))
    (root / "manifest.json").write_text("{not json")
    return root


def break_params_hash(root: Path) -> Path:
    build_store(root, SETTINGS, three_cases(SETTINGS))
    write_params(root, sha="0" * 64)
    return root


def break_no_params(root: Path) -> Path:
    build_store(root, SETTINGS, three_cases(SETTINGS))
    (root / "retrieval_params.json").unlink()
    return root


def break_cases_dir_missing(root: Path) -> Path:
    build_store(root, SETTINGS, three_cases(SETTINGS))
    import shutil

    shutil.rmtree(root / "cases")
    return root


@pytest.mark.parametrize(
    "breaker",
    [break_missing_dir, break_no_manifest, break_bad_manifest, break_params_hash, break_no_params,
     break_cases_dir_missing],
)  # fmt: skip
def test_a_broken_memory_directory_runs_the_service_without_memory(
    fresh_memory_cache, monkeypatch, tmp_path, breaker
):
    monkeypatch.setenv("NMR_AGENT_MEMORY_DIR", str(breaker(tmp_path / "mem")))
    assert get_memory() is None
    r = post(FakeAPI([message(hypothesis_json())]))
    assert r.status_code == 200
    assert r.json()["memory_cases"] == [] and r.json()["versions"]["memory_version"] == "none"


def test_the_dependency_can_be_overridden(fresh_memory_cache, tmp_path):
    rt = runtime(tmp_path)
    app.dependency_overrides[get_memory] = lambda: rt
    assert post(FakeAPI([message(hypothesis_json())])).json()["memory_cases"]


def test_process_route_ignores_memory(fresh_memory_cache, tmp_path):
    app.dependency_overrides[get_memory] = lambda: runtime(tmp_path)
    r = TestClient(app).post("/v1/process", json=ASPIRIN_DMSO)
    assert r.status_code == 200 and "memory_cases" not in r.json()


def test_case_dict_helper_is_consistent():  # keeps the imported helpers honest
    assert case_dict(5, SETTINGS, ETHYL_ACETATE)["case_id"] == "mem-000005"
    assert ETHYL_BENZOATE["formula"] == "C9H10O2" and SampleContext().molecular_formula is None
