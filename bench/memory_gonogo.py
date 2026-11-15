"""Experience-memory offline go/no-go (spec §7.0, phase M4). No model calls, no network.

    PYTHONPATH=. uv run python bench/memory_gonogo.py samples   # once: the 2 x 5 memory-sized samples
    PYTHONPATH=. uv run python bench/memory_gonogo.py tune      # once: weights on the tuning half, frozen
    PYTHONPATH=. uv run python bench/memory_gonogo.py check     # once: the check half, the verdict

Each command refuses to overwrite its outputs (bench/memory/gonogo_*.json, retrieval_params.json,
gonogo_report.txt): the measurement is run once. `tune` and `check` need the NMRShiftDB2 export
(bench/data/, untracked) to rebuild each molecule's spectra, and a clean working tree for the code
(each output records the commit of the code that produced it).

Cases are built in memory (placeholder provenance and approval, agent_outcome.status "solved"); nothing
is written to a memory/ directory. Because every status is "solved", the spec's missed-first tie-break
is inactive here and the selection is simply highest score first, ties on case_id.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import multiprocessing as mp
import random
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs

from src.memory.case import (
    AgentOutcome,
    Approval,
    HPeak,
    MemoryCase,
    Provenance,
)
from src.memory.features import (
    Features,
    case_features,
    dou_closeness,
    element_closeness,
    peak_closeness,
    query_features,
    solvent_match,
)
from src.memory.leakage import profile, tanimoto
from src.memory.retrieval import RetrievalParams, retrieve, select_random
from src.memory.store import MemoryStore, chemistry_fields
from src.schemas import Nucleus
from src.verify.smarts import SPECS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "memory"
FILES = {
    "order": OUT / "memory_order.json",
    "split": OUT / "split.json",
    "samples": OUT / "gonogo_samples.json",
    "params": OUT / "retrieval_params.json",
    "pairs": OUT / "gonogo_pairs.json",
    "report": OUT / "gonogo_report.txt",
}

# --- fixed by the task and spec §7.0; never changed after a result
HALVES = ("tuning", "check")
SAMPLE_SEEDS = {"tuning": tuple(range(20261117, 20261122)), "check": tuple(range(20261122, 20261127))}
SAMPLE_QUOTA = {">20": 80, "11-20": 80, "<=10": 40}  # drawn in this order; 200 molecules
K = 3
GRID_STEPS = 10  # weights are multiples of 1/GRID_STEPS summing to 1
FIXED = {
    "solvent_bonus": 0.03,
    "c13_tolerance_ppm": 2.0,
    "h1_tolerance_ppm": 0.15,
    "tie_epsilon": 0.02,
    "k": K,
    "min_similarity": 0.0,
}
DEFAULT_WEIGHTS = (0.35, 0.30, 0.20, 0.15)
WEIGHT_NAMES = ("weight_c13", "weight_h1", "weight_elements", "weight_dou")
GO_SIMILARITY, GO_COVERAGE, GO_WIN_RATE = 0.10, 0.15, 0.60
TIE_DIGITS = 12  # objective values equal to this many decimals count as tied
PLACEHOLDER = "PLACEHOLDER-m4-in-memory-case"
CODE_PATHS = ("src", "bench/benchmark.py", "bench/memory.py", "bench/memory_gonogo.py")


# --------------------------------------------------------------------------- small helpers


def id_order(mol_id: str) -> tuple[int, str]:
    """Numeric order for digit-string ids (same rule as bench/memory.py)."""
    return (len(mol_id), mol_id)


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def code_is_dirty() -> bool:
    out = subprocess.run(
        ["git", "status", "--porcelain", "--", *CODE_PATHS, "tests"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return bool(out.strip())


def refuse_if_exists(*names: str) -> None:
    existing = [FILES[n] for n in names if FILES[n].exists()]
    if existing:
        sys.exit(
            f"refusing to run: {', '.join(str(p.relative_to(ROOT)) for p in existing)} exist(s); "
            "the measurement is run once"
        )


def require_clean_code() -> None:
    if code_is_dirty():
        sys.exit("refusing to run: uncommitted changes under src/, bench/ or tests/; commit the code first")


def dump_rows(path: Path, header: dict, list_key: str, rows: list) -> None:
    """Header keys first, then one row per line."""
    lines = [f" {json.dumps(k)}: {json.dumps(v)}," for k, v in header.items()]
    body = ",\n".join(f"  {json.dumps(r)}" for r in rows)
    path.write_text(
        "{\n" + "\n".join(lines) + f"\n {json.dumps(list_key)}: [\n{body}\n ]\n}}\n", encoding="utf-8"
    )


def read_selection() -> tuple[dict[str, dict], dict[str, str]]:
    """(memory_order entries by id, half by id)."""
    entries = json.loads(FILES["order"].read_text())["entries"]
    halves = {a["id"]: a["half"] for a in json.loads(FILES["split"].read_text())["assignments"]}
    by_id = {e["id"]: e for e in entries}
    if set(by_id) != set(halves):
        raise SystemExit("memory_order.json and split.json list different molecules")
    return by_id, halves


# --------------------------------------------------------------------------- samples


def draw_sample(ids_by_stratum: dict[str, Sequence[str]], seed: int) -> list[str]:
    """One stratified sample: SAMPLE_QUOTA[stratum] ids per stratum, drawn with random.Random(seed).

    Strata are drawn in the quota's order (>20, 11-20, <=10) from each stratum's ids sorted by id; the
    result is returned sorted by id.
    """
    rng = random.Random(seed)
    chosen: list[str] = []
    for stratum, quota in SAMPLE_QUOTA.items():
        chosen += rng.sample(sorted(ids_by_stratum[stratum], key=id_order), quota)
    return sorted(chosen, key=id_order)


def draw_all_samples(by_id: dict[str, dict], halves: dict[str, str]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for half in HALVES:
        strata: dict[str, list[str]] = {s: [] for s in SAMPLE_QUOTA}
        for mol_id, h in halves.items():
            if h == half:
                strata[by_id[mol_id]["stratum"]].append(mol_id)
        out[half] = [
            {"index": i, "seed": seed, "ids": draw_sample(strata, seed)}
            for i, seed in enumerate(SAMPLE_SEEDS[half])
        ]
    return out


def cmd_samples() -> None:
    refuse_if_exists("samples")
    require_clean_code()
    by_id, halves = read_selection()
    drawn = draw_all_samples(by_id, halves)
    header = {
        "description": "5 memory-sized samples (80 >20, 80 11-20, 40 <=10 heavy atoms) per half, drawn once",
        "sample_index_base": 0,
        "git_commit": git_commit(),
    }
    rows = [{"half": h, **s} for h in HALVES for s in drawn[h]]
    dump_rows(FILES["samples"], header, "samples", rows)
    print(f"wrote {FILES['samples'].relative_to(ROOT)}: {len(rows)} samples")


def read_samples() -> dict[str, list[list[str]]]:
    rows = json.loads(FILES["samples"].read_text())["samples"]
    out: dict[str, list[list[str]]] = {h: [] for h in HALVES}
    for r in sorted(rows, key=lambda r: (r["half"], r["index"])):
        out[r["half"]].append(r["ids"])
    return out


# --------------------------------------------------------------------------- molecules and cases


@dataclass
class Molecule:
    id: str
    pool: str
    smiles: str
    stratum: str
    scaffold: str
    half: str
    case_id: str
    features: Features
    case: MemoryCase
    groups: frozenset[str]
    fingerprint: object = field(repr=False, default=None)


_PATTERNS: dict[str, list] = {}


def fragment_groups(smiles: str) -> frozenset[str]:
    """Groups of src.verify.smarts.SPECS with at least one SMARTS pattern matching the molecule."""
    if not _PATTERNS:
        for group, spec in SPECS.items():
            _PATTERNS[group.value] = [Chem.MolFromSmarts(s) for s in spec.smarts]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES {smiles!r}")
    return frozenset(g for g, pats in _PATTERNS.items() if any(mol.HasSubstructMatch(p) for p in pats))


def fragment_coverage(truth: frozenset[str], returned: Sequence[frozenset[str]]) -> float | None:
    """|truth ∩ union(returned)| / |truth|; None when the truth has no group."""
    if not truth:
        return None
    union: set[str] = set().union(*returned) if returned else set()
    return len(truth & union) / len(truth)


def best_similarity(truth_smiles: str, returned_smiles: Sequence[str]) -> float:
    """Highest Morgan (r2, 2048) Tanimoto between the truth and a returned case; 0 if none returned."""
    best = 0.0
    for s in returned_smiles:
        t = tanimoto(truth_smiles, s)
        if t is not None and t > best:
            best = t
    return best


def make_case(
    case_id: str, entry: dict, features: Features, h_peaks: list[HPeak], formula: str
) -> MemoryCase:
    chem = chemistry_fields(entry["smiles"])
    assert chem is not None, entry["id"]
    assert features.elements is not None and features.dou is not None
    today = dt.date(2026, 11, 14)
    return MemoryCase(
        case_id=case_id,
        schema_version="1",
        provenance=Provenance(
            source="nmrshiftdb2",
            source_record_id=entry["id"],
            spectrum_ids={},
            truth_basis="database-assigned structure (M4 in-memory case, never stored)",
            extracted_by=PLACEHOLDER,
            proposed_on=today,
        ),
        approval=Approval(approved_by=PLACEHOLDER, date_added=today, note=None),
        formula=formula,
        dou=features.dou,
        carbon_count=features.elements.get("C", 0),
        heavy_atoms=int(entry["heavy_atoms"]),
        solvent=features.solvent,
        h_peaks=h_peaks,
        c_peaks=list(features.c_shifts),
        smiles=entry["smiles"],
        inchikey=str(chem["inchikey"]),
        murcko_scaffold=str(chem["murcko_scaffold"]),
        generic_scaffold=str(chem["generic_scaffold"]),
        assignments=None,
        agent_outcome=AgentOutcome(
            collected_on=today,
            model_id="none",
            prompt_version="none",
            settings_hash="none",
            status="solved",
            proposed_smiles=[entry["smiles"]],
        ),
    )


def load_molecules(by_id: dict[str, dict], halves: dict[str, str]) -> dict[str, Molecule]:
    """Every molecule of memory_order.json with its features, in-memory case and fragment groups.

    Spectra come from the record with the same id in bench.benchmark.pool() (named) or
    pool(unreported=True); features are built the way the service builds them.
    """
    from bench import benchmark
    from src.prompting import prepare_request

    benchmark.SD = str(ROOT / benchmark.SD)
    records = {
        "named": {r["id"]: r for r in benchmark.pool()},
        "unreported": {r["id"]: r for r in benchmark.pool(unreported=True)},
    }
    case_ids = {mid: f"mem-{e['position']:06d}" for mid, e in by_id.items()}
    out: dict[str, Molecule] = {}
    for mol_id in sorted(by_id, key=id_order):
        e = by_id[mol_id]
        rec = records[e["pool"]].get(mol_id)
        if rec is None or rec["smiles"] != e["smiles"]:
            raise SystemExit(f"record {mol_id} not found in the {e['pool']} pool, or its structure differs")
        req, _ = benchmark.request_for(rec)
        table, _, context = prepare_request(req)
        feats = query_features(table, context)
        h_peaks = [HPeak(shift_ppm=p.shift_ppm, integral=p.integral) for p in table.analyte_peaks(Nucleus.H1)]
        case = make_case(case_ids[mol_id], e, feats, h_peaks, context.molecular_formula or "")
        if case_features(case) != feats:
            raise SystemExit(f"case features differ from query features for {mol_id}")
        if case.murcko_scaffold != e["scaffold"]:
            raise SystemExit(f"scaffold of {mol_id} differs from memory_order.json")
        prof = profile(e["smiles"])
        assert prof is not None
        out[mol_id] = Molecule(
            id=mol_id,
            pool=e["pool"],
            smiles=e["smiles"],
            stratum=e["stratum"],
            scaffold=e["scaffold"],
            half=halves[mol_id],
            case_id=case_ids[mol_id],
            features=feats,
            case=case,
            groups=fragment_groups(e["smiles"]),
            fingerprint=prof.fingerprint,
        )
    return out


def build_pool(sample_cases: Sequence[MemoryCase], query: MemoryCase) -> MemoryStore:
    """The sample minus the query itself and minus every case with the query's Murcko scaffold."""
    kept = sorted(
        (
            c
            for c in sample_cases
            if c.case_id != query.case_id and c.murcko_scaffold != query.murcko_scaffold
        ),
        key=lambda c: c.case_id,
    )
    return MemoryStore(cases=tuple(kept), memory_version="m4-in-memory")


# --------------------------------------------------------------------------- pairs


def params_from(weights: Sequence[float]) -> RetrievalParams:
    return RetrievalParams(**dict(zip(WEIGHT_NAMES, weights, strict=True)), **FIXED)


def pair_seed(query_id: str, sample_index: int) -> int:
    return int(hashlib.sha256(f"{query_id}:{sample_index}".encode()).hexdigest()[:8], 16)


def evaluate_pair(
    q: Molecule,
    sample_index: int,
    sample_cases: Sequence[MemoryCase],
    by_case: dict[str, Molecule],
    params: RetrievalParams,
) -> dict:
    pool = build_pool(sample_cases, q.case)
    ret = retrieve(q.features, pool, params)
    rnd = select_random(pool, pair_seed(q.id, sample_index), params)
    ret_mols = [by_case[r.case.case_id] for r in ret]
    rnd_mols = [by_case[r.case.case_id] for r in rnd]
    best_ret = best_similarity(q.smiles, [m.smiles for m in ret_mols])
    best_rnd = best_similarity(q.smiles, [m.smiles for m in rnd_mols])
    return {
        "query_id": q.id,
        "stratum": q.stratum,
        "sample_index": sample_index,
        "retrieved": [
            {"molecule_id": m.id, "case_id": m.case_id, "score": r.similarity}
            for r, m in zip(ret, ret_mols, strict=True)
        ],
        "random": [{"molecule_id": m.id, "case_id": m.case_id} for m in rnd_mols],
        "best_similarity_retrieved": best_ret,
        "best_similarity_random": best_rnd,
        "coverage_retrieved": fragment_coverage(q.groups, [m.groups for m in ret_mols]),
        "coverage_random": fragment_coverage(q.groups, [m.groups for m in rnd_mols]),
        "win": best_ret > best_rnd,
    }


_STATE: dict = {}


def _pairs_for_queries(query_ids: list[str]) -> list[dict]:
    mols: dict[str, Molecule] = _STATE["mols"]
    by_case: dict[str, Molecule] = _STATE["by_case"]
    samples: list[list[MemoryCase]] = _STATE["sample_cases"]
    params: RetrievalParams = _STATE["params"]
    return [
        evaluate_pair(mols[qid], si, cases, by_case, params)
        for qid in query_ids
        for si, cases in enumerate(samples)
    ]


def parallel(fn: Callable, chunks: list, workers: int) -> list:
    if workers <= 1:
        return [fn(c) for c in chunks]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        return list(ex.map(fn, chunks))


def chunked(items: Sequence, n: int) -> list[list]:
    size = max(1, -(-len(items) // (n * 4)))
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def run_half(
    mols: dict[str, Molecule], samples: list[list[str]], half: str, params: RetrievalParams, workers: int
) -> list[dict]:
    """Every query of ``half`` against each of its samples, with the real retrieve() and select_random()."""
    _STATE["mols"] = mols
    _STATE["by_case"] = {m.case_id: m for m in mols.values()}
    _STATE["sample_cases"] = [[mols[i].case for i in ids] for ids in samples]
    _STATE["params"] = params
    queries = sorted((m.id for m in mols.values() if m.half == half), key=id_order)
    parts = parallel(_pairs_for_queries, chunked(queries, workers), workers)
    return [p for part in parts for p in part]


# --------------------------------------------------------------------------- summary and verdict


def _mean(xs: Sequence[float]) -> float:
    return float(np.mean(xs)) if len(xs) else float("nan")


def summarize(pairs: Sequence[dict]) -> dict:
    cov = [p for p in pairs if p["coverage_retrieved"] is not None]
    mean_sim_ret = _mean([p["best_similarity_retrieved"] for p in pairs])
    mean_sim_rnd = _mean([p["best_similarity_random"] for p in pairs])
    mean_cov_ret = _mean([p["coverage_retrieved"] for p in cov])
    mean_cov_rnd = _mean([p["coverage_random"] for p in cov])
    return {
        "pairs": len(pairs),
        "queries": len({p["query_id"] for p in pairs}),
        "sim_retrieved": mean_sim_ret,
        "sim_random": mean_sim_rnd,
        "sim_diff": mean_sim_ret - mean_sim_rnd,
        "cov_retrieved": mean_cov_ret,
        "cov_random": mean_cov_rnd,
        "cov_diff": mean_cov_ret - mean_cov_rnd,
        "cov_excluded": len(pairs) - len(cov),
        "win_rate": sum(p["win"] for p in pairs) / len(pairs) if pairs else float("nan"),
    }


def verdict(sim_diff: float, cov_diff: float, win_rate: float) -> tuple[bool, list[str]]:
    """GO iff all three thresholds are met (>=); spec §7.0. Returns (go, lines naming each test)."""
    tests = [
        ("mean best similarity, retrieved - random", sim_diff, GO_SIMILARITY),
        ("mean fragment coverage, retrieved - random", cov_diff, GO_COVERAGE),
        ("win rate (retrieved > random)", win_rate, GO_WIN_RATE),
    ]
    lines = []
    go = True
    for name, value, threshold in tests:
        ok = bool(value >= threshold)  # NaN compares false: no data is never a GO
        go &= ok
        lines.append(f"{name}: {value:.4f} vs threshold >= {threshold:.2f}: {'PASS' if ok else 'FAIL'}")
    return go, lines


def _block(title: str, s: dict) -> list[str]:
    return [
        f"{title}",
        f"  pairs {s['pairs']}, queries {s['queries']}",
        f"  mean best similarity: retrieved {s['sim_retrieved']:.4f}, random {s['sim_random']:.4f}, "
        f"difference {s['sim_diff']:+.4f}",
        f"  mean fragment coverage: retrieved {s['cov_retrieved']:.4f}, random {s['cov_random']:.4f}, "
        f"difference {s['cov_diff']:+.4f} ({s['cov_excluded']} pairs excluded: truth has no group)",
        f"  win rate: {s['win_rate']:.4f}",
    ]


def report_text(check: Sequence[dict], tuning: Sequence[dict], params: dict, commit: str) -> str:
    s = summarize(check)
    go, lines = verdict(s["sim_diff"], s["cov_diff"], s["win_rate"])
    out = [
        f"experience-memory offline go/no-go (code commit {commit})",
        f"frozen parameters sha256 {params['sha256']}",
        f"weights (13C, 1H, elements, DoU): {tuple(params['parameters'][n] for n in WEIGHT_NAMES)}",
        "",
        *_block("== check half (the decision)", s),
        "",
        "== check half by the query's heavy-atom stratum",
    ]
    for stratum in SAMPLE_QUOTA:
        part = [p for p in check if p["stratum"] == stratum]
        if part:
            out += _block(f"stratum {stratum}", summarize(part))
    out += [
        "",
        *_block("== tuning half with the frozen weights (in-sample, for comparison only)", summarize(tuning)),
    ]
    out += ["", "== thresholds (spec 7.0), check half", *lines, ""]
    out.append(f"VERDICT: {'GO' if go else 'NO-GO'}")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- tune: fast path


def grid_weights() -> list[tuple[float, float, float, float]]:
    """Every (13C, 1H, elements, DoU) of multiples of 0.1, each >= 0, summing to 1.0."""
    out = []
    for a, b, c in itertools.product(range(GRID_STEPS + 1), repeat=3):
        d = GRID_STEPS - a - b - c
        if d >= 0:
            out.append((a / GRID_STEPS, b / GRID_STEPS, c / GRID_STEPS, d / GRID_STEPS))
    return out


def _closeness_rows(q_ids: list[str]) -> list[np.ndarray]:
    mols: dict[str, Molecule] = _STATE["mols"]
    col_feats: list[Features] = _STATE["col_feats"]
    p = _STATE["base_params"]
    rows = []
    for qid in q_ids:
        qf = mols[qid].features
        row = np.zeros((5, len(col_feats)))
        for j, cf in enumerate(col_feats):
            if qf.c_shifts:
                row[0, j] = peak_closeness(qf.c_shifts, cf.c_shifts, p.c13_tolerance_ppm)
            if qf.h_shifts:
                row[1, j] = peak_closeness(qf.h_shifts, cf.h_shifts, p.h1_tolerance_ppm)
            if qf.elements is not None and cf.elements is not None:
                row[2, j] = element_closeness(qf.elements, cf.elements)
            if qf.dou is not None and cf.dou is not None:
                row[3, j] = dou_closeness(qf.dou, cf.dou)
            row[4, j] = 1.0 if solvent_match(qf.solvent, cf.solvent) else 0.0
        rows.append(row)
    return rows


class FastScorer:
    """Per-feature closeness precomputed once per (query, case); any weights are then a few numpy ops.

    The arithmetic mirrors ``retrieval.score_case`` term by term (same summation order, unavailable
    features contribute exact zeros), so the selection equals ``retrieve()`` (tested).
    """

    def __init__(
        self, mols: dict[str, Molecule], query_ids: list[str], sample_ids: list[list[str]], workers: int = 1
    ):
        self.mols = mols
        self.query_ids = query_ids
        self.cols = sorted({mols[i].case_id: i for ids in sample_ids for i in ids}.items())  # by case_id
        self.col_mol = [mols[i] for _, i in self.cols]
        self.col_of = {cid: j for j, (cid, _) in enumerate(self.cols)}
        self.samples = [np.array([self.col_of[mols[i].case_id] for i in ids]) for ids in sample_ids]
        for s in self.samples:
            s.sort()  # column order is case_id order
        _STATE["mols"] = mols
        _STATE["col_feats"] = [case_features(m.case) for m in self.col_mol]
        _STATE["base_params"] = params_from(DEFAULT_WEIGHTS)
        parts = parallel(_closeness_rows, chunked(query_ids, workers), workers)
        self.closeness = np.stack([r for part in parts for r in part])  # (Q, 5, N): 4 features, solvent match
        q_feats = [mols[i].features for i in query_ids]
        self.avail = np.array(
            [[bool(f.c_shifts), bool(f.h_shifts), f.elements is not None, f.dou is not None] for f in q_feats]
        )
        fps = [m.fingerprint for m in self.col_mol]
        self.truth = np.array(
            [DataStructs.BulkTanimotoSimilarity(mols[i].fingerprint, fps) for i in query_ids]
        )
        self.col_scaffold = np.array([m.scaffold for m in self.col_mol], dtype=object)
        self.col_cid = np.array([m.case_id for m in self.col_mol], dtype=object)

    def sims(self, qi: int, cols: np.ndarray, weights: np.ndarray, bonus: float) -> np.ndarray:
        """(G, len(cols)) similarities for weight rows ``weights`` (G, 4)."""
        c = self.closeness[qi][:, cols]  # (5, n)
        w = weights * self.avail[qi][None, :]  # unavailable features get weight 0.0
        num = np.zeros((len(weights), len(cols)))
        total = np.zeros((len(weights), 1))
        for i in range(4):
            num = num + w[:, i : i + 1] * c[i][None, :]
            total = total + w[:, i : i + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            base = np.where(total > 0, num / total, 0.0)
        s = np.minimum(1.0, base + bonus * c[4][None, :])
        if not self.avail[qi].any():
            s = np.zeros_like(s)
        return s

    def select(self, qi: int, si: int, weights: np.ndarray, bonus: float, k: int = K) -> list[list[int]]:
        """Chosen column indices (best first) per weight row, for query ``qi`` against sample ``si``."""
        q = self.mols[self.query_ids[qi]]
        cols = self.samples[si]
        valid = (self.col_cid[cols] != q.case_id) & (self.col_scaffold[cols] != q.scaffold)
        cols = cols[valid]
        sims = self.sims(qi, cols, weights, bonus)
        order = np.argsort(-sims, axis=1, kind="stable")  # ties: lower case_id first (columns are sorted)
        out = []
        for g in range(len(weights)):
            seen: set[str] = set()
            chosen: list[int] = []
            for pos in order[g]:
                j = int(cols[pos])
                sc = self.col_scaffold[j]
                if sc in seen:
                    continue
                seen.add(sc)
                chosen.append(j)
                if len(chosen) == k:
                    break
            out.append(chosen)
        return out


def _objective_chunk(qis: list[int]) -> np.ndarray:
    fs: FastScorer = _STATE["fast"]
    weights: np.ndarray = _STATE["weights"]
    bonus: float = _STATE["bonus"]
    total = np.zeros(len(weights))
    for qi in qis:
        for si in range(len(fs.samples)):
            for g, chosen in enumerate(fs.select(qi, si, weights, bonus)):
                total[g] += max((fs.truth[qi, j] for j in chosen), default=0.0)
    return total


def objective(fs: FastScorer, weights: np.ndarray, bonus: float, workers: int) -> np.ndarray:
    """Mean best similarity of the retrieved sets over all (query, sample) pairs, per weight row."""
    _STATE.update(fast=fs, weights=weights, bonus=bonus)
    parts = parallel(_objective_chunk, chunked(list(range(len(fs.query_ids))), workers), workers)
    return np.sum(parts, axis=0) / (len(fs.query_ids) * len(fs.samples))


def pick_best(grid: Sequence[tuple[float, ...]], values: Sequence[float]) -> list[int]:
    """Grid indices best first: highest objective (ties at TIE_DIGITS decimals), then the smallest sum of
    absolute differences from the provisional defaults, then lexicographic weights."""
    return sorted(
        range(len(grid)),
        key=lambda i: (
            -round(float(values[i]), TIE_DIGITS),
            round(sum(abs(a - b) for a, b in zip(grid[i], DEFAULT_WEIGHTS, strict=True)), 9),
            grid[i],
        ),
    )


def params_hash(parameters: dict) -> str:
    return hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()


def cmd_tune(workers: int) -> None:
    refuse_if_exists("params")
    require_clean_code()
    if not FILES["samples"].exists():
        sys.exit("run `samples` first")
    t0 = time.time()
    by_id, halves = read_selection()
    samples = read_samples()["tuning"]
    mols = load_molecules(by_id, halves)
    t1 = time.time()
    queries = sorted((m for m in mols if mols[m].half == "tuning"), key=id_order)
    fs = FastScorer(mols, queries, samples, workers)
    t2 = time.time()
    grid = grid_weights()
    bonus = FIXED["solvent_bonus"]
    values = objective(fs, np.array(grid), bonus, workers)
    default_value = float(objective(fs, np.array([DEFAULT_WEIGHTS]), bonus, 1)[0])
    ranked = pick_best(grid, values)
    best = grid[ranked[0]]
    parameters = {**dict(zip(WEIGHT_NAMES, best, strict=True)), **FIXED}
    doc = {
        "description": "retrieval parameters tuned on the tuning half (memory-sized pools) and frozen",
        "git_commit": git_commit(),
        "parameters": parameters,
        "sha256": params_hash(parameters),
        "objective": "mean best Morgan Tanimoto of the retrieved sets over all tuning (query, sample) pairs",
        "objective_value": float(values[ranked[0]]),
        "pairs": len(queries) * len(samples),
        "grid_points": len(grid),
        "top10": [
            {"weights": dict(zip(WEIGHT_NAMES, grid[i], strict=True)), "objective": float(values[i])}
            for i in ranked[:10]
        ],
        "provisional_default": {
            "weights": dict(zip(WEIGHT_NAMES, DEFAULT_WEIGHTS, strict=True)),
            "objective": default_value,
        },
        "run_seconds": {
            "load_molecules": round(t1 - t0, 1),
            "closeness": round(t2 - t1, 1),
            "grid": round(time.time() - t2, 1),
        },
    }
    FILES["params"].write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(
        f"wrote {FILES['params'].relative_to(ROOT)}: weights {best}, "
        f"objective {doc['objective_value']:.4f}, default {default_value:.4f}"
    )


# --------------------------------------------------------------------------- check


def cmd_check(workers: int) -> None:
    if not FILES["params"].exists():
        sys.exit("refusing to run: bench/memory/retrieval_params.json is missing; run `tune` first")
    refuse_if_exists("pairs", "report")
    require_clean_code()
    if not FILES["samples"].exists():
        sys.exit("run `samples` first")
    doc = json.loads(FILES["params"].read_text())
    if params_hash(doc["parameters"]) != doc["sha256"]:
        sys.exit("retrieval_params.json does not match its recorded SHA-256")
    params = RetrievalParams(**doc["parameters"])
    by_id, halves = read_selection()
    samples = read_samples()
    t0 = time.time()
    mols = load_molecules(by_id, halves)
    check_pairs = run_half(mols, samples["check"], "check", params, workers)
    t1 = time.time()
    tuning_pairs = run_half(mols, samples["tuning"], "tuning", params, workers)
    commit = git_commit()
    header = {
        "description": "check half: every query against each of its 5 samples; frozen parameters",
        "git_commit": commit,
        "params_sha256": doc["sha256"],
        "sample_index_base": 0,
        "random_seed": "int(sha256(f'{query_id}:{sample_index}').hexdigest()[:8], 16)",
    }
    dump_rows(FILES["pairs"], header, "pairs", check_pairs)
    FILES["report"].write_text(report_text(check_pairs, tuning_pairs, doc, commit), encoding="utf-8")
    print(FILES["report"].read_text())
    print(f"check half {t1 - t0:.0f} s, tuning half {time.time() - t1:.0f} s", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("samples", "tune", "check"))
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    args = ap.parse_args()
    if args.command == "samples":
        cmd_samples()
    elif args.command == "tune":
        cmd_tune(args.workers)
    else:
        cmd_check(args.workers)


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
