# ruff: noqa: E501
"""Experience-memory selections and leakage checks (spec §7.1, §7.2, §9).

    PYTHONPATH=. uv run python bench/memory.py select           # once: writes the four files in bench/memory/
    PYTHONPATH=. uv run python bench/memory.py check [--store <dir>]

Phase M6 (spec §6, §7.2a, §15.4):

    select-large                 write bench/memory/memory_large.json (once)
    setup-store                  create the empty memory/ store skeleton
    collect [--ids ..] [--cap 35] [--dry-run]   THE PAID STEP: run the service on each memory molecule, memory off
    propose [--ids ..]           pending cases + review sheets from the collected outcomes (never approves)
    approve <case_id>.. --by "<name>"           a person only: asks for the typed word APPROVE
    reject <case_id>.. --reason "<text>"

Phase M7 (spec §15.6-15.8, §15.11):

    eval [--arms ..] [--ids ..] [--cap 40] [--dry-run]   THE PAID STEP: M0-a, M0-b, M200L, R200L on the 50 large MEM-EVAL molecules
    compare                      the §15.7 measures and the §15.8 verdict from bench/memory/eval/ (no model call)

`select` needs the NMRShiftDB2 export (bench/data/, untracked) and refuses to run if any output exists:
the selections are written once, before any model call, and never rewritten. `check` reads only the
committed files (and a store, if given) and exits non-zero on any violation.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import hashlib
import json
import random
import re
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors

from src.memory import load_store
from src.memory.leakage import (
    BENZENE_SCAFFOLD,
    SIMILARITY_LIMIT,
    rule1_structure_key,
    rule2_scaffold,
    rule3_eval_scaffolds,
    rule4_similarity,
    rule5_proposed,
)
from src.memory.store import chemistry_fields
from src.schemas import Nucleus

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "bench" / "memory"
FILES = {
    "mem_eval": OUT / "mem_eval.json",
    "memory_order": OUT / "memory_order.json",
    "split": OUT / "split.json",
    "report": OUT / "selection_report.txt",
}
OLD_SELECTIONS = [
    ROOT / "bench" / d / "selection.json" for d in ("results", "results_main", "results_unreported")
]

EVAL_SEED, ORDER_SEED, SPLIT_SEED = 20261114, 20261115, 20261116
MIN_EVAL_HEAVY = 11
BIN_SIZE = 20  # upper edge of the 11-20 bin
PER_BIN = 50
MAX_PER_SCAFFOLD = 2
MEMORY_SIZE = 200
PREFIXES = (25, 50, 100, 200)
PATTERN = (">20", "11-20", ">20", "11-20", "<=10")  # every 5 entries: 40% / 40% / 20%
EVAL_FIELDS = ("id", "smiles", "inchikey", "formula", "solvent", "h_spec", "c_spec")


# --------------------------------------------------------------------------- shared helpers


def key_of(inchikey: str) -> str:
    return inchikey.split("-")[0]


def stratum(heavy: int) -> str:
    return ">20" if heavy > BIN_SIZE else "11-20" if heavy >= MIN_EVAL_HEAVY else "<=10"


def id_order(r: dict) -> tuple[int, str]:
    return (len(r["id"]), r["id"])  # numeric order for digit strings, deterministic for anything else


def load_old_sets() -> list[dict]:
    return [r for f in OLD_SELECTIONS for r in json.loads(f.read_text())["records"]]


def git_state() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    code = ["src", "bench/memory.py", "bench/benchmark.py"]
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_dirty_code": bool(run("status", "--porcelain", "--", *code)),
    }


def dump_json(path: Path, header: dict, list_key: str, rows: list) -> None:
    """Header keys first, then one row per line (the files are large lists; this keeps diffs readable)."""
    lines = [f" {json.dumps(k)}: {json.dumps(v)}," for k, v in header.items()]
    body = ",\n".join(f"  {json.dumps(r)}" for r in rows)
    path.write_text(
        "{\n" + "\n".join(lines) + f"\n {json.dumps(list_key)}: [\n{body}\n ]\n}}\n", encoding="utf-8"
    )


# --------------------------------------------------------------------------- select


def build_pool(records: list[dict]) -> tuple[list[dict], dict]:
    """Drop incomplete 13C lists, then keep the first record per structure key."""
    from bench import benchmark

    complete = [r for r in records if not benchmark.c13_incomplete(r)]
    seen: set[str] = set()
    kept = []
    for r in complete:
        k = key_of(r["inchikey"])
        if k not in seen:
            seen.add(k)
            kept.append(r)
    return kept, {
        "records": len(records),
        "removed_13c_incomplete": len(records) - len(complete),
        "removed_duplicate_structure": len(complete) - len(kept),
        "kept": len(kept),
    }


def annotate(r: dict, pool_name: str) -> dict:
    chem = chemistry_fields(r["smiles"])
    assert chem is not None, r["id"]
    mol = Chem.MolFromSmiles(r["smiles"])
    return {
        **{f: r[f] for f in EVAL_FIELDS},
        "pool": pool_name,
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "scaffold": chem["murcko_scaffold"],
        "generic_scaffold": chem["generic_scaffold"],
    }


def draw_mem_eval(named: list[dict], old_keys: set[str], old_scaffolds: set[str]) -> tuple[list[dict], dict]:
    counts = collections.OrderedDict()
    pool = list(named)
    counts["named pool"] = len(pool)
    rules = [
        (f"fewer than {MIN_EVAL_HEAVY} heavy atoms", lambda r: r["heavy_atoms"] < MIN_EVAL_HEAVY),
        ("old-set structure", lambda r: key_of(r["inchikey"]) in old_keys),
        ("scaffold of an old-set molecule", lambda r: r["scaffold"] in old_scaffolds),
        ("acyclic", lambda r: r["scaffold"] == ""),
        ("benzene scaffold", lambda r: r["scaffold"] == BENZENE_SCAFFOLD),
    ]
    removed = {}
    for name, test in rules:  # sequential: each molecule is counted under the first rule that removes it
        before = len(pool)
        pool = [r for r in pool if not test(r)]
        removed[name] = before - len(pool)
    eligible_bins = collections.Counter(stratum(r["heavy_atoms"]) for r in pool)
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for r in pool:
        groups[r["scaffold"]].append(r)
    rng = random.Random(EVAL_SEED)
    order = sorted(groups)
    rng.shuffle(order)
    chosen: list[dict] = []
    in_bin = collections.Counter()
    quota = {"11-20": PER_BIN, ">20": PER_BIN}
    for scaffold in order:
        if all(in_bin[b] >= quota[b] for b in quota):
            break
        members = sorted(groups[scaffold], key=id_order)
        rng.shuffle(members)
        taken = 0
        for m in members:
            b = stratum(m["heavy_atoms"])
            if taken < MAX_PER_SCAFFOLD and in_bin[b] < quota[b]:
                chosen.append(m)
                in_bin[b] += 1
                taken += 1
    if any(in_bin[b] < quota[b] for b in quota):
        sys.exit(f"cannot fill MEM-EVAL: got {dict(in_bin)} from {len(pool)} eligible molecules")
    return chosen, {
        "named_pool_after_dedup": counts["named pool"],
        "removed_sequentially": removed,
        "eligible": len(pool),
        "eligible_by_bin": dict(eligible_bins),
        "eligible_scaffolds": len(groups),
        "scaffolds_used": len({m["scaffold"] for m in chosen}),
    }


def interleave(strata: dict[str, list[dict]]) -> list[dict]:
    queues = {k: collections.deque(v) for k, v in strata.items()}
    out = []
    while any(queues.values()):
        for s in PATTERN:
            if queues[s]:
                out.append(queues[s].popleft())
    return out


def split_halves(cands: list[dict]) -> dict[str, str]:
    rng = random.Random(SPLIT_SEED)
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for c in cands:
        groups[c["scaffold"]].append(c)
    half: dict[str, str] = {}
    size = {"tuning": 0, "check": 0}

    def fewer() -> str:
        return "tuning" if size["tuning"] <= size["check"] else "check"

    whole = sorted(s for s in groups if s not in ("", BENZENE_SCAFFOLD))
    rng.shuffle(whole)
    for s in whole:
        h = fewer()
        for c in groups[s]:
            half[c["id"]] = h
        size[h] += len(groups[s])
    for s in ("", BENZENE_SCAFFOLD):  # molecule by molecule, alternating, starting with the smaller half
        members = sorted(groups.get(s, []), key=id_order)
        rng.shuffle(members)
        h = fewer()
        for c in members:
            half[c["id"]] = h
            size[h] += 1
            h = "check" if h == "tuning" else "tuning"
    return half


def select() -> None:
    existing = [p for p in FILES.values() if p.exists()]
    if existing:
        sys.exit(
            f"refusing to run: {', '.join(str(p.relative_to(ROOT)) for p in existing)} exist(s); selections are written once"
        )
    sys.path.insert(0, str(ROOT))
    from bench import benchmark

    benchmark.SD = str(ROOT / benchmark.SD)  # the export is found from any working directory
    git = git_state()
    if git["git_dirty_code"]:
        print(
            "warning: uncommitted changes under src/ or bench/; the recorded commit does not describe the code"
        )

    named_raw, named_counts = build_pool(benchmark.pool())
    unrep_raw, unrep_counts = build_pool(benchmark.pool(unreported=True))
    named = [annotate(r, "named") for r in named_raw]
    named_keys = {key_of(r["inchikey"]) for r in named}
    unrep_all = [annotate(r, "unreported") for r in unrep_raw]
    unrep = [r for r in unrep_all if key_of(r["inchikey"]) not in named_keys]
    unrep_counts["removed_structure_also_in_named"] = len(unrep_all) - len(unrep)
    unrep_counts["kept_after_named_overlap"] = len(unrep)
    union = named + unrep

    old = load_old_sets()
    old_keys = {key_of(r["inchikey"]) for r in old}
    old_scaffolds = {s for r in old if (s := chemistry_fields(r["smiles"])["murcko_scaffold"])}

    # ---- MEM-EVAL
    drawn, eval_counts = draw_mem_eval(named, old_keys, old_scaffolds)
    mem_eval = [{k: r[k] for k in (*EVAL_FIELDS, "heavy_atoms", "scaffold")} for r in drawn]
    eval_keys = {key_of(r["inchikey"]) for r in drawn}
    eval_scaffolds = {r["scaffold"] for r in drawn}
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def fp(smiles: str):
        return gen.GetFingerprint(Chem.MolFromSmiles(smiles))

    eval_fps = [fp(r["smiles"]) for r in drawn]

    # ---- candidates
    cand_counts = collections.OrderedDict(union_of_both_pools=len(union))
    cands = []
    removed = collections.Counter()
    for r in union:  # sequential: counted under the first rule that removes it
        if key_of(r["inchikey"]) in eval_keys:
            removed["MEM-EVAL structure"] += 1
        elif key_of(r["inchikey"]) in old_keys:
            removed["old-set structure"] += 1
        elif r["scaffold"] and r["scaffold"] in eval_scaffolds:
            removed["scaffold shared with MEM-EVAL"] += 1
        else:
            r["_max_sim"] = max(DataStructs.BulkTanimotoSimilarity(fp(r["smiles"]), eval_fps))
            if r["_max_sim"] >= SIMILARITY_LIMIT:
                removed[f"Tanimoto >= {SIMILARITY_LIMIT:.2f} to a MEM-EVAL molecule"] += 1
            else:
                cands.append(r)
    cand_counts["removed_sequentially"] = dict(removed)
    cand_counts["candidates"] = len(cands)
    cand_counts["candidates_by_pool"] = dict(collections.Counter(r["pool"] for r in cands))

    # ---- memory order
    rng = random.Random(ORDER_SEED)
    strata = {s: sorted((c for c in cands if stratum(c["heavy_atoms"]) == s), key=id_order) for s in PATTERN}
    for s in strata:
        rng.shuffle(strata[s])
    ordered = interleave(strata)
    order_rows = [
        {
            "position": i,
            "id": c["id"],
            "pool": c["pool"],
            "smiles": c["smiles"],
            "inchikey": c["inchikey"],
            "heavy_atoms": c["heavy_atoms"],
            "stratum": stratum(c["heavy_atoms"]),
            "scaffold": c["scaffold"],
            "solvent": c["solvent"],
        }
        for i, c in enumerate(ordered, 1)
    ]

    # ---- split
    half = split_halves(cands)
    split_counts = dict(collections.Counter(half.values()))

    # ---- write
    OUT.mkdir(parents=True, exist_ok=True)
    meta = lambda seed, counts: {"seed": seed, "filter_counts": counts, **git}  # noqa: E731
    dump_json(
        FILES["mem_eval"],
        meta(EVAL_SEED, {"named_pool": named_counts, "mem_eval": eval_counts}),
        "records",
        mem_eval,
    )
    dump_json(
        FILES["memory_order"],
        meta(
            ORDER_SEED,
            {
                "named_pool": named_counts,
                "unreported_pool": unrep_counts,
                "candidates": cand_counts,
                "pattern": list(PATTERN),
                "memory_size": MEMORY_SIZE,
            },
        ),
        "entries",
        order_rows,
    )
    dump_json(
        FILES["split"],
        meta(SPLIT_SEED, {"candidates": len(cands), "per_half": split_counts}),
        "assignments",
        [{"id": r["id"], "half": half[r["id"]]} for r in order_rows],
    )
    FILES["report"].write_text(
        report_text(
            named_counts,
            unrep_counts,
            eval_counts,
            cand_counts,
            drawn,
            order_rows,
            ordered,
            split_counts,
            half,
            git,
        )
    )
    print(FILES["report"].read_text())


def report_text(named_c, unrep_c, eval_c, cand_c, drawn, order_rows, ordered, split_counts, half, git) -> str:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    efp = [gen.GetFingerprint(Chem.MolFromSmiles(r["smiles"])) for r in drawn]
    first = ordered[:MEMORY_SIZE]
    L: list[str] = []
    L.append(
        f"selection report (code commit {git['git_commit']}, uncommitted code changes: {git['git_dirty_code']})"
    )
    L.append("")
    L.append("== pools (13C-incomplete records dropped, then one record per structure key)")
    L.append(f"named-solvent pool:  {json.dumps(named_c)}")
    L.append(f"unreported pool:     {json.dumps(unrep_c)}")
    L.append("")
    L.append(f"== MEM-EVAL (seed {EVAL_SEED})")
    L.append(f"named pool after dedup: {eval_c['named_pool_after_dedup']}")
    for name, n in eval_c["removed_sequentially"].items():
        L.append(f"  removed ({name}): {n}")
    L.append(
        f"eligible: {eval_c['eligible']} in {eval_c['eligible_scaffolds']} scaffolds, by bin {eval_c['eligible_by_bin']}"
    )
    bins = collections.Counter(stratum(r["heavy_atoms"]) for r in drawn)
    L.append(f"chosen: {len(drawn)}; 11-20 heavy atoms: {bins['11-20']}; >20: {bins['>20']}")
    L.append(
        f"scaffolds used: {eval_c['scaffolds_used']}; max per scaffold: {max(collections.Counter(r['scaffold'] for r in drawn).values())}"
    )
    L.append(f"solvents: {dict(collections.Counter(r['solvent'] for r in drawn))}")
    L.append("")
    L.append(f"== memory candidates (seed {ORDER_SEED} for the order)")
    L.append(f"union of both pools: {cand_c['union_of_both_pools']}")
    for name, n in cand_c["removed_sequentially"].items():
        L.append(f"  removed ({name}): {n}")
    L.append(f"candidates: {cand_c['candidates']} {cand_c['candidates_by_pool']}")
    L.append(f"by stratum: {dict(collections.Counter(r['stratum'] for r in order_rows))}")
    L.append("")
    L.append("== stratified mix of every prefix (>20 / 11-20 / <=10)")
    for n in PREFIXES:
        c = collections.Counter(r["stratum"] for r in order_rows[:n])
        L.append(f"first {n}: {c['>20']} / {c['11-20']} / {c['<=10']}")
    L.append(f"first 200 by pool: {dict(collections.Counter(r['pool'] for r in order_rows[:MEMORY_SIZE]))}")
    L.append(
        f"first 200 solvents: {dict(collections.Counter(r['solvent'] for r in order_rows[:MEMORY_SIZE]))}"
    )
    L.append("")
    L.append(
        "== nearest-neighbour Morgan Tanimoto (radius 2, 2048 bits): per MEM-EVAL molecule, the highest value"
    )
    L.append("percentiles: numpy linear interpolation; max is over the 100 MEM-EVAL molecules")
    for name, pool in (("any memory candidate", ordered), ("any of the first 200", first)):
        fps = [gen.GetFingerprint(Chem.MolFromSmiles(r["smiles"])) for r in pool]
        best = [max(DataStructs.BulkTanimotoSimilarity(e, fps)) for e in efp]
        L.append(
            f"{name}: median {np.median(best):.3f}, 90th percentile {np.percentile(best, 90):.3f}, max {max(best):.3f}"
        )
    L.append("")
    L.append("== generic-scaffold overlap (reported, not enforced; spec §9)")
    eval_gen = collections.defaultdict(list)
    for r in drawn:
        g = chemistry_fields(r["smiles"])["generic_scaffold"]
        eval_gen[g].append(r["id"])
    first_gen = collections.Counter(chemistry_fields(r["smiles"])["generic_scaffold"] for r in first)
    shared = {g for g in eval_gen if g and g in first_gen}
    L.append(
        f"MEM-EVAL molecules whose generic scaffold also occurs in the first 200: {sum(len(eval_gen[g]) for g in shared)} of {len(drawn)}"
    )
    L.append(
        f"(eval molecule, memory item) pairs with the same generic scaffold: {sum(len(eval_gen[g]) * first_gen[g] for g in shared)}"
    )
    L.append("")
    L.append(f"== tuning/check split (seed {SPLIT_SEED})")
    L.append(f"per half: {split_counts}")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- check


def load_selection_files() -> tuple[list[dict], list[dict]]:
    mem_eval = json.loads(FILES["mem_eval"].read_text())["records"]
    entries = json.loads(FILES["memory_order"].read_text())["entries"]
    return mem_eval, entries


def old_structure_keys() -> set[str]:
    return {key_of(r["inchikey"]) for r in load_old_sets()}


def report_rule(name: str, violations: list) -> bool:
    print(f"  {name}: {'ok' if not violations else f'FAIL ({len(violations)})'}")
    for v in violations[:10]:
        print(f"      {v}")
    if len(violations) > 10:
        print(f"      ... {len(violations) - 10} more")
    return not violations


def check(store_dir: str | None) -> int:
    mem_eval, entries = load_selection_files()
    old_keys = old_structure_keys()
    ok = True
    print(f"rule 3 (MEM-EVAL has no acyclic or benzene molecule), {len(mem_eval)} molecules")
    ok &= report_rule("rule 3", rule3_eval_scaffolds(mem_eval))
    for scope, items in (("first 200", entries[:MEMORY_SIZE]), ("all candidates", entries)):
        print(f"selection files, {scope} ({len(items)} items)")
        ok &= report_rule("rule 1 structure key", rule1_structure_key(items, mem_eval, old_keys))
        ok &= report_rule("rule 2 scaffold", rule2_scaffold(items, mem_eval))
        ok &= report_rule("rule 4 similarity", rule4_similarity(items, mem_eval))
    if store_dir:
        # No evaluation configuration exists yet, so there is nothing to compare each case's outcome
        # configuration with: expected_outcome_config=None skips that rule explicitly (M6 will freeze one).
        store = load_store(Path(store_dir), strict=True, expected_outcome_config=None)
        print(f"store {store_dir}, {len(store.cases)} cases")
        ok &= report_rule("rule 1 structure key", rule1_structure_key(store.cases, mem_eval, old_keys))
        ok &= report_rule("rule 2 scaffold", rule2_scaffold(store.cases, mem_eval))
        ok &= report_rule("rule 4 similarity", rule4_similarity(store.cases, mem_eval))
        ok &= report_rule("rule 5 proposed SMILES", rule5_proposed(store.cases, mem_eval))
    print("clean" if ok else "VIOLATIONS FOUND")
    return 0 if ok else 1


# --------------------------------------------------------------------------- M6: the large-weighted memory list (§15.4)

LARGE_FILE = OUT / "memory_large.json"
COLLECT_DIR = OUT / "collect"
STORE = ROOT / "memory"
LARGE_QUOTAS = {">20": 120, "11-20": 60, "<=10": 20}
LEAKAGE_RULES_LARGE = (1, 2, 4)
SD_PATH = ROOT / "bench" / "data" / "nmrshiftdb2" / "nmrshiftdb2withsignals.sd"
SPEND_CAP_DEFAULT = 35.0
BATCH_SIZE = 25
CONFIRM_WORD = "APPROVE"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def utc_today() -> dt.date:
    return dt.datetime.now(dt.UTC).date()


def build_large(entries: list[dict], excluded: set[str]) -> tuple[list[dict], dict[str, list[str]]]:
    """§15.4: per stratum, the first LARGE_QUOTAS[s] usable entries in memory_order; the rest of that stratum, in
    the same order, is its replacement order. The chosen rows are listed in memory_order (position) order."""
    usable = [e for e in entries if e["id"] not in excluded]
    chosen: list[dict] = []
    replacements: dict[str, list[str]] = {}
    for s, n in LARGE_QUOTAS.items():
        rows = [e for e in usable if e["stratum"] == s]
        if len(rows) < n:
            sys.exit(f"stratum {s}: only {len(rows)} usable entries, need {n}")
        chosen += rows[:n]
        replacements[s] = [e["id"] for e in rows[n:]]
    order = {e["id"]: i for i, e in enumerate(entries)}
    chosen.sort(key=lambda e: order[e["id"]])
    keep = ("stratum", "position", "id", "pool", "smiles", "inchikey", "heavy_atoms", "scaffold", "solvent")
    return [{"index": i, **{k: e[k] for k in keep}} for i, e in enumerate(chosen, 1)], replacements


def select_large(
    out: Path = LARGE_FILE,
    order_file: Path | None = None,
    samples_file: Path | None = None,
    eval_file: Path | None = None,
    old_keys: set[str] | None = None,
    git: dict | None = None,
) -> dict:
    """Write bench/memory/memory_large.json once; refuses to overwrite it, and refuses to write a list that
    breaks leakage rules 1, 2 or 4 against MEM-EVAL (spec §15.4: it must be clean)."""
    order_file = order_file or FILES["memory_order"]
    samples_file = samples_file or OUT / "gonogo_samples.json"
    eval_file = eval_file or FILES["mem_eval"]
    if out.exists():
        sys.exit(f"refusing to run: {rel(out)} exists; the list is written once, before any model call")
    entries = json.loads(order_file.read_text())["entries"]
    excluded = [x["id"] for x in json.loads(samples_file.read_text())["excluded"]]
    mem_eval = json.loads(eval_file.read_text())["records"]
    rows, replacements = build_large(entries, set(excluded))
    old = old_structure_keys() if old_keys is None else old_keys
    violations = [
        *rule1_structure_key(rows, mem_eval, old),
        *rule2_scaffold(rows, mem_eval),
        *rule4_similarity(rows, mem_eval),
    ]
    leakage = {
        "rules": list(LEAKAGE_RULES_LARGE),
        "memory_items": len(rows),
        "mem_eval_molecules": len(mem_eval),
        "rule_1_also_against_old_set_structure_keys": True,
        "violations": [str(v) for v in violations],
        "clean": not violations,
    }
    if violations:
        sys.exit(
            "the selected list breaks the leakage rules, nothing written:\n  "
            + "\n  ".join(leakage["violations"][:20])
        )
    git = git or git_state()
    header = {
        "description": "Large-weighted 200-case memory list (spec §15.4); written once, before any model call",
        "git_commit": git["git_commit"],
        "git_dirty_code": git["git_dirty_code"],
        "sources": {rel(p): sha256_file(p) for p in (order_file, samples_file, eval_file)},
        "excluded_ids": excluded,
        "quotas": LARGE_QUOTAS,
        "counts": {
            "selected_by_stratum": dict(collections.Counter(r["stratum"] for r in rows)),
            "selected_by_pool": dict(collections.Counter(r["pool"] for r in rows)),
            "available_by_stratum_after_exclusions": {
                s: LARGE_QUOTAS[s] + len(replacements[s]) for s in LARGE_QUOTAS
            },
        },
        "leakage": leakage,
        "replacement_order": replacements,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    dump_json(out, header, "entries", rows)
    return json.loads(out.read_text())


def load_large(path: Path = LARGE_FILE) -> dict:
    if not path.exists():
        sys.exit(f"{rel(path)} does not exist; run select-large first")
    return json.loads(path.read_text())


def resolve_rows(large: dict, ids: list[str] | None, order_file: Path | None = None) -> list[dict]:
    """The memory_large entries (all, or those named); a replacement id is resolved through memory_order."""
    by_id = {r["id"]: r for r in large["entries"]}
    if not ids:
        return list(large["entries"])
    replacement_ids = {i for lst in large["replacement_order"].values() for i in lst}
    order = None
    rows = []
    for i in ids:
        if i in by_id:
            rows.append(by_id[i])
        elif i in replacement_ids:
            if order is None:
                order = {
                    e["id"]: e
                    for e in json.loads((order_file or FILES["memory_order"]).read_text())["entries"]
                }
            rows.append(
                {
                    k: order[i][k]
                    for k in (
                        "stratum",
                        "position",
                        "id",
                        "pool",
                        "smiles",
                        "inchikey",
                        "heavy_atoms",
                        "scaffold",
                        "solvent",
                    )
                }
            )
        else:
            sys.exit(f"{i} is neither in memory_large.json nor in its replacement order")
    return rows


# --------------------------------------------------------------------------- M6: the SD record behind a pool record


@dataclass(frozen=True)
class SdRecord:
    """What pool() leaves out: the spectrum numbers and the molecule, for one record."""

    id: str
    mol: Chem.Mol
    smiles: str
    inchikey: str
    formula: str
    solvent: str | None
    h_spec: str
    c_spec: str
    h_spectrum: int
    c_spectrum: int
    # file atom index -> atom index of ``mol`` (see bench.benchmark.atom_index_map); None: indices are used as they are
    index_map: tuple[int | None, ...] | None = None

    def pool_record(self) -> dict:
        """The dict pool() returns for this record (the input of bench.benchmark.request_for)."""
        return {
            "id": self.id,
            "smiles": self.smiles,
            "inchikey": self.inchikey,
            "formula": self.formula,
            "solvent": self.solvent,
            "h_spec": self.h_spec,
            "c_spec": self.c_spec,
        }


def sd_record_from_text(rec: str, unreported: bool) -> SdRecord | None:
    """The same filters and the same "first matching pair" rule as bench.benchmark.pool(), for one record."""
    from bench import benchmark as b

    end = rec.find("M  END")
    mol = Chem.MolFromMolBlock(rec[: end + 6])
    if mol is None or len(Chem.GetMolFrags(mol)) != 1 or Chem.GetFormalCharge(mol) != 0:
        return None
    if {a.GetSymbol() for a in mol.GetAtoms()} - b.ALLOWED or mol.GetNumAtoms() > 30:
        return None
    if Descriptors.MolWt(mol) > 450:
        return None
    tags = dict(b.TAG.findall(rec[end:]))
    pred = {i for t in ("Program", "NMRProgram") for i, v in b.idx(tags.get(t, "")).items() if v}
    solv = b.idx(tags.get("Solvent", ""))
    sp: dict[str, dict[int, str]] = {"1H": {}, "13C": {}}
    for k, v in tags.items():
        m = re.fullmatch(r"Spectrum (1H|13C) (\d+)", k)
        if m and int(m[2]) not in pred:
            sp[m[1]][int(m[2])] = v
    if unreported:
        pairs = [
            (h, c)
            for h in sp["1H"]
            for c in sp["13C"]
            if solv.get(h, "").lower() in b.UNREPORTED and solv.get(c, "").lower() in b.UNREPORTED
        ]
    else:
        pairs = [
            (h, c)
            for h in sp["1H"]
            for c in sp["13C"]
            if b.map_solv(solv.get(h, "")) and b.map_solv(solv.get(h, "")) == b.map_solv(solv.get(c, ""))
        ]
    if not pairs:
        return None
    h, c = pairs[0]
    index_map = b.atom_index_map(rec[: end + 6], mol)
    return SdRecord(
        id=tags.get("nmrshiftdb2 ID", "?").strip(),
        mol=mol,
        smiles=Chem.MolToSmiles(mol),
        inchikey=Chem.MolToInchiKey(mol),
        formula=rdMolDescriptors.CalcMolFormula(mol),
        solvent=None if unreported else b.map_solv(solv[h]).value,
        h_spec=sp["1H"][h],
        c_spec=sp["13C"][c],
        h_spectrum=h,
        c_spectrum=c,
        index_map=index_map,
    )


_SD_ID = re.compile(r"> <nmrshiftdb2 ID>\n(\S+)")


def read_sd_records(wanted: Mapping[str, str], path: Path | None = None) -> dict[str, SdRecord]:
    """One pass over the SD export. ``wanted`` maps an nmrshiftdb2 id to its pool ("named" or "unreported").
    Ids that are absent, or that pool() would not have kept, are missing from the result."""
    path = path or SD_PATH
    found: dict[str, SdRecord] = {}
    buf: list[str] = []
    with open(path, errors="replace") as fh:
        for line in fh:
            if not line.startswith("$$$$"):
                buf.append(line)
                continue
            rec = "".join(buf)
            buf = []
            m = _SD_ID.search(rec)
            if m and m[1] in wanted and m[1] not in found:
                r = sd_record_from_text(rec, wanted[m[1]] == "unreported")
                if r is not None:
                    found[r.id] = r
    return found


def read_sd_record(record_id: str, pool: str = "named", path: Path | None = None) -> SdRecord | None:
    return read_sd_records({record_id: pool}, path).get(record_id)


# --------------------------------------------------------------------------- M6: collect (the paid step)


def outcome_status(candidates: list[str], truth_inchikey: str) -> str:
    """ "solved" if the first candidate has the truth's InChIKey connectivity block, else "missed"."""
    from bench import benchmark as b

    return "solved" if candidates and b.key(candidates[0]) == key_of(truth_inchikey) else "missed"


def build_request(row: dict, rec: SdRecord | None):
    """The benchmark's request for one memory molecule, plus its prompt. Raises ValueError on a problem."""
    from bench import benchmark as b
    from src.prompting import prepare_request

    if rec is None:
        raise ValueError("no usable record in the SD export (missing, or no same-solvent 1H + 13C pair)")
    if rec.smiles != row["smiles"] or rec.inchikey != row["inchikey"]:
        raise ValueError(f"SD record structure {rec.smiles} differs from the listed {row['smiles']}")
    req, complete = b.request_for(rec.pool_record(), rec.mol, rec.index_map)
    table, prompt, context = prepare_request(req)
    return req, complete, table, prompt, context


def request_hash(prompt) -> str:
    """SHA-256 of the exact user message a collect run sends (memory is off, so it is the whole request content)."""
    return hashlib.sha256(prompt.user_text.encode("utf-8")).hexdigest()


def stale_results(rows: list[dict], records: Mapping[str, SdRecord], out_dir: Path) -> dict[str, str]:
    """Saved results whose request is no longer the one the code builds now (or that carry no request hash)."""
    stale: dict[str, str] = {}
    for row in rows:
        f = out_dir / f"{row['id']}.json"
        if not f.exists():
            continue
        saved = json.loads(f.read_text()).get("request_sha256")
        try:
            *_, prompt, _ = build_request(row, records.get(row["id"]))
        except Exception as e:  # noqa: BLE001 - reported, the run does not continue
            stale[row["id"]] = f"the request cannot be rebuilt ({type(e).__name__}: {e})"
            continue
        if saved is None:
            stale[row["id"]] = "the saved result has no request_sha256"
        elif saved != request_hash(prompt):
            stale[row["id"]] = "the request built now differs from the one that was sent"
    return stale


def collect_dry_run(rows: list[dict], records: Mapping[str, SdRecord], large: dict) -> dict:
    failures: dict[str, str] = {}
    tokens = static = n = 0
    for row in rows:
        try:
            *_, prompt, _ = build_request(row, records.get(row["id"]))
        except Exception as e:  # noqa: BLE001 - every failure is reported, none stops the dry run
            failures[row["id"]] = f"{type(e).__name__}: {e}"
            continue
        n += 1
        tokens += prompt.total_tokens_est
        static += prompt.static_tokens_est
    print(f"dry run: {len(rows)} molecules, {n} buildable, {len(failures)} failed; no model call was made")
    print(f"estimated input tokens: {tokens} in total ({tokens // max(n, 1)} per molecule; "
          f"{static // max(n, 1)} of them the cacheable system prompt)")  # fmt: skip
    for i, why in failures.items():
        print(f"  FAILS {i}: {why}")
    return {"molecules": len(rows), "buildable": n, "failures": failures, "input_tokens_est": tokens}


def replacements_for(failed: list[dict], large: dict, taken: set[str]) -> dict[str, str]:
    """For each failed row, the next unused id of its stratum's replacement order."""
    queues = {s: [i for i in ids if i not in taken] for s, ids in large["replacement_order"].items()}
    return {r["id"]: queues[r["stratum"]].pop(0) for r in failed if queues[r["stratum"]]}


def collect(
    ids: list[str] | None = None,
    cap: float = SPEND_CAP_DEFAULT,
    dry_run: bool = False,
    large_file: Path = LARGE_FILE,
    out_dir: Path = COLLECT_DIR,
    read_records=read_sd_records,
    client_factory=None,
    settings=None,
    today: dt.date | None = None,
) -> dict:
    """Run the analysis once per memory molecule with memory disabled and save one file per molecule.

    A molecule with a result file is skipped (resume). The run stops before the next molecule once the money
    spent (the files' ``cost_usd``, including earlier runs) reaches ``cap``.
    """
    from src.analysis import CallMeter, run_analysis
    from src.core.settings import Settings
    from src.memory.runtime import outcome_config

    settings = settings or Settings()
    large = load_large(large_file)
    rows = resolve_rows(large, ids)
    records = read_records({r["id"]: r["pool"] for r in rows})
    if dry_run:
        report = collect_dry_run(rows, records, large)
        report["stale"] = stale_results(rows, records, out_dir)
        for i, why in report["stale"].items():
            print(f"  STALE RESULT {i}: {why}")
        failed = [r for r in rows if r["id"] in report["failures"]]
        if failed:
            taken = {r["id"] for r in large["entries"]}
            subs = replacements_for(failed, large, taken)
            for i, new in subs.items():
                print(f"  replacement for {i}: {new} (next unused in its stratum's replacement order)")
            report["replacements"] = subs
        return report

    stale = stale_results(rows, records, out_dir)
    if stale:
        lines = "\n".join(f"  {i}: {why}" for i, why in stale.items())
        sys.exit(
            f"stopping before any model call: {len(stale)} saved result(s) do not match the request built now.\n"
            f"{lines}\nMove them to {rel(out_dir.parent / 'collect_superseded')}/ (or fix the cause) and run again."
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = outcome_config(settings).model_dump()
    client = (client_factory or default_client)(settings)
    spent = sum(json.loads(f.read_text()).get("cost_usd", 0) for f in out_dir.glob("*.json"))
    done: list[str] = []
    for row in rows:
        out = out_dir / f"{row['id']}.json"
        if out.exists():
            continue
        if spent >= cap:
            print(f"spend cap ${cap:.2f} reached (${spent:.2f}); stopping")
            break
        try:
            req, complete, _, prompt, _ = build_request(row, records.get(row["id"]))
        except Exception as e:  # noqa: BLE001 - not an analysis: nothing was sent, nothing is saved
            print(row["id"], f"REQUEST NOT BUILT, skipped: {type(e).__name__}: {e}")
            continue
        trace: dict = {}
        meter = CallMeter()
        base = {
            "id": row["id"], "stratum": row["stratum"], "position": row["position"],
            "truth": row["smiles"], "truth_inchikey": row["inchikey"], "solvent": row["solvent"],
            "formula": records[row["id"]].formula, "h_integrals_given": complete,
            "request_sha256": request_hash(prompt),
            "collected_on": (today or utc_today()).isoformat(), "outcome_config": cfg,
        }  # fmt: skip
        try:
            resp = run_analysis(req, settings, client, trace=trace, meter=meter, memory=None)
        except Exception as e:  # noqa: BLE001 - classified below
            if not is_service_failure(e):
                sys.exit(
                    f"STOPPED at {row['id']}: the call did not reach the model or the account failed "
                    f"({type(e).__name__}: {str(e)[:300]}); nothing was saved for this molecule.\n"
                    f"{len(done)} molecule(s) collected in this run, ${spent:.2f} spent in {rel(out_dir)}.\n"
                    f"Fix the cause (credit, key, rate limit, network) and resume with the same command: "
                    f"finished molecules are skipped."
                )
            result = {**base, "status": "no_answer", "error": f"{type(e).__name__}: {str(e)[:200]}",
                      "error_reason": getattr(e, "reason", type(e).__name__), "candidates": [], "trace": trace, "cost_usd": 0, "llm_calls": meter.calls,
                      "timing_ms": {}, "warnings": []}  # fmt: skip
            out.write_text(json.dumps(result, indent=1))
            print(row["id"], "no_answer", result["error"])
            done.append(row["id"])
            continue
        cands = [
            {"smiles": v.candidate.smiles, "model_rank": v.candidate.rank, "status": v.status,
             "score": v.verifier_score, "confidence": v.final_confidence}
            for v in resp.candidates
        ]  # fmt: skip
        spent += resp.usage.estimated_cost_usd
        result = {
            **base, "status": outcome_status([c["smiles"] for c in cands], row["inchikey"]),
            "candidates": cands, "trace": trace, "cost_usd": resp.usage.estimated_cost_usd,
            "llm_calls": resp.usage.llm_calls, "timing_ms": resp.timing_ms, "warnings": resp.warnings,
        }  # fmt: skip
        out.write_text(json.dumps(result, indent=1))
        print(row["id"], result["status"], f"${resp.usage.estimated_cost_usd:.3f}")
        done.append(row["id"])
    print(f"collected {len(done)} molecule(s) this run; total spent in {rel(out_dir)}: ${spent:.2f}")
    return {"collected": done, "spent": spent}


# What counts as the service failing on a molecule (a no_answer outcome): the model ran out of time, or its answer could
# not be used. Everything else (the request never reached the model, the account or the network failed) says nothing
# about the service and stops the run. Derived from src.llm's exception types and LLMUnavailable.reason codes.
SERVICE_FAILURE_REASONS = frozenset(
    {"timeout", "output_truncated", "unexpected_stop_reason", "call_budget_exhausted"}
)


def is_service_failure(e: BaseException) -> bool:
    from src.llm.errors import LLMRefused, LLMSchemaFailure, LLMUnavailable

    if isinstance(e, LLMRefused | LLMSchemaFailure):
        return True
    return isinstance(e, LLMUnavailable) and e.reason in SERVICE_FAILURE_REASONS


def default_client(settings):
    import anthropic

    return anthropic.Anthropic(max_retries=0, timeout=settings.llm_call_timeout_s)


# --------------------------------------------------------------------------- M6: propose


def environment_of(atom: Chem.Atom) -> str:
    """ "C, 2 H, ring" or "C, aromatic, 1 H": element, aromatic, attached hydrogens, ring membership.
    Built only from the molecule graph, never from text in the record."""
    parts = [atom.GetSymbol()]
    if atom.GetIsAromatic():
        parts.append("aromatic")
    parts.append(f"{atom.GetTotalNumHs()} H")
    if atom.IsInRing() and not atom.GetIsAromatic():
        parts.append("ring")
    return ", ".join(parts)


def build_assignments(rec: SdRecord) -> tuple[list[dict], dict[str, int]]:
    """One assignment per spectrum entry that names an atom of the molecule; the rest are counted, not guessed."""
    out: list[dict] = []
    stats = {"entries": 0, "left_out": 0}
    n_atoms = rec.mol.GetNumAtoms()
    for nucleus, spec, digits in (("1H", rec.h_spec, 2), ("13C", rec.c_spec, 1)):
        for entry in spec.strip().strip("|").split("|"):
            p = entry.split(";")
            if not p[0]:
                continue
            stats["entries"] += 1
            k = int(p[2]) if len(p) >= 3 and p[2].strip().isdigit() else None
            if k is not None and rec.index_map is not None:
                k = rec.index_map[k] if k < len(rec.index_map) else None
            if k is not None and k < n_atoms:
                atom = rec.mol.GetAtomWithIdx(k)
                out.append(
                    {
                        "nucleus": nucleus,
                        "shift_ppm": round(float(p[0]), digits),
                        "environment": environment_of(atom),
                    }
                )
            else:
                stats["left_out"] += 1
    return out, stats


def build_pending(row: dict, rec: SdRecord, result: dict, case_id: str, git_label: str, today: dt.date):
    """The pending case and its notes. Raises ValueError for a record that cannot become a case."""
    from src.memory.case import AgentOutcome, Provenance
    from src.memory.pending import PendingCase

    _, _, table, _, context = build_request(row, rec)
    chem = chemistry_fields(rec.smiles)
    if chem is None:
        raise ValueError(f"RDKit cannot parse {rec.smiles}")
    if chem["formula"] != context.molecular_formula:
        raise ValueError(f"formula {chem['formula']} differs from the request's {context.molecular_formula}")
    assignments, stats = build_assignments(rec)
    cfg = result["outcome_config"]
    status = result["status"]
    proposed = [] if status == "no_answer" else [c["smiles"] for c in result["candidates"]]
    pending = PendingCase(
        case_id=case_id,
        schema_version="1",
        provenance=Provenance(
            source="nmrshiftdb2",
            source_record_id=rec.id,
            spectrum_ids={"1H": rec.h_spectrum, "13C": rec.c_spectrum},
            truth_basis="database-assigned structure",
            extracted_by=f"bench/memory.py propose @ {git_label}, RDKit {rdkit.__version__}",
            proposed_on=today,
        ),
        **{
            k: chem[k]
            for k in (
                "formula",
                "dou",
                "carbon_count",
                "heavy_atoms",
                "smiles",
                "inchikey",
                "murcko_scaffold",
                "generic_scaffold",
            )
        },
        solvent=context.solvent if context.solvent is not None else table.solvent,
        h_peaks=[{"shift_ppm": p.shift_ppm, "integral": p.integral} for p in table.analyte_peaks(Nucleus.H1)],
        c_peaks=[p.shift_ppm for p in table.analyte_peaks(Nucleus.C13)],
        assignments=assignments or None,
        agent_outcome=AgentOutcome(
            collected_on=dt.date.fromisoformat(result["collected_on"]),
            model_id=cfg["model_id"],
            prompt_version=cfg["prompt_version"],
            settings_hash=cfg["settings_hash"],
            status=status,
            proposed_smiles=proposed,
        ),
    )
    return pending, {"assignments_entries": stats["entries"], "assignments_left_out": stats["left_out"]}


def case_ids_in(store: Path) -> dict[str, str]:
    """source_record_id -> case id, over pending, approved and rejected cases (ids are never reused)."""
    found: dict[str, str] = {}
    for sub in ("pending", "cases", "rejected"):
        for f in sorted((store / sub).glob("mem-*.json")):
            doc = json.loads(f.read_text())
            doc = doc.get("case", doc)
            found[doc["provenance"]["source_record_id"]] = doc["case_id"]
    return found


def used_case_numbers(store: Path) -> set[int]:
    return {
        int(f.stem[4:]) for sub in ("pending", "cases", "rejected") for f in (store / sub).glob("mem-*.json")
    }


def structure_keys_in(store: Path) -> dict[str, str]:
    """Structure key -> case id for the pending and the approved cases (the duplicate rule)."""
    keys: dict[str, str] = {}
    for sub in ("pending", "cases"):
        for f in sorted((store / sub).glob("mem-*.json")):
            doc = json.loads(f.read_text())
            keys.setdefault(key_of(doc["inchikey"]), doc["case_id"])
    return keys


def leakage_problems(case, mem_eval: list[dict], old_keys: set[str]) -> list[str]:
    """Rules 1, 2, 4 and 5 for one case (a MemoryCase, or a placeholder view of a pending case)."""
    v = [
        *rule1_structure_key([case], mem_eval, old_keys),
        *rule2_scaffold([case], mem_eval),
        *rule4_similarity([case], mem_eval),
        *rule5_proposed([case], mem_eval),
    ]
    return [str(x) for x in v]


def propose(
    ids: list[str] | None = None,
    store: Path = STORE,
    collect_dir: Path = COLLECT_DIR,
    large_file: Path = LARGE_FILE,
    eval_file: Path | None = None,
    read_records=read_sd_records,
    settings=None,
    today: dt.date | None = None,
    git: dict | None = None,
    old_keys: set[str] | None = None,
) -> dict:
    """Build pending cases for every collected molecule that has no case yet, refusing the ones that fail a
    check, then write the review sheets. Never touches cases/ or the manifest."""
    from src.core.settings import Settings
    from src.memory.pending import chemistry_problems, dump_case, placeholder_case
    from src.memory.runtime import outcome_config

    settings = settings or Settings()
    today = today or utc_today()
    git = git or git_state()
    git_label = git["git_commit"] + ("+uncommitted changes" if git["git_dirty_code"] else "")
    large = load_large(large_file)
    rows = resolve_rows(large, ids)
    mem_eval = json.loads((eval_file or FILES["mem_eval"]).read_text())["records"]
    old = old_structure_keys() if old_keys is None else old_keys
    expected = outcome_config(settings).model_dump()
    pending_dir = store / "pending"
    pending_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("cases", "rejected"):
        (store / sub).mkdir(parents=True, exist_ok=True)

    have = case_ids_in(store)
    index_of = {r["id"]: r["index"] for r in large["entries"]}
    next_free = max([len(large["entries"]), *used_case_numbers(store)]) + 1
    todo = []
    skipped: dict[str, str] = {}
    for row in rows:
        f = collect_dir / f"{row['id']}.json"
        if row["id"] in have:
            skipped[row["id"]] = f"already has case {have[row['id']]}"
        elif not f.exists():
            skipped[row["id"]] = "not collected"
        else:
            todo.append((row, json.loads(f.read_text())))
    records = read_records({r["id"]: r["pool"] for r, _ in todo}) if todo else {}
    notes_file = pending_dir / "propose-notes.json"
    notes = json.loads(notes_file.read_text()) if notes_file.exists() else {}
    keys = structure_keys_in(store)
    refused: dict[str, list[str]] = {}
    written: list[str] = []
    for row, result in todo:
        problems: list[str] = []
        if result["outcome_config"] != expected:
            problems.append("collected under a different configuration than the current Settings")
        if result["truth_inchikey"] != row["inchikey"]:
            problems.append("collect result is for a different structure than memory_large lists")
        if problems:
            refused[row["id"]] = problems
            continue
        if row["id"] in index_of:
            case_id = f"mem-{index_of[row['id']]:06d}"
        else:
            case_id, next_free = f"mem-{next_free:06d}", next_free + 1
        try:
            pending, meta = build_pending(row, records.get(row["id"]), result, case_id, git_label, today)
            problems += chemistry_problems(pending)
        except Exception as e:  # noqa: BLE001 - reported as a refusal
            refused[row["id"]] = [f"inconsistent or unusable record: {type(e).__name__}: {e}"]
            continue
        if key_of(pending.inchikey) in keys:
            problems.append(f"structure key duplicates case {keys[key_of(pending.inchikey)]}")
        problems += leakage_problems(placeholder_case(pending), mem_eval, old)
        if problems:
            refused[row["id"]] = problems
            continue
        (pending_dir / f"{case_id}.json").write_text(dump_case(pending), encoding="utf-8")
        keys[key_of(pending.inchikey)] = case_id
        notes[case_id] = {**meta, "molecule": row["id"]}
        written.append(case_id)
    if written:
        notes_file.write_text(json.dumps(dict(sorted(notes.items())), indent=1) + "\n", encoding="utf-8")
    sheets = write_review_sheets(store, collect_dir, mem_eval, old)
    print(
        f"proposed {len(written)} case(s) into {rel(pending_dir)}; skipped {len(skipped)}; refused {len(refused)}"
    )
    for i, why in skipped.items():
        print(f"  skipped {i}: {why}")
    for i, why in refused.items():
        print(f"  REFUSED {i}: {'; '.join(why)}")
    print(f"review sheets: {', '.join(p.name for p in sheets) or 'none'}")
    return {"written": written, "skipped": skipped, "refused": refused, "sheets": [p.name for p in sheets]}


def batch_of(case_id: str) -> int:
    return (int(case_id[4:]) - 1) // BATCH_SIZE + 1


def write_review_sheets(
    store: Path, collect_dir: Path, mem_eval: list[dict], old_keys: set[str]
) -> list[Path]:
    """One sheet per batch of 25 case numbers, listing the cases of that batch that are pending now."""
    from src.memory.pending import PendingCase, placeholder_case

    pending_dir = store / "pending"
    for stale in pending_dir.glob("review-batch-*.md"):
        stale.unlink()
    cases = [PendingCase.model_validate_json(f.read_text()) for f in sorted(pending_dir.glob("mem-*.json"))]
    notes_file = pending_dir / "propose-notes.json"
    notes = json.loads(notes_file.read_text()) if notes_file.exists() else {}
    batches: dict[int, list[PendingCase]] = collections.defaultdict(list)
    for c in cases:
        batches[batch_of(c.case_id)].append(c)
    paths = []
    for n, members in sorted(batches.items()):
        ids = " ".join(c.case_id for c in members)
        lines = [
            f"# Review sheet, batch {n:02d} ({len(members)} cases)",
            "",
            "**The owner approves or rejects each case with the commands below.** Nothing in this sheet is "
            "approved; an AI agent must not run them.",
            "",
            "```",
            f'.venv/bin/python bench/memory.py approve {ids} --by "<your name>"',
            '.venv/bin/python bench/memory.py reject <case_id>... --reason "<why>"',
            "```",
            "",
            "Check at least: the structure matches the formula; the peak list is plausible for it; for a `missed` "
            "case, that the proposal really is wrong and the miss is not a database problem (wrong structure, "
            "missing peaks, a difference only in stereochemistry); nothing personal or confidential. A doubtful "
            "`missed` case is rejected, not relabelled.",
            "",
        ]
        for c in members:
            lines += review_entry(c, notes.get(c.case_id, {}), mem_eval, old_keys, placeholder_case(c))
        path = pending_dir / f"review-batch-{n:02d}.md"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def first_proposal_text(proposed: list[str]) -> str:
    if not proposed:
        return "none"
    mol = Chem.MolFromSmiles(proposed[0])
    return f"`{Chem.MolToSmiles(mol)}`" if mol is not None else "not shown (does not parse as SMILES)"


def review_entry(c, note: dict, mem_eval: list[dict], old_keys: set[str], as_case) -> list[str]:
    leak = leakage_problems(as_case, mem_eval, old_keys)
    out = c.agent_outcome
    n_assign = len(c.assignments or [])
    left = note.get("assignments_left_out")
    h = ", ".join(
        f"{p.shift_ppm}" + (f" ({p.integral} H)" if p.integral is not None else "") for p in c.h_peaks
    )
    return [
        f"## {c.case_id}: {c.formula}, {c.solvent.value if c.solvent else 'solvent not reported'}",
        "",
        f"- nmrshiftdb2 record {c.provenance.source_record_id}, spectra {c.provenance.spectrum_ids}",
        f"- structure: `{c.smiles}`; scaffold: `{c.murcko_scaffold or '(acyclic)'}`; {c.heavy_atoms} heavy atoms",
        f"- 1H peaks ({len(c.h_peaks)}): {h}",
        f"- 13C peaks ({len(c.c_peaks)}): {', '.join(str(p) for p in c.c_peaks)}",
        f"- agent outcome: **{out.status}**, {len(out.proposed_smiles)} candidate(s); first proposal: "
        f"{first_proposal_text(out.proposed_smiles)}",
        "- leakage rules 1, 2, 4, 5 against the 100 MEM-EVAL molecules: "
        + ("clean" if not leak else "FAILED: " + "; ".join(leak)),
        f"- assignments: {n_assign}"
        + (f" ({left} spectrum entries without an atom index left out)" if left is not None else ""),
        "",
    ]


# --------------------------------------------------------------------------- M6: approve and reject


def confirm(stdin, stdout, prompt: str) -> bool:
    """True only if the typed line is exactly CONFIRM_WORD on an interactive input."""
    if not (hasattr(stdin, "isatty") and stdin.isatty()):
        print("refusing: approval needs an interactive terminal (stdin is not one)", file=sys.stderr)
        return False
    stdout.write(prompt)
    stdout.flush()
    return stdin.readline().rstrip("\r\n") == CONFIRM_WORD


def append_log(store: Path, line: str) -> None:
    with open(store / "APPROVALS.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def oneline(text: str) -> str:
    return " ".join(text.split())


def report_store(store: Path, settings, run_check) -> int:
    from src.memory.runtime import outcome_config

    s = load_store(store, strict=True, expected_outcome_config=outcome_config(settings))
    print(
        f"memory_version: {s.memory_version} ({len(s.cases)} approved case(s), strict load with the current outcome configuration: ok)"
    )
    rc = run_check(str(store))
    print(f"check --store {rel(store)}: exit {rc}")
    return rc


def approve(
    case_ids: list[str],
    by: str,
    store: Path = STORE,
    stdin=None,
    stdout=None,
    eval_file: Path | None = None,
    settings=None,
    today: dt.date | None = None,
    run_check=None,
    old_keys: set[str] | None = None,
) -> int:
    """Spec §6. All checks run before anything is written; the typed word is read from an interactive stdin."""
    from src.core.settings import Settings
    from src.memory.pending import PendingCase, dump_case, finalize, placeholder_case
    from src.memory.runtime import outcome_config
    from src.memory.store import ManifestEntry, load_manifest, sha256_hex, write_manifest

    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    settings = settings or Settings()
    today = today or utc_today()
    run_check = run_check or (lambda s: check(s))
    by = oneline(by)
    if not by:
        print("refusing: --by needs the approver's name", file=sys.stderr)
        return 2
    if len(set(case_ids)) != len(case_ids):
        print("refusing: a case id is given twice", file=sys.stderr)
        return 2
    mem_eval = json.loads((eval_file or FILES["mem_eval"]).read_text())["records"]
    old = old_structure_keys() if old_keys is None else old_keys
    expected = outcome_config(settings).model_dump()
    manifest = load_manifest(store)
    approved_keys = structure_keys_in(store)
    staged = []
    for cid in case_ids:
        path = store / "pending" / f"{cid}.json"
        if not path.is_file():
            print(f"refusing: {cid} is not in {rel(store / 'pending')}", file=sys.stderr)
            return 2
        pending = PendingCase.model_validate_json(path.read_text())
        problems = leakage_problems(placeholder_case(pending), mem_eval, old)
        cfg = pending.agent_outcome
        if (cfg.model_id, cfg.prompt_version, cfg.settings_hash) != tuple(expected.values()):
            problems.append("outcome configuration differs from the current Settings; recollect it")
        dup = [c for k, c in approved_keys.items() if k == key_of(pending.inchikey) and c != cid]
        if dup:
            problems.append(f"structure key duplicates case {dup[0]}")
        if problems:
            print(f"refusing: {cid}: " + "; ".join(problems), file=sys.stderr)
            return 2
        case = finalize(pending, by, today)
        staged.append((path, case))
    print("About to APPROVE these cases into the store (this adds them to the memory the service reads):")
    for _, c in staged:
        print(f"  {c.case_id}  {c.formula}  {c.agent_outcome.status}  {c.smiles}")
    print(f"Approver: {by}")
    if not confirm(stdin, stdout, f"Type {CONFIRM_WORD} to approve {len(staged)} case(s): "):
        print("not approved; nothing changed")
        return 1
    entries = list(manifest.cases)
    for path, case in staged:
        text = dump_case(case)
        (store / "cases").mkdir(parents=True, exist_ok=True)
        (store / "cases" / f"{case.case_id}.json").write_text(text, encoding="utf-8")
        path.unlink()
        entries.append(ManifestEntry(case_id=case.case_id, sha256=sha256_hex(text.encode("utf-8"))))
        write_manifest(store, entries)
        append_log(store, f"{today.isoformat()} {case.case_id} approve {by}")
        print(f"approved {case.case_id}")
    return report_store(store, settings, run_check)


def reject(
    case_ids: list[str],
    reason: str,
    store: Path = STORE,
    settings=None,
    today: dt.date | None = None,
    run_check=None,
) -> int:
    from src.core.settings import Settings

    settings = settings or Settings()
    today = today or utc_today()
    run_check = run_check or (lambda s: check(s))
    reason = oneline(reason)
    if not reason:
        print("refusing: --reason needs a text", file=sys.stderr)
        return 2
    if len(set(case_ids)) != len(case_ids):
        print("refusing: a case id is given twice", file=sys.stderr)
        return 2
    for cid in case_ids:
        if not (store / "pending" / f"{cid}.json").is_file():
            print(f"refusing: {cid} is not in {rel(store / 'pending')}", file=sys.stderr)
            return 2
    (store / "rejected").mkdir(parents=True, exist_ok=True)
    for cid in case_ids:
        src = store / "pending" / f"{cid}.json"
        doc = {"rejected_on": today.isoformat(), "reason": reason, "case": json.loads(src.read_text())}
        (store / "rejected" / f"{cid}.json").write_text(
            json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        src.unlink()
        append_log(store, f"{today.isoformat()} {cid} reject {reason}")
        print(f"rejected {cid}")
    return report_store(store, settings, run_check)


# --------------------------------------------------------------------------- M6: the store skeleton

NOTICE_TEXT = """# NMRShiftDB2 data notice

Contains information from nmrshiftdb2 ([www.nmrshiftdb.org](https://nmrshiftdb.nmr.uni-koeln.de)), which is made available here under the [nmrshiftdb2 Database License](https://nmrshiftdb.nmr.uni-koeln.de/nmrshiftdbhtml/nmrshiftdb2datalicense.txt).

A verbatim copy of that license is in [bench/NMRSHIFTDB2-DATABASE-LICENSE.txt](../bench/NMRSHIFTDB2-DATABASE-LICENSE.txt). This data is not covered by the repository's MIT License.

## What in this directory comes from nmrshiftdb2

| Path | Contents |
|---|---|
| `cases/` | Approved memory cases: for each, the nmrshiftdb2 record id and spectrum numbers, the structure (SMILES, scaffold), the formula, the solvent, the peak lists, and the atom-level environments taken from the database's atom assignments |
| `rejected/` | Cases that were proposed from nmrshiftdb2 records and rejected, with the reason |

Each case also holds the service's own output on that molecule (its proposed structures), which is not nmrshiftdb2 data. The derived database stays under the nmrshiftdb2 Database License; the machine-readable form is the JSON in `cases/`.
"""


def setup_store(store: Path = STORE, params_source: Path | None = None) -> None:
    """Create the empty repository store (committed). Refuses if a manifest already exists."""
    from src.memory.store import write_manifest

    params_source = params_source or OUT / "retrieval_params.json"
    if (store / "manifest.json").exists():
        sys.exit(f"refusing to run: {rel(store / 'manifest.json')} exists")
    for sub in ("cases", "rejected"):
        (store / sub).mkdir(parents=True, exist_ok=True)
        (store / sub / ".gitkeep").write_bytes(b"")
    write_manifest(store, [])
    (store / "retrieval_params.json").write_bytes(params_source.read_bytes())
    (store / "NMRSHIFTDB2-NOTICE.md").write_text(NOTICE_TEXT, encoding="utf-8")
    (store / "APPROVALS.log").write_bytes(b"")
    print(f"created the empty store in {rel(store)}")


# --------------------------------------------------------------------------- M7: the four H-large arms (spec §15.6, §15.11)

EVAL_DIR = OUT / "eval"
ARMS = ("M0-a", "M0-b", "M200L", "R200L")
MEMORY_ARMS = ("M200L", "R200L")
EVAL_MIN_HEAVY = 20  # the molecules with heavy_atoms > 20
EVAL_STORE_CASES = 200
EVAL_SPEND_CAP_DEFAULT = 40.0
NEAR_DUPLICATE = 0.5  # §15.11: an injected case at or above this Tanimoto to the truth
RUN_FILE = "run.json"
RUN_FILE_IGNORED = ("created_on", "git_commit")  # recorded, not compared on resume


def seed_of(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def order_seed(rid: str) -> int:
    """§15.11 run order: int(sha256(f"{id}:order").hexdigest()[:8], 16)."""
    return seed_of(f"{rid}:order")


def random_seed(rid: str) -> int:
    """§15.11 R200L seed: the M4 rule with sample index 0, int(sha256(f"{id}:0").hexdigest()[:8], 16)."""
    return seed_of(f"{rid}:0")


def arm_order(rid: str, arms: tuple[str, ...] = ARMS) -> list[str]:
    """The four arms of one molecule in the seeded shuffled order (the shuffle of all four, then the
    chosen subset in that order)."""
    order = list(ARMS)
    random.Random(order_seed(rid)).shuffle(order)
    return [a for a in order if a in arms]


def eval_rows(eval_file: Path | None = None) -> list[dict]:
    """The MEM-EVAL records with more than 20 heavy atoms, in file order. ``pool`` is "unreported" for a
    record without a solvent and "named" otherwise (the records of mem_eval.json carry no pool field)."""
    records = json.loads((eval_file or FILES["mem_eval"]).read_text())["records"]
    return [
        {**r, "pool": r.get("pool") or ("unreported" if r.get("solvent") is None else "named")}
        for r in records
        if r["heavy_atoms"] > EVAL_MIN_HEAVY
    ]


_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def tanimoto(smiles_a: str, smiles_b: str) -> float:
    """Morgan (radius 2, 2048 bits) Tanimoto similarity; 0.0 when either SMILES does not parse."""
    a, b = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    if a is None or b is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(_MORGAN.GetFingerprint(a), _MORGAN.GetFingerprint(b))


@dataclass(frozen=True)
class Built:
    """One analysis ready to run: the request, the prompt with and without the examples, the runtime."""

    req: object
    complete: bool
    base_prompt: object
    prompt: object
    memory: object  # MemoryRuntime | None


def arm_memory(arm: str, runtime, rid: str):
    if arm == "M200L":
        return runtime
    if arm == "R200L":
        return dataclasses.replace(runtime, mode="random", seed=random_seed(rid))
    return None


def build_arm(row: dict, rec: SdRecord | None, arm: str, runtime) -> Built:
    from src.prompting import prepare_request

    req, complete, _, base_prompt, _ = build_request(row, rec)
    memory = arm_memory(arm, runtime, row["id"])
    prompt = base_prompt if memory is None else prepare_request(req, None, memory)[1]
    return Built(req, complete, base_prompt, prompt, memory)


def build_all(
    rows: list[dict], records: Mapping[str, SdRecord], arms: tuple[str, ...], runtime
) -> tuple[dict[tuple[str, str], Built], dict[tuple[str, str], str]]:
    """Every (arm, molecule) analysis built, and the failures: (arm, id) -> reason."""
    built: dict[tuple[str, str], Built] = {}
    failures: dict[tuple[str, str], str] = {}
    for row in rows:
        rec = records.get(row["id"])
        if rec is not None and (rec.h_spec != row["h_spec"] or rec.c_spec != row["c_spec"]):
            for arm in arms:
                failures[(arm, row["id"])] = "the SD record's spectra differ from mem_eval.json"
            continue
        for arm in arms:
            try:
                built[(arm, row["id"])] = build_arm(row, rec, arm, runtime)
            except Exception as e:  # noqa: BLE001 - reported, the run does not start
                failures[(arm, row["id"])] = f"{type(e).__name__}: {e}"
    return built, failures


def injected_of(prompt, truth_smiles: str) -> list[dict]:
    """The cases the prompt carries, best first, with the Tanimoto of each to the truth."""
    return [
        {
            "case_id": r.case.case_id,
            "similarity": r.similarity,
            "tanimoto_to_truth": round(tanimoto(r.case.smiles, truth_smiles), 6),
        }
        for r in prompt.examples
    ]


def store_leakage_violations(cases, mem_eval: list[dict], old_keys: set[str]) -> list[str]:
    """Rules 1, 2, 4 and 5 for a store against all MEM-EVAL records (what ``check --store`` runs)."""
    out: list[str] = []
    for name, rule in (
        ("rule 1 structure key", lambda: rule1_structure_key(cases, mem_eval, old_keys)),
        ("rule 2 scaffold", lambda: rule2_scaffold(cases, mem_eval)),
        ("rule 4 similarity", lambda: rule4_similarity(cases, mem_eval)),
        ("rule 5 proposed SMILES", lambda: rule5_proposed(cases, mem_eval)),
    ):
        out += [f"{name}: {v}" for v in rule()]
    return out


def eval_run_record(store: Path, runtime, settings, rows: list[dict], cases_expected: int) -> dict:
    from src.memory.runtime import outcome_config, params_hash

    doc = json.loads((store / "retrieval_params.json").read_text())
    return {
        "manifest_sha256": sha256_file(store / "manifest.json"),
        "memory_version": runtime.memory_version,
        "cases": len(runtime.store.cases),
        "cases_expected": cases_expected,
        "retrieval_params_sha256": params_hash(doc["parameters"]),
        "outcome_config": outcome_config(settings).model_dump(),
        "arms": list(ARMS),
        "molecule_ids": [r["id"] for r in rows],
        "created_on": utc_today().isoformat(),
        "git_commit": git_state().get("git_commit"),
    }


def check_run_file(path: Path, record: dict, results_exist: bool) -> None:
    """Write run.json once; on a resume refuse if anything in it differs from this run."""
    if not path.exists():
        if results_exist:
            sys.exit(f"{rel(path)} is missing but result files exist; refusing to continue")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1) + "\n")
        return
    saved = json.loads(path.read_text())
    differ = [
        k for k in sorted(set(saved) | set(record))
        if k not in RUN_FILE_IGNORED and saved.get(k) != record.get(k)
    ]  # fmt: skip
    if differ:
        lines = "\n".join(f"  {k}: saved {saved.get(k)!r}, now {record.get(k)!r}" for k in differ)
        sys.exit(f"{rel(path)} does not match this run; refusing to start.\n{lines}")


def eval_result_files(out_dir: Path) -> list[Path]:
    return [f for arm in ARMS for f in sorted((out_dir / arm).glob("*.json"))]


def eval_spent(out_dir: Path) -> float:
    return sum(json.loads(f.read_text()).get("cost_usd", 0) for f in eval_result_files(out_dir))


def eval_stale(built: Mapping[tuple[str, str], Built], out_dir: Path) -> dict[tuple[str, str], str]:
    """Saved results whose request (or examples block) is not the one built now, or that carry no hash."""
    stale: dict[tuple[str, str], str] = {}
    for (arm, rid), b in built.items():
        f = out_dir / arm / f"{rid}.json"
        if not f.exists():
            continue
        saved = json.loads(f.read_text())
        if saved.get("request_sha256") is None:
            stale[(arm, rid)] = "the saved result has no request_sha256"
        elif saved["request_sha256"] != request_hash(b.base_prompt):
            stale[(arm, rid)] = "the request built now differs from the one that was sent"
        elif saved.get("prompt_sha256") != request_hash(b.prompt):
            stale[(arm, rid)] = "the prompt (examples block) built now differs from the one that was sent"
    return stale


def eval_cost_estimate(
    arm_tokens: Mapping[str, int], collect_dir: Path, large: dict, records, n_per_arm: int
):
    """Estimated cost of the run from the collect results, or None when there are none.

    Method 1 (per molecule): the mean cost of the priced collect results of the ">20" stratum, times the number
    of analyses. Method 2 (per input token): total cost over the estimated input tokens of those same collect
    requests (rebuilt now), times each arm's estimated input tokens. Output and thinking tokens dominate real
    cost, so method 2 only scales the input side; neither method is a bound."""
    priced = {}
    for f in sorted(collect_dir.glob("*.json")):
        r = json.loads(f.read_text())
        if r.get("cost_usd", 0) > 0 and r.get("stratum") == ">20":
            priced[r["id"]] = r["cost_usd"]
    if not priced:
        return None
    per_molecule = sum(priced.values()) / len(priced)
    rows = [r for r in resolve_rows(large, None) if r["id"] in priced]
    tokens = 0
    used_cost = 0.0
    for row in rows:
        try:
            *_, prompt, _ = build_request(row, records.get(row["id"]))
        except Exception:  # noqa: BLE001 - such a molecule is left out of method 2
            continue
        tokens += prompt.total_tokens_est
        used_cost += priced[row["id"]]
    per_token = used_cost / tokens if tokens else None
    return {
        "priced_collect_results": len(priced),
        "mean_cost_per_molecule": per_molecule,
        "method_1_total": per_molecule * sum(n_per_arm for _ in arm_tokens),
        "cost_per_input_token": per_token,
        "method_2_total": per_token * sum(arm_tokens.values()) if per_token else None,
    }


def eval_run(
    arms: tuple[str, ...] = ARMS,
    ids: list[str] | None = None,
    cap: float = EVAL_SPEND_CAP_DEFAULT,
    dry_run: bool = False,
    eval_file: Path | None = None,
    store: Path = STORE,
    out_dir: Path = EVAL_DIR,
    collect_dir: Path = COLLECT_DIR,
    large_file: Path = LARGE_FILE,
    read_records=read_sd_records,
    client_factory=None,
    settings=None,
    cases_expected: int = EVAL_STORE_CASES,
    old_keys: set[str] | None = None,
) -> dict:
    """The four arms on the large MEM-EVAL molecules (§15.6, §15.11). The paid step unless ``dry_run``.

    Molecule by molecule in mem_eval.json order; the arms of a molecule in its seeded order. An analysis with a
    result file is skipped (resume). A service failure is saved as no_answer; any other failure stops the run
    and saves nothing for that analysis. The run stops before the next analysis once the money in the result
    files reaches ``cap``.
    """
    from src.analysis import CallMeter, run_analysis
    from src.core.settings import Settings
    from src.memory import MemoryRuntime
    from src.memory.runtime import outcome_config

    settings = settings or Settings()
    all_rows = eval_rows(eval_file)
    rows = all_rows
    if ids:
        unknown = [i for i in ids if i not in {r["id"] for r in all_rows}]
        if unknown:
            sys.exit(f"not among the {len(all_rows)} large MEM-EVAL molecules: {', '.join(unknown)}")
        rows = [r for r in all_rows if r["id"] in set(ids)]

    try:
        runtime = MemoryRuntime.from_directory(store, settings, strict=True)
    except Exception as e:  # noqa: BLE001 - any load problem refuses the run
        sys.exit(f"the memory store {rel(store)} cannot be loaded strictly: {type(e).__name__}: {e}")
    if len(runtime.store.cases) != cases_expected:
        sys.exit(f"the store holds {len(runtime.store.cases)} cases, not {cases_expected}; refusing to run")
    mem_eval = json.loads((eval_file or FILES["mem_eval"]).read_text())["records"]
    problems = store_leakage_violations(
        runtime.store.cases, mem_eval, old_structure_keys() if old_keys is None else old_keys
    )
    if problems:
        sys.exit("the store breaks the leakage rules; refusing to run:\n  " + "\n  ".join(problems[:10]))

    records = read_records({r["id"]: r["pool"] for r in rows})
    built, failures = build_all(rows, records, arms, runtime)
    if dry_run:
        return eval_dry_run(
            rows, arms, built, failures, runtime, collect_dir, large_file, read_records, out_dir
        )
    if failures:
        lines = "\n".join(f"  {a} {i}: {why}" for (a, i), why in failures.items())
        sys.exit(f"stopping before any model call: {len(failures)} request(s) cannot be built.\n{lines}")
    stale = eval_stale(built, out_dir)
    if stale:
        lines = "\n".join(f"  {a} {i}: {why}" for (a, i), why in stale.items())
        sys.exit(
            f"stopping before any model call: {len(stale)} saved result(s) do not match the request built now.\n"
            f"{lines}"
        )
    record = eval_run_record(store, runtime, settings, all_rows, cases_expected)
    check_run_file(out_dir / RUN_FILE, record, bool(eval_result_files(out_dir)))

    cfg = outcome_config(settings).model_dump()
    client = (client_factory or default_client)(settings)
    spent = eval_spent(out_dir)
    ran: list[tuple[str, str]] = []
    for row in rows:
        for arm in arm_order(row["id"], arms):
            out = out_dir / arm / f"{row['id']}.json"
            if out.exists():
                continue
            if spent >= cap:
                print(f"spend cap ${cap:.2f} reached (${spent:.2f}); stopping")
                return {"ran": ran, "spent": spent, "stopped": "cap"}
            b = built[(arm, row["id"])]
            injected = injected_of(b.prompt, row["smiles"])
            trace: dict = {}
            meter = CallMeter()
            base = {
                "arm": arm, "id": row["id"], "truth": row["smiles"], "truth_inchikey": row["inchikey"],
                "heavy_atoms": row["heavy_atoms"], "h_integrals_given": b.complete,
                "request_sha256": request_hash(b.base_prompt), "prompt_sha256": request_hash(b.prompt),
                "outcome_config": cfg, "run_on": utc_today().isoformat(),
                "memory_version": None if b.memory is None else runtime.memory_version,
                "injected": injected,
            }  # fmt: skip
            try:
                resp = run_analysis(b.req, settings, client, trace=trace, meter=meter, memory=b.memory)
            except Exception as e:  # noqa: BLE001 - classified below
                if not is_service_failure(e):
                    sys.exit(
                        f"STOPPED at {arm} {row['id']}: the call did not reach the model or the account failed "
                        f"({type(e).__name__}: {str(e)[:300]}); nothing was saved for this analysis.\n"
                        f"{len(ran)} analysis(es) run this time, ${spent:.2f} spent in {rel(out_dir)}.\n"
                        f"Fix the cause (credit, key, rate limit, network) and resume with the same command: "
                        f"finished analyses are skipped."
                    )
                result = {
                    **base, "status": "no_answer", "error": f"{type(e).__name__}: {str(e)[:200]}",
                    "error_reason": getattr(e, "reason", type(e).__name__), "candidates": [],
                    "trace": trace, "cost_usd": 0, "llm_calls": meter.calls, "timing_ms": {}, "warnings": [],
                }  # fmt: skip
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(result, indent=1))
                print(arm, row["id"], "no_answer", result["error"])
                ran.append((row["id"], arm))
                continue
            cands = [
                {"smiles": v.candidate.smiles, "model_rank": v.candidate.rank, "status": v.status,
                 "score": v.verifier_score, "confidence": v.final_confidence}
                for v in resp.candidates
            ]  # fmt: skip
            spent += resp.usage.estimated_cost_usd
            result = {
                **base, "status": outcome_status([c["smiles"] for c in cands], row["inchikey"]),
                "candidates": cands, "trace": trace, "cost_usd": resp.usage.estimated_cost_usd,
                "llm_calls": resp.usage.llm_calls, "timing_ms": resp.timing_ms, "warnings": resp.warnings,
            }  # fmt: skip
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(result, indent=1))
            print(arm, row["id"], result["status"], f"${resp.usage.estimated_cost_usd:.3f}")
            ran.append((row["id"], arm))
    print(f"ran {len(ran)} analysis(es) this time; total spent in {rel(out_dir)}: ${spent:.2f}")
    return {"ran": ran, "spent": spent, "stopped": None}


def eval_dry_run(
    rows, arms, built, failures, runtime, collect_dir, large_file, read_records, out_dir
) -> dict:
    """No model call: what the run would send, how much input, and a cost estimate with its method."""
    tokens = {a: sum(b.prompt.total_tokens_est for (x, _), b in built.items() if x == a) for a in arms}
    n = collections.Counter(a for a, _ in built)
    examples = {
        a: [len(b.prompt.examples) for (x, _), b in built.items() if x == a] for a in MEMORY_ARMS if a in arms
    }
    done = sum(1 for a in arms for r in rows if (out_dir / a / f"{r['id']}.json").exists())
    print(
        f"dry run: {len(rows)} molecules x {len(arms)} arm(s) = {len(rows) * len(arms)} analyses, "
        f"{len(built)} buildable, {len(failures)} failed; {done} already have a result; no model call was made"
    )
    for a in arms:
        print(f"  {a}: estimated input tokens {tokens[a]} ({tokens[a] // max(n[a], 1)} per molecule)")
    for a, counts in examples.items():
        print(
            f"  {a}: examples injected per molecule: mean {sum(counts) / max(len(counts), 1):.2f}, "
            f"counts {dict(sorted(collections.Counter(counts).items()))}"
        )
    for (a, i), why in failures.items():
        print(f"  FAILS {a} {i}: {why}")
    stale = eval_stale(built, out_dir)
    for (a, i), why in stale.items():
        print(f"  STALE RESULT {a} {i}: {why}")
    estimate = None
    if collect_dir.exists() and built:
        large = load_large(large_file)
        wanted = {r["id"]: r["pool"] for r in resolve_rows(large, None)}
        extra = read_records(wanted)
        estimate = eval_cost_estimate(tokens, collect_dir, large, extra, len(rows))
    if estimate is None:
        print("cost estimate: no priced collect results to estimate from")
    else:
        print(
            f"cost estimate, method 1 (mean cost per molecule of the {estimate['priced_collect_results']} priced "
            f"collect results of the >20 stratum, ${estimate['mean_cost_per_molecule']:.3f}, times "
            f"{len(rows) * len(arms)} analyses): ${estimate['method_1_total']:.2f}"
        )
        if estimate["method_2_total"] is not None:
            print(
                f"cost estimate, method 2 (collect cost per estimated input token, "
                f"${estimate['cost_per_input_token']:.6f}, times the estimated input tokens above): "
                f"${estimate['method_2_total']:.2f}"
            )
        print("  Output and thinking tokens dominate real cost, so neither figure is a bound.")
    return {
        "analyses": len(rows) * len(arms), "buildable": len(built), "failures": {f"{a}/{i}": w for (a, i), w in failures.items()},
        "input_tokens_est": tokens, "examples_per_molecule": examples, "estimate": estimate, "already_done": done,
        "stale": {f"{a}/{i}": w for (a, i), w in stale.items()},
    }  # fmt: skip


# --------------------------------------------------------------------------- M7: compare (spec §15.7, §15.8, §15.11)

COMPARE_JSON = "compare.json"
COMPARE_TXT = "compare.txt"


def first_smiles(res: dict | None) -> str | None:
    if not res or res.get("status") == "no_answer" or not res.get("candidates"):
        return None
    return res["candidates"][0]["smiles"]


def trace_smiles(res: dict) -> list[str]:
    """Every candidate SMILES of every round of the trace (first answer, repair) plus the final candidates."""
    out = [c["smiles"] for c in res.get("candidates", [])]
    for round_ in ("first", "repaired"):
        h = (res.get("trace") or {}).get(round_)
        if isinstance(h, dict):
            out += [c.get("smiles") for c in h.get("candidates", []) if c.get("smiles")]
    return out


def molecule_measures(res: dict | None, row: dict) -> dict | None:
    """The §15.7 numbers of one analysis; None when the result file is missing."""
    from bench import benchmark as b

    if res is None:
        return None
    truth_key = key_of(row["inchikey"])
    first = first_smiles(res)
    keys = [b.key(c["smiles"]) for c in res.get("candidates", [])] if first is not None else []
    own = sorted(res.get("candidates", []), key=lambda c: c.get("model_rank", 0)) if first is not None else []
    injected = res.get("injected") or []
    timing = res.get("timing_ms") or {}
    return {
        "similarity": tanimoto(first, row["smiles"]) if first is not None else 0.0,
        "no_answer": res.get("status") == "no_answer",
        "timeout": res.get("status") == "no_answer" and res.get("error_reason") == "timeout",
        "top1": bool(keys) and keys[0] == truth_key,
        "top3": truth_key in keys[:3],
        "proposed": truth_key in {b.key(s) for s in trace_smiles(res)},
        "own_top1": bool(own) and b.key(own[0]["smiles"]) == truth_key,
        "cost": res.get("cost_usd", 0.0),
        "time_ms": timing.get("total"),
        "examples": len(injected),
        "m": max((i["tanimoto_to_truth"] for i in injected), default=None),
    }


def _mean(xs) -> float | None:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def arm_summary(per: dict[str, dict | None]) -> dict:
    got = [m for m in per.values() if m is not None]
    n = len(got)
    return {
        "n": n,
        "S": _mean(m["similarity"] for m in got),
        "top1": sum(m["top1"] for m in got),
        "top3": sum(m["top3"] for m in got),
        "proposed_any_round": sum(m["proposed"] for m in got),
        "own_order_top1": sum(m["own_top1"] for m in got),
        "no_answer": sum(m["no_answer"] for m in got),
        "timeouts": sum(m["timeout"] for m in got),
        "mean_time_s": (lambda t: None if t is None else t / 1000)(
            _mean(m["time_ms"] for m in got if m["time_ms"] is not None)
        ),
        "mean_cost_usd": _mean(m["cost"] for m in got),
        "mean_examples": _mean(m["examples"] for m in got),
    }


def _r(x: float) -> float:
    """Rounded before a threshold comparison, so a difference of means that is 0.05 on paper is not 0.04999999."""
    return round(x, 9)


def wilcoxon_p(d: list[float]) -> tuple[float | None, int]:
    """(p, number of non-zero differences); the §15.11 call. No non-zero difference: p is None."""
    import warnings

    from scipy.stats import wilcoxon

    nonzero = sum(1 for x in d if x != 0)
    if nonzero == 0:
        return None, 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = wilcoxon(d, alternative="two-sided", zero_method="wilcox", method="auto")
    return float(res.pvalue), nonzero


def verdict_of(per: dict[str, dict[str, dict]], arms: dict[str, dict], ids: list[str]) -> dict:
    """§15.8 as §15.11 defines it. ``per[arm][id]`` are complete molecule measures."""
    s = {a: arms[a]["S"] for a in ARMS}
    m0 = (s["M0-a"] + s["M0-b"]) / 2
    delta = s["M200L"] - m0
    noise = abs(s["M0-a"] - s["M0-b"])
    d = [
        per["M200L"][i]["similarity"] - (per["M0-a"][i]["similarity"] + per["M0-b"][i]["similarity"]) / 2
        for i in ids
    ]
    p, nonzero = wilcoxon_p(d)
    gap = s["M200L"] - s["R200L"]
    low = [
        k for k, i in enumerate(ids) if per["M200L"][i]["m"] is None or per["M200L"][i]["m"] < NEAR_DUPLICATE
    ]
    delta_low = _mean(d[k] for k in low)
    confined = True if delta_low is None else _r(delta_low) <= 0.02
    c = []

    def add(role: str, name: str, value: str, holds: bool) -> None:
        c.append({"role": role, "condition": name, "value": value, "holds": bool(holds)})

    add("supports", "Δ ≥ +0.05", f"Δ = {delta:+.4f}", _r(delta) >= 0.05)
    add("supports", "Δ > 2N", f"Δ = {delta:+.4f}, 2N = {2 * noise:.4f}", _r(delta) > _r(2 * noise))
    add(
        "supports", "Wilcoxon two-sided p < 0.05",
        f"p = {'n/a' if p is None else f'{p:.4f}'} over {nonzero} non-zero d_i of {len(d)}",
        p is not None and p < 0.05,
    )  # fmt: skip
    add("supports", "S(M200L) − S(R200L) ≥ +0.03", f"{gap:+.4f}", _r(gap) >= 0.03)
    add("falsifies", "Δ ≤ +0.02", f"Δ = {delta:+.4f}", _r(delta) <= 0.02)
    add("falsifies", "Δ ≤ N", f"Δ = {delta:+.4f}, N = {noise:.4f}", _r(delta) <= _r(noise))
    add(
        "falsifies",
        "S(R200L) ≥ S(M200L)",
        f"S(R200L) = {s['R200L']:.4f}, S(M200L) = {s['M200L']:.4f}",
        _r(s["R200L"]) >= _r(s["M200L"]),
    )
    add(
        "falsifies", f"gain confined to molecules with m_i ≥ {NEAR_DUPLICATE}",
        f"Δ over the {len(low)} molecule(s) with m_i < {NEAR_DUPLICATE} = "
        + ("n/a (none: counts as confined)" if delta_low is None else f"{delta_low:+.4f}"),
        confined,
    )  # fmt: skip
    t_m = arms["M200L"]["timeouts"]
    t_0 = (arms["M0-a"]["timeouts"] + arms["M0-b"]["timeouts"]) / 2
    add("harm", "timeouts: M200L − mean(M0) > 3", f"{t_m} − {t_0:g} = {t_m - t_0:g}", t_m - t_0 > 3)
    cost_m = arms["M200L"]["mean_cost_usd"]
    cost_0 = (arms["M0-a"]["mean_cost_usd"] + arms["M0-b"]["mean_cost_usd"]) / 2
    rise = None if cost_0 == 0 else cost_m / cost_0 - 1
    add(
        "harm", "cost per molecule rises by more than 30%",
        f"${cost_m:.4f} vs ${cost_0:.4f} ({'n/a' if rise is None else f'{rise:+.1%}'})",
        rise is not None and _r(rise) > 0.30,
    )  # fmt: skip
    falsified = [x for x in c if x["role"] == "falsifies" and x["holds"]]
    supported = all(x["holds"] for x in c if x["role"] == "supports")
    verdict = "H-large falsified" if falsified else "H-large supported" if supported else "inconclusive"
    return {
        "verdict": verdict, "delta": delta, "N": noise, "mean_S_M0": m0, "S_M200L_minus_S_R200L": gap,
        "wilcoxon_p": p, "nonzero_d": nonzero, "d": dict(zip(ids, d, strict=True)),
        "delta_low_m": delta_low, "n_low_m": len(low), "conditions": c,
        "harm": [x for x in c if x["role"] == "harm" and x["holds"]],
    }  # fmt: skip


def compare(out_dir: Path = EVAL_DIR, eval_file: Path | None = None, write: bool = True) -> dict:
    """§15.7 per arm and the §15.8 verdict, from the result files only (no model call). Fewer than all
    analyses: the partial measures, labelled PARTIAL, and no verdict."""
    rows = eval_rows(eval_file)
    ids = [r["id"] for r in rows]
    per: dict[str, dict[str, dict | None]] = {}
    for a in ARMS:
        per[a] = {}
        for r in rows:
            f = out_dir / a / f"{r['id']}.json"
            per[a][r["id"]] = molecule_measures(json.loads(f.read_text()) if f.exists() else None, r)
    arms = {a: arm_summary(per[a]) for a in ARMS}
    have = sum(v["n"] for v in arms.values())
    total = len(ARMS) * len(rows)
    complete = have == total and total > 0
    report = {
        "complete": complete, "results": have, "expected": total, "molecules": len(rows),
        "arms": arms, "verdict": None,
        "m_i_M200L": {i: per["M200L"][i]["m"] for i in ids if per["M200L"][i] is not None},
        "per_molecule": {
            a: {i: m for i, m in per[a].items() if m is not None} for a in ARMS
        },
    }  # fmt: skip
    if complete:
        report["verdict"] = verdict_of(per, arms, ids)
    text = compare_text(report)
    if write:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / COMPARE_JSON).write_text(json.dumps(report, indent=1) + "\n")
        (out_dir / COMPARE_TXT).write_text(text)
    print(text, end="")
    return report


def _fmt(x, spec: str = ".4f") -> str:
    return "n/a" if x is None else format(x, spec)


def compare_text(report: dict) -> str:
    tag = "" if report["complete"] else "PARTIAL "
    lines = [
        f"{tag}H-large comparison: {report['results']} of {report['expected']} analyses have a result "
        f"({report['molecules']} molecules x {len(ARMS)} arms)"
    ]
    if not report["complete"]:
        lines.append(
            "PARTIAL: no verdict is given until all analyses have results; the measures below use what exists."
        )
    head = f"{'arm':7} {'n':>3} {'S':>7} {'top1':>5} {'top3':>5} {'any':>4} {'own1':>5} {'noans':>5} {'tmo':>4} {'time_s':>7} {'cost$':>7} {'ex':>5}"
    lines += ["", head]
    for a, v in report["arms"].items():
        lines.append(
            f"{a:7} {v['n']:>3} {_fmt(v['S']):>7} {v['top1']:>5} {v['top3']:>5} {v['proposed_any_round']:>4} "
            f"{v['own_order_top1']:>5} {v['no_answer']:>5} {v['timeouts']:>4} {_fmt(v['mean_time_s'], '.1f'):>7} "
            f"{_fmt(v['mean_cost_usd'], '.4f'):>7} {_fmt(v['mean_examples'], '.2f'):>5}"
        )
    lines.append(
        "S: mean Morgan r2 2048-bit Tanimoto of the first candidate to the truth (0 without an answer)."
    )
    lines.append(
        "top1/top3: InChIKey connectivity match among the first 1/3 candidates; any: truth proposed in any round;"
    )
    lines.append(
        "own1: truth is the model's own rank-1 candidate; noans: no_answer; tmo: no_answer with reason timeout;"
    )
    lines.append("time_s, cost$: mean per analysis with a value; ex: mean examples injected.")
    ms = [v for v in report["m_i_M200L"].values() if v is not None]
    if ms:
        lines.append(
            f"m_i (highest Tanimoto of an injected M200L case to the truth): mean {sum(ms) / len(ms):.3f}, "
            f"max {max(ms):.3f}, {sum(1 for v in ms if v >= NEAR_DUPLICATE)} molecule(s) at or above {NEAR_DUPLICATE}"
        )
    v = report["verdict"]
    if v is None:
        return "\n".join(lines) + "\n"
    lines += [
        "",
        f"Δ = S(M200L) − mean(S(M0-a), S(M0-b)) = {v['delta']:+.4f}; N = |S(M0-a) − S(M0-b)| = {v['N']:.4f}",
        f"Wilcoxon (two-sided, zero_method=wilcox, method=auto) on d_i: p = {_fmt(v['wilcoxon_p'])}, non-zero d_i = {v['nonzero_d']}",
        f"S(M200L) − S(R200L) = {v['S_M200L_minus_S_R200L']:+.4f}",
        "",
        "conditions:",
    ]
    for c in v["conditions"]:
        lines.append(f"  [{'HOLDS' if c['holds'] else 'no   '}] {c['role']:9} {c['condition']}: {c['value']}")
    lines += ["", f"VERDICT: {v['verdict']}"]
    lines.append("harm: " + ("; ".join(x["condition"] for x in v["harm"]) if v["harm"] else "none"))
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select", help="write the four selection files (once)")
    c = sub.add_parser("check", help="run leakage rules 1-4 on the selections, 1, 2, 4, 5 on a store")
    c.add_argument("--store", help="a memory store directory (manifest.json + cases/)")
    sub.add_parser("select-large", help="write bench/memory/memory_large.json (once)")
    sub.add_parser("setup-store", help="create the empty memory/ store skeleton")
    co = sub.add_parser("collect", help="the paid step: run the service on the memory molecules, memory off")
    co.add_argument("--ids", nargs="+", help="only these nmrshiftdb2 ids")
    co.add_argument("--cap", type=float, default=SPEND_CAP_DEFAULT, help="spend cap in USD (default 35)")
    co.add_argument("--dry-run", action="store_true", help="build every request, call nothing")
    pr = sub.add_parser("propose", help="build pending cases and review sheets from collected outcomes")
    pr.add_argument("--ids", nargs="+")
    pr.add_argument("--store", default=str(STORE))
    ap_ = sub.add_parser("approve", help="a person approves pending cases (asks for a typed confirmation)")
    ap_.add_argument("case_ids", nargs="+")
    ap_.add_argument("--by", required=True, help="the approver's name")
    ap_.add_argument("--store", default=str(STORE))
    rj = sub.add_parser("reject", help="move pending cases to rejected/ with a reason")
    rj.add_argument("case_ids", nargs="+")
    rj.add_argument("--reason", required=True)
    rj.add_argument("--store", default=str(STORE))
    ev = sub.add_parser("eval", help="the paid step: the four H-large arms on the large MEM-EVAL molecules")
    ev.add_argument("--arms", nargs="+", choices=ARMS, help="only these arms (default: all four)")
    ev.add_argument("--ids", nargs="+", help="only these nmrshiftdb2 ids (default: all 50)")
    ev.add_argument("--cap", type=float, default=EVAL_SPEND_CAP_DEFAULT, help="spend cap in USD (default 40)")
    ev.add_argument("--dry-run", action="store_true", help="build every request and prompt, call nothing")
    sub.add_parser(
        "compare", help="the §15.7 measures and the §15.8 verdict from bench/memory/eval/ (no model call)"
    )
    args = ap.parse_args()
    if args.cmd == "select":
        select()
    elif args.cmd == "check":
        sys.exit(check(args.store))
    elif args.cmd == "select-large":
        out = select_large()
        print(json.dumps({k: out[k] for k in ("counts", "leakage")}, indent=1))
    elif args.cmd == "setup-store":
        setup_store()
    elif args.cmd == "collect":
        collect(args.ids, args.cap, args.dry_run)
    elif args.cmd == "eval":
        eval_run(tuple(args.arms or ARMS), args.ids, args.cap, args.dry_run)
    elif args.cmd == "compare":
        compare()
    elif args.cmd == "propose":
        propose(args.ids, Path(args.store))
    elif args.cmd == "approve":
        sys.exit(approve(args.case_ids, args.by, Path(args.store)))
    else:
        sys.exit(reject(args.case_ids, args.reason, Path(args.store)))


if __name__ == "__main__":
    main()
