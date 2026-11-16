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

`select` needs the NMRShiftDB2 export (bench/data/, untracked) and refuses to run if any output exists:
the selections are written once, before any model call, and never rewritten. `check` reads only the
committed files (and a store, if given) and exits non-zero on any violation.
"""

from __future__ import annotations

import argparse
import collections
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
    elif args.cmd == "propose":
        propose(args.ids, Path(args.store))
    elif args.cmd == "approve":
        sys.exit(approve(args.case_ids, args.by, Path(args.store)))
    else:
        sys.exit(reject(args.case_ids, args.reason, Path(args.store)))


if __name__ == "__main__":
    main()
