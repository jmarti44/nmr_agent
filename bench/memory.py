# ruff: noqa: E501
"""Experience-memory selections and leakage checks (spec §7.1, §7.2, §9).

    PYTHONPATH=. uv run python bench/memory.py select           # once: writes the four files in bench/memory/
    PYTHONPATH=. uv run python bench/memory.py check [--store <dir>]

`select` needs the NMRShiftDB2 export (bench/data/, untracked) and refuses to run if any output exists:
the selections are written once, before any model call, and never rewritten. `check` reads only the
committed files (and a store, if given) and exits non-zero on any violation.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select", help="write the four selection files (once)")
    c = sub.add_parser("check", help="run leakage rules 1-4 on the selections, 1, 2, 4, 5 on a store")
    c.add_argument("--store", help="a memory store directory (manifest.json + cases/)")
    args = ap.parse_args()
    if args.cmd == "select":
        select()
    else:
        sys.exit(check(args.store))


if __name__ == "__main__":
    main()
