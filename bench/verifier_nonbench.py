# ruff: noqa: E402, E501, B018
"""Verifier on NMRShiftDB2 records OUTSIDE the Phase 5 benchmark (P5R-1 before/after measurement).

Adapted from the reviewer's Phase 1 probe (docs/review-phase1-probes/realdata_probe.py): the same record filter
and table builder, plus (1) exclusion of every molecule in the three benchmark selections (by id and by InChIKey
connectivity block), (2) a fixed seed, (3) a per-status/rule summary that is printed identically before and
after a rule change, so the two outputs can be diffed.

Usage: uv run python -m bench.verifier_nonbench [N=300] [SEED=11]
"""

import collections
import json
import random
import re
import sys
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
from src.schemas import Candidate, Nucleus, Peak, PeakTable, RuleStatus, SampleContext, Solvent
from src.verify.chem import parse_smiles
from src.verify.rules import Ctx, v08_structure_demands_peaks, v09_peaks_explained
from src.verify.verifier import verify_candidate

V08_ONLY = (
    len(sys.argv) > 3 and sys.argv[3] == "v08"
)  # all records with 1H+13C, V08 only (more aldehydes etc.)
SD = "bench/data/nmrshiftdb2/nmrshiftdb2withsignals.sd"
EX_IDS, EX_KEYS = set(), set()
for _f in (
    "bench/results/selection.json",
    "bench/results_main/selection.json",
    "bench/results_unreported/selection.json",
):
    for _r in json.loads(Path(_f).read_text())["records"]:
        EX_IDS.add(str(_r["id"]))
        EX_KEYS.add(_r["inchikey"].split("-")[0])
TAG = re.compile(r"^> +<([^>]+)>\n(.*?)(?=\n\n|\Z)", re.M | re.S)
IDX = re.compile(r"(\d+):(.*?)(?=\s+\d+:|\s*$)", re.S)
SOLV = [
    ("Chloroform-D1", Solvent.CDCL3),
    ("CDCl3", Solvent.CDCL3),
    ("Dimethylsulphoxide-D6", Solvent.DMSO_D6),
    ("DMSO", Solvent.DMSO_D6),
    ("Deuteriumoxide", Solvent.D2O),
    ("Methanol-D4", Solvent.CD3OD),
    ("Acetone-D6", Solvent.ACETONE_D6),
    ("Benzene-D6", Solvent.C6D6),
    ("Acetonitrile-D3", Solvent.CD3CN),
]
SUBGROUPS = {
    "aldehyde": "[CX3H1](=O)[#6]",
    "heteroaromatic": "[n,o,s]1aaaa1",
    "amide_NH": "[CX3](=O)[NX3;!H0]",
}
ALLOWED = {"C", "H", "N", "O", "S", "F", "Cl", "Br"}


def v08_status(table, ctx, smiles, rule=v08_structure_demands_peaks) -> str:
    """Status of one rule alone (the full verifier stops early on a wrong H count, which incomplete records have)."""
    parsed, _ = parse_smiles(smiles)
    cand = Candidate(rank=1, smiles=smiles, rationale="", assignments=[], confidence=0.5)
    return rule(Ctx(table=table, context=ctx, parsed=parsed, candidate=cand)).status.value


def idx(v):
    return {int(i): s.strip() for i, s in IDX.findall(v.strip())}


def map_solv(s):
    for k, v in SOLV:
        if s.lower().startswith(k.lower()):
            return v
    return None


def records():
    buf = []
    with open(SD, errors="replace") as fh:
        for line in fh:
            if line.startswith("$$$$"):
                yield "".join(buf)
                buf = []
            else:
                buf.append(line)


pool, by_formula = [], collections.defaultdict(list)
for rec in records():
    end = rec.find("M  END")
    mol = Chem.MolFromMolBlock(rec[: end + 6])
    if mol is None or len(Chem.GetMolFrags(mol)) != 1 or Chem.GetFormalCharge(mol) != 0:
        continue
    if (
        {a.GetSymbol() for a in mol.GetAtoms()} - ALLOWED
        or mol.GetNumAtoms() > 30
        or Descriptors.MolWt(mol) > 450
    ):
        continue
    smi = Chem.MolToSmiles(mol)
    f = rdMolDescriptors.CalcMolFormula(mol)
    by_formula[f].append(smi)
    tags = dict(TAG.findall(rec[end:]))
    pred = {i for t in ("Program", "NMRProgram") for i, v in idx(tags.get(t, "")).items() if v}
    solv = idx(tags.get("Solvent", ""))
    sp = {"1H": {}, "13C": {}}
    for k, v in tags.items():
        m = re.fullmatch(r"Spectrum (1H|13C) (\d+)", k)
        if m and int(m[2]) not in pred:
            sp[m[1]][int(m[2])] = v
    pairs = [
        (h, c)
        for h in sp["1H"]
        for c in sp["13C"]
        if map_solv(solv.get(h, "")) and map_solv(solv.get(h, "")) == map_solv(solv.get(c, ""))
    ]
    rid_ = tags.get("nmrshiftdb2 ID", "?").strip()
    from rdkit.Chem import inchi as _inchi

    if rid_ in EX_IDS or _inchi.MolToInchiKey(mol).split("-")[0] in EX_KEYS:
        continue
    if pairs:
        h, c = pairs[0]
        pool.append(
            (tags.get("nmrshiftdb2 ID", "?").strip(), smi, f, map_solv(solv[h]), sp["1H"][h], sp["13C"][c])
        )


def table(mol, hspec, cspec, solvent):
    hs = collections.defaultdict(set)
    shifts_of = collections.defaultdict(set)
    for e in hspec.strip().strip("|").split("|"):
        p = e.split(";")
        if len(p) >= 3 and p[0] and p[2].strip().isdigit():
            s, a = round(float(p[0]), 2), int(p[2])
            hs[s].add(a)
            shifts_of[a].add(s)
    assigned = {a for a in shifts_of if a < mol.GetNumAtoms()}
    need = {a.GetIdx() for a in mol.GetAtoms() if a.GetTotalNumHs() and a.GetSymbol() == "C"}
    if not need <= assigned and not V08_ONLY:
        raise LookupError("incomplete 1H assignment")
    peaks = []
    for shift in sorted(hs, reverse=True):
        atoms = [i for i in hs[shift] if i < mol.GetNumAtoms()]
        if not atoms:
            continue
        n_h = sum(mol.GetAtomWithIdx(i).GetTotalNumHs() / len(shifts_of[i]) for i in atoms)
        exch = any(mol.GetAtomWithIdx(i).GetSymbol() in ("O", "N", "S") for i in atoms)
        if n_h == 0:
            continue
        peaks.append(
            Peak(
                id=len(peaks) + 1,
                nucleus=Nucleus.H1,
                shift_ppm=shift,
                integral=round(float(n_h), 2),
                exchangeable_hint=exch,
                multiplicity="br" if exch else "unknown",
            )
        )
    cs = sorted(
        {round(float(e.split(";")[0]), 1) for e in cspec.strip().strip("|").split("|") if e.split(";")[0]},
        reverse=True,
    )
    for s in cs:
        peaks.append(Peak(id=len(peaks) + 1, nucleus=Nucleus.C13, shift_ppm=s))
    return PeakTable(peaks=peaks, solvent=solvent)


random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 11)
sample = random.sample(pool, min(int(sys.argv[1]) if len(sys.argv) > 1 else 300, len(pool)))
stat = collections.Counter()
fails = collections.Counter()
hard_ex = []
soft_ex = collections.defaultdict(list)
v08t = collections.Counter()
nonacc = []
dec = collections.Counter()
per_solv = collections.Counter()
per_solv_ok = collections.Counter()
for rid, smi, f, solv, hspec, cspec in sample:
    mol = Chem.MolFromSmiles(smi)
    try:
        t = table(mol, hspec, cspec, solv)
        ctx = SampleContext(molecular_formula=f, solvent=solv.value)
        v = verify_candidate(
            t,
            Candidate(rank=1, smiles=smi, rationale="truth", assignments=[], confidence=0.5),
            ctx,
            use_model_confidence=False,
        )
    except Exception as e:
        stat["error:" + type(e).__name__] += 1
        continue
    subs = [k for k, sm in SUBGROUPS.items() if mol.HasSubstructMatch(Chem.MolFromSmarts(sm))] + ["all"]
    for k in subs:
        v08t[(k, "truth", v08_status(t, ctx, smi))] += 1
    v08t[("V09 (all)", "truth", v08_status(t, ctx, smi, v09_peaks_explained))] += 1
    if v.status != "accepted":
        nonacc.append(
            (
                rid,
                v.status,
                smi,
                solv.value,
                [
                    r.rule_id.split("_")[0] + ":" + r.message[:90]
                    for r in v.rules
                    if r.status == RuleStatus.FAIL
                ],
            )
        )
    stat[v.status] += 1
    per_solv[solv.value] += 1
    per_solv_ok[solv.value] += v.status == "accepted"
    for r in v.rules:
        if r.status == RuleStatus.FAIL:
            fails[f"{r.rule_id.split('_')[0]}:{r.severity}"] += 1
            if r.severity == "hard":
                hard_ex.append((rid, smi, solv.value, r.rule_id, r.message[:140]))
            elif len(soft_ex[r.rule_id.split("_")[0]]) < 4:
                soft_ex[r.rule_id.split("_")[0]].append((rid, smi, solv.value, r.message[:150]))
    iso = [s for s in by_formula[f] if s != smi]
    if iso:
        decoy_smi = random.choice(iso)
        d = verify_candidate(
            t,
            Candidate(rank=2, smiles=decoy_smi, rationale="decoy", assignments=[], confidence=0.5),
            ctx,
            use_model_confidence=False,
        )
        for k in subs:
            v08t[(k, "decoy", v08_status(t, ctx, decoy_smi))] += 1
        v08t[("V09 (all)", "decoy", v08_status(t, ctx, decoy_smi, v09_peaks_explained))] += 1
        dec["decoys"] += 1
        dec["decoy_" + d.status] += 1
        dec[
            "truth>decoy"
            if v.verifier_score > d.verifier_score
            else "tie"
            if v.verifier_score == d.verifier_score
            else "decoy>truth"
        ] += 1
print("pool (same named solvent, filtered):", len(pool), "| sample:", len(sample))
print("TRUTH status:", dict(stat))
print("per solvent accepted/total:", {k: f"{per_solv_ok[k]}/{per_solv[k]}" for k in per_solv})
print("rule FAILs on truth:", dict(fails.most_common()))
print("DECOYS:", dict(dec))
print("\nHARD FAIL examples:")
[print("  ", x) for x in hard_ex[:12]]
print("\nSOFT FAIL examples by rule:")
for k, xs in soft_ex.items():
    for x in xs:
        print("  ", k, x)

print("\nV08 status by group (truth, and same-formula decoy verified against the truth's spectrum):")
for k in sorted({a for a, _, _ in v08t}):
    print("  ", k, {f"{w}:{st}": n for (a, w, st), n in sorted(v08t.items()) if a == k})
print("\nTRUTH NOT ACCEPTED (all):")
[print("  ", x) for x in sorted(nonacc)]
