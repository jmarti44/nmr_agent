# ruff: noqa: E501
"""NMRShiftDB2 pilot benchmark: peak list + formula + solvent in, is the true structure among the candidates?

    PYTHONPATH=. uv run python bench/benchmark.py select 20          # choose molecules, write the selection
    # all commands take --set pilot|main|unreported (default pilot): main = 50 molecules, unreported = 20 with
    # no solvent given; each set keeps its own selection and results in bench/results[_main|_unreported]
    PYTHONPATH=. uv run python bench/benchmark.py dry                # build every request, no model call, no cost
    set -a; . ./.env; set +a
    PYTHONPATH=. uv run python bench/benchmark.py run [--cap 3.0] [--retry-errors]    # call the model (resumes; stops at the cap)
    PYTHONPATH=. uv run python bench/benchmark.py rescore            # re-verify saved answers, no model calls
    PYTHONPATH=. uv run python bench/benchmark.py report

Input given to the service: 1H and 13C shifts of one experimental spectrum pair (same solvent), the formula
and the solvent. NOT given: multiplicity, exchangeable flags, spectrometer frequency or the structure. 1H
integrals are given only where the database assignment is complete for the 1H carbons.
The selection is written before any model call and is never changed afterwards.
"""

from __future__ import annotations

import collections
import json
import random
import re
import sys
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

from src.schemas import AnalysisRequest, Nucleus, Peak, PeakListInput, SampleContext, Solvent

RDLogger.DisableLog("rdApp.*")
SD = "bench/data/nmrshiftdb2/nmrshiftdb2withsignals.sd"
SELECTION = Path("bench/results/selection.json")  # the pilot; a larger run uses --selection to name its own
RESULTS = Path("bench/results")
SEED = 20261019
QUOTAS = {"CDCl3": 8, "DMSO-d6": 4, "D2O": 3, "CD3OD": 3, "other": 2}  # 20 molecules

TAG = re.compile(r"^> +<([^>]+)>\n(.*?)(?=\n\n|\Z)", re.M | re.S)
IDX = re.compile(r"(\d+):(.*?)(?=\s+\d+:|\s*$)", re.S)
SOLV = [
    ("Chloroform-D1", Solvent.CDCL3), ("CDCl3", Solvent.CDCL3),
    ("Dimethylsulphoxide-D6", Solvent.DMSO_D6), ("DMSO", Solvent.DMSO_D6),
    ("Deuteriumoxide", Solvent.D2O), ("Methanol-D4", Solvent.CD3OD),
    ("Acetone-D6", Solvent.ACETONE_D6), ("Benzene-D6", Solvent.C6D6), ("Acetonitrile-D3", Solvent.CD3CN),
]  # fmt: skip
ALLOWED = {"C", "H", "N", "O", "S", "F", "Cl", "Br"}
UNREPORTED = {"unreported", "unknown"}  # the database writes the word in the Solvent tag


def idx(v: str) -> dict[int, str]:
    return {int(i): s.strip() for i, s in IDX.findall(v.strip())}


def map_solv(s: str) -> Solvent | None:
    return next((v for k, v in SOLV if s.lower().startswith(k.lower())), None)


def records():
    buf: list[str] = []
    with open(SD, errors="replace") as fh:
        for line in fh:
            if line.startswith("$$$$"):
                yield "".join(buf)
                buf = []
            else:
                buf.append(line)


def pool(unreported: bool = False) -> list[dict]:
    """Records with a same-solvent experimental 1H + 13C pair. `unreported=True` instead takes pairs whose
    Solvent tag says Unreported or Unknown for both spectra (the robustness set: the solvent is then not given to the service)."""
    out = []
    for rec in records():
        end = rec.find("M  END")
        mol = Chem.MolFromMolBlock(rec[: end + 6])
        if mol is None or len(Chem.GetMolFrags(mol)) != 1 or Chem.GetFormalCharge(mol) != 0:
            continue
        if {a.GetSymbol() for a in mol.GetAtoms()} - ALLOWED or mol.GetNumAtoms() > 30:
            continue
        if Descriptors.MolWt(mol) > 450:
            continue
        tags = dict(TAG.findall(rec[end:]))
        pred = {i for t in ("Program", "NMRProgram") for i, v in idx(tags.get(t, "")).items() if v}
        solv = idx(tags.get("Solvent", ""))
        sp: dict[str, dict[int, str]] = {"1H": {}, "13C": {}}
        for k, v in tags.items():
            m = re.fullmatch(r"Spectrum (1H|13C) (\d+)", k)
            if m and int(m[2]) not in pred:  # experimental spectra only
                sp[m[1]][int(m[2])] = v
        if unreported:
            pairs = [
                (h, c)
                for h in sp["1H"]
                for c in sp["13C"]
                if solv.get(h, "").lower() in UNREPORTED and solv.get(c, "").lower() in UNREPORTED
            ]
        else:
            pairs = [
                (h, c) for h in sp["1H"] for c in sp["13C"]
                if map_solv(solv.get(h, "")) and map_solv(solv.get(h, "")) == map_solv(solv.get(c, ""))
            ]  # fmt: skip
        if pairs:
            h, c = pairs[0]
            out.append(
                {
                    "id": tags.get("nmrshiftdb2 ID", "?").strip(),
                    "smiles": Chem.MolToSmiles(mol),
                    "inchikey": Chem.MolToInchiKey(mol),
                    "formula": rdMolDescriptors.CalcMolFormula(mol),
                    "solvent": None if unreported else map_solv(solv[h]).value,
                    "h_spec": sp["1H"][h],
                    "c_spec": sp["13C"][c],
                }
            )
    return out


def distinct_carbons(smiles: str) -> int:
    from src.verify.chem import carbon_environment_count

    return carbon_environment_count(Chem.MolFromSmiles(smiles))


def c13_incomplete(r: dict) -> bool:
    """True if the 13C list has fewer signals than the structure's distinct carbons minus one (V06's slack).

    Uses only the database record, never a model output, so it can be fixed before any call (review P5-2).
    1H is not filtered the same way: only 113 of 828 records have a complete 1H assignment, so integrals are
    given or withheld per record instead (`h_integrals_given`).
    """
    n = len(
        {
            round(float(e.split(";")[0]), 1)
            for e in r["c_spec"].strip().strip("|").split("|")
            if e.split(";")[0]
        }
    )
    return n < distinct_carbons(r["smiles"]) - 1


def solvent_group(s: str | None) -> str:
    if s is None:
        return "unreported"
    return s if s in QUOTAS else "other"


MAIN_QUOTAS = {"CDCl3": 25, "D2O": 8, "DMSO-d6": 8, "CD3OD": 5, "other": 4}  # 50 molecules
DOU_BINS = ((0, 1), (2, 4), (5, 7), (8, 99))
MIN_PER_DOU_BIN = 5


def dou(formula: str) -> int:
    from src.schemas import degree_of_unsaturation

    return int(degree_of_unsaturation(SampleContext(molecular_formula=formula).formula_counts))


def dou_bin(formula: str) -> str:
    d = dou(formula)
    lo, hi = next(b for b in DOU_BINS if b[0] <= d <= b[1])
    return f"{lo}-{hi}" if hi < 99 else f"{lo}+"


def configure(set_name: str) -> None:
    """Point the script at one of the three sets: pilot (bench/results), main, unreported."""
    global SELECTION, RESULTS
    if set_name != "pilot":
        RESULTS = Path(f"bench/results_{set_name}")
        SELECTION = RESULTS / "selection.json"
        RESULTS.mkdir(parents=True, exist_ok=True)


def _unique_filtered(p: list[dict], exclude: set[str]) -> tuple[list[dict], int]:
    seen = set(exclude)
    uniq = []
    for r in p:  # one record per structure, never one already used in another set
        k = r["inchikey"].split("-")[0]
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    kept = [r for r in uniq if not c13_incomplete(r)]
    return kept, len(uniq) - len(kept)


def _used_keys() -> set[str]:
    used = set()
    for d in ("bench/results", "bench/results_main", "bench/results_unreported"):
        f = Path(d) / "selection.json"
        if f.exists():
            used |= {r["inchikey"].split("-")[0] for r in json.loads(f.read_text())["records"]}
    return used


def select(n: int, set_name: str = "pilot") -> None:
    if SELECTION.exists():
        sys.exit(f"{SELECTION} exists; the selection is fixed once written")
    rng = random.Random(SEED)
    uniq, removed = _unique_filtered(pool(unreported=set_name == "unreported"), _used_keys())
    print(f"13C-completeness filter removed {removed}; {len(uniq)} unique structures left")
    by = collections.defaultdict(list)
    for r in uniq:
        by[solvent_group(r["solvent"])].append(r)
    quotas = {"pilot": QUOTAS, "main": MAIN_QUOTAS, "unreported": {"unreported": n}}[set_name]
    if set_name == "main":
        # Redraw (fixed seeds) until every DoU bin that exists in the pool has at least MIN_PER_DOU_BIN molecules.
        avail = collections.Counter(dou_bin(r["formula"]) for r in uniq)
        for attempt in range(5000):
            rng = random.Random(SEED + attempt)
            chosen = []
            for group, k in quotas.items():
                g = list(by[group])
                rng.shuffle(g)
                chosen += g[:k]
            have = collections.Counter(dou_bin(r["formula"]) for r in chosen)
            if all(have[b] >= min(MIN_PER_DOU_BIN, avail[b]) for b in avail):
                print(
                    f"DoU bins satisfied on draw {attempt}: {dict(have)} (available in pool: {dict(avail)})"
                )
                break
        else:
            sys.exit("could not satisfy the DoU bins; relax them in the script, then record why")
    else:
        chosen = []
        for group, k in quotas.items():
            g = list(by[group])
            rng.shuffle(g)
            chosen += g[:k]
            if len(g) < k:
                print(f"warning: only {len(g)} records for {group}")
    chosen = chosen[:n]
    SELECTION.write_text(
        json.dumps(
            {
                "seed": SEED,
                "set": set_name,
                "quotas": quotas,
                "pool": len(uniq),
                "removed_by_13c_filter": removed,
                "records": chosen,
            },
            indent=1,
        )
    )
    print(f"selected {len(chosen)}: {dict(collections.Counter(solvent_group(r['solvent']) for r in chosen))}")


_ID_TAG = re.compile(r"> <nmrshiftdb2 ID>\n(\S+)")
_MOLBLOCKS: dict[str, dict[str, str]] = {}  # SD path -> {nmrshiftdb2 id: molblock text}; one read per path


class MolblockError(LookupError):
    """The record's molblock is missing or is not the record's structure. Deliberately not a ValueError: callers
    that treat ValueError as "the service rejects this request" (memory_gonogo.request_error) must not absorb it."""


def molblock_of(record_id: str) -> str | None:
    """The molblock text of one record. The SD file is read once per process (per path) for all records."""
    if SD not in _MOLBLOCKS:
        blocks: dict[str, str] = {}
        for rec in records():
            m = _ID_TAG.search(rec)
            if m and m[1] not in blocks:
                blocks[m[1]] = rec[: rec.find("M  END") + 6]
        _MOLBLOCKS[SD] = blocks
    return _MOLBLOCKS[SD].get(record_id)


def atom_index_map(block: str, mol: Chem.Mol) -> tuple[int | None, ...]:
    """File atom index -> atom index in ``mol`` (the H-removed molecule), for every atom of the molblock.

    The nmrshiftdb2 spectrum indices count the atoms of the SD file in file order, explicit H atoms included; RDKit
    removes those on reading, so every heavy atom after an H shifts down. An index naming a removed H atom maps to its
    single heavy neighbour; an index naming an atom RDKit kept maps to itself; None if the H has no heavy neighbour or
    several. Raises MolblockError unless the surviving atoms are exactly ``mol``'s atoms, element by element, in order."""
    raw = Chem.MolFromMolBlock(block, removeHs=False, sanitize=False)
    if raw is None:
        raise MolblockError("the molblock cannot be parsed")
    for a in raw.GetAtoms():
        a.SetIntProp("file_idx", a.GetIdx())
    try:
        kept = Chem.RemoveHs(Chem.Mol(raw))
    except Exception as e:  # noqa: BLE001 - RDKit raises several exception types
        raise MolblockError(f"the molblock's explicit H atoms cannot be removed: {e}") from e
    survivors = [a.GetIntProp("file_idx") for a in kept.GetAtoms()]
    if len(survivors) != mol.GetNumAtoms() or any(
        raw.GetAtomWithIdx(f).GetSymbol() != a.GetSymbol()
        for f, a in zip(survivors, mol.GetAtoms(), strict=True)
    ):
        raise MolblockError("the molblock's atoms do not correspond one to one to the H-removed molecule")
    new_of = {f: i for i, f in enumerate(survivors)}
    out: list[int | None] = []
    for a in raw.GetAtoms():
        if a.GetIdx() in new_of:
            out.append(new_of[a.GetIdx()])
            continue
        heavy = [n.GetIdx() for n in a.GetNeighbors() if n.GetIdx() in new_of]
        out.append(new_of[heavy[0]] if len(heavy) == 1 and a.GetDegree() == 1 else None)
    return tuple(out)


def molblock_mol_map(
    r: dict, mol: Chem.Mol | None = None, index_map: tuple[int | None, ...] | None = None
) -> tuple[Chem.Mol, tuple[int | None, ...] | None]:
    """The molecule the record's spectrum atom indices refer to (the H-removed SD molblock molecule, which pool()
    parses) and the file-index map onto it (see atom_index_map). Raises MolblockError if the record is not found, the
    molecule's canonical SMILES differs from ``r["smiles"]`` or the map cannot be verified; never falls back to the
    SMILES molecule. ``mol`` (and ``index_map``) may be passed by a caller that holds them; a passed ``mol`` without a
    map means "no explicit H atoms in the file" (indices are used as they are): None is returned for the map."""
    if mol is None:
        block = molblock_of(r["id"])
        if block is None:
            raise MolblockError(f"record {r['id']}: no molblock in {SD}")
        mol = Chem.MolFromMolBlock(block)
        if mol is not None and Chem.MolToSmiles(mol) == r["smiles"]:
            index_map = atom_index_map(block, mol)
    if mol is None or Chem.MolToSmiles(mol) != r["smiles"]:
        raise MolblockError(
            f"record {r['id']}: the molblock's structure does not match the stored SMILES {r['smiles']}"
        )
    return mol, index_map


def molblock_mol(r: dict, mol: Chem.Mol | None = None) -> Chem.Mol:
    """The H-removed molblock molecule of a record (see molblock_mol_map)."""
    return molblock_mol_map(r, mol)[0]


def request_for(
    r: dict, mol: Chem.Mol | None = None, index_map: tuple[int | None, ...] | None = None
) -> tuple[AnalysisRequest, bool]:
    """The request the service sees for one record. 1H integrals use the molblock molecule and the file-order index
    map (see molblock_mol_map); pass ``mol`` (and ``index_map``) to supply them instead of looking them up."""
    mol, index_map = molblock_mol_map(r, mol, index_map)

    def atom_of(k: int) -> int | None:
        if index_map is None:
            return k if k < mol.GetNumAtoms() else None
        return index_map[k] if k < len(index_map) else None

    hs: dict[float, set[int]] = collections.defaultdict(set)
    shifts_of: dict[int, set[float]] = collections.defaultdict(set)
    unmapped = False
    for e in r["h_spec"].strip().strip("|").split("|"):
        p = e.split(";")
        if len(p) >= 2 and p[0]:
            s = round(float(p[0]), 2)
            hs[s]
            if len(p) >= 3 and p[2].strip().isdigit():
                i = atom_of(int(p[2]))
                if i is None:
                    unmapped = True
                else:
                    hs[s].add(i)
                    shifts_of[i].add(s)
    need = {a.GetIdx() for a in mol.GetAtoms() if a.GetTotalNumHs() and a.GetSymbol() == "C"}
    # Complete: every H-bearing carbon has a 1H shift, and every 1H entry names an atom of the molecule that has at
    # least one attached H (an explicit H atom counts as its heavy neighbour). An entry that names no atom, an atom
    # that does not exist or an atom without H makes the assignment inconsistent: no integrals are given then, and
    # no observed peak is ever dropped.
    consistent = not unmapped and all(
        hs[s] and all(mol.GetAtomWithIdx(i).GetTotalNumHs() > 0 for i in hs[s]) for s in hs
    )
    complete = need <= set(shifts_of) and consistent
    peaks: list[Peak] = []
    for shift in sorted(hs, reverse=True):
        integral = None
        if complete:
            n_h = sum(mol.GetAtomWithIdx(i).GetTotalNumHs() / len(shifts_of[i]) for i in hs[shift])
            integral = round(float(n_h), 2)
        peaks.append(Peak(id=len(peaks) + 1, nucleus=Nucleus.H1, shift_ppm=shift, integral=integral))
    cs = sorted(
        {
            round(float(e.split(";")[0]), 1)
            for e in r["c_spec"].strip().strip("|").split("|")
            if e.split(";")[0]
        },
        reverse=True,
    )
    peaks += [Peak(id=len(peaks) + i, nucleus=Nucleus.C13, shift_ppm=s) for i, s in enumerate(cs, start=1)]
    ctx = SampleContext(molecular_formula=r["formula"], solvent=r["solvent"])
    req = AnalysisRequest(
        input_format="peaklist", peaks=PeakListInput(peaks=peaks), context=ctx, max_candidates=3
    )
    return req, complete


def load_selection() -> list[dict]:
    return json.loads(SELECTION.read_text())["records"]


def dry() -> None:
    from src.prompting import prepare_request

    tot = 0
    for r in load_selection():
        req, complete = request_for(r)
        _, prompt, _ = prepare_request(req)
        tot += prompt.user_tokens_est
        print(f"{r['id']:>9} {str(r['solvent']):<8} {r['formula']:<12} 1H integrals {'yes' if complete else 'no ':<3} "
              f"{sum(p.nucleus == Nucleus.H1 for p in req.peaks.peaks):>2} H / "
              f"{sum(p.nucleus == Nucleus.C13 for p in req.peaks.peaks):>2} C peaks, user ~{prompt.user_tokens_est} tok")  # fmt: skip
    print("total user tokens (estimate):", tot)


def run(cap: float, retry_errors: bool = False) -> None:
    import anthropic

    from src.analysis import run_analysis
    from src.core.settings import Settings

    settings = Settings()
    client = anthropic.Anthropic(max_retries=0, timeout=settings.llm_call_timeout_s)
    spent = sum(json.loads(f.read_text()).get("cost_usd", 0) for f in RESULTS.glob("mol_*.json"))
    for r in load_selection():
        out = RESULTS / f"mol_{r['id']}.json"
        first_error = None
        if out.exists():
            previous = json.loads(out.read_text())
            if "error" not in previous or not retry_errors or previous.get("first_attempt_error"):
                continue
            first_error = previous[
                "error"
            ]  # kept on the rerun's row: the first attempt still counts as a miss
        if spent >= cap:
            print(f"spend cap ${cap} reached (${spent:.2f}); stopping")
            break
        req, complete = request_for(r)
        trace: dict = {}
        try:
            resp = run_analysis(req, settings, client, trace=trace)
        except Exception as e:  # noqa: BLE001 - record and carry on
            out.write_text(
                json.dumps({"id": r["id"], "error": f"{type(e).__name__}: {str(e)[:200]}", "cost_usd": 0})
            )
            print(r["id"], "ERROR", type(e).__name__)
            continue
        spent += resp.usage.estimated_cost_usd
        row = result_row(r, complete, resp, trace)
        if first_error:
            row["first_attempt_error"] = first_error
        out.write_text(json.dumps(row, indent=1))
        print(r["id"], [v.status for v in resp.candidates], f"${resp.usage.estimated_cost_usd:.3f}")


def result_row(r: dict, complete: bool, resp, trace: dict) -> dict:
    """Everything needed to re-verify offline: the table, context, each round's answer, all verified candidates."""
    cands = [
        {"smiles": v.candidate.smiles, "model_rank": v.candidate.rank, "status": v.status,
         "score": v.verifier_score, "confidence": v.final_confidence}
        for v in resp.candidates
    ]  # fmt: skip
    return {
        "id": r["id"], "truth": r["smiles"], "truth_inchikey": r["inchikey"], "solvent": r["solvent"],
        "formula": r["formula"], "h_integrals_given": complete, "candidates": cands,
        "cost_usd": resp.usage.estimated_cost_usd, "llm_calls": resp.usage.llm_calls,
        "llm_ms": resp.timing_ms["llm"], "warnings": resp.warnings, "trace": trace,
    }  # fmt: skip


def rescore() -> None:
    """Re-verify saved answers with the current verifier. No model calls. Pilot files saved before the trace
    existed hold only the three returned candidates: they are re-verified from those, ordered by score then the
    model's own rank (the model's confidence was not saved), which is an approximation."""
    from src.analysis import reverify
    from src.schemas import LLMHypothesis, PeakTable, VerifiedCandidate
    from src.verify.verifier import verify_candidates

    changed = 0
    for f in sorted(RESULTS.glob("mol_*.json")):
        row = json.loads(f.read_text())
        if "candidates" not in row:
            continue
        sel = next(r for r in load_selection() if r["id"] == row["id"])
        t = row.get("trace")
        if t:
            table, ctx = PeakTable.model_validate(t["table"]), SampleContext.model_validate(t["context"])
            first = LLMHypothesis.model_validate(t["first"])
            rep = LLMHypothesis.model_validate(t["repaired"]) if t["repaired"] else None
            ver: list[VerifiedCandidate] = reverify(table, ctx, first, rep)
        else:
            req, _ = request_for(sel)
            from src.processing.normalize import normalize_peaklist
            from src.schemas import Candidate

            table = normalize_peaklist(req.peaks, req.context)
            cands = [Candidate(rank=c["model_rank"], smiles=c["smiles"], rationale="saved", assignments=[], confidence=0.5)
                     for c in row["candidates"]]  # fmt: skip
            ver = verify_candidates(table, cands, req.context, use_model_confidence=False)
        new = [{"smiles": v.candidate.smiles, "model_rank": v.candidate.rank, "status": v.status,
                "score": v.verifier_score, "confidence": v.final_confidence} for v in ver][:3]  # fmt: skip
        if [(c["smiles"], c["status"], round(c["score"], 3)) for c in new] != [
            (c["smiles"], c["status"], round(c["score"], 3)) for c in row["candidates"]
        ]:
            changed += 1
        row["rescored"] = new
        row["rescored_all"] = [{"smiles": v.candidate.smiles, "model_rank": v.candidate.rank, "status": v.status}
                               for v in ver]  # fmt: skip
        f.write_text(json.dumps(row, indent=1))
    print(f"rescored; {changed} molecules differ from the stored statuses/scores/order")


def key(smiles: str) -> str | None:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(m).split("-")[0] if m else None  # connectivity layer: stereo is not scored


def _pos(smiles_in_order: list[str], truth: str) -> int | None:
    keys = [key(s) for s in smiles_in_order]
    return keys.index(truth) if truth in keys else None


def _orders(r: dict) -> dict[str, list[str]]:
    """Candidate SMILES in each ordering: the service's (as returned), the model's own, and the rescored one."""
    if r.get("trace"):
        cands = [c["candidate"] for c in r["trace"]["verified_untrimmed"]]
        model = [c["smiles"] for c in sorted(cands, key=lambda c: c["rank"])]
    else:
        model = [c["smiles"] for c in sorted(r["candidates"], key=lambda c: c["model_rank"])]
    out = {"service": [c["smiles"] for c in r["candidates"]], "model": model}
    if "rescored" in r:
        out["rescored"] = [c["smiles"] for c in r["rescored"]]
    return out


def report() -> None:
    """Every number comes from here. A molecule whose first attempt failed (timeout or API error) counts as a
    miss in every headline row; its rerun is reported separately."""
    rows = [json.loads(f.read_text()) for f in sorted(RESULTS.glob("mol_*.json"))]
    sel = {r["id"]: r for r in load_selection()}
    scored = [r for r in rows if "candidates" in r]
    failed_first = {r["id"] for r in rows if "error" in r} | {
        r["id"] for r in scored if r.get("first_attempt_error")
    }
    n = len(rows)

    def table(label: str, ids: set[str]) -> None:
        for name in ("service", "model", "rescored"):
            have = [r for r in scored if r["id"] in ids and name in _orders(r)]
            if len(have) < len([r for r in scored if r["id"] in ids]):
                continue
            t1 = t3 = 0
            for r in have:
                if r["id"] in failed_first:
                    continue
                pos = _pos(_orders(r)[name], r["truth_inchikey"].split("-")[0])
                t1 += pos == 0
                t3 += pos is not None and pos < 3
            print(f"  {label:<28} {name:<9} top-1 {t1}/{len(ids)}  top-3 {t3}/{len(ids)}")

    ids = {r["id"] for r in rows}
    print(f"molecules {n}; first attempt failed (counted as a miss): {sorted(failed_first) or 'none'}")
    table("all", ids)
    incomplete = {i for i in ids if c13_incomplete(sel[i])}
    print(f"13C-incomplete records in this selection: {sorted(incomplete) or 'none'}")
    if incomplete:
        table("excluding 13C-incomplete", ids - incomplete)
    print("by solvent (service order, top-1):", end=" ")
    by = collections.defaultdict(lambda: [0, 0])
    done = {r["id"]: r for r in scored}
    for i, rec in sel.items():  # every selected molecule is in the denominator; failures are misses
        g = solvent_group(rec["solvent"])
        by[g][1] += 1
        r = done.get(i)
        if r is not None and i not in failed_first:
            by[g][0] += _pos(_orders(r)["service"], r["truth_inchikey"].split("-")[0]) == 0
    print({k: f"{a}/{b}" for k, (a, b) in by.items()})
    print(f"cost ${sum(r['cost_usd'] for r in scored):.2f} (rerun molecules included), "
          f"mean LLM time {sum(r['llm_ms'] for r in scored) / max(len(scored), 1) / 1000:.0f} s, "
          f"max {max((r['llm_ms'] for r in scored), default=0) / 1000:.0f} s, repair rounds {sum(r['llm_calls'] > 1 for r in scored)}")  # fmt: skip
    for r in scored:
        if r["id"] in failed_first:
            print(f"  rerun after a failed first attempt: {r['id']} ({r.get('first_attempt_error')}), "
                  f"{r['llm_ms'] / 1000:.0f} s, ${r['cost_usd']:.3f}")  # fmt: skip
    verifier_section(rows, scored, failed_first)


def verifier_section(rows: list[dict], scored: list[dict], failed_first: set[str]) -> None:
    """What the verifier did (review P5R-2): where the misses come from, its effect on order, and how its
    status relates to correctness. "now" uses the saved traces re-verified by the current rules (`rescore`)."""

    def correct(r: dict, smi: str) -> bool:
        return key(smi) == r["truth_inchikey"].split("-")[0]

    print("verifier:")
    no_answer = [r["id"] for r in rows if "candidates" not in r]
    never = proposed = 0
    for r in scored:
        if r["id"] in failed_first:
            continue
        allc = [c["candidate"]["smiles"] for c in r["trace"]["verified_untrimmed"]] if r.get("trace") else [
            c["smiles"] for c in r["candidates"]]  # fmt: skip
        if not any(correct(r, s) for s in allc):
            never += 1
        elif _pos(_orders(r)["service"], r["truth_inchikey"].split("-")[0]) != 0:
            proposed += 1
    print(f"  misses: no answer {len(no_answer)} (+{len(failed_first) - len(no_answer)} recovered on rerun and "
          f"counted as a miss); never proposed {never}; proposed but not first {proposed}")  # fmt: skip
    for label, name in (("as run", "service"), ("now", "rescored")):
        eff = collections.Counter()
        status = collections.Counter()
        for r in scored:
            if name not in _orders(r) or r["id"] in failed_first:
                continue
            o = _orders(r)
            tk = r["truth_inchikey"].split("-")[0]
            m_top, v_top = _pos(o["model"], tk) == 0, _pos(o[name], tk) == 0
            eff["promoted" if v_top and not m_top else "demoted" if m_top and not v_top else "same"] += 1
            top = (r["candidates"] if name == "service" else r["rescored"])[0]
            status[(top["status"], v_top)] += 1
        if not eff:
            continue
        print(f"  ordering effect, {label}: correct first by verifier but not by model {eff['promoted']}; "
              f"correct first by model but not by verifier {eff['demoted']}")  # fmt: skip
        print(f"  top-1 status vs correct, {label}: "
              + ", ".join(f"{st} {sum(v for (s2, ok), v in status.items() if s2 == st and ok)}/"
                          f"{sum(v for (s2, _), v in status.items() if s2 == st)}"
                          for st in ("accepted", "flagged", "rejected")))  # fmt: skip
    for label, field in (("as run", None), ("now", "rescored_all")):
        tally: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])  # status -> [correct, wrong]
        for r in scored:
            if r["id"] in failed_first:
                continue
            if field is None and r.get("trace"):
                cands = [(c["candidate"]["smiles"], c["status"]) for c in r["trace"]["verified_untrimmed"]]
            elif field and r.get(field):
                cands = [(c["smiles"], c["status"]) for c in r[field]]
            else:
                continue
            for smi, st in cands:
                tally[st][0 if correct(r, smi) else 1] += 1
        print(f"  every candidate the model proposed, {label} (correct/wrong): "
              + ", ".join(f"{st} {a}/{b}" for st, (a, b) in sorted(tally.items())))  # fmt: skip


if __name__ == "__main__":
    cmd = sys.argv[1]
    configure(sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "pilot")
    if cmd == "select":
        select(int(sys.argv[2]), sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else "pilot")
    elif cmd == "dry":
        dry()
    elif cmd == "run":
        run(
            float(sys.argv[sys.argv.index("--cap") + 1]) if "--cap" in sys.argv else 3.0,
            "--retry-errors" in sys.argv,
        )
    elif cmd == "rescore":
        rescore()
    elif cmd == "report":
        report()
