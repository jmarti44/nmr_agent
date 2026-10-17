"""Verification rules V01-V11 (SPEC §8.6). Deterministic; RDKit only, no network.

Each rule returns one RuleResult. Deviations from the SPEC text are marked DEVIATION and covered by tests.
V12 (starting-material similarity) is not implemented in Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem

from src.schemas import (
    Candidate,
    FunctionalGroupClaim,
    Multiplicity,
    Nucleus,
    Peak,
    PeakRole,
    PeakTable,
    RuleResult,
    RuleStatus,
    SampleContext,
    Solvent,
    degree_of_unsaturation,
    parse_formula,
)
from src.verify.chem import (
    HEnv,
    ParsedMol,
    carbon_environment_count,
    formula_counts,
    h_environments,
    hill_equal,
    matches,
)
from src.verify.shift_tables import (
    ALL_RULES,
    DMSO_EXCHANGEABLE_WINDOW,
    SOURCES,
    ShiftRule,
    slack_for,
)
from src.verify.smarts import SPECS, Evidence

# Top of the plain-alkyl range for CH3 / CH2 / CH; the starting point for additive substituent effects.
_BASE_HI = {3: 1.3, 2: 1.7, 1: 2.0}


def env_hull(env: HEnv) -> tuple[float, float] | None:
    """Additive range for a carbon with two or more direct substituents (e.g. benzylic + alpha-amine).

    Each substituent adds the amount its own range exceeds the plain-alkyl top (hi_k - base), so a single
    substituent gives exactly its own range, and two give the sum. The lower bound is the smallest lo.
    """
    if len(env.mods) < 2:
        return None
    base = _BASE_HI.get(env.n_h, 2.0)
    lo = min(ALL_RULES[k].lo for k in env.mods)
    hi = base + sum(max(0.0, ALL_RULES[k].hi - base) for k in env.mods)
    return lo, hi


_NON_ANALYTE = (PeakRole.SOLVENT, PeakRole.WATER, PeakRole.REFERENCE, PeakRole.IMPURITY_SUSPECT)
_EXCHANGING_SOLVENTS = (Solvent.D2O, Solvent.CD3OD)


@dataclass
class Ctx:
    """Everything a rule needs about one candidate."""

    table: PeakTable
    context: SampleContext
    parsed: ParsedMol
    candidate: Candidate
    claims: list[FunctionalGroupClaim] | None = None
    claims_note: str | None = None  # why claims were not applied to this candidate
    hypothesis_dou: int | None = None
    envs: list[HEnv] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.envs = h_environments(self.parsed.mol)
        self.slack_h = slack_for(Nucleus.H1, self.table.referencing)
        self.slack_c = slack_for(Nucleus.C13, self.table.referencing)
        by_id = {p.id: p for p in self.table.peaks}
        self.by_id = by_id
        self.h_peaks = self.table.analyte_peaks(Nucleus.H1)
        self.c_peaks = self.table.analyte_peaks(Nucleus.C13)
        imp = PeakRole.IMPURITY_SUSPECT
        self.h_ext = [
            p for p in self.table.peaks if p.nucleus == Nucleus.H1 and p.role in (PeakRole.ANALYTE, imp)
        ]
        self.c_ext = [
            p for p in self.table.peaks if p.nucleus == Nucleus.C13 and p.role in (PeakRole.ANALYTE, imp)
        ]

    def h_in_ranges(self, p: Peak, keys: tuple[str, ...]) -> bool:
        """Is a 1H peak in one of the rules, or in the additive range of an environment that uses one?"""
        if any(ALL_RULES[k].contains(p.shift_ppm, self.slack_h) for k in keys):
            return True
        for env in self.envs:
            hull = None if env.exchangeable or not any(k in env.mods for k in keys) else env_hull(env)
            if hull and hull[0] - self.slack_h <= p.shift_ppm <= hull[1] + self.slack_h:
                return True
        return False

    @property
    def solvent(self) -> Solvent | None:
        return self.context.solvent or self.table.solvent

    @property
    def exchanging(self) -> bool:
        return self.context.d2o_exchange_performed or self.solvent in _EXCHANGING_SOLVENTS

    @property
    def exchangeable_h(self) -> int:
        return sum(e.n_h for e in self.envs if e.exchangeable)

    @property
    def total_h(self) -> int:
        return formula_counts(self.parsed.mol).get("H", 0)


def _res(rule_id, status, severity, message, evidence=None, source=None) -> RuleResult:
    return RuleResult(
        rule_id=rule_id,
        status=status,
        severity=severity,
        message=message,
        evidence=evidence or {},
        source=source,
    )


def _cite(*ids: str) -> str:
    return "; ".join(SOURCES[i] for i in ids)


# --------------------------------------------------------------------------- V02-V04


def v02_formula_match(c: Ctx) -> RuleResult:
    rid = "V02_formula_match"
    if not c.context.molecular_formula:
        return _res(rid, RuleStatus.SKIPPED, "hard", "no molecular formula supplied")
    try:
        got = parse_formula(c.parsed.formula)
    except ValueError as e:
        return _res(rid, RuleStatus.FAIL, "hard", f"candidate formula not comparable: {e}")
    want = c.context.formula_counts or {}
    ok = hill_equal(got, want)
    msg = f"{c.parsed.formula} {'==' if ok else '!='} {c.context.molecular_formula}"
    return _res(rid, RuleStatus.PASS if ok else RuleStatus.FAIL, "hard", msg)


def v03_dou_consistency(c: Ctx) -> RuleResult:
    rid = "V03_dou_consistency"
    structural = c.parsed.dou
    if c.context.molecular_formula:
        expected, severity, basis = degree_of_unsaturation(c.context.formula_counts or {}), "hard", "formula"
    elif c.hypothesis_dou is not None:
        expected, severity, basis = float(c.hypothesis_dou), "soft", "model-reported"
    else:
        return _res(rid, RuleStatus.SKIPPED, "soft", "no formula or model DoU to compare against")
    ok = abs(structural - expected) < 1e-6
    return _res(
        rid,
        RuleStatus.PASS if ok else RuleStatus.FAIL,
        severity,
        f"structure DoU {structural} {'==' if ok else '!='} {basis} DoU {expected:g}",
        {"structural": structural, "expected": expected, "basis": basis},
    )


def v04_exact_mass(c: Ctx) -> RuleResult:
    rid = "V04_exact_mass"
    obs = c.context.observed_exact_mass
    if obs is None:
        return _res(rid, RuleStatus.SKIPPED, "soft", "no observed exact mass supplied")
    d = abs(c.parsed.exact_mass - obs)
    ok = d <= 0.003
    return _res(
        rid,
        RuleStatus.PASS if ok else RuleStatus.FAIL,
        "soft",
        f"calc {c.parsed.exact_mass:.4f} vs obs {obs:.4f} (delta {d:.4f} Da, limit 0.003)",
    )


# --------------------------------------------------------------------------- V05-V07


def v05_proton_accounting(c: Ctx) -> RuleResult:
    rid = "V05_proton_accounting"
    if not c.h_peaks:
        return _res(rid, RuleStatus.SKIPPED, "soft", "no analyte 1H peaks")
    if any(p.integral is None for p in c.h_peaks):
        return _res(rid, RuleStatus.SKIPPED, "soft", "not every analyte 1H peak carries an integral")
    h = c.total_h  # candidate H count; equals the supplied formula's H when V02 passed
    e = c.exchangeable_h
    tol = max(0.5, min(1.0, 0.1 * h))
    s_a = sum(p.integral for p in c.h_peaks)
    s_ai = sum(p.integral for p in c.h_ext if p.integral is not None)
    # Exchangeable protons may be unobserved (D2O, broad); impurity-tagged peaks may be real analyte signal.
    ok = any(h - e - tol <= s <= h + tol for s in (s_a, s_ai))
    severity = "hard" if c.context.molecular_formula else "soft"
    ev = {"candidate_H": h, "exchangeable_H": e, "integral_sum": round(s_a, 2), "tolerance": tol}
    if not ok:
        return _res(
            rid,
            RuleStatus.FAIL,
            severity,
            f"integrals sum to {s_a:g}; candidate has {h} H ({e} exchangeable), "
            f"allowed [{h - e - tol:g}, {h + tol:g}]",
            ev,
        )
    assigned = sum(a.proton_count for a in c.candidate.assignments)
    if c.candidate.assignments and not (h - e <= assigned <= h):
        return _res(
            rid,
            RuleStatus.WARN,
            severity,
            f"integrals agree but model assignments account for {assigned} H of {h}",
            {**ev, "assigned": assigned},
        )
    return _res(rid, RuleStatus.PASS, severity, f"integrals sum to {s_a:g}; candidate has {h} H", ev)


def v06_c13_environment_count(c: Ctx) -> RuleResult:
    rid = "V06_c13_environment_count"
    if not c.c_peaks:
        return _res(rid, RuleStatus.SKIPPED, "soft", "no analyte 13C peaks")
    u = carbon_environment_count(c.parsed.mol)
    counts = {len(c.c_peaks), len(c.c_ext)}
    ok = any(u - 2 <= n <= u + 1 for n in counts)
    return _res(
        rid,
        RuleStatus.PASS if ok else RuleStatus.FAIL,
        "soft",
        f"{len(c.c_peaks)} observed 13C signals vs {u} distinct carbons (allowed {u - 2}..{u + 1})",
        {"observed": len(c.c_peaks), "unique_carbons": u},
    )


def v07_h_environment_count(c: Ctx) -> RuleResult:
    """Fail only when there are clearly more signals than environments; fewer is normal (overlap)."""
    rid = "V07_h_environment_count"
    obs_a = [p for p in c.h_peaks if not p.exchangeable_hint]
    obs_ai = [p for p in c.h_ext if not p.exchangeable_hint]
    if not obs_a:
        return _res(rid, RuleStatus.SKIPPED, "soft", "no non-exchangeable analyte 1H peaks")
    env = len({e.rank for e in c.envs if not e.exchangeable})
    # In a chiral or ring molecule the two protons of a CH2 can be diastereotopic: two signals.
    mol = c.parsed.mol
    stereo = bool(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False))
    diastereotopic = (
        len({e.rank for e in c.envs if e.n_h == 2 and not e.exchangeable})
        if (stereo or mol.GetRingInfo().NumRings())
        else 0
    )
    n = min((len(obs_a), len(obs_ai)), key=lambda x: abs(x - env))  # impurity-tagged peaks count optionally
    ev = {"observed": len(obs_a), "environments": env, "diastereotopic_allowance": diastereotopic}
    base = f"{len(obs_a)} non-exchangeable 1H signals vs {env} distinct H environments"
    if n > env + diastereotopic + 2:
        return _res(rid, RuleStatus.FAIL, "soft", f"{base}: more than {2 + diastereotopic} extra signals", ev)
    if n < env - 2:
        return _res(
            rid, RuleStatus.PASS, "soft", f"{base}: fewer signals than environments (overlap is normal)", ev
        )
    return _res(rid, RuleStatus.PASS, "soft", f"{base} (allowed up to +2)", ev)


# --------------------------------------------------------------------------- V08


def _in_rule(peaks: list[Peak], keys: tuple[str, ...], slack: float) -> list[Peak]:
    rules: list[ShiftRule] = [ALL_RULES[k] for k in keys]
    return [p for p in peaks if any(r.contains(p.shift_ppm, slack) for r in rules)]


def _evidence_applicable(c: Ctx, ev: Evidence) -> bool:
    if ev.applies_if and not matches(c.parsed.mol, ev.applies_if):
        return False
    return not (ev.unless_exchanged and c.exchanging)


def v08_structure_demands_peaks(c: Ctx) -> RuleResult:
    rid = "V08_structure_demands_peaks"
    details: dict[str, dict] = {}
    hard: list[str] = []
    soft: list[str] = []
    weak: list[str] = []  # missing soft-only 13C evidence, or evidence that rests on impurity-tagged peaks
    for group, spec in SPECS.items():
        if not any(matches(c.parsed.mol, s) for s in spec.smarts):
            continue
        info: dict = {}
        for ev in spec.h:
            if not _evidence_applicable(c, ev):
                continue
            found = [p for p in c.h_ext if c.h_in_ranges(p, ev.keys)]
            if not found:
                is_hard = (
                    bool(c.context.molecular_formula)
                    and c.solvent is not None
                    and c.solvent.value in ev.hard_in
                )
                info["1H"] = {"missing": list(ev.keys)}
                msg = f"{group.value}: no 1H peak in {'/'.join(ev.keys)}"
                (weak if ev.weak else hard if is_hard else soft).append(msg)
            else:
                info["1H"] = {"peak_ids": [p.id for p in found]}
                if all(p.role == PeakRole.IMPURITY_SUSPECT for p in found):
                    weak.append(f"{group.value}: 1H evidence only from impurity-tagged peaks")
        if c.c_peaks:
            for ev in spec.c:
                if ev.applies_if and not matches(c.parsed.mol, ev.applies_if):
                    continue
                found = _in_rule(c.c_ext, ev.keys, c.slack_c)
                rule = ALL_RULES[ev.keys[0]]
                if not found:
                    info["13C"] = {"missing": list(ev.keys)}
                    msg = f"{group.value}: no 13C peak in {'/'.join(ev.keys)}"
                    (weak if rule.soft_only else soft).append(msg)
                else:
                    info["13C"] = {"peak_ids": [p.id for p in found]}
        details[group.value] = info
    src = _cite("OS-H", "UW-H", "PNA-C", "OS-C")
    if hard:
        return _res(rid, RuleStatus.FAIL, "hard", "; ".join(hard + soft), details, src)
    if soft:
        return _res(rid, RuleStatus.FAIL, "soft", "; ".join(soft + weak), details, src)
    if weak:
        return _res(rid, RuleStatus.WARN, "soft", "; ".join(weak), details, src)
    return _res(
        rid,
        RuleStatus.PASS,
        "soft",
        f"required evidence present for {len(details)} matched group(s)",
        details,
        src,
    )


# --------------------------------------------------------------------------- V09


def _explained_specifically(c: Ctx, p: Peak) -> bool:
    """Explained by an environment with a sourced range (everything except the h_unclassified catch-all)."""
    exch_envs = [e for e in c.envs if e.exchangeable]
    exch_rules = [ALL_RULES[k] for e in exch_envs for k in e.keys]
    if exch_envs and any(r.contains(p.shift_ppm, c.slack_h) for r in exch_rules):
        return True
    lo, hi = DMSO_EXCHANGEABLE_WINDOW
    if (
        exch_envs
        and c.solvent == Solvent.DMSO_D6
        and (p.exchangeable_hint or p.multiplicity == Multiplicity.BR)
        and lo <= p.shift_ppm <= hi
    ):
        return True
    if p.exchangeable_hint:
        return False  # flagged exchangeable: only OH/NH/COOH environments may explain it
    for env in c.envs:
        if env.exchangeable:
            continue
        keys = tuple(k for k in env.keys if k != "h_unclassified")
        if keys and c.h_in_ranges(p, keys):
            return True
    return False


def v09_peaks_explained(c: Ctx) -> RuleResult:
    rid = "V09_peaks_explained_by_structure"
    if not c.h_peaks:
        return _res(rid, RuleStatus.SKIPPED, "soft", "no analyte 1H peaks")
    # The catch-all range (0-12 ppm) stands in for environments with no sourced range (S, F, small rings...).
    # It may explain at most one peak per such environment, or one S or F atom would explain any spectrum.
    slots = len({e.rank for e in c.envs if not e.exchangeable and "h_unclassified" in e.keys})
    unexplained: list[int] = []
    by_catch_all: list[int] = []
    for p in c.h_peaks:
        if _explained_specifically(c, p):
            continue
        if (
            slots > 0
            and not p.exchangeable_hint
            and ALL_RULES["h_unclassified"].contains(p.shift_ppm, c.slack_h)
        ):
            slots -= 1
            by_catch_all.append(p.id)
        else:
            unexplained.append(p.id)
    tainted: list[int] = []
    for cl in c.claims or []:
        for pid in cl.evidence.peak_ids:
            p = c.by_id.get(pid)
            if p is not None and p.role in _NON_ANALYTE and pid not in tainted:
                tainted.append(pid)
    src = _cite("OS-H", "UW-H", "SK", "NSDB2")
    ev = {
        "unexplained_peak_ids": unexplained,
        "tainted_evidence_peak_ids": tainted,
        "catch_all_peak_ids": by_catch_all,
    }
    if unexplained:
        return _res(
            rid, RuleStatus.FAIL, "soft", f"1H peaks not explained by the structure: {unexplained}", ev, src
        )
    if tainted:
        return _res(
            rid, RuleStatus.WARN, "soft", f"claims cite solvent/impurity/reference peaks: {tainted}", ev, src
        )
    return _res(
        rid,
        RuleStatus.PASS,
        "soft",
        "every analyte 1H peak falls in a range for an environment present",
        ev,
        src,
    )


# --------------------------------------------------------------------------- V10


def v10_claimed_groups(c: Ctx) -> RuleResult:
    rid = "V10_claimed_groups_consistent"
    if c.claims is None:
        return _res(rid, RuleStatus.SKIPPED, "soft", c.claims_note or "no model claims to check")
    problems: list[str] = []
    for cl in c.claims:
        spec = SPECS[cl.group]
        present = any(matches(c.parsed.mol, s) for s in spec.smarts)
        if not cl.present:
            if present:
                problems.append(f"{cl.group.value}: claimed absent but present in structure")
            continue
        if not present:
            problems.append(f"{cl.group.value}: claimed present but absent from structure")
            continue
        cited = [
            c.by_id[i] for i in cl.evidence.peak_ids if i in c.by_id and c.by_id[i].role == PeakRole.ANALYTE
        ]
        c_applicable = [e for e in spec.c if not e.applies_if or matches(c.parsed.mol, e.applies_if)]
        applicable = [e for e in spec.h if _evidence_applicable(c, e)] + c_applicable
        if not applicable:
            continue
        hit = False
        for ev in spec.h:
            if _evidence_applicable(c, ev) and _in_rule(
                [p for p in cited if p.nucleus == Nucleus.H1], ev.keys, c.slack_h
            ):
                hit = True
        for ev in c_applicable:
            if _in_rule([p for p in cited if p.nucleus == Nucleus.C13], ev.keys, c.slack_c):
                hit = True
        if not hit:
            problems.append(
                f"{cl.group.value}: cited peaks {cl.evidence.peak_ids} not in a required shift range"
            )
    if problems:
        return _res(
            rid, RuleStatus.FAIL, "soft", "; ".join(problems), {"problems": problems}, _cite("OS-H", "PNA-C")
        )
    return _res(
        rid,
        RuleStatus.PASS,
        "soft",
        f"{len(c.claims)} claim(s) consistent with the structure and cited peaks",
    )


RULES_AFTER_PARSE = (
    v02_formula_match,
    v03_dou_consistency,
    v04_exact_mass,
    v05_proton_accounting,
    v06_c13_environment_count,
    v07_h_environment_count,
    v08_structure_demands_peaks,
    v09_peaks_explained,
    v10_claimed_groups,
)
