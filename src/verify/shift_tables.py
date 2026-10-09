"""Literature chemical-shift rules (SPEC §8.1-8.3). RDKit-free so the reference endpoint stays light.

`lo`/`hi` is the union range, used for "can this peak be explained" and "is required evidence present"
checks (avoids false rejections). `confirm` is the narrower intersection, used only to confirm that a
diagnostic peak supports a claim (avoids false confirmations). `soft_only` ranges rest on thin sourcing and
must never be the sole reason to reject a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.schemas import Nucleus

SHIFT_TABLE_VERSION = "2026.10.2"  # 2026.10.2: ranges re-sourced from NMRShiftDB2 after the Phase 1 review

SOURCES: dict[str, str] = {
    "OS-H": "Organic Chemistry (OpenStax, McMurry) §13.4, Table 13.3",
    "SK": "Skoog et al., Principles of Instrumental Analysis, 7th ed., ch. 19 (Fig. 19-17 converted tau→delta; Fig. 19-29 visual)",
    "LT-H": "Chemistry LibreTexts, Organic Chemistry (Morsch et al.) §13.3",
    "OSU-H": "Oregon State University, K. Gable, CH 362 chemical shift pages",
    "UW-H": "University of Windsor, J. Green, 1H NMR chemical shifts table (2015)",
    "OS-C": "Organic Chemistry (OpenStax) §13.11",
    "HW-C": "Heriot-Watt University, 13C NMR spectroscopy handout",
    "PNA-C": "Process NMR Associates, Table 2 - 13C NMR",
    "FUL": "Fulmer et al., Organometallics 2010, 29, 2176, Tables 1-2",
    "UCSD": "UC San Diego CHEM 40 Table Appendix 2023-24 (after Gottlieb et al., J. Org. Chem. 1997, 62, 7512)",
    "NSDB2": "NMRShiftDB2 experimental spectra; distributions computed 2026-10-09 by bench/nmrshiftdb2_shift_distributions.py (n and percentiles in its output file)",
    "NONE": "No verified range; accepted as neutral (wide) so unsourced environments cannot cause false rejections",
}

Mode = Literal["explain", "diagnostic"]


@dataclass(frozen=True)
class ShiftRule:
    key: str
    nucleus: Nucleus
    description: str
    lo: float
    hi: float
    mode: Mode
    sources: tuple[str, ...]
    confirm: tuple[float, float] | None = None
    exchangeable: bool = False
    soft_only: bool = False

    def contains(self, ppm: float, slack: float = 0.0) -> bool:
        return self.lo - slack <= ppm <= self.hi + slack

    def confirms(self, ppm: float, slack: float = 0.0) -> bool:
        lo, hi = self.confirm if self.confirm else (self.lo, self.hi)
        return lo - slack <= ppm <= hi + slack


def _h(key, desc, lo, hi, mode, sources, confirm=None, exchangeable=False) -> ShiftRule:
    return ShiftRule(key, Nucleus.H1, desc, lo, hi, mode, tuple(sources), confirm, exchangeable)


def _c(key, desc, lo, hi, mode, sources, soft_only=False, confirm=None) -> ShiftRule:
    return ShiftRule(key, Nucleus.C13, desc, lo, hi, mode, tuple(sources), confirm, False, soft_only)


_H_RULES = [
    _h("h_alkyl_ch3", "R-CH3 (saturated, no alpha-heteroatom)", 0.7, 1.3, "explain", ["OS-H", "UW-H", "OSU-H"]),
    _h("h_alkyl_ch2", "R-CH2-R", 1.2, 1.7, "explain", ["OS-H", "UW-H", "LT-H"]),
    _h("h_alkyl_ch", "R3CH", 1.4, 2.0, "explain", ["OS-H", "UW-H", "LT-H"]),
    _h("h_allylic", "C=C-CH", 1.6, 2.4, "explain", ["OS-H", "UW-H"]),
    _h("h_propargylic", "C#C-CH", 1.7, 2.8, "explain", ["UW-H"]),
    _h("h_benzylic", "Ar-CH", 2.2, 2.8, "explain", ["OS-H", "UW-H"]),
    _h("h_alpha_carbonyl", "CH alpha to C=O (ketone, aldehyde, acid, ester)", 1.8, 3.4, "explain", ["OS-H", "UW-H", "OSU-H", "NSDB2"]),
    _h("h_alpha_amine", "CH-NR2", 2.2, 3.7, "explain", ["UW-H", "NSDB2"]),
    _h("h_n_alkyl_amide", "CH-N-C(=O)R", 2.9, 3.9, "explain", ["UW-H"]),
    _h("h_alkyne_terminal", "R-C#C-H", 1.7, 3.5, "diagnostic", ["SK", "UW-H", "OS-H", "OSU-H", "NSDB2"], confirm=(2.3, 3.3)),
    # Protons on an sp3 carbon beta to Cl/Br/I/O/N/NO2: union of the per-class 5th-95th percentiles of experimental shifts
    # (n = 63-4790 per class), rounded outward. CH3-C-X also agrees with SK (1.0-2.0 outer limits).
    _h("h_beta_ch3", "CH3 beta to a heteroatom (CH3-C-X)", 0.9, 2.5, "explain", ["SK", "NSDB2"]),
    _h("h_beta_ch2", "CH2 beta to a heteroatom", 1.2, 3.7, "explain", ["NSDB2"]),
    _h("h_beta_ch", "CH beta to a heteroatom", 1.3, 3.7, "explain", ["NSDB2"]),
    _h("h_alpha_halide", "CH-X (Cl, Br, I)", 2.2, 4.2, "explain", ["UW-H", "OS-H"]),
    _h("h_alpha_oxygen", "CH-OH, CH-OR", 3.2, 4.5, "explain", ["OS-H", "UW-H"]),
    _h("h_methoxy", "O-CH3", 3.2, 4.0, "diagnostic", ["UW-H", "LT-H"]),
    _h("h_ester_och", "R-C(=O)O-CH", 3.6, 5.0, "explain", ["UW-H"]),
    _h("h_aryl_ether", "Ar-O-CH", 3.9, 4.5, "explain", ["UW-H"]),
    _h("h_vinyl", "C=C-H", 4.5, 8.1, "diagnostic", ["OS-H", "SK", "UW-H", "OSU-H", "NSDB2"], confirm=(4.7, 6.5)),
    _h("h_aromatic", "Ar-H (carbocyclic)", 6.0, 9.0, "diagnostic", ["OS-H", "SK", "UW-H", "OSU-H", "NSDB2"], confirm=(6.8, 8.0)),
    _h("h_heteroaromatic", "pyridine/furan/thiophene C-H", 6.0, 9.0, "explain", ["UW-H", "UCSD"]),
    _h("h_aldehyde", "R-CH=O", 9.0, 10.1, "diagnostic", ["OS-H", "SK", "UW-H", "OSU-H"], confirm=(9.7, 10.0)),
    _h("h_carboxylic_acid", "R-COOH (broad)", 10.0, 14.0, "diagnostic", ["OS-H", "SK", "UW-H", "OSU-H", "LT-H"], confirm=(11.0, 12.0), exchangeable=True),
    _h("h_alcohol_oh", "R-OH (variable)", 0.5, 5.5, "explain", ["UW-H", "OS-H", "LT-H"], exchangeable=True),
    _h("h_amine_nh", "R-NH (variable)", 0.5, 5.0, "explain", ["UW-H", "LT-H", "OSU-H"], exchangeable=True),
    _h("h_amide_nh", "R-C(=O)N-H", 5.0, 8.5, "explain", ["UW-H"], exchangeable=True),
    _h("h_phenol_oh", "Ar-OH (free or associated 4-8; H-bonded 10.5-12.5)", 4.0, 12.5, "explain", ["SK", "LT-H"], exchangeable=True),
    _h("h_unclassified", "environment with no verified range (wide, neutral)", 0.0, 12.0, "explain", ["NONE"]),
]  # fmt: skip

_C_RULES = [
    _c("c_sp3_any", "any sp3 carbon", 0, 90, "explain", ["OS-C", "HW-C"]),
    _c("c_alkyl", "sp3 C without heteroatom", 0, 50, "explain", ["HW-C", "PNA-C", "FUL"]),
    _c("c_alpha_oxygen", "C-O (alcohol, ether, ester O-C)", 50, 90, "explain", ["HW-C", "OS-C", "FUL", "UCSD"]),
    _c("c_alpha_nitrogen", "C-N (amine, amide N-C)", 25, 70, "explain", ["SK", "OS-C", "UCSD"], soft_only=True),
    _c("c_alpha_halide", "C-Cl, C-Br", 20, 75, "explain", ["FUL", "OS-C"], soft_only=True),
    _c("c_alkyne", "C#C", 65, 100, "explain", ["SK", "OS-C", "HW-C"], soft_only=True),
    _c("c_nitrile", "C#N", 112, 125, "diagnostic", ["UCSD", "FUL", "OS-C"]),
    _c("c_sp2_any", "alkene + aromatic", 100, 160, "explain", ["HW-C", "OS-C"]),
    _c("c_aromatic_ch", "protonated aromatic", 110, 135, "explain", ["PNA-C", "FUL", "UCSD"]),
    _c("c_aromatic_quat", "substituted/ipso aromatic, incl. C-O/C-N aromatic", 125, 165, "explain", ["PNA-C"]),
    _c("c_carbonyl_any", "any C=O", 160, 220, "diagnostic", ["OS-C", "HW-C"]),
    _c("c_ester_amide", "ester / amide / acid C=O", 160, 185, "diagnostic", ["PNA-C", "UCSD", "FUL"]),
    _c("c_aldehyde", "R-CH=O", 190, 208, "diagnostic", ["PNA-C"]),
    _c("c_ketone", "R-C(=O)-R", 175, 220, "diagnostic", ["PNA-C", "FUL", "UCSD", "NSDB2"], confirm=(195, 220)),
]  # fmt: skip

H_RULES: dict[str, ShiftRule] = {r.key: r for r in _H_RULES}
C_RULES: dict[str, ShiftRule] = {r.key: r for r in _C_RULES}
ALL_RULES: dict[str, ShiftRule] = {**H_RULES, **C_RULES}

# Keys of exchangeable-proton rules, used when a peak may be an OH/NH/COOH signal.
EXCHANGEABLE_H_KEYS: tuple[str, ...] = tuple(k for k, r in H_RULES.items() if r.exchangeable)

# Extra slack (ppm) when the axis was not referenced (SPEC §5.5 rule 5), and the base slack that absorbs
# rounding in the supplied peak list.
BASE_SLACK = {Nucleus.H1: 0.05, Nucleus.C13: 0.5}
UNREFERENCED_EXTRA_SLACK = {Nucleus.H1: 0.1, Nucleus.C13: 1.0}
# DMSO-d6: any broad 1H signal in this window may be an exchangeable proton. DEVIATION: SPEC §8.2 says 4-11,
# but its own sourcing for that range is an unverified remark, and indole/imide N-H sit at about 11.1 in DMSO-d6.
DMSO_EXCHANGEABLE_WINDOW = (4.0, 12.0)


def slack_for(nucleus: Nucleus, referencing: str) -> float:
    extra = UNREFERENCED_EXTRA_SLACK[nucleus] if referencing == "uncorrected" else 0.0
    return BASE_SLACK[nucleus] + extra


def rule_as_dict(r: ShiftRule) -> dict:
    return {
        "key": r.key,
        "nucleus": r.nucleus.value,
        "description": r.description,
        "range_ppm": [r.lo, r.hi],
        "confirm_range_ppm": list(r.confirm) if r.confirm else None,
        "mode": r.mode,
        "exchangeable": r.exchangeable,
        "soft_only": r.soft_only,
        "sources": [{"id": s, "citation": SOURCES[s]} for s in r.sources],
    }
