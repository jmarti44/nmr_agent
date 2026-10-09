"""Residual-solvent, water and common-impurity tables (SPEC §8.5) and peak tagging (SPEC §5.5 step 4).

Tagged peaks are never deleted: the role is changed so the prompt and the verifier can treat them differently.
Impurity shifts are CDCl3 values (Fulmer / Gottlieb), so impurities are only tagged when the solvent is CDCl3.
"""

from __future__ import annotations

from src.schemas import Nucleus, Peak, PeakRole, Solvent

TOL_H = 0.05
TOL_C = 0.5

# solvent -> (residual 1H shifts, residual 13C shifts, water/HOD 1H shift). Source: UCSD (residual), FUL (water).
SOLVENT_TABLE: dict[Solvent, dict] = {
    Solvent.CDCL3: {"h": (7.27,), "c": (77.2,), "water": 1.56},
    Solvent.DMSO_D6: {"h": (2.50,), "c": (39.5,), "water": 3.33},
    Solvent.ACETONE_D6: {"h": (2.05,), "c": (206.7, 29.9), "water": 2.84},
    Solvent.CD3OD: {"h": (3.31,), "c": (49.2,), "water": 4.87},
    Solvent.D2O: {"h": (), "c": (), "water": 4.80},
    Solvent.C6D6: {"h": (7.16,), "c": (128.4,), "water": 0.40},
    Solvent.CD3CN: {"h": (1.94,), "c": (118.7, 1.4), "water": 2.13},
}

# CDCl3 1H impurities (Fulmer Table 1 / Gottlieb): name -> shifts.
IMPURITIES_H_CDCL3: dict[str, tuple[float, ...]] = {
    "acetone": (2.17,),
    "acetic acid": (2.10,),
    "ethyl acetate": (2.05, 4.12, 1.26),
    "methanol": (3.49,),
    "ethanol": (1.25, 3.72),
    "dichloromethane": (5.30,),
    "diethyl ether": (3.48, 1.21),
    "THF": (3.76, 1.85),
    "DMF": (8.02, 2.96, 2.88),
    "DMSO": (2.62,),
    "cyclohexane": (1.43,),
    "n-hexane": (0.88, 1.26),
    "toluene": (2.36, 7.17, 7.29),
    "silicone grease": (0.07,),
    "hydrocarbon grease": (0.86, 1.26),
}

# CDCl3 13C impurity anchors (Fulmer).
IMPURITIES_C_CDCL3: dict[str, tuple[float, ...]] = {
    "acetone": (207.07, 30.92),
    "acetic acid": (175.99, 20.81),
    "methanol": (50.41,),
    "dichloromethane": (53.52,),
    "cyclohexane": (26.94,),
    "n-hexane": (14.14,),
    "toluene": (21.46,),
    "ethyl acetate": (21.04,),
}

REFERENCE_H = (0.0,)  # TMS


def classify_peak(
    peak: Peak, solvent: Solvent | None, impurities: bool = True
) -> tuple[PeakRole, str | None]:
    """Role and label for one peak given the solvent. Peaks that already carry a non-analyte role keep it.

    impurities=False skips the common-impurity table (solvent, water and reference are still tagged).
    """
    if peak.role != PeakRole.ANALYTE:
        return peak.role, peak.label
    tol = TOL_H if peak.nucleus == Nucleus.H1 else TOL_C
    if peak.nucleus == Nucleus.H1 and abs(peak.shift_ppm - REFERENCE_H[0]) <= tol:
        return PeakRole.REFERENCE, "TMS"
    entry = SOLVENT_TABLE.get(solvent) if solvent else None
    if entry:
        key = "h" if peak.nucleus == Nucleus.H1 else "c"
        if peak.nucleus == Nucleus.H1 and abs(peak.shift_ppm - entry["water"]) <= tol:
            return PeakRole.WATER, f"{solvent.value} water/HOD"
        for ref in entry[key]:
            if abs(peak.shift_ppm - ref) <= tol:
                return PeakRole.SOLVENT, f"{solvent.value} residual"
    if impurities and solvent == Solvent.CDCL3:
        table = IMPURITIES_H_CDCL3 if peak.nucleus == Nucleus.H1 else IMPURITIES_C_CDCL3
        for name, shifts in table.items():
            if any(abs(peak.shift_ppm - s) <= tol for s in shifts):
                return PeakRole.IMPURITY_SUSPECT, f"possible {name}"
    return PeakRole.ANALYTE, peak.label


def tag_peaks(peaks: list[Peak], solvent: Solvent | None, impurities: bool = True) -> list[Peak]:
    """Return copies of the peaks with solvent/water/reference/impurity roles applied. Nothing is dropped."""
    out = []
    for p in peaks:
        role, label = classify_peak(p, solvent, impurities)
        out.append(
            p.model_copy(update={"role": role, "label": label}) if (role, label) != (p.role, p.label) else p
        )
    return out


def table_as_dict() -> dict:
    return {
        "residual_solvents": {
            s.value: {
                "residual_1h_ppm": list(v["h"]),
                "residual_13c_ppm": list(v["c"]),
                "water_1h_ppm": v["water"],
            }
            for s, v in SOLVENT_TABLE.items()
        },
        "impurities_1h_cdcl3_ppm": {k: list(v) for k, v in IMPURITIES_H_CDCL3.items()},
        "impurities_13c_cdcl3_ppm": {k: list(v) for k, v in IMPURITIES_C_CDCL3.items()},
        "tolerance_ppm": {"1H": TOL_H, "13C": TOL_C},
        "sources": ["UCSD", "FUL"],
        "note": "Impurity shifts are CDCl3 values and are only used when the solvent is CDCl3.",
    }
