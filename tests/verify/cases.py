"""Ground-truth structures with typical literature-style peak lists (illustrative values, not database data).

Used for the invariant that a true structure is never hard-rejected. Benchmark data from NMRShiftDB2
replaces these in Phase 5.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.schemas import Multiplicity, Nucleus, Peak, PeakTable, SampleContext, Solvent


@dataclass
class Case:
    name: str
    smiles: str
    formula: str
    solvent: str
    # (shift, multiplicity, integral, exchangeable_hint)
    h: list[tuple[float, str, float, bool]]
    c: list[float] = field(default_factory=list)

    def table(self, with_c: bool = True, **overrides) -> PeakTable:
        peaks: list[Peak] = []
        for shift, mult, integral, hint in self.h:
            peaks.append(
                Peak(
                    id=len(peaks) + 1,
                    nucleus=Nucleus.H1,
                    shift_ppm=shift,
                    multiplicity=Multiplicity(mult),
                    integral=integral,
                    exchangeable_hint=hint,
                )
            )
        if with_c:
            for shift in self.c:
                peaks.append(Peak(id=len(peaks) + 1, nucleus=Nucleus.C13, shift_ppm=shift))
        return PeakTable(peaks=peaks, solvent=Solvent(self.solvent), **overrides)

    def context(self, **kw) -> SampleContext:
        return SampleContext(molecular_formula=self.formula, solvent=self.solvent, **kw)


CASES: dict[str, Case] = {
    c.name: c
    for c in [
        Case("aspirin", "CC(=O)Oc1ccccc1C(=O)O", "C9H8O4", "DMSO-d6",
             [(12.95, "br", 1, True), (7.95, "dd", 1, False), (7.64, "td", 1, False), (7.37, "td", 1, False),
              (7.17, "dd", 1, False), (2.24, "s", 3, False)],
             [169.1, 166.3, 150.3, 134.2, 131.2, 126.1, 124.0, 123.8, 20.8]),
        Case("benzaldehyde", "O=Cc1ccccc1", "C7H6O", "CDCl3",
             [(10.00, "s", 1, False), (7.88, "d", 2, False), (7.62, "t", 1, False), (7.52, "t", 2, False)],
             [192.3, 136.4, 134.5, 129.8, 129.1]),
        Case("acetophenone", "CC(=O)c1ccccc1", "C8H8O", "CDCl3",
             [(7.96, "d", 2, False), (7.56, "t", 1, False), (7.46, "t", 2, False), (2.61, "s", 3, False)],
             [198.1, 137.1, 133.1, 128.5, 128.3, 26.6]),
        Case("ethyl_acetate", "CCOC(C)=O", "C4H8O2", "CDCl3",
             [(4.12, "q", 2, False), (2.05, "s", 3, False), (1.26, "t", 3, False)],
             [171.1, 60.4, 21.0, 14.2]),
        Case("n_methylacetamide", "CNC(C)=O", "C3H7NO", "CDCl3",
             [(6.30, "br", 1, True), (2.80, "d", 3, False), (2.00, "s", 3, False)],
             [171.0, 26.3, 23.1]),
        Case("pyridine", "c1ccncc1", "C5H5N", "CDCl3",
             [(8.60, "m", 2, False), (7.64, "m", 1, False), (7.27, "m", 2, False)],
             [149.9, 135.9, 123.8]),
        Case("1_hexyne", "C#CCCCC", "C6H10", "CDCl3",
             [(2.18, "td", 2, False), (1.93, "t", 1, False), (1.52, "m", 2, False), (1.46, "m", 2, False),
              (0.92, "t", 3, False)],
             [84.4, 68.2, 30.7, 21.9, 18.1, 13.5]),
        Case("benzonitrile", "N#Cc1ccccc1", "C7H5N", "CDCl3",
             [(7.64, "m", 3, False), (7.47, "t", 2, False)],
             [132.8, 132.1, 129.1, 118.8, 112.4]),
        Case("nitrobenzene", "O=[N+]([O-])c1ccccc1", "C6H5NO2", "CDCl3",
             [(8.23, "d", 2, False), (7.71, "t", 1, False), (7.56, "t", 2, False)],
             [148.3, 134.7, 129.4, 123.5]),
        Case("phenol_dmso", "Oc1ccccc1", "C6H6O", "DMSO-d6",
             [(9.35, "br", 1, True), (7.15, "t", 2, False), (6.75, "d", 2, False), (6.72, "t", 1, False)],
             [157.8, 129.4, 118.9, 115.4]),
        Case("indole_dmso", "c1ccc2[nH]ccc2c1", "C8H7N", "DMSO-d6",
             [(11.08, "br", 1, True), (7.52, "d", 1, False), (7.39, "d", 1, False), (7.34, "t", 1, False),
              (7.08, "t", 1, False), (6.98, "t", 1, False), (6.42, "m", 1, False)],
             [136.0, 127.7, 124.4, 121.0, 120.1, 119.1, 111.4, 101.4]),
        Case("benzenesulfonamide", "NS(=O)(=O)c1ccccc1", "C6H7NO2S", "DMSO-d6",
             [(7.82, "d", 2, False), (7.60, "m", 3, False), (7.35, "s", 2, True)],
             [144.2, 131.8, 129.1, 125.7]),
        Case("tert_butanol", "CC(C)(C)O", "C4H10O", "CDCl3",
             [(1.28, "s", 9, False), (1.60, "br", 1, True)],
             [69.0, 31.2]),
        Case("aniline", "Nc1ccccc1", "C6H7N", "CDCl3",
             [(7.17, "t", 2, False), (6.77, "t", 1, False), (6.69, "d", 2, False), (3.60, "br", 2, True)],
             [146.4, 129.3, 118.5, 115.1]),
        Case("benzoic_acid_no_cooh", "OC(=O)c1ccccc1", "C7H6O2", "CDCl3",
             [(8.12, "d", 2, False), (7.62, "t", 1, False), (7.48, "t", 2, False)],
             [172.6, 133.9, 130.3, 129.4, 128.5]),
        Case("ethyl_bromide", "CCBr", "C2H5Br", "CDCl3",
             [(3.43, "q", 2, False), (1.67, "t", 3, False)], [28.0, 19.0]),
        Case("1_bromobutane", "CCCCBr", "C4H9Br", "CDCl3",
             [(3.41, "t", 2, False), (1.85, "m", 2, False), (1.45, "m", 2, False), (0.93, "t", 3, False)],
             [33.4, 33.2, 21.5, 13.4]),
        Case("benzyl_alcohol", "OCc1ccccc1", "C7H8O", "CDCl3",
             [(7.35, "m", 5, False), (4.70, "s", 2, False), (2.00, "br", 1, True)],
             [140.9, 128.6, 127.7, 127.0, 65.2]),
        Case("benzylamine", "NCc1ccccc1", "C7H9N", "CDCl3",
             [(7.33, "m", 4, False), (7.24, "m", 1, False), (3.86, "s", 2, False), (1.50, "br", 2, True)],
             [143.4, 128.5, 127.1, 126.7, 46.5]),
        Case("1_2_epoxybutane", "CCC1CO1", "C4H8O", "CDCl3",
             [(2.90, "m", 1, False), (2.74, "t", 1, False), (2.46, "dd", 1, False), (1.55, "m", 2, False),
              (1.02, "t", 3, False)],
             [52.4, 47.0, 25.7, 9.8]),
        Case("thioanisole", "CSc1ccccc1", "C7H8S", "CDCl3",
             [(7.32, "m", 4, False), (7.17, "m", 1, False), (2.50, "s", 3, False)],
             [138.5, 128.8, 126.7, 125.0, 15.8]),
        Case("1_fluorohexane", "CCCCCCF", "C6H13F", "CDCl3",
             [(4.43, "dt", 2, False), (1.70, "m", 2, False), (1.40, "m", 2, False), (1.33, "m", 4, False),
              (0.90, "t", 3, False)],
             [84.2, 31.4, 30.4, 25.2, 22.5, 14.0]),
        Case("chalcone", "O=C(C=Cc1ccccc1)c1ccccc1", "C15H12O", "CDCl3",
             [(8.02, "d", 2, False), (7.81, "d", 1, False), (7.63, "m", 2, False), (7.56, "t", 1, False),
              (7.53, "d", 1, False), (7.50, "t", 2, False), (7.41, "m", 3, False)],
             [190.5, 144.9, 138.2, 134.8, 132.8, 130.5, 129.0, 128.6, 128.5, 128.4, 122.1]),
        Case("nonadecane", "CCCCCCCCCCCCCCCCCCC", "C19H40", "CDCl3",
             [(1.25, "br", 34, False), (0.88, "t", 6, False)], []),
        Case("1_octanol", "CCCCCCCCO", "C8H18O", "CDCl3",
             [(3.64, "t", 2, False), (1.57, "m", 2, False), (1.29, "m", 10, False), (0.89, "t", 3, False),
              (1.50, "br", 1, True)],
             [63.1, 32.8, 31.8, 29.4, 29.3, 25.7, 22.7, 14.1]),
        Case("2_methylnaphthalene", "Cc1ccc2ccccc2c1", "C11H10", "CDCl3",
             [(7.80, "m", 3, False), (7.62, "s", 1, False), (7.43, "m", 2, False), (7.33, "d", 1, False),
              (2.50, "s", 3, False)],
             [135.4, 133.6, 131.7, 128.0, 127.6, 127.5, 127.4, 126.0, 125.8, 125.1, 21.7]),
        Case("acetic_acid_d2o", "CC(=O)O", "C2H4O2", "D2O",
             [(2.08, "s", 3, False)],
             [177.0, 20.8]),
    ]
}  # fmt: skip
