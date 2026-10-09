"""Peak-list normalization (SPEC §3.2 step 4, §5.5)."""

import pytest

from src.processing.normalize import normalize_peaklist
from src.schemas import Nucleus, Peak, PeakRole, SampleContext, Solvent

H, C = Nucleus.H1, Nucleus.C13


def p(id, nuc, shift, mult="unknown", integral=None, **kw):
    return Peak(id=id, nucleus=nuc, shift_ppm=shift, multiplicity=mult, integral=integral, **kw)


def ctx(**kw):
    return SampleContext(**kw)


def integrals(table):
    return [x.integral for x in table.peaks if x.nucleus == H and x.role == PeakRole.ANALYTE]


def test_peaks_are_ordered_downfield_first_1h_before_13c_and_renumbered():
    peaks = [p(1, C, 20.8), p(2, H, 2.24, integral=3), p(3, C, 169.1), p(4, H, 7.9, integral=1)]
    t = normalize_peaklist(peaks, ctx())
    assert [(x.nucleus.value, x.shift_ppm) for x in t.peaks] == [
        ("1H", 7.9),
        ("1H", 2.24),
        ("13C", 169.1),
        ("13C", 20.8),
    ]
    assert [x.id for x in t.peaks] == [1, 2, 3, 4]
    assert any("renumbered" in line for line in t.processing_log)


def test_already_ordered_peaks_log_no_renumbering():
    t = normalize_peaklist([p(1, H, 7.9, integral=1), p(2, H, 2.2, integral=3)], ctx())
    assert not any("renumbered" in line for line in t.processing_log)


def test_nothing_is_dropped_and_input_is_not_mutated():
    peaks = [
        p(1, H, 7.27, integral=1),
        p(2, H, 1.56, integral=2),
        p(3, H, 0.0),
        p(4, C, 77.0),
        p(5, H, 5.0, integral=1),
    ]
    before = [x.model_copy() for x in peaks]
    t = normalize_peaklist(peaks, ctx(solvent="CDCl3"))
    assert len(t.peaks) == len(peaks)
    assert peaks == before


def test_solvent_water_and_reference_are_tagged_but_impurities_are_not_for_peak_lists():
    peaks = [
        p(1, H, 7.26),
        p(2, H, 1.56),
        p(3, H, 0.0),
        p(4, H, 2.17, "s", 3),
        p(5, H, 5.0, integral=1),
        p(6, C, 77.0),
    ]
    t = normalize_peaklist(peaks, ctx(solvent="CDCl3"))
    roles = {x.shift_ppm: x.role for x in t.peaks}
    assert (
        roles[7.26] == PeakRole.SOLVENT and roles[1.56] == PeakRole.WATER and roles[0.0] == PeakRole.REFERENCE
    )
    assert (
        roles[2.17] == PeakRole.ANALYTE and roles[5.0] == PeakRole.ANALYTE and roles[77.0] == PeakRole.SOLVENT
    )
    assert not any(x.role == PeakRole.IMPURITY_SUSPECT for x in t.peaks)
    assert any("tagged solvent" in line for line in t.processing_log)


def test_impurity_tagging_is_available_for_the_raw_spectrum_path():
    peaks = [p(1, H, 2.17, "s", 3), p(2, H, 5.0, integral=1)]
    t = normalize_peaklist(peaks, ctx(solvent="CDCl3"), tag_impurities=True)
    assert {x.shift_ppm: x.role for x in t.peaks} == {2.17: PeakRole.IMPURITY_SUSPECT, 5.0: PeakRole.ANALYTE}


def test_tagged_impurity_peak_keeps_its_integral():
    t = normalize_peaklist([p(1, H, 2.17, "s", 3.0)], ctx(solvent="CDCl3"), tag_impurities=True)
    assert t.peaks[0].role == PeakRole.IMPURITY_SUSPECT and t.peaks[0].integral == 3.0


def test_impurities_are_only_tagged_in_cdcl3():
    t = normalize_peaklist([p(1, H, 2.17, "s", 3.0)], ctx(solvent="DMSO-d6"), tag_impurities=True)
    assert t.peaks[0].role == PeakRole.ANALYTE


def test_real_aspirin_peaks_are_not_mislabelled_for_peak_lists():
    """Ring H 7.17 and methyl C 20.8 match toluene / acetic acid entries; untagged they stay analyte."""
    peaks = [p(1, H, 7.17, integral=1), p(2, C, 20.8)]
    t = normalize_peaklist(peaks, ctx(solvent="CDCl3"))
    assert [x.role for x in t.peaks] == [PeakRole.ANALYTE, PeakRole.ANALYTE]
    t2 = normalize_peaklist(
        peaks, ctx(solvent="CDCl3"), tag_impurities=True
    )  # why it is off: it mislabels both
    assert [x.role for x in t2.peaks] == [PeakRole.IMPURITY_SUSPECT, PeakRole.IMPURITY_SUSPECT]


def test_broad_downfield_signal_is_marked_exchangeable_but_a_sharp_aldehyde_is_not():
    t = normalize_peaklist(
        [p(1, H, 12.9, "br", 1), p(2, H, 10.0, "s", 1), p(3, H, 9.6, "br", 1), p(4, H, 4.0, "br", 1)], ctx()
    )
    flags = {x.shift_ppm: x.exchangeable_hint for x in t.peaks}
    assert flags == {12.9: True, 10.0: False, 9.6: True, 4.0: False}


# ---- integrals ------------------------------------------------------------------------------------------


def test_integrals_consistent_with_the_formula_are_kept_exactly():
    peaks = [
        p(1, H, 7.9, integral=1),
        p(2, H, 7.6, integral=1),
        p(3, H, 7.4, integral=1),
        p(4, H, 7.2, integral=1),
        p(5, H, 2.2, integral=3),
    ]
    t = normalize_peaklist(
        peaks, ctx(molecular_formula="C9H8O4")
    )  # sum 7 of 8 H: the COOH proton is unlisted
    assert integrals(t) == [1, 1, 1, 1, 3] and t.integral_normalization == "formula_H"
    assert any("look like proton counts" in line for line in t.processing_log)


def test_arbitrary_unit_integrals_are_scaled_to_the_formula_h_count():
    peaks = [
        p(1, H, 7.9, integral=100),
        p(2, H, 7.6, integral=100),
        p(3, H, 2.2, integral=300),
        p(4, H, 12.9, "br", 100),
    ]
    t = normalize_peaklist(peaks, ctx(molecular_formula="C9H8O4"))
    # 100 : 100 : 300 : 100 (sum 600) -> 8 H; the table is ordered downfield first (12.9, 7.9, 7.6, 2.2)
    assert integrals(t) == pytest.approx([8 / 6, 8 / 6, 8 / 6, 4.0], abs=0.001)
    assert t.integral_normalization == "formula_H"


def test_hydrocarbon_has_no_exchangeable_allowance():
    """No O/N/S means no unlisted OH to excuse a low sum: 6 of 14 H is treated as arbitrary units."""
    t = normalize_peaklist(
        [p(1, H, 1.3, integral=4), p(2, H, 0.9, integral=2)], ctx(molecular_formula="C7H14")
    )
    assert sum(integrals(t)) == pytest.approx(14.0, abs=0.01)


def test_scaling_in_an_exchange_solvent_is_logged_as_possibly_inflated():
    peaks = [p(1, H, 4.0, integral=50), p(2, H, 1.3, integral=150)]
    t = normalize_peaklist(peaks, ctx(molecular_formula="C4H10O", solvent="D2O"))
    assert any("inflated" in line for line in t.processing_log)


def test_solvent_and_water_integrals_are_never_scaled():
    peaks = [
        p(1, H, 7.26, integral=0.2),
        p(2, H, 1.56, integral=2.0),
        p(3, H, 7.5, integral=100),
        p(4, H, 2.0, integral=100),
    ]
    t = normalize_peaklist(peaks, ctx(molecular_formula="C2H2", solvent="CDCl3"))
    by_shift = {x.shift_ppm: x.integral for x in t.peaks}
    assert (
        by_shift[7.26] == 0.2 and by_shift[1.56] == 2.0 and sum(integrals(t)) == pytest.approx(2.0, abs=0.01)
    )


def test_without_a_formula_the_smallest_multiplet_becomes_one_and_values_round_to_half():
    t = normalize_peaklist(
        [p(1, H, 7.9, integral=100), p(2, H, 3.7, integral=203), p(3, H, 1.2, integral=298)], ctx()
    )
    assert integrals(t) == [1.0, 2.0, 3.0] and t.integral_normalization == "smallest_peak"


def test_without_a_formula_counts_near_one_are_not_rescaled():
    t = normalize_peaklist([p(1, H, 7.9, integral=1.04), p(2, H, 1.2, integral=3.0)], ctx())
    assert integrals(t) == [1.0, 3.0]


def test_missing_integrals_leave_values_untouched():
    t = normalize_peaklist([p(1, H, 7.9, integral=1), p(2, H, 1.2)], ctx(molecular_formula="C4H8O2"))
    assert t.integral_normalization == "none" and integrals(t) == [1, None]
    assert any("not every analyte" in line for line in t.processing_log)


def test_normalization_is_idempotent():
    peaks = [
        p(1, H, 7.9, integral=100),
        p(2, H, 7.6, integral=100),
        p(3, H, 2.2, integral=300),
        p(4, C, 169.1),
    ]
    once = normalize_peaklist(peaks, ctx(molecular_formula="C9H8O4", solvent="CDCl3"))
    twice = normalize_peaklist(once.peaks, ctx(molecular_formula="C9H8O4", solvent="CDCl3"))
    assert [x.model_dump() for x in once.peaks] == [x.model_dump() for x in twice.peaks]


# ---- table metadata -------------------------------------------------------------------------------------


def test_frequencies_solvent_and_referencing():
    t = normalize_peaklist(
        [p(1, H, 7.9, integral=1)], ctx(spectrometer_mhz=400.13, solvent="CDCl3"), referencing="procs"
    )
    assert t.nucleus_frequencies_mhz == {"1H": 400.13, "13C": 100.61}
    assert t.solvent == Solvent.CDCL3 and t.referencing == "procs"
    assert normalize_peaklist([p(1, H, 7.9, integral=1)], ctx()).nucleus_frequencies_mhz == {}


# ---- residual-size guard on solvent / water tags ---------------------------------------------------------


def test_without_a_formula_nothing_is_restored_and_a_large_solvent_position_peak_is_logged():
    """Restoring needs the formula; without one the tag stays and the log says it may hide a signal."""
    peaks = [p(1, H, 7.27, "s", 1), p(2, H, 7.1, integral=2), p(3, H, 2.3, integral=3)]
    t = normalize_peaklist(peaks, ctx(solvent="CDCl3"))
    assert {x.shift_ppm: x.role for x in t.peaks}[7.27] == PeakRole.SOLVENT
    assert any("may overlap an analyte signal" in line for line in t.processing_log)


def test_a_small_residual_solvent_peak_stays_tagged():
    peaks = [p(1, H, 7.27, "s", 0.1), p(2, H, 7.1, integral=2), p(3, H, 2.3, integral=3)]
    t = normalize_peaklist(peaks, ctx(solvent="CDCl3"))
    assert {x.shift_ppm: x.role for x in t.peaks}[7.27] == PeakRole.SOLVENT


def test_water_without_a_formula_is_not_restored_and_costs_the_scale():
    """No formula, no fit test: a real CH2 at the water position stays water."""
    peaks = [p(1, H, 1.56, "m", 2), p(2, H, 3.6, integral=2), p(3, H, 0.9, integral=3)]
    t = normalize_peaklist(peaks, ctx(solvent="CDCl3"))
    assert {x.shift_ppm: x.role for x in t.peaks}[1.56] == PeakRole.WATER
    # Cost: the lost CH2 leaves 2 H as the smallest signal, so with no formula the scale halves.
    assert sorted(integrals(t)) == [1.0, 1.5]


def test_in_arbitrary_units_with_a_formula_a_big_solvent_position_peak_stays_tagged_and_is_logged():
    """The scale comes from the other peaks (100/200/300 -> 6 H), so it cannot also decide this peak."""
    peaks = [
        p(1, H, 7.27, "s", 100),
        p(2, H, 7.1, integral=100),
        p(3, H, 3.6, integral=200),
        p(4, H, 0.9, integral=300),
    ]
    t = normalize_peaklist(peaks, ctx(solvent="CDCl3", molecular_formula="C6H6"))
    assert {x.shift_ppm: x.role for x in t.peaks}[7.27] == PeakRole.SOLVENT
    assert sum(integrals(t)) == pytest.approx(6.0, abs=0.01)
    assert any("may overlap an analyte signal" in line for line in t.processing_log)
    small = [
        p(1, H, 7.27, "s", 8),
        p(2, H, 7.1, integral=100),
        p(3, H, 3.6, integral=200),
        p(4, H, 0.9, integral=300),
    ]
    t2 = normalize_peaklist(small, ctx(solvent="CDCl3", molecular_formula="C6H6"))
    assert {x.shift_ppm: x.role for x in t2.peaks}[7.27] == PeakRole.SOLVENT
    assert not any("may overlap" in line for line in t2.processing_log)


def test_without_integrals_the_position_tag_stands():
    t = normalize_peaklist([p(1, H, 7.27, "s"), p(2, H, 2.3)], ctx(solvent="CDCl3"))
    assert {x.shift_ppm: x.role for x in t.peaks}[7.27] == PeakRole.SOLVENT


# ---- cases from the reviewer's probe -------------------------------------------------------------------

PROBE_CASES = [
    ("ethyl acetate in DMSO-d6 with a 2 H water peak",
     [(4.03, 2, "q"), (3.33, 2.0, "br"), (2.50, 0.1, "quint"), (1.99, 3, "s"), (1.17, 3, "t")],
     "C4H8O2", "DMSO-d6", [2, 3, 3]),
    ("alanine in D2O with HOD", [(4.80, 5.0, "s"), (3.77, 1, "q"), (1.47, 3, "d")], "C3H7NO2", "D2O", [1, 3]),
    ("ethyl acetate as the analyte in CDCl3", [(4.12, 2, "q"), (2.05, 3, "s"), (1.26, 3, "t")],
     "C4H8O2", "CDCl3", [2, 3, 3]),
    ("methyl propanoate, CH2 at the toluene position",
     [(3.67, 2.0, "s"), (2.33, 1.33, "q"), (1.14, 2.0, "t")], "C4H8O2", "CDCl3", [3, 2, 3]),
    ("methyl propanoate in arbitrary units inside the window",
     [(3.67, 2.0, "s"), (2.28, 1.33, "q"), (1.12, 2.0, "t")], "C4H8O2", "CDCl3", [3, 2, 3]),
    # aromatic multiplets on the CDCl3 residual position, count-like integrals (review follow-up)
    ("anisole, 2 ArH at 7.29", [(7.29, 2, "m"), (6.93, 3, "m"), (3.80, 3, "s")], "C7H8O", "CDCl3", [2, 3, 3]),
    ("ethylbenzene, 2 ArH at 7.27", [(7.27, 2, "m"), (7.18, 3, "m"), (2.65, 2, "q"), (1.24, 3, "t")],
     "C8H10", "CDCl3", [2, 3, 2, 3]),
    ("benzyl acetate, 5 ArH at 7.30", [(7.30, 5, "m"), (5.10, 2, "s"), (2.10, 3, "s")],
     "C9H10O2", "CDCl3", [5, 2, 3]),
    ("toluene, 2 ArH at 7.27", [(7.27, 2, "m"), (7.17, 3, "m"), (2.36, 3, "s")], "C7H8", "CDCl3", [2, 3, 3]),
    # one restoration rule for solvent and water alike: only when the formula-fit test discriminates
    ("1-butanol, CH2 at 1.56 (water position)",
     [(3.64, 2, "t"), (1.56, 2, "m"), (1.39, 2, "m"), (0.94, 3, "t")], "C4H10O", "CDCl3", [2, 2, 2, 3]),
]  # fmt: skip


@pytest.mark.parametrize("name,rows,formula,solvent,want", PROBE_CASES, ids=[c[0] for c in PROBE_CASES])
def test_probe_case(name, rows, formula, solvent, want):
    peaks = [p(i + 1, H, s, m, n) for i, (s, n, m) in enumerate(rows)]
    t = normalize_peaklist(peaks, ctx(molecular_formula=formula, solvent=solvent))
    got = [x.integral for x in t.peaks if x.role == PeakRole.ANALYTE]
    assert got == pytest.approx(want, abs=0.01), (name, t.processing_log)


def test_integrals_that_fit_the_sum_window_but_are_not_counts_are_scaled():
    peaks = [p(1, H, 3.7, integral=2.0), p(2, H, 2.3, integral=1.33), p(3, H, 1.1, integral=2.0)]
    t = normalize_peaklist(peaks, ctx(molecular_formula="C4H8O2"))
    assert integrals(t) == pytest.approx([3.0, 2.0, 3.0], abs=0.01)
    assert any("not near whole or half numbers" in line for line in t.processing_log)


def test_exact_counts_with_an_unlisted_exchangeable_proton_are_still_kept():
    shifts_and_counts = [(7.9, 1), (7.6, 1), (7.4, 1), (7.2, 1), (2.2, 3)]
    peaks = [p(i + 1, H, s_, integral=n) for i, (s_, n) in enumerate(shifts_and_counts)]
    assert integrals(normalize_peaklist(peaks, ctx(molecular_formula="C9H8O4"))) == [1, 1, 1, 1, 3]


def test_half_number_counts_count_as_count_like():
    peaks = [p(1, H, 3.7, integral=0.5), p(2, H, 2.3, integral=2.5), p(3, H, 1.1, integral=3.0)]
    assert integrals(normalize_peaklist(peaks, ctx(molecular_formula="C6H6"))) == [0.5, 2.5, 3.0]


def test_a_solvent_position_peak_is_left_tagged_when_the_analyte_already_fits_the_formula():
    """C6H6O: 2+3 = 5 already fits H = 6 (the OH may be unlisted), so the 7.27 peak is not restored."""
    peaks = [p(1, H, 7.27, "s", 1), p(2, H, 7.1, integral=2), p(3, H, 2.3, integral=3)]
    t = normalize_peaklist(peaks, ctx(solvent="CDCl3", molecular_formula="C6H6O"))
    assert {x.shift_ppm: x.role for x in t.peaks}[7.27] == PeakRole.SOLVENT
    assert integrals(t) == [2, 3]
    assert any("may overlap an analyte signal" in line for line in t.processing_log)


def test_a_hydrocarbon_with_a_phenyl_peak_at_the_solvent_position_is_resolved_by_formula_fit():
    """C6H6 has no OH to excuse a missing proton: 2+3 = 5 misses H = 6 but 1+2+3 = 6 fits, so 7.27 is real."""
    peaks = [p(1, H, 7.27, "s", 1), p(2, H, 7.1, integral=2), p(3, H, 2.3, integral=3)]
    t = normalize_peaklist(peaks, ctx(solvent="CDCl3", molecular_formula="C6H6"))
    assert {x.shift_ppm: x.role for x in t.peaks}[7.27] == PeakRole.ANALYTE
    assert integrals(t) == [1, 2, 3]
    assert any("with it the integrals fit" in line for line in t.processing_log)


def test_when_neither_reading_fits_the_formula_the_peak_stays_tagged_and_the_scale_is_logged():
    """Counts 1+2+3 = 6 and 2+3 = 5 both miss H = 8 (no O/N/S): fall back to tagging and rescaling."""
    peaks = [p(1, H, 7.27, "s", 1), p(2, H, 7.1, integral=2), p(3, H, 2.3, integral=3)]
    t = normalize_peaklist(peaks, ctx(solvent="CDCl3", molecular_formula="C6H8"))
    assert {x.shift_ppm: x.role for x in t.peaks}[7.27] == PeakRole.SOLVENT
    assert sum(integrals(t)) == pytest.approx(8.0, abs=0.01)
    assert any("may overlap an analyte signal" in line for line in t.processing_log)


def test_restoring_by_formula_fit_leaves_the_supplied_integrals_untouched():
    peaks = [p(1, H, 7.29, "m", 2), p(2, H, 6.93, "m", 3), p(3, H, 3.80, "s", 3)]
    t = normalize_peaklist(peaks, ctx(solvent="CDCl3", molecular_formula="C7H8O"))  # anisole
    assert integrals(t) == [2, 3, 3] and t.integral_normalization == "formula_H"


def test_an_integral_that_is_too_small_to_matter_is_never_restored_by_fit():
    """A 0.1 H solvent peak cannot be what makes the sum fit; the analyte integrals are scaled as before."""
    peaks = [
        p(1, H, 7.27, "s", 0.1),
        p(2, H, 7.1, integral=100),
        p(3, H, 3.6, integral=200),
        p(4, H, 0.9, integral=300),
    ]
    t = normalize_peaklist(peaks, ctx(solvent="CDCl3", molecular_formula="C6H6"))
    assert {x.shift_ppm: x.role for x in t.peaks}[7.27] == PeakRole.SOLVENT


def test_log_says_so_when_no_analyte_1h_peak_remains():
    t = normalize_peaklist([p(1, H, 7.27, "s"), p(2, H, 1.56, "s")], ctx(solvent="CDCl3"))
    assert any("no analyte 1H peaks remain" in line for line in t.processing_log)


# ---- the single restoration rule: formula fit, solvent and water alike -----------------------------------


def test_a_real_ch2_at_the_water_position_is_restored_and_the_restoration_is_logged():
    peaks = [p(1, H, 3.64, "t", 2), p(2, H, 1.56, "m", 2), p(3, H, 1.39, "m", 2), p(4, H, 0.94, "t", 3)]
    t = normalize_peaklist(peaks, ctx(solvent="CDCl3", molecular_formula="C4H10O"))
    assert {x.shift_ppm: x.role for x in t.peaks}[1.56] == PeakRole.ANALYTE and integrals(t) == [2, 2, 2, 3]
    assert any("at the water position" in line and "restored as analyte" in line for line in t.processing_log)


def test_a_real_residual_is_not_restored_when_the_analyte_already_fits():
    """Benzoic acid, COOH unlisted: 5 of 6 H fits. A 0.6 H CHCl3 residual must stay solvent."""
    peaks = [p(1, H, 8.12, "d", 2), p(2, H, 7.62, "t", 1), p(3, H, 7.48, "t", 2), p(4, H, 7.26, "s", 0.6)]
    t = normalize_peaklist(peaks, ctx(solvent="CDCl3", molecular_formula="C7H6O2"))
    assert {x.shift_ppm: x.role for x in t.peaks}[7.26] == PeakRole.SOLVENT
    assert integrals(t) == [2, 1, 2]


@pytest.mark.parametrize("solvent,hod", [("D2O", 4.80), ("CD3OD", 4.87)])
def test_hod_in_d2o_and_cd3od_is_never_restored_even_if_it_would_make_the_sum_fit(solvent, hod):
    """C5H10: counts 3+3 = 6 miss H = 10 but 6+4 would fit; in these solvents the peak is mostly HOD."""
    peaks = [p(1, H, hod, "s", 4), p(2, H, 1.5, integral=3), p(3, H, 0.9, integral=3)]
    t = normalize_peaklist(peaks, ctx(solvent=solvent, molecular_formula="C5H10"))
    assert {x.shift_ppm: x.role for x in t.peaks}[hod] == PeakRole.WATER
    # the same numbers with the water at the CDCl3 position are restored (the fit test discriminates)
    t2 = normalize_peaklist([p(1, H, 1.56, "s", 4), p(2, H, 1.2, integral=3), p(3, H, 0.9, integral=3)],
                            ctx(solvent="CDCl3", molecular_formula="C5H10"))  # fmt: skip
    assert {x.shift_ppm: x.role for x in t2.peaks}[1.56] == PeakRole.ANALYTE


def test_the_smallest_fitting_set_is_restored_and_ties_go_to_the_sum_closest_to_h():
    """C4H10O: 7 H misses [8, 11]. Adding the water CH2 (9) or the 7.27 peak (8) would each fit; the sum
    nearer H = 10 wins and only one peak is restored."""
    peaks = [
        p(1, H, 7.27, "s", 1),
        p(2, H, 3.64, "t", 2),
        p(3, H, 1.56, "m", 2),
        p(4, H, 1.39, "m", 2),
        p(5, H, 0.94, "t", 3),
    ]
    t = normalize_peaklist(peaks, ctx(solvent="CDCl3", molecular_formula="C4H10O"))
    roles = {x.shift_ppm: x.role for x in t.peaks}
    assert roles[1.56] == PeakRole.ANALYTE and roles[7.27] == PeakRole.SOLVENT


def test_a_peak_below_half_a_proton_is_never_a_restoration_candidate():
    peaks = [p(1, H, 7.27, "s", 0.3), p(2, H, 3.64, "t", 2), p(3, H, 1.39, "m", 2), p(4, H, 0.94, "t", 3)]
    t = normalize_peaklist(peaks, ctx(solvent="CDCl3", molecular_formula="C4H10O"))
    assert {x.shift_ppm: x.role for x in t.peaks}[7.27] == PeakRole.SOLVENT
