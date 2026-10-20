import numpy as np
import pytest

from src.processing.peaks import classify_multiplet, peaks_from_spectrum
from src.processing.referencing import find_anchor
from src.schemas import Multiplicity, Nucleus, Peak, Solvent
from tests.spectra import MHZ, h1_spectrum, lorentz

UNIT = 4000.0  # h1_spectrum scales areas by this


@pytest.fixture(scope="module")
def picked():
    spec, ppm = h1_spectrum()
    peaks, sigma, notes = peaks_from_spectrum(spec, ppm, MHZ, "1H")
    return {round(p.shift_ppm, 2): p for p in peaks}, sigma, notes


def test_noise_is_close_to_the_known_gaussian_sigma(picked):
    # Lorentzian tails of the big lines slope across the window, so the estimate runs a little high.
    assert picked[1] == pytest.approx(0.5, rel=0.2)


def test_every_programmed_signal_is_found_and_nothing_else(picked):
    peaks, _, notes = picked
    assert sorted(peaks) == [0.0, 1.25, 4.1, 6.9, 7.26, 8.3] and notes == []


@pytest.mark.parametrize(
    ("shift", "mult", "j"),
    [
        (1.25, Multiplicity.T, 7.0),
        (4.1, Multiplicity.Q, 7.0),
        (6.9, Multiplicity.D, 8.5),
        (8.3, Multiplicity.S, None),
    ],
)
def test_multiplicity_and_coupling(picked, shift, mult, j):
    p = picked[0][shift]
    assert p.multiplicity == mult
    if j:
        assert p.coupling_hz[0] == pytest.approx(j, abs=0.4)


def test_multiplet_centre_is_the_programmed_shift(picked):
    for shift in (1.25, 4.1, 6.9):
        assert picked[0][shift].shift_ppm == pytest.approx(shift, abs=0.002)


def test_integrals_are_proportional_to_the_programmed_areas(picked):
    peaks = picked[0]
    # areas 3 : 2 : 2 : 1 (TMS is 1) ; the +-8 FWHM window holds ~96% of a Lorentzian
    assert peaks[1.25].integral / peaks[0.0].integral == pytest.approx(3.0, rel=0.06)
    assert peaks[4.1].integral / peaks[0.0].integral == pytest.approx(2.0, rel=0.06)
    assert peaks[6.9].integral / peaks[0.0].integral == pytest.approx(2.0, rel=0.06)


def test_ids_run_downfield_to_upfield(picked):
    ids = [p.id for p in sorted(picked[0].values(), key=lambda p: -p.shift_ppm)]
    assert ids == list(range(1, 7))


def test_a_noise_only_spectrum_gives_no_peaks():
    rng = np.random.default_rng(0)
    ppm = np.linspace(14, -1, 8192)
    peaks, _, _ = peaks_from_spectrum(rng.normal(0, 1, ppm.size), ppm, MHZ, "1H")
    assert peaks == []


def test_a_flat_spectrum_is_refused():
    with pytest.raises(ValueError, match="noise"):
        peaks_from_spectrum(np.zeros(2048), np.linspace(14, -1, 2048), MHZ, "1H")


def test_spikes_narrower_than_the_line_width_floor_are_rejected():
    ppm = np.linspace(14, -1, 32768)
    rng = np.random.default_rng(1)
    spec = rng.normal(0, 1, ppm.size)
    spec[12000] += 500.0  # one-point spike, ~0.18 Hz wide
    peaks, _, _ = peaks_from_spectrum(spec, ppm, MHZ, "1H")
    assert peaks == []


def test_close_signals_of_very_different_height_stay_separate():
    ppm = np.linspace(14, -1, 32768)
    spec = lorentz(ppm, 7.24, 1.0) * UNIT + lorentz(ppm, 7.26, 30.0, 1.2) * UNIT
    spec = spec + np.random.default_rng(2).normal(0, 0.5, ppm.size)
    shifts = sorted(round(p.shift_ppm, 2) for p in peaks_from_spectrum(spec, ppm, MHZ, "1H")[0])
    assert shifts == [7.24, 7.26]


def test_more_than_200_signals_keep_the_tallest_and_say_so():
    ppm = np.linspace(14, -1, 65536)
    centres = np.linspace(0.5, 12.5, 230)
    spec = sum(lorentz(ppm, c, 0.5 + (i % 7) * 0.1) for i, c in enumerate(centres)) * UNIT
    spec = spec + np.random.default_rng(3).normal(0, 0.5, ppm.size)
    peaks, _, notes = peaks_from_spectrum(spec, ppm, MHZ, "1H")
    assert len(peaks) == 200 and any("tallest 200" in n for n in notes)


def test_carbon_lines_get_relative_heights_and_no_integrals():
    ppm = np.linspace(230, -10, 32768)
    spec = lorentz(ppm, 77.2, 5.0, 3.0) + lorentz(ppm, 128.5, 2.0, 3.0) + lorentz(ppm, 20.0, 1.0, 3.0)
    spec = spec * 100 + np.random.default_rng(4).normal(0, 0.002, ppm.size)
    peaks, _, _ = peaks_from_spectrum(spec, ppm, 100.0, "13C")
    assert [round(p.shift_ppm) for p in peaks] == [128, 77, 20]
    assert max(p.height for p in peaks) == 1.0 and all(p.integral is None for p in peaks)


@pytest.mark.parametrize(
    ("positions", "mult"),
    [
        ([0.0], Multiplicity.S),
        ([0.0, 7.0], Multiplicity.D),
        ([0.0, 7.0, 14.0], Multiplicity.T),
        ([0.0, 7.0, 14.0, 21.0], Multiplicity.Q),
        ([0.0, 8.0, 10.0, 18.0], Multiplicity.DD),
        ([0.0, 7.0, 16.0], Multiplicity.M),
        ([0.0, 7.0, 14.0, 21.0, 28.0], Multiplicity.M),
    ],
)
def test_classify_multiplet(positions, mult):
    assert classify_multiplet(positions)[0] == mult


def test_dd_reports_both_couplings():
    assert classify_multiplet([0.0, 8.0, 10.0, 18.0])[1] == [8.0, 2.0]


def _p(ppm, height, nucleus=Nucleus.H1):
    return Peak(id=1, nucleus=nucleus, shift_ppm=ppm, height=height)


def test_tms_is_preferred_as_the_anchor():
    a = find_anchor([_p(0.02, 50), _p(7.30, 100)], Nucleus.H1, Solvent.CDCL3)
    assert a.name == "TMS" and a.shift == pytest.approx(-0.02)


def test_residual_solvent_is_used_when_there_is_no_tms():
    a = find_anchor([_p(7.30, 100), _p(2.0, 80)], Nucleus.H1, Solvent.CDCL3)
    assert a.name == "CDCl3 residual" and a.shift == pytest.approx(7.27 - 7.30)


def test_a_tiny_peak_near_zero_is_not_an_anchor():
    assert find_anchor([_p(0.02, 0.5), _p(5.0, 100)], Nucleus.H1, None) is None


def test_peaks_outside_the_search_window_do_not_anchor():
    assert find_anchor([_p(7.9, 100)], Nucleus.H1, Solvent.CDCL3) is None


def test_carbon_uses_a_wider_window():
    a = find_anchor([_p(78.0, 1.0, Nucleus.C13)], Nucleus.C13, Solvent.CDCL3)
    assert a.shift == pytest.approx(77.2 - 78.0)


def test_an_analyte_line_near_but_not_at_the_solvent_position_is_not_an_anchor():
    """DMSO-d6 is at 2.50; a methyl at 2.35 must not pull the axis by 0.15 ppm."""
    assert find_anchor([_p(2.35, 100)], Nucleus.H1, Solvent.DMSO_D6) is None


def _triplet(heights, mhz=100.6, centre=77.16 + 0.02, nucleus=Nucleus.C13):
    sep = 32.0 / mhz
    return [
        Peak(id=i + 1, nucleus=nucleus, shift_ppm=round(centre + d, 3), height=h)
        for i, (d, h) in enumerate(zip((sep, 0.0, -sep), heights, strict=True))
    ]


@pytest.mark.parametrize("heights", [(1.05, 1.0, 0.98), (0.98, 1.0, 1.05), (1.04, 0.97, 1.0)])
@pytest.mark.parametrize("mhz", [100.6, 0.0])
def test_cdcl3_13c_anchor_is_the_triplet_centre_even_when_an_outer_line_is_taller(heights, mhz):
    a = find_anchor(_triplet(heights), Nucleus.C13, Solvent.CDCL3, mhz)
    assert a.observed_ppm == pytest.approx(77.18, abs=0.002)


def test_a_dmso_septet_is_centred_on_its_middle_line():
    mhz, j = 125.7, 21.0
    weights = [1, 3, 6, 7, 6, 3, 1]
    peaks = [
        Peak(id=i + 1, nucleus=Nucleus.C13, shift_ppm=round(39.55 + (i - 3) * j / mhz, 3), height=w / 7)
        for i, w in enumerate(weights)
    ]
    peaks.append(Peak(id=9, nucleus=Nucleus.C13, shift_ppm=38.9, height=0.9))  # an analyte line beside it
    a = find_anchor(peaks, Nucleus.C13, Solvent.DMSO_D6, mhz)
    assert a.observed_ppm == pytest.approx(39.55, abs=0.002)


def test_silyl_line_near_zero_does_not_override_the_solvent_line():
    """TBS SiMe2 at 0.05, CHCl3 already correct at 7.27 and no TMS: the axis must not move."""
    peaks = [_p(7.27, 0.3), _p(3.6, 0.4), _p(0.89, 1.0), _p(0.05, 0.7)]
    a = find_anchor(peaks, Nucleus.H1, Solvent.CDCL3)
    assert a.name == "CDCl3 residual" and a.shift == pytest.approx(0.0)
    assert "silyl" in (a.note or "")


def test_tms_confirmed_by_the_solvent_is_used_for_its_exactness():
    a = find_anchor([_p(0.02, 0.5), _p(7.29, 1.0)], Nucleus.H1, Solvent.CDCL3)
    assert a.name == "TMS" and a.shift == pytest.approx(-0.02)


def test_the_solvent_candidate_that_agrees_with_tms_is_chosen_over_a_taller_neighbour():
    """A taller aromatic line near the solvent position is not taken for it when TMS says otherwise."""
    a = find_anchor([_p(0.06, 0.5), _p(7.33, 1.0), _p(7.32, 0.6)], Nucleus.H1, Solvent.CDCL3)
    assert a.name == "TMS" and a.shift == pytest.approx(-0.06)
