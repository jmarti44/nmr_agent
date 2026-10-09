"""Synthetic Bruker round trip: the NumPy 2 / nmrglue 0.12 regression guard (SPEC §11.2, F3)."""

import warnings

import numpy as np
import pytest
from scipy.signal import find_peaks

from src.processing.bruker import process_bruker_fid
from tests.synthetic import CARRIER_PPM, PEAKS, write_synthetic_bruker


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    exp = write_synthetic_bruker(tmp_path_factory.mktemp("bruker") / "1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # guess_udic warns that SR is not applied (no procs)
        return process_bruker_fid(str(exp))


def _peak_ppms(spec, ppm):
    idx, _ = find_peaks(spec, height=spec.max() * 0.05)
    return ppm[idx]


def test_every_programmed_peak_recovered(processed):
    spec, ppm, _ = processed
    found = _peak_ppms(spec, ppm)
    for shift, _ in PEAKS:
        assert np.min(np.abs(found - shift)) < 0.005, f"{shift} ppm not recovered; found {found}"
    assert len(found) == len(PEAKS)


def test_axis_is_descending_and_not_mirrored(processed):
    spec, ppm, _ = processed
    assert ppm[0] > ppm[-1]
    peak_max = spec.max()
    for shift, _ in PEAKS:
        mirrored = 2 * CARRIER_PPM - shift
        if any(abs(mirrored - s) < 0.1 for s, _ in PEAKS):
            continue  # mirror image coincides with a real peak
        window = np.abs(ppm - mirrored) < 0.02
        assert spec[window].max() < 0.05 * peak_max, f"mirrored peak present at {mirrored} ppm"


def test_relative_integrals_within_10_percent(processed):
    spec, ppm, _ = processed
    areas = np.array([spec[np.abs(ppm - s) < 0.02].sum() for s, _ in PEAKS])
    expected = np.array([a for _, a in PEAKS])
    ratio = (areas / areas.sum()) / (expected / expected.sum())
    assert np.all(np.abs(ratio - 1) < 0.10), ratio


def test_meta(processed):
    _, _, meta = processed
    assert meta["nucleus"] == "1H"
    assert meta["sfo1_mhz"] == pytest.approx(400.13 + 0.0024, abs=1e-3)
    assert meta["referenced"] is False
