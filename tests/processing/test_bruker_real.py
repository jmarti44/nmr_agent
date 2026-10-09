"""Real-data check (SPEC F1): our FID pipeline vs the TopSpin-processed 1r of a CC-BY dataset."""

import shutil
import warnings
from pathlib import Path

import nmrglue as ng
import numpy as np
import pytest
from scipy.signal import find_peaks

from src.processing.bruker import process_bruker_fid

EXP = Path(__file__).parents[1] / "fixtures/bruker/wadpressine_1h/10"
MATCH_PPM = 0.01  # SPEC §11.4 position target


def _peaks(spec, ppm, rel_height=0.01):
    """Peaks above rel_height of the spectrum maximum (same footing for both spectra)."""
    spec = np.asarray(spec, float)
    idx, _ = find_peaks(spec / spec.max(), height=rel_height, distance=10)
    return ppm[idx]


def _frac_matched(a, b):
    """Fraction of a's peaks that have a peak of b within MATCH_PPM."""
    return float(np.mean([np.min(np.abs(b - x)) < MATCH_PPM for x in a]))


@pytest.fixture(scope="module")
def ours():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return process_bruker_fid(str(EXP))


@pytest.fixture(scope="module")
def topspin():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dic, ref = ng.bruker.read_pdata(str(EXP / "pdata/1"), scale_data=True)
        uc = ng.fileiobase.uc_from_udic(ng.bruker.guess_udic(dic, ref))
    return np.asarray(ref, float), uc.ppm_scale()


def test_baseline_has_no_deep_negative_lobes(ours):
    """Regression for review R0-1: a median window narrower than a multiplet subtracts real signal."""
    spec, _, _ = ours
    assert spec.min() / spec.max() > -0.01


def test_fid_pipeline_matches_topspin_peaks_both_directions(ours, topspin):
    spec, ppm, meta = ours
    ref, rppm = topspin
    assert meta["nucleus"] == "1H" and meta["referenced"] is True
    mine, theirs = _peaks(spec, ppm), _peaks(ref, rppm)
    # Precision: our peaks are real. Recall: TopSpin's peaks are found. Axes here both come from procs,
    # so this validates mirroring/phasing/baseline, not referencing (see the FID-only test).
    assert _frac_matched(mine, theirs) >= 0.95
    assert _frac_matched(theirs, mine) >= 0.80
    for target in (3.31, 4.86):  # residual MeOD and HOD
        assert np.min(np.abs(mine - target)) < MATCH_PPM


def test_fid_only_without_procs_is_unreferenced(tmp_path, ours):
    """No procs/pdata: SR is not applied. Auto-referencing (SPEC §5.5) is a Phase 3 gate."""
    dst = tmp_path / "10"
    dst.mkdir()
    for name in ("acqus", "fid", "pulseprogram"):
        shutil.copy(EXP / name, dst / name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, ppm_raw, meta = process_bruker_fid(str(dst))
    assert meta["referenced"] is False
    _, ppm_ref, _ = ours
    # Recorded offset vs the TopSpin-referenced axis (~ +0.019 ppm on this dataset). It is the number the
    # Phase 3 auto-referencing rule must remove, so assert the known size rather than "close to zero".
    assert ppm_raw[0] - ppm_ref[0] == pytest.approx(0.0194, abs=0.005)
