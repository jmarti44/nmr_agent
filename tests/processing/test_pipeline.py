import io
import zipfile
from pathlib import Path

import pytest

from src.ingest.errors import IngestError
from src.processing.pipeline import process_upload
from src.schemas import Nucleus, PeakRole, ProcessingOptions, SampleContext, Solvent
from tests.spectra import h1_spectrum, jcamp_text, peaktable_text
from tests.synthetic import write_synthetic_bruker

OPTS = ProcessingOptions()


def zip_dir(root, prefix="exp/1"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for f in sorted(root.rglob("*")):
            if f.is_file():
                z.write(f, f"{prefix}/{f.relative_to(root)}")
    return buf.getvalue()


@pytest.fixture(scope="module")
def synthetic_zip(tmp_path_factory):
    return zip_dir(write_synthetic_bruker(tmp_path_factory.mktemp("s") / "x"))


@pytest.fixture(scope="module")
def offset_zip(tmp_path_factory):
    return zip_dir(write_synthetic_bruker(tmp_path_factory.mktemp("o") / "x", offset_ppm=0.06))


def by_shift(peaks, nucleus=Nucleus.H1):
    return {round(p.shift_ppm, 2): p for p in peaks if p.nucleus == nucleus}


def test_bruker_zip_fid_is_processed_and_the_header_fills_the_context(synthetic_zip):
    r = process_upload("bruker_zip", synthetic_zip, SampleContext(), OPTS)
    assert r.context.solvent == Solvent.CDCL3 and r.context.spectrometer_mhz == pytest.approx(
        400.13, abs=0.01
    )
    assert {0.0, 2.35, 7.26, 7.3, 7.65, 9.8} <= set(by_shift(r.peaks))
    assert any("taken from the acquisition parameters" in line for line in r.log)


def test_caller_context_wins_over_the_header(synthetic_zip):
    ctx = SampleContext(solvent=Solvent.DMSO_D6, spectrometer_mhz=500.0)
    r = process_upload("bruker_zip", synthetic_zip, ctx, OPTS)
    assert r.context.solvent == Solvent.DMSO_D6 and r.context.spectrometer_mhz == 500.0


def test_fid_without_procs_is_referenced_on_tms(offset_zip):
    """The line positions are shifted by +0.06 ppm (an unapplied SR); TMS puts them back."""
    r = process_upload("bruker_zip", offset_zip, SampleContext(), OPTS)
    assert r.referencing == "auto:TMS"
    shifts = by_shift(r.peaks)
    assert {0.0, 2.35, 7.65, 9.8} <= set(shifts)
    assert shifts[7.65].shift_ppm == pytest.approx(7.651, abs=0.005)


def test_without_an_anchor_the_axis_is_left_uncorrected(tmp_path):
    exp = write_synthetic_bruker(
        tmp_path / "x", peaks=[(3.601, 2.0), (7.651, 1.0), (9.801, 1.0)], offset_ppm=0.05
    )
    r = process_upload("bruker_zip", zip_dir(exp), SampleContext(solvent=Solvent.DMSO_D6), OPTS)
    assert r.referencing == "uncorrected"
    assert any("uncorrected" in line for line in r.log)


def test_two_experiments_of_the_same_nucleus_are_refused(tmp_path):
    a = write_synthetic_bruker(tmp_path / "a")
    b = write_synthetic_bruker(tmp_path / "b")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, root in (("e/1", a), ("e/2", b)):
            for f in root.rglob("*"):
                if f.is_file():
                    z.write(f, f"{name}/{f.relative_to(root)}")
    with pytest.raises(IngestError, match="two 1H"):
        process_upload("bruker_zip", buf.getvalue(), SampleContext(), OPTS)


def test_zip_without_an_experiment_is_refused():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("readme.txt", "hi")
    with pytest.raises(IngestError, match="no Bruker experiment"):
        process_upload("bruker_zip", buf.getvalue(), SampleContext(), OPTS)


def test_jcamp_spectrum_is_picked_and_referenced():
    spec, ppm = h1_spectrum(offset=0.04)
    r = process_upload("jcampdx", jcamp_text(spec, ppm).encode(), SampleContext(), OPTS)
    assert r.referencing == "auto:TMS" and 1.25 in by_shift(r.peaks)
    assert r.context.solvent == Solvent.CDCL3


def test_jcamp_peak_table_becomes_a_peak_list():
    data = peaktable_text([(7.26, 100.0), (2.35, 40.0)]).encode()
    r = process_upload("jcampdx", data, SampleContext(), OPTS)
    assert [p.shift_ppm for p in r.peaks] == [7.26, 2.35] and r.referencing == "as_supplied"
    assert all(p.integral is None for p in r.peaks)


def test_a_spectrum_without_proton_peaks_is_refused():
    spec, ppm = h1_spectrum(npts=1024)
    spec[:] = 0.0
    with pytest.raises(ValueError):
        process_upload("jcampdx", jcamp_text(spec, ppm).encode(), SampleContext(), OPTS)


def test_small_impurity_position_peaks_are_tagged_but_large_ones_are_not(tmp_path):
    # ethyl acetate CH3CO at 2.05 (tiny) next to an analyte singlet of the same shift region
    peaks = [(0.0, 0.3), (2.05, 0.04), (3.0, 1.0), (4.5, 1.0), (6.0, 1.0), (7.26, 0.3)]
    r = process_upload(
        "bruker_zip", zip_dir(write_synthetic_bruker(tmp_path / "x", peaks=peaks)), SampleContext(), OPTS
    )
    roles = {round(p.shift_ppm, 2): p.role for p in r.peaks}
    assert roles[2.05] == PeakRole.IMPURITY_SUSPECT
    assert roles[3.0] == PeakRole.ANALYTE


def test_unknown_format_is_refused():
    with pytest.raises(IngestError, match="cannot process"):
        process_upload("peaklist", b"x", SampleContext(), OPTS)


REAL = "tests/fixtures/bruker/wadpressine_1h/10"


def _real_zip(tmp_path, names):
    import shutil
    from pathlib import Path

    dst = tmp_path / "10"
    dst.mkdir(parents=True)
    for n in names:
        src = Path(REAL) / n
        shutil.copytree(src, dst / n) if src.is_dir() else shutil.copy(src, dst / n)
    return zip_dir(dst, "wad/10")


def test_real_fid_only_data_is_auto_referenced_to_the_topspin_axis(tmp_path):
    """Phase 3 gate: the +0.019 ppm unreferenced offset on this dataset is removed using the MeOD line."""
    fid_only = process_upload("bruker_zip", _real_zip(tmp_path, ["acqus", "fid"]), SampleContext(), OPTS)
    topspin = process_upload(
        "bruker_zip", _real_zip(tmp_path / "t", ["acqus", "pdata"]), SampleContext(), OPTS
    )
    assert fid_only.referencing == "auto:CD3OD residual" and topspin.referencing == "procs"
    meod = [p for p in fid_only.peaks if abs(p.shift_ppm - 3.31) < 0.03]
    ref = [p for p in topspin.peaks if abs(p.shift_ppm - 3.31) < 0.03]
    assert meod and ref and meod[0].shift_ppm == pytest.approx(ref[0].shift_ppm, abs=0.01)


def test_real_1h_and_13c_experiments_in_one_zip(tmp_path):
    import shutil
    from pathlib import Path

    fixtures = Path("tests/fixtures/bruker")
    root = tmp_path / "wad"
    shutil.copytree(fixtures / "wadpressine_1h/10", root / "10")
    shutil.copytree(fixtures / "wadpressine_13c/12", root / "12")
    r = process_upload("bruker_zip", zip_dir(root, "wad"), SampleContext(), OPTS)
    assert r.referencing == "procs" and set(r.noise_sigma) == {"1H", "13C"}
    carbon = [p for p in r.peaks if p.nucleus == Nucleus.C13]
    assert carbon and max(p.height for p in carbon) == 1.0
    assert any(abs(p.shift_ppm - 214.3) < 0.2 for p in carbon)  # the ketone
    assert all(p.integral is None for p in carbon)


def test_a_peak_table_over_200_entries_keeps_the_tallest_and_says_so():
    pairs = [(0.9, 1000.0)] + [(2.0 + i * 0.01, 1.0 + i * 0.001) for i in range(250)]
    r = process_upload("jcampdx", peaktable_text(pairs).encode(), SampleContext(), OPTS)
    assert len(r.peaks) == 200 and any(abs(p.shift_ppm - 0.9) < 1e-6 for p in r.peaks)
    assert any("kept the tallest 200" in line for line in r.log)


def test_a_13c_peak_table_is_refused():
    data = peaktable_text([(77.0, 10.0), (128.0, 5.0)]).replace("^1H", "^13C").encode()
    with pytest.raises(IngestError, match="1H"):
        process_upload("jcampdx", data, SampleContext(), OPTS)


def test_13c_from_a_raw_fid_carries_the_unvalidated_note(tmp_path):
    from src.processing.pipeline import UNVALIDATED_13C_FID, _process_experiment

    exp = write_synthetic_bruker(tmp_path / "c", nucleus="13C")
    nucleus, _, _, log = _process_experiment(exp, OPTS)
    assert nucleus == "13C" and UNVALIDATED_13C_FID in log


def test_13c_from_processed_data_does_not_carry_it():
    from src.processing.pipeline import UNVALIDATED_13C_FID, _process_experiment

    _, _, _, log = _process_experiment(Path("tests/fixtures/bruker/wadpressine_13c/12"), OPTS)
    assert UNVALIDATED_13C_FID not in log


def test_an_unexpected_parser_error_becomes_a_400_style_ingest_error(monkeypatch):
    from src.processing import pipeline

    def boom(*a, **k):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(pipeline, "process_jcamp", boom)
    with pytest.raises(IngestError, match="could not be processed") as e:
        process_upload("jcampdx", b"##TITLE=x\n##END=\n", SampleContext(), OPTS)
    assert "secret" not in str(e.value)


def _silyl_zip(tmp_path, offset=0.0, tms=False):
    peaks = [(0.05, 1.0), (0.89, 1.5), (3.62, 0.7), (1.17, 0.5), (7.26, 0.4)]
    if tms:
        peaks.append((0.0, 0.3))
    return zip_dir(write_synthetic_bruker(tmp_path / "x", peaks=peaks, offset_ppm=offset))


def test_raw_data_keeps_a_silyl_line_the_referencing_rejected_as_analyte(tmp_path):
    from src.prompting.builder import prepare_request
    from src.schemas import AnalysisRequest

    req = AnalysisRequest(input_format="bruker_zip", context=SampleContext(solvent=Solvent.CDCL3))
    table, prompt, _ = prepare_request(req, _silyl_zip(tmp_path))
    near_zero = [p for p in table.peaks if abs(p.shift_ppm - 0.05) < 0.02]
    assert near_zero and near_zero[0].role == PeakRole.ANALYTE
    assert "silyl" in " ".join(table.processing_log)
    assert table.referencing.startswith("auto:CDCl3")


def test_raw_data_with_a_real_tms_line_still_tags_it(tmp_path):
    from src.prompting.builder import prepare_request
    from src.schemas import AnalysisRequest

    req = AnalysisRequest(input_format="bruker_zip", context=SampleContext(solvent=Solvent.CDCL3))
    table, _, _ = prepare_request(req, _silyl_zip(tmp_path, tms=True))
    zero = [p for p in table.peaks if abs(p.shift_ppm) < 0.02]
    assert zero and zero[0].role == PeakRole.REFERENCE


def test_already_referenced_data_with_a_silyl_line_is_checked_too():
    """A TopSpin-referenced spectrum whose solvent line is right and whose near-zero line is not at 0."""
    from src.processing.pipeline import _zero_line_rejected
    from src.schemas import Peak

    peaks = [
        Peak(id=1, nucleus=Nucleus.H1, shift_ppm=7.27, height=0.4),
        Peak(id=2, nucleus=Nucleus.H1, shift_ppm=0.05, height=1.0),
    ]
    log: list[str] = []
    assert _zero_line_rejected(peaks, "1H", Solvent.CDCL3, 400.0, log) and log
    peaks[1] = peaks[1].model_copy(update={"shift_ppm": 0.0})
    assert not _zero_line_rejected(peaks, "1H", Solvent.CDCL3, 400.0, [])
