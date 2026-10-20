import numpy as np
import pytest

from src.ingest.errors import IngestError
from src.ingest.jcamp import parse_peak_table, read_jcamp
from tests.spectra import MHZ, h1_spectrum, jcamp_text, peaktable_text


def write(tmp_path, text):
    f = tmp_path / "a.jdx"
    f.write_text(text)
    return f


def test_spectrum_roundtrip_keeps_axis_and_header(tmp_path):
    spec, ppm = h1_spectrum()
    j = read_jcamp(write(tmp_path, jcamp_text(spec, ppm)))
    assert j.kind == "spectrum" and j.nucleus == "1H" and j.sfo1_mhz == MHZ and j.solvent == "CDCl3"
    assert j.ppm[0] > j.ppm[-1]
    assert j.ppm[0] == pytest.approx(14.0) and j.ppm[-1] == pytest.approx(-1.0)
    assert np.allclose(j.y, spec, atol=1e-3)


def test_ascending_axis_is_flipped_to_descending(tmp_path):
    spec, ppm = h1_spectrum(npts=1024)
    j = read_jcamp(write(tmp_path, jcamp_text(spec[::-1], ppm[::-1])))
    assert j.ppm[0] > j.ppm[-1]
    assert np.allclose(j.y, spec, atol=1e-3)  # data follow the axis


def test_hz_axis_is_converted_with_the_observe_frequency(tmp_path):
    spec, ppm = h1_spectrum(npts=1024)
    j = read_jcamp(write(tmp_path, jcamp_text(spec, ppm, units="HZ")))
    assert j.ppm[0] == pytest.approx(14.0)


def test_hz_without_observe_frequency_is_refused(tmp_path):
    spec, ppm = h1_spectrum(npts=1024)
    text = jcamp_text(spec, ppm, units="HZ").replace("##.OBSERVE FREQUENCY=400.0\n", "")
    with pytest.raises(IngestError, match="OBSERVE FREQUENCY"):
        read_jcamp(write(tmp_path, text))


def test_carbon_nucleus_is_recognised(tmp_path):
    spec, ppm = h1_spectrum(npts=1024)
    assert read_jcamp(write(tmp_path, jcamp_text(spec, ppm, nucleus="13C"))).nucleus == "13C"


def test_other_nuclei_are_refused(tmp_path):
    spec, ppm = h1_spectrum(npts=1024)
    with pytest.raises(IngestError, match="31P"):
        read_jcamp(write(tmp_path, jcamp_text(spec, ppm, nucleus="31P")))


def test_peak_table_block_is_read_as_peaks(tmp_path):
    j = read_jcamp(write(tmp_path, peaktable_text([(7.26, 100.0), (2.35, 40.0)])))
    assert j.kind == "peaktable" and j.peaks == [(7.26, 100.0), (2.35, 40.0)]


def test_peak_table_in_hz_is_converted(tmp_path):
    j = read_jcamp(write(tmp_path, peaktable_text([(MHZ * 2.0, 10.0)], units="HZ")))
    assert j.peaks[0][0] == pytest.approx(2.0)


def test_peak_table_pair_syntax_variants():
    text = "##PEAK TABLE=(XY..XY)\n7.26, 100; 2.35, 40\n1.2 5 3.3 7 $$ comment\n##END=\n"
    assert parse_peak_table(text) == [(7.26, 100.0), (2.35, 40.0), (1.2, 5.0), (3.3, 7.0)]


def test_peak_table_with_an_odd_number_of_values_is_refused():
    with pytest.raises(IngestError, match="pair"):
        parse_peak_table("##PEAK TABLE=(XY..XY)\n7.26, 100; 2.35\n##END=\n")


def test_file_without_any_data_is_refused(tmp_path):
    with pytest.raises(IngestError):
        read_jcamp(write(tmp_path, "##TITLE=empty\n##JCAMP-DX=5.00\n##END=\n"))


def test_garbage_is_refused(tmp_path):
    with pytest.raises(IngestError):
        read_jcamp(write(tmp_path, "this is not a jcamp file\n" * 5))


def _dup_file(lines: int) -> str:
    body = "\n".join(f"{i} @T999999" for i in range(lines))
    head = "##TITLE= dup\n##JCAMP-DX= 5.01\n##DATA TYPE= NMR SPECTRUM\n##XUNITS= PPM\n##FIRSTX= 12\n"
    head += "##LASTX= -1\n##NPOINTS= 1000\n##.OBSERVE FREQUENCY= 400\n##.OBSERVE NUCLEUS= ^1H\n"
    return f"{head}##XYDATA= (X++(Y..Y))\n{body}\n##END=\n"


def test_dup_runs_that_expand_beyond_the_limit_are_refused_before_decoding(tmp_path):
    with pytest.raises(IngestError, match="expand"):
        read_jcamp(write(tmp_path, _dup_file(200)))


def test_a_declared_point_count_beyond_the_limit_is_refused(tmp_path):
    spec, ppm = h1_spectrum(npts=1024)
    text = jcamp_text(spec, ppm).replace("##NPOINTS=1024", "##NPOINTS=99999999")
    with pytest.raises(IngestError, match="declares"):
        read_jcamp(write(tmp_path, text))


def test_a_large_ordinary_file_is_not_mistaken_for_a_bomb(tmp_path):
    spec, ppm = h1_spectrum(npts=65536)
    assert read_jcamp(write(tmp_path, jcamp_text(spec, ppm))).y.size == 65536
