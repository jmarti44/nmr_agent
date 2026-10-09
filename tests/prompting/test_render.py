from pathlib import Path

from src.processing.normalize import normalize_peaklist
from src.prompting.render import render_context_block, render_peak_table
from src.schemas import Nucleus, Peak, PeakRole, PeakTable, SampleContext
from tests.prompting.helpers import aspirin_request

GOLDEN = Path(__file__).parent / "golden"


def _table(req=None):
    req = req or aspirin_request()
    return normalize_peaklist(req.peaks, req.context), req.context


def test_aspirin_message_matches_the_golden_file():
    from src.prompting import build_prompt_for_request

    _, prompt = build_prompt_for_request(aspirin_request())
    assert prompt.user_text + "\n" == (GOLDEN / "aspirin_dmso_user.txt").read_text()


def test_rendering_is_deterministic():
    t, c = _table()
    assert render_peak_table(t, c) == render_peak_table(t, c)


def test_numbers_are_rounded_for_cache_stability():
    peaks = [
        Peak(id=1, nucleus=Nucleus.H1, shift_ppm=7.26493, integral=1.04, coupling_hz=[7.84, 1.61]),
        Peak(id=2, nucleus=Nucleus.C13, shift_ppm=77.0449),
    ]
    text = render_peak_table(PeakTable(peaks=peaks), SampleContext())
    assert "7.26 " in text and "7.26493" not in text
    assert "7.8, 1.6" in text and "1.0 " in text
    assert "2:77.0" in text and "77.04" not in text


def test_header_shows_frequency_solvent_and_referencing_when_known():
    t, c = _table()
    assert (
        render_peak_table(t, c).splitlines()[0]
        == "Nucleus: 1H, 400.13 MHz, solvent DMSO-d6, referencing: as_supplied"
    )
    bare = render_peak_table(
        PeakTable(peaks=[Peak(id=1, nucleus=Nucleus.H1, shift_ppm=1.0)]), SampleContext()
    )
    assert bare.splitlines()[0] == "Nucleus: 1H, solvent not given, referencing: as_supplied"


def test_formula_line_has_h_count_and_dou_or_says_not_supplied():
    t, c = _table()
    assert "Formula (HRMS): C9H8O4   (H = 8, DoU = 6)" in render_peak_table(t, c)
    assert "Formula: not supplied" in render_peak_table(t, SampleContext())


def test_roles_labels_and_missing_values_are_visible_to_the_model():
    t, c = _table()
    rows = {
        int(line.split()[0]): line
        for line in render_peak_table(t, c).splitlines()[3:]
        if line.split()[0].isdigit()
    }
    assert "analyte   (exchangeable?)" in rows[1]
    assert "solvent   (DMSO-d6 residual)" in rows[6]
    assert rows[6].split()[4] == "-"  # a residual-solvent integral is not shown
    assert "dd" in rows[2] and "7.8, 1.6" in rows[2]


def test_13c_line_lists_ids_so_claims_can_cite_them_and_marks_non_analyte():
    t, c = _table()
    line = [x for x in render_peak_table(t, c).splitlines() if x.startswith("13C")][0]
    assert line == (
        "13C (100.61 MHz) signals as id:shift(ppm): 8:169.1, 9:166.3, 10:150.3, 11:134.2, 12:131.2, "
        "13:126.1, 14:124.0, 15:123.8, 16:39.5[solvent], 17:20.8"
    )


def test_no_13c_line_when_there_are_no_13c_peaks():
    t = PeakTable(peaks=[Peak(id=1, nucleus=Nucleus.H1, shift_ppm=2.0, integral=3.0)])
    assert "13C" not in render_peak_table(t, SampleContext())


def test_every_peak_id_in_the_table_appears_in_the_text():
    t, c = _table()
    text = render_peak_table(t, c)
    for p in t.peaks:
        if p.nucleus == Nucleus.H1:
            assert any(
                line.split()[0] == str(p.id) for line in text.splitlines() if line.split()[0].isdigit()
            )
        else:
            assert f"{p.id}:{p.shift_ppm:.1f}" in text


def test_context_block_lists_only_what_was_given_and_the_request_size():
    assert (
        render_context_block(SampleContext(), 3)
        == "Propose up to 3 candidate structure(s) (1 to 3), best first."
    )
    ctx = SampleContext(
        observed_exact_mass=180.0423,
        starting_material_smiles="OC(=O)c1ccccc1O",
        d2o_exchange_performed=True,
        notes="n",
    )
    block = render_context_block(ctx, 1)
    assert block.startswith("Sample details:")
    for needle in (
        "180.0423 Da",
        "OC(=O)c1ccccc1O",
        "D2O exchange was performed",
        "Notes from the chemist: n",
    ):
        assert needle in block
    assert block.endswith("(1 to 1), best first.")


def test_role_enum_values_in_the_text_are_the_schema_values():
    t, c = _table()
    text = render_peak_table(t, c)
    assert PeakRole.SOLVENT.value in text and PeakRole.ANALYTE.value in text
