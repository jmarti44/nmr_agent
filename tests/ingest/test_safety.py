import io
import zipfile
from pathlib import Path

import pytest

from src.ingest import safety
from src.ingest.errors import IngestError
from src.ingest.safety import check_upload_size, extract_zip, find_experiments


def make_zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_extracts_into_the_destination(tmp_path):
    extract_zip(make_zip({"exp/1/fid": b"x", "exp/1/acqus": b"y"}), tmp_path)
    assert (tmp_path / "exp/1/fid").read_bytes() == b"x"


@pytest.mark.parametrize("name", ["../evil", "a/../../evil", "/abs/evil", "a\\..\\evil"])
def test_path_traversal_and_absolute_members_are_refused(tmp_path, name):
    with pytest.raises(IngestError, match="unsafe path"):
        extract_zip(make_zip({name: b"x"}), tmp_path)
    assert not list(tmp_path.parent.glob("evil"))


def test_symlink_members_are_refused(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        info = zipfile.ZipInfo("link")
        info.external_attr = (0o120777) << 16
        z.writestr(info, "/etc/passwd")
    with pytest.raises(IngestError, match="symbolic link"):
        extract_zip(buf.getvalue(), tmp_path)


def test_too_many_files_are_refused(tmp_path):
    members = {f"f{i}": b"" for i in range(safety.MAX_ZIP_FILES + 1)}
    with pytest.raises(IngestError, match="entries"):
        extract_zip(make_zip(members), tmp_path)


def test_zip_that_expands_too_much_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(safety, "MAX_UNPACKED_BYTES", 100)
    with pytest.raises(IngestError, match="expands"):
        extract_zip(make_zip({"a": b"0" * 1000}), tmp_path)


def test_a_corrupt_member_is_refused(tmp_path):
    data = bytearray(make_zip({"a": b"0123456789" * 50}))
    data[data.index(b"0123")] ^= 0xFF  # flips a stored byte, so the CRC no longer matches
    with pytest.raises(IngestError, match="corrupt"):
        extract_zip(bytes(data), tmp_path)


def test_not_a_zip_is_refused(tmp_path):
    with pytest.raises(IngestError, match="not a valid zip"):
        extract_zip(b"hello", tmp_path)


def test_upload_size_limit():
    check_upload_size(b"x")
    with pytest.raises(IngestError, match="limit"):
        check_upload_size(b"x" * (safety.MAX_UPLOAD_BYTES + 1))
    with pytest.raises(IngestError, match="empty"):
        check_upload_size(b"")


def test_macos_metadata_is_skipped(tmp_path):
    extract_zip(make_zip({"__MACOSX/._fid": b"x", "a/.DS_Store": b"x", "a/acqus": b"y"}), tmp_path)
    assert sorted(p.name for p in tmp_path.rglob("*") if p.is_file()) == ["acqus"]


def _touch(root: Path, *names: str):
    for n in names:
        (root / n).parent.mkdir(parents=True, exist_ok=True)
        (root / n).write_bytes(b"x")


def test_experiments_are_found_up_to_depth_three(tmp_path):
    _touch(tmp_path, "a/b/10/fid", "a/b/10/acqus")
    assert find_experiments(tmp_path) == [tmp_path / "a/b/10"]


def test_deeper_experiments_are_not_searched(tmp_path):
    _touch(tmp_path, "a/b/c/d/10/fid", "a/b/c/d/10/acqus")
    assert find_experiments(tmp_path) == []


def test_processed_only_experiments_count(tmp_path):
    _touch(tmp_path, "10/acqus", "10/pdata/1/1r")
    assert find_experiments(tmp_path) == [tmp_path / "10"]


def test_2d_data_is_refused(tmp_path):
    _touch(tmp_path, "10/ser", "10/acqus")
    with pytest.raises(IngestError, match="multidimensional"):
        find_experiments(tmp_path)


def test_compressed_zero_fid_is_refused_before_anything_is_written(tmp_path):
    """300 MB of zeros deflates to ~300 KB; the directory shows its size, so nothing is extracted or read."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("e/acqus", "##$TD= 65536\n")
        with z.open("e/fid", "w", force_zip64=True) as f:
            for _ in range(300):
                f.write(b"\0" * (1 << 20))
    assert len(buf.getvalue()) < 1_000_000
    with pytest.raises(IngestError, match="expands"):
        extract_zip(buf.getvalue(), tmp_path)
    assert not list(tmp_path.rglob("fid"))


def test_one_oversized_member_below_the_total_limit_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(safety, "MAX_MEMBER_BYTES", 1000)
    with pytest.raises(IngestError, match="larger than"):
        extract_zip(make_zip({"a": b"0" * 2000}), tmp_path)


def test_fid_much_larger_than_td_says_is_refused_without_reading(tmp_path):
    _touch(tmp_path, "10/acqus")
    (tmp_path / "10/acqus").write_text("##$TD= 1000\n")
    (tmp_path / "10/fid").write_bytes(b"\0" * 2_000_000)
    with pytest.raises(IngestError, match="much larger than its parameters"):
        safety.check_data_file_sizes(tmp_path / "10")


def test_processed_spectrum_much_larger_than_si_says_is_refused(tmp_path):
    (tmp_path / "pdata/1").mkdir(parents=True)
    (tmp_path / "pdata/1/procs").write_text("##$SI= 1024\n")
    (tmp_path / "pdata/1/1r").write_bytes(b"\0" * 2_000_000)
    with pytest.raises(IngestError, match="much larger than its parameters"):
        safety.check_data_file_sizes(tmp_path)


def test_a_td_beyond_the_point_limit_is_refused(tmp_path):
    (tmp_path / "acqus").write_text(f"##$TD= {4 * safety.MAX_FID_POINTS}\n")
    (tmp_path / "fid").write_bytes(b"\0" * 100)
    with pytest.raises(IngestError, match="limit"):
        safety.check_data_file_sizes(tmp_path)


def test_a_normal_experiment_passes_the_size_check():
    safety.check_data_file_sizes(Path("tests/fixtures/bruker/wadpressine_1h/10"))


def test_member_named_dot_is_refused(tmp_path):
    with pytest.raises(IngestError):
        extract_zip(make_zip({".": b"x"}), tmp_path)


def test_a_file_and_a_directory_with_the_same_name_is_refused(tmp_path):
    with pytest.raises(IngestError, match="cannot be extracted"):
        extract_zip(make_zip({"a": b"x", "a/acqus": b"y"}), tmp_path)


def test_td_at_the_limit_passes_and_one_block_over_is_refused(tmp_path):
    (tmp_path / "fid").write_bytes(b"\0" * 1024)
    (tmp_path / "acqus").write_text(f"##$TD= {2 * safety.MAX_FID_POINTS}\n")
    (tmp_path / "fid").write_bytes(b"\0" * (2 * safety.MAX_FID_POINTS * 4))
    safety.check_data_file_sizes(tmp_path)
    (tmp_path / "acqus").write_text(f"##$TD= {2 * safety.MAX_FID_POINTS + 256}\n")
    with pytest.raises(IngestError, match="limit"):
        safety.check_data_file_sizes(tmp_path)


def test_peak_memory_at_the_largest_accepted_fid_fits_the_host(tmp_path):
    """The service imports plus the biggest FID the limits accept must stay well under the 512 MB host."""
    import re
    import subprocess
    import sys

    import numpy as np

    td = 2 * safety.MAX_FID_POINTS
    acqus = Path("tests/fixtures/bruker/wadpressine_1h/10/acqus").read_text(errors="replace")
    acqus = re.sub(r"##\$TD= *\d+", f"##$TD= {td}", acqus)
    acqus = re.sub(r"##\$DTYPA= *\d+", "##$DTYPA= 0", acqus)
    order = "<" if re.search(r"##\$BYTORDA= *0", acqus) else ">"
    t = np.arange(td // 2)
    rng = np.random.default_rng(1)
    sig = 1e6 * np.exp(-t / 4e5) * np.exp(2j * np.pi * 0.05 * t) + rng.normal(0, 1e3, t.size)
    inter = np.empty(td)
    inter[0::2], inter[1::2] = sig.real, sig.imag
    (tmp_path / "up.zip").write_bytes(
        make_zip({"e/acqus": acqus.encode(), "e/fid": inter.astype(order + "i4").tobytes()})
    )
    child = (
        "import resource, sys\n"
        "from src.processing.pipeline import process_upload\n"
        "from src.schemas import ProcessingOptions, SampleContext\n"
        "data = open(sys.argv[1], 'rb').read()\n"
        "process_upload('bruker_zip', data, SampleContext(), ProcessingOptions())\n"
        "scale = 1 if sys.platform == 'darwin' else 1024\n"
        "print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale // 2**20)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", child, str(tmp_path / "up.zip")],
        capture_output=True,
        text=True,
        timeout=300,
        env={**__import__("os").environ, "PYTHONPATH": "."},
    )
    assert out.returncode == 0, out.stderr[-500:]
    assert int(out.stdout.strip().splitlines()[-1]) < 400
