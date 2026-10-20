"""Upload limits and zip extraction (SPEC §5.1). Checks run before anything is parsed."""

from __future__ import annotations

import stat
import zipfile
import zlib
from pathlib import Path, PurePosixPath

from src.ingest.errors import IngestError

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_ZIP_FILES = 200
# Zip bombs: sizes are checked from the zip directory before anything is written, and again while copying.
# MAX_FID_POINTS (complex points, TD = 2x) is the largest FID the 512 MB host can process: the 2^21 first
# chosen needed 659 MiB. Real 1D data are far smaller (1H TD 32k-128k, 13C 64k). The byte limits follow:
# a double-precision FID at the limit is 8 MB, so one member may be 8 MB and one 1H + 13C experiment with
# processed data fits in 32 MB.
MAX_FID_POINTS = 2**18
MAX_MEMBER_BYTES = 2 * MAX_FID_POINTS * 8
MAX_UNPACKED_BYTES = 32 * 1024 * 1024
MAX_SEARCH_DEPTH = 3


def check_upload_size(data: bytes) -> None:
    if len(data) > MAX_UPLOAD_BYTES:
        raise IngestError(f"upload is {len(data) / 1e6:.1f} MB; the limit is {MAX_UPLOAD_BYTES // 2**20} MB")
    if not data:
        raise IngestError("upload is empty")


def _safe_member(name: str) -> PurePosixPath | None:
    """Relative path of a zip member, or None for entries to skip (directories, macOS metadata)."""
    if "\\" in name or "\x00" in name:
        raise IngestError(f"unsafe path in zip: {name!r}")
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts:
        raise IngestError(f"unsafe path in zip: {name!r}")
    if name.endswith("/") or "__MACOSX" in p.parts or p.name == ".DS_Store":
        return None
    return p


def extract_zip(data: bytes, dest: Path) -> None:
    """Extract an in-memory zip into `dest` (an empty per-request directory), enforcing the limits."""
    import io

    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise IngestError("the upload is not a valid zip file") from e
    with zf:
        infos = zf.infolist()
        if len(infos) > MAX_ZIP_FILES:
            raise IngestError(f"zip has {len(infos)} entries; the limit is {MAX_ZIP_FILES}")
        if sum(i.file_size for i in infos) > MAX_UNPACKED_BYTES:
            raise IngestError("zip expands to more than the allowed size")
        if any(i.file_size > MAX_MEMBER_BYTES for i in infos):
            raise IngestError(f"a file in the zip is larger than {MAX_MEMBER_BYTES // 2**20} MB")
        root = dest.resolve()
        for info in infos:
            rel = _safe_member(info.filename)
            if rel is None:
                continue
            if stat.S_ISLNK(info.external_attr >> 16):
                raise IngestError(f"zip contains a symbolic link: {info.filename!r}")
            if info.flag_bits & 0x1:
                raise IngestError("zip is password protected")
            target = (root / rel).resolve()
            if target == root or not target.is_relative_to(root):
                raise IngestError(f"unsafe path in zip: {info.filename!r}")
            written = 0
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as out:
                    while chunk := src.read(1 << 20):
                        written += len(chunk)
                        if written > info.file_size:  # the header lied about the size
                            raise IngestError("zip member is larger than its declared size")
                        out.write(chunk)
            except OSError as e:  # e.g. a file and a directory with the same name
                raise IngestError(f"zip member {info.filename!r} cannot be extracted") from e
            except (zipfile.BadZipFile, zlib.error, EOFError, NotImplementedError) as e:
                raise IngestError(
                    f"zip member {info.filename!r} is corrupt or uses an unsupported method"
                ) from e


def find_experiments(root: Path) -> list[Path]:
    """Directories (up to MAX_SEARCH_DEPTH below root) with `acqus` and a `fid` or a processed pdata/N/1r."""
    found = []
    for acqus in sorted(root.rglob("acqus")):
        d = acqus.parent
        if len(d.relative_to(root).parts) > MAX_SEARCH_DEPTH:
            continue
        if (d / "fid").is_file() or any(d.glob("pdata/*/1r")):
            found.append(d)
        elif (d / "ser").is_file():
            raise IngestError("the experiment is multidimensional (ser file); only 1D data is supported")
    return found


def check_data_file_sizes(exp: Path) -> None:
    """Refuse FID and processed-data files too big to be what their parameter files say, before reading them.

    nmrglue loads a whole file into memory, so an all-zero 300 MB `fid` that compressed to 300 KB would
    otherwise cost gigabytes. The expected size comes from TD (acqus) and SI (procs).
    """
    import re

    def number(path: Path, key: str) -> int | None:
        try:
            m = re.search(rf"##\${key}=\s*(\d+)", path.read_text(errors="replace")[:200_000])
        except OSError:
            return None
        return int(m.group(1)) if m else None

    def check(data: Path, count: int | None, label: str) -> None:
        size = data.stat().st_size
        if size > MAX_MEMBER_BYTES:
            raise IngestError(f"the {label} file is larger than {MAX_MEMBER_BYTES // 2**20} MB")
        if count is not None and count > 2 * MAX_FID_POINTS:
            raise IngestError(f"the {label} claims {count} points; the limit is {2 * MAX_FID_POINTS}")
        # 8 bytes per value covers doubles; Bruker pads files to 1024-byte blocks.
        if count is not None and size > count * 8 + 4096:
            raise IngestError(f"the {label} file is much larger than its parameters say ({count} points)")

    fid = exp / "fid"
    if fid.is_file():
        check(fid, number(exp / "acqus", "TD"), "FID")
    for r in exp.glob("pdata/*/1r"):
        check(r, number(r.parent / "procs", "SI"), "processed spectrum")
