# ruff: noqa: E501
"""Peak memory of a whole /v1/analyze request (upload -> processing -> prompt -> model -> RDKit verify), with a
fake model so no credits are spent. Run each case in a fresh process:

    PYTHONPATH=. uv run python bench/memory_probe.py real      # the Wadpressine 1H experiment, zipped
    PYTHONPATH=. uv run python bench/memory_probe.py largest   # a FID at the size limit
    PYTHONPATH=. uv run python bench/memory_probe.py peaklist  # JSON peak list only
"""

import io
import json
import re
import resource
import sys
import zipfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from src.api.analyze_routes import get_llm_client
from src.api.main import app
from src.core.settings import Settings, get_settings
from src.ingest.safety import MAX_FID_POINTS
from tests.llm.fake_api import FakeAPI, hypothesis_json, message
from tests.prompting.helpers import ASPIRIN_DMSO

case = sys.argv[1] if len(sys.argv) > 1 else "real"
fix = Path("tests/fixtures/bruker/wadpressine_1h/10")


def rss_mib() -> int:
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale // 2**20


def build_upload() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        if case == "real":
            for f in fix.rglob("*"):
                if f.is_file():
                    z.write(f, f"e/{f.relative_to(fix)}")
        else:
            td = 2 * MAX_FID_POINTS
            acqus = re.sub(r"##\$TD= *\d+", f"##$TD= {td}", (fix / "acqus").read_text(errors="replace"))
            order = "<" if re.search(r"##\$BYTORDA= *0", acqus) else ">"
            t = np.arange(td // 2)
            rng = np.random.default_rng(1)
            sig = 1e6 * np.exp(-t / 4e5) * np.exp(2j * np.pi * 0.05 * t) + rng.normal(0, 1e3, t.size)
            inter = np.empty(td)
            inter[0::2], inter[1::2] = sig.real, sig.imag
            z.writestr("e/acqus", acqus)
            z.writestr("e/fid", inter.astype(order + "i4").tobytes())
    return buf.getvalue()


upload = build_upload() if case != "peaklist" else b""
api = FakeAPI([message(hypothesis_json()), message(hypothesis_json())])
app.dependency_overrides[get_llm_client] = lambda: api.client()
app.dependency_overrides[get_settings] = lambda: Settings()
client = TestClient(app)
before = rss_mib()
if case == "peaklist":
    r = client.post("/v1/analyze", json=ASPIRIN_DMSO)
else:
    meta = {"input_format": "bruker_zip", "context": {"molecular_formula": "C9H8O4"}}
    r = client.post("/v1/analyze", files={"file": ("x.zip", upload)}, data={"metadata": json.dumps(meta)})
print(
    f"{case}: HTTP {r.status_code}, peak RSS {rss_mib()} MiB (before the request {before} MiB), upload {len(upload) / 1e6:.1f} MB"
)
