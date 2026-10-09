"""FastAPI app. /healthz imports nothing heavy so it answers while RDKit/nmrglue are still cold (SPEC F8)."""

from fastapi import FastAPI

from src.core.version import SERVICE_VERSION

app = FastAPI(
    title="NMR-Agent",
    version=SERVICE_VERSION,
    description="LLM-orchestrated 1D NMR interpretation with deterministic verification.",
)


@app.get("/healthz", tags=["ops"])
def healthz() -> dict[str, str]:
    return {"status": "ok"}
