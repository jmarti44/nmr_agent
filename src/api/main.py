"""FastAPI app. /healthz imports nothing heavy so it answers while RDKit/nmrglue are still cold (SPEC F8)."""

import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.verify_routes import router as verify_router
from src.core.version import SERVICE_VERSION

app = FastAPI(
    title="NMR-Agent",
    version=SERVICE_VERSION,
    description="LLM-orchestrated 1D NMR interpretation with deterministic verification.",
    openapi_tags=[
        {"name": "verification", "description": "Check candidate structures against a peak table."},
        {"name": "reference", "description": "The literature tables the verifier uses, with sources."},
        {"name": "ops", "description": "Liveness."},
    ],
)
app.include_router(verify_router)


@app.exception_handler(RequestValidationError)
async def _invalid_input(request: Request, exc: RequestValidationError) -> JSONResponse:
    """SPEC §9.3: malformed requests are 400 INVALID_INPUT, not FastAPI's default 422."""
    detail = "; ".join(f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()[:5])
    body = {"error_code": "INVALID_INPUT", "detail": detail, "request_id": f"req_{uuid.uuid4().hex[:16]}"}
    return JSONResponse(status_code=400, content=body)


@app.get("/healthz", tags=["ops"])
def healthz() -> dict[str, str]:
    return {"status": "ok"}
