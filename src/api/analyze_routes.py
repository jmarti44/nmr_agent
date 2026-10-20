"""POST /v1/analyze (peak-list input). The model and RDKit are imported lazily so /healthz stays light.

NOT SAFE TO EXPOSE PUBLICLY YET: this route spends Anthropic credits and has no rate limit, daily budget,
API key or demo mode (SPEC §9.4, planned for the hardening phase).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from src.api.uploads import REQUEST_BODY_DOC, read_analysis_request
from src.core.settings import Settings, get_settings
from src.llm.errors import LLMRefused, LLMSchemaFailure, LLMUnavailable
from src.prompting.builder import PromptTooLarge
from src.schemas import AnalysisResponse, ErrorResponse, PeakTable

router = APIRouter(prefix="/v1")

_ERRORS = {code: {"model": ErrorResponse} for code in (400, 451, 502, 503, 504)}


@lru_cache(maxsize=1)
def _client() -> Any:
    import anthropic

    s = get_settings()
    # Retries are done by the orchestrator inside the analysis deadline, so the SDK's own are off.
    return anthropic.Anthropic(max_retries=0, timeout=s.llm_call_timeout_s)


def get_llm_client() -> Any:
    """Dependency so tests can substitute a client; the real one is built on first use."""
    return _client()


def _error(status: int, code: str, detail: str, request_id: str, headers: dict | None = None) -> JSONResponse:
    body = ErrorResponse(error_code=code, detail=detail, request_id=request_id).model_dump()
    return JSONResponse(status_code=status, content=body, headers=headers)


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    tags=["analysis"],
    responses=_ERRORS,
    openapi_extra=REQUEST_BODY_DOC,
    summary="Analyze 1D NMR data: propose and verify candidate structures",
    description=(
        "Normalizes the peak list, asks the model for candidate structures, verifies every candidate with "
        "RDKit and literature shift rules, and returns them ranked. `accepted` means consistent with the "
        "supplied data, not proven correct. Send a JSON peak list, or multipart with a Bruker zip or "
        "JCAMP-DX file (`file`) and the request JSON (`metadata`)."
    ),
)
async def analyze(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    client: Annotated[Any, Depends(get_llm_client)],
):
    from src.analysis import new_request_id, run_analysis

    request_id = new_request_id()
    try:
        req, upload = await read_analysis_request(request)
        return await run_in_threadpool(run_analysis, req, settings, client, request_id, upload)
    except (PromptTooLarge, ValueError) as e:
        return _error(400, "INVALID_INPUT", str(e), request_id)
    except LLMRefused as e:
        return _error(
            451, "LLM_REFUSED", f"the model declined this request (category: {e.category})", request_id
        )
    except LLMSchemaFailure as e:
        return _error(502, "LLM_SCHEMA_FAILURE", f"the model's output never validated: {e}"[:600], request_id)
    except LLMUnavailable as e:
        if e.reason == "timeout":
            return _error(504, "TIMEOUT", e.detail, request_id)
        if e.reason == "no_credentials":
            return _error(503, "LLM_UNAVAILABLE", e.detail, request_id)
        headers = {"Retry-After": e.retry_after} if e.retry_after else None
        return _error(502, "LLM_UNAVAILABLE", f"{e.reason}: {e.detail}", request_id, headers)


@router.post(
    "/process",
    response_model=PeakTable,
    tags=["analysis"],
    responses={code: {"model": ErrorResponse} for code in (400,)},
    openapi_extra=REQUEST_BODY_DOC,
    summary="Preprocess only: raw data or a peak list to a normalized peak table (no model call)",
    description=(
        "Runs everything before the model: unpacking, FID processing or spectrum reading, peak picking, "
        "referencing, solvent tagging and integral normalization. Free to call. Use it to check what the "
        "model would be shown, and lower `options.k_sigma` effects before spending an analysis."
    ),
)
async def process(request: Request):
    from src.analysis import new_request_id
    from src.prompting.builder import prepare_request

    request_id = new_request_id()
    try:
        req, upload = await read_analysis_request(request)
        table, _, _ = await run_in_threadpool(prepare_request, req, upload)
        return table
    except (PromptTooLarge, ValueError) as e:
        return _error(400, "INVALID_INPUT", str(e), request_id)
