"""Parse a request that is either a JSON AnalysisRequest or multipart (`file` + `metadata` JSON string)."""

from __future__ import annotations

import json

from fastapi import Request
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from src.ingest import safety
from src.schemas import AnalysisRequest

REQUEST_BODY_DOC = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {"schema": {"$ref": "#/components/schemas/AnalysisRequest"}},
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file", "metadata"],
                    "properties": {
                        "file": {"type": "string", "format": "binary", "description": "zip or .jdx/.dx"},
                        "metadata": {
                            "type": "string",
                            "description": "JSON string of an AnalysisRequest (jcampdx or bruker_zip)",
                        },
                    },
                }
            },
        },
    }
}


def _validation_message(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()[:5])


async def read_analysis_request(request: Request) -> tuple[AnalysisRequest, bytes | None]:
    """-> (validated request, uploaded bytes or None). Raises ValueError with a caller-safe message."""
    ctype = request.headers.get("content-type", "")
    try:
        if ctype.startswith("multipart/form-data"):
            form = await request.form(max_files=1, max_fields=2)
            file, meta = form.get("file"), form.get("metadata")
            if not isinstance(file, UploadFile):
                raise ValueError("multipart request needs a 'file' part")
            if not isinstance(meta, str):
                raise ValueError("multipart request needs a 'metadata' part holding the request JSON")
            data = await file.read(safety.MAX_UPLOAD_BYTES + 1)
            return AnalysisRequest.model_validate(json.loads(meta)), data
        return AnalysisRequest.model_validate(await request.json()), None
    except ValidationError as e:
        raise ValueError(_validation_message(e)) from e
    except json.JSONDecodeError as e:
        raise ValueError("the request body (or 'metadata') is not valid JSON") from e
