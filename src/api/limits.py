"""Request body size limit, enforced while the body is received (not after it has been read in full)."""

from __future__ import annotations

import json
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.ingest import safety

JSON_LIMIT_BYTES = 1024 * 1024
MULTIPART_OVERHEAD_BYTES = 64 * 1024


class BodyTooLarge(Exception):
    pass


def limit_for(content_type: str) -> int:
    """Largest body accepted for this content type (read at call time so tests can lower it)."""
    if content_type.startswith("multipart/form-data"):
        return safety.MAX_UPLOAD_BYTES + MULTIPART_OVERHEAD_BYTES
    return JSON_LIMIT_BYTES


class BodyLimitMiddleware:
    """Return 413 once a request body passes its limit, by Content-Length or by counting received bytes."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        limit = limit_for(headers.get(b"content-type", b"").decode("latin-1"))
        length = headers.get(b"content-length", b"")
        if length.isdigit() and int(length) > limit:
            await self._reject(send, limit)
            return

        received = 0
        started = False

        async def counting_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise BodyTooLarge
            return message

        async def tracking_send(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except BodyTooLarge:
            if started:
                raise
            await self._reject(send, limit)

    @staticmethod
    async def _reject(send: Send, limit: int) -> None:
        body = json.dumps(
            {
                "error_code": "PAYLOAD_TOO_LARGE",
                "detail": f"request body is larger than the {limit} byte limit for this content type",
                "request_id": f"req_{uuid.uuid4().hex[:16]}",
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
