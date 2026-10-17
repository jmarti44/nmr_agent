"""A fake Anthropic API behind the real SDK (an httpx2 mock transport), so retries, headers, usage parsing
and exception types are exercised for real. Responses are hand-written to the documented Messages API
shape; they are NOT recordings of live responses.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx2

from src.schemas import LLMHypothesis


def message(
    text: str,
    *,
    stop_reason: str = "end_turn",
    thinking: bool = True,
    usage: dict | None = None,
    iterations: list | None = None,
    stop_details: dict | None = None,
    msg_id: str = "msg_01",
    model: str = "claude-opus-5-5",
    blocks: list[dict] | None = None,
) -> dict:
    content: list[dict] = list(blocks) if blocks is not None else []
    if blocks is None and thinking:
        content.append({"type": "thinking", "thinking": "", "signature": "sig_abc"})
    if text and blocks is None:
        content.append({"type": "text", "text": text})
    u = {
        "input_tokens": 400,
        "output_tokens": 300,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
    }
    u.update(usage or {})
    if iterations is not None:
        u["iterations"] = iterations
    return {
        "id": msg_id, "type": "message", "role": "assistant", "model": model, "content": content,
        "stop_reason": stop_reason, "stop_sequence": None, "stop_details": stop_details, "usage": u,
    }  # fmt: skip


def error(status: int, kind: str, text: str, headers: dict | None = None) -> httpx2.Response:
    body = {"type": "error", "error": {"type": kind, "message": text}}
    return httpx2.Response(status, json=body, headers=headers or {})


@dataclass
class FakeAPI:
    """Queue of replies: a dict (200 message), an httpx2.Response, or an Exception raised by the transport."""

    replies: list[Any]
    requests: list[dict] = field(default_factory=list)

    def handler(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append(
            {"path": request.url.path, "beta": request.headers.get("anthropic-beta"), "body": body}
        )
        if not self.replies:
            raise AssertionError("the fake API ran out of replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, httpx2.Response):
            return reply
        return httpx2.Response(200, json=reply)

    def client(self) -> anthropic.Anthropic:
        http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(self.handler))
        return anthropic.Anthropic(api_key="sk-ant-test", http_client=http, max_retries=0)


# ---- hypothesis payloads -----------------------------------------------------------------------------------

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ISOMER = "COC(=O)c1ccccc1C(=O)O"  # wrong structure with the same formula


def hypothesis(smiles: list[str] | None = None, groups: list[dict] | None = None, **over: Any) -> dict:
    smiles = smiles or [ASPIRIN]
    h = {
        "formula_inferred": None,
        "degree_of_unsaturation": 6,
        "functional_groups": groups
        if groups is not None
        else [
            {
                "group": "carboxylic_acid", "present": True, "confidence": 0.9,
                "evidence": {"peak_ids": [1], "note": "broad signal at 12.95"},
            },
            {
                "group": "aromatic_ring", "present": True, "confidence": 0.9,
                "evidence": {"peak_ids": [2, 3], "note": "aromatic doublets of doublets"},
            },
        ],
        "candidates": [
            {
                "rank": i + 1, "smiles": s, "name": None, "rationale": "fits the data", "assignments": [],
                "confidence": 0.8,
            }
            for i, s in enumerate(smiles)
        ],
        "unexplained_peak_ids": [],
        "reasoning_summary": "summary",
    }  # fmt: skip
    h.update(over)
    return h


def hypothesis_json(*a: Any, **k: Any) -> str:
    return json.dumps(hypothesis(*a, **k))


def valid(text: str) -> bool:
    LLMHypothesis.model_validate_json(text)
    return True


Handler = Callable[[httpx2.Request], httpx2.Response]
