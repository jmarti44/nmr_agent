"""Environment-driven configuration (SPEC §10.6). Variables use the NMR_AGENT_ prefix; the SDK reads
ANTHROPIC_API_KEY itself.

DEVIATION: SPEC §7.1 says to use effort "high" for the elucidation call and "low" for the repair call,
while §10.6 lists a default of "medium". The per-call guidance in §7.1 is followed: `effort` is the
elucidation effort and `repair_effort` the repair effort, both configurable.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NMR_AGENT_", extra="ignore")

    model: str = "claude-opus-5-5"
    effort: Effort = "high"
    repair_effort: Effort = "low"
    enable_fallbacks: bool = True
    # Limits that bound one analysis (SPEC §7.3): 1 primary + 2 schema corrections + 1 repair.
    max_llm_calls: int = Field(4, ge=1, le=8)
    max_primary_attempts: int = Field(3, ge=1, le=6)
    initial_max_tokens: int = Field(16000, ge=1024)
    max_tokens_cap: int = Field(32000, ge=1024)
    llm_call_timeout_s: float = Field(120.0, gt=0)
    request_timeout_s: float = Field(180.0, gt=0)  # the whole analysis, all calls together
    # Retries per model call: 429, 408/409, 5xx, connection errors, and timeouts if a full attempt still fits.
    # Done by the orchestrator inside the request deadline; the SDK's own retries are off. The name is kept.
    sdk_max_retries: int = Field(3, ge=0)
    # Abuse controls for /v1/analyze (SPEC §9.4); in-memory counters, per process.
    daily_llm_budget: int = Field(100, ge=0)  # model calls per UTC day
    per_ip_per_hour: int = Field(10, ge=1)  # analyses per client address per hour
    api_key: str | None = None  # when set, /v1/analyze requires X-API-Key
    # Experience memory (docs/experience-memory-spec.md): a directory holding manifest.json, cases/ and
    # retrieval_params.json. Unset means memory is disabled and the service behaves as without it.
    memory_dir: Path | None = None
    # Defined for later phases; not used yet.
    max_upload_mb: int = 25
    pubchem: bool = False
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
