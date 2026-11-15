"""Memory at run time: the frozen outcome configuration and the runtime that picks examples for a query.

``outcome_config`` is what phase M6 records in every case's ``agent_outcome`` and what the loader checks.
Changing any Settings field it hashes (see ``OUTCOME_SETTINGS_FIELDS``) changes ``settings_hash`` and so
invalidates every collected outcome: the store then rejects those cases (strict mode raises, lenient mode
skips them) until outcomes are collected again.

Nothing here reaches the verifier, the scoring or the ranking (constraint 1).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from src.core.settings import Settings
from src.memory.render import case_cost
from src.memory.retrieval import RetrievalParams, RetrievalResult, retrieve, select_random
from src.memory.store import MemoryStore, OutcomeConfig, load_store
from src.prompting.templates import FULL_VERSION
from src.schemas import PeakTable, SampleContext

# Cap on the whole examples block, in estimated tokens (spec §5.4), enforced through the retrieval budget.
EXAMPLES_TOKEN_BUDGET = 2000
PARAMS_FILE = "retrieval_params.json"

OUTCOME_SETTINGS_FIELDS = (
    "model",
    "effort",
    "repair_effort",
    "enable_fallbacks",
    "max_llm_calls",
    "max_primary_attempts",
    "initial_max_tokens",
    "max_tokens_cap",
    "llm_call_timeout_s",
    "request_timeout_s",
)


class ParamsError(ValueError):
    """The retrieval parameter file is missing, malformed or does not match its recorded hash."""


def settings_hash(settings: Settings) -> str:
    """SHA-256 of the canonical JSON (sorted keys, no spaces) of the OUTCOME_SETTINGS_FIELDS values."""
    values = {name: getattr(settings, name) for name in OUTCOME_SETTINGS_FIELDS}
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def outcome_config(settings: Settings) -> OutcomeConfig:
    return OutcomeConfig(
        model_id=settings.model,
        prompt_version=FULL_VERSION,
        settings_hash=settings_hash(settings),
    )


def params_hash(parameters: dict) -> str:
    """The hash recorded in retrieval_params.json (same definition as bench/memory_gonogo.py)."""
    return hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()


def load_params(path: Path) -> RetrievalParams:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ParamsError(f"{path}: cannot read retrieval parameters ({e})") from e
    if not isinstance(doc, dict) or not isinstance(doc.get("parameters"), dict):
        raise ParamsError(f"{path}: expected an object with a 'parameters' object")
    parameters = doc["parameters"]
    recorded = doc.get("sha256")
    actual = params_hash(parameters)
    if recorded != actual:
        raise ParamsError(f"{path}: sha256 {recorded!r} does not match the parameters ({actual})")
    try:
        return RetrievalParams(**parameters)
    except (TypeError, ValueError) as e:
        raise ParamsError(f"{path}: invalid retrieval parameters ({e})") from e


@dataclass(frozen=True)
class MemoryRuntime:
    """A store, the frozen retrieval parameters and how to pick examples.

    mode "retrieved": the service (best matches). mode "random": the benchmark's R200L arm, ``seed`` required;
    use ``with_seed`` to change the seed per molecule.
    """

    store: MemoryStore
    params: RetrievalParams
    mode: Literal["retrieved", "random"] = "retrieved"
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in ("retrieved", "random"):
            raise ValueError(f"unknown memory mode {self.mode!r}")
        if self.mode == "random" and self.seed is None:
            raise ValueError("random mode needs a seed")

    @classmethod
    def from_directory(
        cls,
        directory: Path,
        settings: Settings,
        *,
        strict: bool,
        mode: Literal["retrieved", "random"] = "retrieved",
        seed: int | None = None,
    ) -> MemoryRuntime:
        """Raises ManifestError (store unusable), CaseRejectedError (strict only) or ParamsError."""
        directory = Path(directory)
        params = load_params(directory / PARAMS_FILE)
        store = load_store(directory, strict=strict, expected_outcome_config=outcome_config(settings))
        return cls(store, params, mode, seed)

    def with_seed(self, seed: int) -> MemoryRuntime:
        return replace(self, seed=seed)

    @property
    def memory_version(self) -> str:
        return self.store.memory_version

    def examples_for(self, table: PeakTable, context: SampleContext) -> list[RetrievalResult]:
        """Up to ``params.k`` cases within the examples token budget. No exclusions are ever passed here
        (they are offline-only)."""
        if self.mode == "random":
            assert self.seed is not None
            return select_random(
                self.store, self.seed, self.params, token_budget=EXAMPLES_TOKEN_BUDGET, cost_fn=case_cost
            )
        from src.memory.features import query_features

        return retrieve(
            query_features(table, context),
            self.store,
            self.params,
            token_budget=EXAMPLES_TOKEN_BUDGET,
            cost_fn=case_cost,
        )
