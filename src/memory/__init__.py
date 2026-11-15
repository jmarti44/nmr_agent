"""Experience memory: stored-case format, store manifest and loader (spec §4), retrieval, leakage rules."""

from src.memory.case import CASE_SCHEMA_VERSION, MemoryCase
from src.memory.features import Features, case_features, query_features
from src.memory.leakage import (
    Violation,
    check_injected,
    rule1_structure_key,
    rule2_scaffold,
    rule3_eval_scaffolds,
    rule4_similarity,
    rule5_proposed,
)
from src.memory.pending import PendingCase, finalize
from src.memory.render import case_cost, render_case, render_examples
from src.memory.retrieval import RetrievalParams, RetrievalResult, retrieve, score_case, select_random
from src.memory.runtime import MemoryRuntime, ParamsError, outcome_config
from src.memory.store import (
    STORE_SCHEMA_VERSION,
    CaseRejectedError,
    ManifestError,
    MemoryStore,
    OutcomeConfig,
    Rejection,
    load_store,
    memory_version,
)

__all__ = [
    "CASE_SCHEMA_VERSION",
    "STORE_SCHEMA_VERSION",
    "CaseRejectedError",
    "Features",
    "ManifestError",
    "MemoryCase",
    "MemoryRuntime",
    "MemoryStore",
    "PendingCase",
    "OutcomeConfig",
    "ParamsError",
    "RetrievalParams",
    "RetrievalResult",
    "Rejection",
    "Violation",
    "case_cost",
    "case_features",
    "check_injected",
    "finalize",
    "load_store",
    "memory_version",
    "outcome_config",
    "query_features",
    "render_case",
    "render_examples",
    "retrieve",
    "rule1_structure_key",
    "rule2_scaffold",
    "rule3_eval_scaffolds",
    "rule4_similarity",
    "rule5_proposed",
    "score_case",
    "select_random",
]
