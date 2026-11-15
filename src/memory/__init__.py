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
from src.memory.retrieval import RetrievalParams, RetrievalResult, retrieve, score_case, select_random
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
    "MemoryStore",
    "OutcomeConfig",
    "RetrievalParams",
    "RetrievalResult",
    "Rejection",
    "Violation",
    "case_features",
    "check_injected",
    "load_store",
    "memory_version",
    "query_features",
    "retrieve",
    "rule1_structure_key",
    "rule2_scaffold",
    "rule3_eval_scaffolds",
    "rule4_similarity",
    "rule5_proposed",
    "score_case",
    "select_random",
]
