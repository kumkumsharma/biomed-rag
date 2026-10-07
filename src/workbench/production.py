"""Declared production path vs library default vs playground/experiment overrides.

Does not change ranking implementations. RERANKER.enabled remains the library/env
toggle (default False). Production is a separate, explicit configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import GENERATION, RERANKER, RETRIEVAL

LANES = ("production", "playground", "experiment", "library")
PLAYGROUND_DEFAULT_RERANK_ENABLED = True


@dataclass(frozen=True)
class ProductionPath:
    """Intended production retrieval/generation path.

    Independent of ``RERANKER.enabled``, which is the library/env toggle
    (default False) so BM25+Dense+RRF stays callable without a cross-encoder.
    """

    retriever: str = "hybrid"
    dense_model_id: str = "minilm"
    dense_model_name: str = RETRIEVAL.dense_model_name
    fusion: str = "rrf"
    rrf_k: int = RETRIEVAL.rrf_k
    candidate_pool: int = 50
    reranker_id: str = "medcpt-ce"
    reranker_model_name: str = RERANKER.model_name
    reranker_enabled: bool = True
    query_expansion: bool = False
    generation_model_id: str = "gpt-4o-mini"
    generation_model_name: str = GENERATION.model_name
    final_top_k: int = 10


PRODUCTION = ProductionPath()


def library_reranker_enabled() -> bool:
    """Env/library default. False unless BIOMED_RERANKER_ENABLED is set."""

    return bool(RERANKER.enabled)


def resolve_rerank_enabled(*, override: bool | None = None, lane: str = "library") -> bool:
    """Select rerank on/off without silently mixing lanes.

    production: declared production path (MedCPT on)
    playground: production defaults; explicit request override wins
    experiment: override is required; missing override does not enable MedCPT
    library: RERANKER.enabled (default False) — not production
    """

    if lane not in LANES:
        raise ValueError(f"Unknown lane {lane!r}. Expected one of {LANES}")
    if override is not None:
        return bool(override)
    if lane in {"production", "playground"}:
        return bool(PRODUCTION.reranker_enabled)
    return library_reranker_enabled()


def resolve_pipeline_config(*, lane: str = "library", override: bool | None = None) -> dict[str, Any]:
    """Authoritative stack description for a caller lane. Does not construct retrievers."""

    rerank = resolve_rerank_enabled(override=override, lane=lane)
    label = {
        "production": "Production",
        "playground": "Playground",
        "experiment": "Experiment",
        "library": "Library (not production)",
    }[lane]
    return {
        "lane": lane,
        "label": label,
        "reranker_enabled": rerank,
        "reranker_id": PRODUCTION.reranker_id if rerank else None,
        "reranker_model_name": PRODUCTION.reranker_model_name if rerank else None,
        "candidate_pool": PRODUCTION.candidate_pool if rerank else None,
        "rrf_k": PRODUCTION.rrf_k,
        "dense_model_name": PRODUCTION.dense_model_name,
        "final_top_k": PRODUCTION.final_top_k,
        "is_production_stack": lane == "production" and rerank is True,
        "uses_production_defaults": lane in {"production", "playground"} and override is None,
    }


def production_runtime_report() -> dict[str, Any]:
    """Entry-point map. Does not enable MedCPT on the library path."""

    library = library_reranker_enabled()
    return {
        "declared_production": {
            "reranker_enabled": PRODUCTION.reranker_enabled,
            "reranker_id": PRODUCTION.reranker_id,
            "candidate_pool": PRODUCTION.candidate_pool,
            "rrf_k": PRODUCTION.rrf_k,
            "path": (
                f"BM25 + {PRODUCTION.dense_model_name} + RRF k={PRODUCTION.rrf_k} → "
                f"Top {PRODUCTION.candidate_pool} → {PRODUCTION.reranker_model_name} → "
                f"Top {PRODUCTION.final_top_k} → {PRODUCTION.generation_model_name}"
            ),
        },
        "library_default": {
            "RERANKER.enabled": library,
            "env": "BIOMED_RERANKER_ENABLED",
            "HybridRetriever_runs_medcpt": False,
            "wrap_retriever_without_enabled_runs_medcpt": False,
            "load_pipeline_no_args_runs_medcpt": False,
            "is_production": False,
            "note": (
                "Library defaults are not production. HybridRetriever and "
                "wrap_retriever(config=RERANKER) skip MedCPT unless enabled=True."
            ),
        },
        "streamlit_playground": {
            "checkbox_default": PLAYGROUND_DEFAULT_RERANK_ENABLED,
            "lane_default_without_override": True,
            "passes_explicit_override": True,
            "load_pipeline_signature_default": None,
            "note": (
                "Playground lane defaults to declared production (MedCPT on). "
                "Unchecking is a request-scoped override, not a library default change."
            ),
        },
        "outside_streamlit": {
            "GroundedQA_on_HybridRetriever_runs_medcpt": False,
            "RerankingRetriever_with_RERANKER_default_runs_medcpt": library,
            "evaluation_cli_enables_medcpt_explicitly": True,
            "load_production_pipeline_runs_medcpt": True,
        },
        "can_accidentally_run_without_medcpt": True,
        "reason": (
            "Declared production includes MedCPT. The library default is "
            "RERANKER.enabled=False. HybridRetriever, wrap_retriever() without "
            "enabled=True, and load_pipeline() with lane=library skip MedCPT. "
            "Those callers are not the production lane. "
            "load_pipeline(lane='production') and load_production_pipeline() "
            "always resolve MedCPT on unless an explicit override disables it. "
            "Playground does not skip MedCPT unless the user unchecks it."
        ),
        "ranking_behavior_changed": False,
    }
