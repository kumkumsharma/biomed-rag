"""Read-only runtime configuration for the inspector UI."""

from __future__ import annotations

from typing import Any

from src.config import DATASET, EVALUATION, GENERATION, RERANKER, RETRIEVAL, UPGRADE, UI
from src.retrieval.embedding_models import MINILM_DIMENSION, resolve_embedding_model
from src.retrieval.reranker import resolve_device


def runtime_configuration() -> dict[str, Any]:
    """Current env/config values. Not a control panel."""

    return {
        "retrieval": {
            "bm25_enabled": True,
            "dense_enabled": True,
            "rrf_enabled": True,
            "rrf_k": RETRIEVAL.rrf_k,
            "dense_model": resolve_embedding_model(),
            "dense_baseline_model": RETRIEVAL.dense_model_name,
            "dense_top_k": RETRIEVAL.dense_top_k,
            "bm25_top_k": RETRIEVAL.bm25_top_k,
            "hybrid_top_k": RETRIEVAL.hybrid_top_k,
            "minilm_dimension": MINILM_DIMENSION,
            "sparse_enabled": UPGRADE.sparse_enabled,
            "sparse_model": UPGRADE.sparse_model_name,
            "sparse_cache_dir": UPGRADE.sparse_cache_dir,
            "sparse_status": "EXPERIMENTAL",
        },
        "reranking": {
            "medcpt_enabled": RERANKER.enabled,
            "model_name": RERANKER.model_name,
            "candidate_pool": RERANKER.candidate_pool_size,
            "final_top_k": RERANKER.final_top_k,
            "device": RERANKER.device,
            "resolved_device": resolve_device(RERANKER.device),
            "batch_size": RERANKER.batch_size,
        },
        "query_processing": {
            "pipeline_enabled": UPGRADE.pipeline_enabled,
            "normalization_enabled": UPGRADE.terminology_enabled,
            "deterministic_expansion_enabled": UPGRADE.expansion_enabled,
            "llm_expansion_enabled": GENERATION.query_expansion_enabled,
            "decomposition_enabled": UPGRADE.decomposition_enabled,
            "adaptive_routing_enabled": UPGRADE.adaptive_routing_enabled,
            "filtering_enabled": UPGRADE.filtering_enabled,
            "biomedical_embeddings_enabled": UPGRADE.biomedical_embeddings_enabled,
        },
        "generation": {
            "answer_model": GENERATION.model_name,
            "temperature": GENERATION.temperature,
            "max_tokens": GENERATION.max_tokens,
            "min_evidence_count": GENERATION.min_evidence_count,
            "evidence_top_k": GENERATION.evidence_top_k,
            "query_expansion_model": GENERATION.query_expansion_model_name,
            "query_expansion_count": GENERATION.query_expansion_count,
        },
        "evaluation": {
            "dataset": DATASET.name,
            "query_count": EVALUATION.n_queries,
            "queries_path": EVALUATION.queries_path,
            "metrics": ["Recall@5", "Recall@10", "Recall@50", "MRR@10", "nDCG@10", "latency"],
            "answer_query_count": EVALUATION.n_answer_queries,
            "judge_model": EVALUATION.judge_model_name,
        },
        "ui": {
            "corpus_split": UI.corpus_split,
            "page_title": UI.page_title,
        },
    }
