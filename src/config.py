"""Project configuration.

Holds dataset identifiers and retrieval defaults that later stages will use.
Does not store API keys or other secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    """Hugging Face dataset identifiers and field names."""

    name: str = "rag-datasets/rag-mini-bioasq"
    corpus_config: str = "text-corpus"
    corpus_split: str = "passages"
    qa_config: str = "question-answer-passages"
    qa_split: str = "test"
    passage_id_field: str = "id"
    passage_text_field: str = "passage"
    question_id_field: str = "id"
    question_field: str = "question"
    answer_field: str = "answer"
    relevant_passage_ids_field: str = "relevant_passage_ids"


@dataclass(frozen=True)
class RetrievalConfig:
    """Retrieval defaults for BM25, dense, and later hybrid stages."""

    bm25_top_k: int = 10
    dense_top_k: int = 10
    hybrid_top_k: int = 10
    rrf_k: int = 60
    dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return int(raw)


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip()


@dataclass(frozen=True)
class RerankerConfig:
    """Optional second-stage cross-encoder after Hybrid RRF.

    Disabled by default so the baseline BM25 + Dense + RRF pipeline stays
    available for A/B evaluation. Enable with BIOMED_RERANKER_ENABLED=1.
    The reranker runs locally (Hugging Face / PyTorch). It does not call OpenAI.
    """

    enabled: bool = False
    candidate_pool_size: int = 50
    final_top_k: int = 10
    model_name: str = "ncbi/MedCPT-Cross-Encoder"
    device: str = "auto"
    batch_size: int = 16
    max_length: int = 512


def reranker_config_from_env() -> RerankerConfig:
    """Build reranker settings from environment variables."""

    return RerankerConfig(
        enabled=_env_flag("BIOMED_RERANKER_ENABLED", False),
        candidate_pool_size=_env_int("BIOMED_RERANKER_CANDIDATE_POOL_SIZE", 50),
        final_top_k=_env_int("BIOMED_RERANKER_FINAL_TOP_K", 10),
        model_name=_env_str("BIOMED_RERANKER_MODEL", "ncbi/MedCPT-Cross-Encoder"),
        device=_env_str("BIOMED_RERANKER_DEVICE", "auto"),
        batch_size=_env_int("BIOMED_RERANKER_BATCH_SIZE", 16),
        max_length=_env_int("BIOMED_RERANKER_MAX_LENGTH", 512),
    )


@dataclass(frozen=True)
class UpgradeConfig:
    """Next-generation retrieval flags. All optional stages default off.

    Disabled flags keep Hybrid BM25 + Dense + RRF (+ optional MedCPT) unchanged.
    """

    pipeline_enabled: bool = False
    terminology_enabled: bool = False
    expansion_enabled: bool = False
    llm_expansion_enabled: bool = False
    decomposition_enabled: bool = False
    adaptive_routing_enabled: bool = False
    filtering_enabled: bool = False
    biomedical_embeddings_enabled: bool = False
    biomedical_embedding_model: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    embedding_cache_dir: str = "experiments/cache/embeddings"
    filter_drop_duplicates: bool = True
    filter_min_token_overlap: float = 0.0
    filter_drop_empty: bool = True
    expansion_max_variants: int = 4
    decomposition_min_aspects: int = 2
    candidate_pool_size: int = 50
    final_top_k: int = 10
    sparse_enabled: bool = False
    sparse_model_name: str = "naver/splade-v3-distilbert"
    sparse_cache_dir: str = "experiments/cache/sparse"


def upgrade_config_from_env() -> UpgradeConfig:
    return UpgradeConfig(
        pipeline_enabled=_env_flag("BIOMED_PIPELINE_ENABLED", False),
        terminology_enabled=_env_flag("BIOMED_TERMINOLOGY_ENABLED", False),
        expansion_enabled=_env_flag("BIOMED_DETERMINISTIC_EXPANSION_ENABLED", False),
        llm_expansion_enabled=_env_flag("BIOMED_LLM_EXPANSION_ENABLED", False),
        decomposition_enabled=_env_flag("BIOMED_DECOMPOSITION_ENABLED", False),
        adaptive_routing_enabled=_env_flag("BIOMED_ADAPTIVE_ROUTING_ENABLED", False),
        filtering_enabled=_env_flag("BIOMED_FILTERING_ENABLED", False),
        biomedical_embeddings_enabled=_env_flag("BIOMED_BIOMEDICAL_EMBEDDINGS_ENABLED", False),
        biomedical_embedding_model=_env_str(
            "BIOMED_EMBEDDING_MODEL",
            "pritamdeka/S-PubMedBert-MS-MARCO",
        ),
        embedding_cache_dir=_env_str("BIOMED_EMBEDDING_CACHE_DIR", "experiments/cache/embeddings"),
        filter_drop_duplicates=_env_flag("BIOMED_FILTER_DROP_DUPLICATES", True),
        filter_min_token_overlap=float(os.environ.get("BIOMED_FILTER_MIN_TOKEN_OVERLAP", "0") or 0),
        filter_drop_empty=_env_flag("BIOMED_FILTER_DROP_EMPTY", True),
        expansion_max_variants=_env_int("BIOMED_EXPANSION_MAX_VARIANTS", 4),
        decomposition_min_aspects=_env_int("BIOMED_DECOMPOSITION_MIN_ASPECTS", 2),
        candidate_pool_size=_env_int("BIOMED_PIPELINE_CANDIDATE_POOL_SIZE", 50),
        final_top_k=_env_int("BIOMED_PIPELINE_FINAL_TOP_K", 10),
        sparse_enabled=_env_flag("BIOMED_SPARSE_ENABLED", False),
        sparse_model_name=_env_str("BIOMED_SPARSE_MODEL", "naver/splade-v3-distilbert"),
        sparse_cache_dir=_env_str("BIOMED_SPARSE_CACHE_DIR", "experiments/cache/sparse"),
    )


DATASET = DatasetConfig()
RETRIEVAL = RetrievalConfig()
RERANKER = reranker_config_from_env()
UPGRADE = upgrade_config_from_env()


@dataclass(frozen=True)
class GenerationConfig:
    """LLM and evidence settings. API keys stay in environment variables."""

    provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 512
    min_evidence_count: int = 1
    evidence_top_k: int = 5
    query_expansion_enabled: bool = True
    query_expansion_count: int = 3
    query_expansion_model_name: str = "gpt-4o-mini"
    query_expansion_estimated_cost_per_call_usd: float = 0.0001


GENERATION = GenerationConfig()


@dataclass(frozen=True)
class UIConfig:
    """Local demo UI settings. Not used by retrieval internals.

    The demo uses the full BioASQ passage split. For a faster local smoke
    index, set BIOMED_UI_CORPUS_SPLIT to a Hugging Face slice such as
    passages[:200].
    """

    corpus_split: str = "passages"
    page_title: str = "RAG Evaluation Workbench"


UI = UIConfig()


@dataclass(frozen=True)
class EvaluationConfig:
    """Fixed 100-query retrieval evaluation. Uses the full passage corpus."""

    n_queries: int = 100
    top_k: int = 10
    queries_path: str = "experiments/configs/evaluation_100.json"
    details_path: str = "experiments/results/retrieval_evaluation_details.json"
    summary_path: str = "experiments/results/retrieval_evaluation_summary.json"
    hybrid_qe_details_path: str = "experiments/results/retrieval_evaluation_hybrid_qe_details.json"
    hybrid_qe_summary_path: str = "experiments/results/retrieval_evaluation_hybrid_qe_summary.json"
    comparison_path: str = "experiments/results/retrieval_evaluation_comparison.json"
    rerank_details_path: str = "experiments/results/retrieval_evaluation_rerank_details.json"
    rerank_summary_path: str = "experiments/results/retrieval_evaluation_rerank_summary.json"
    rerank_comparison_details_path: str = "experiments/results/retrieval_evaluation_rerank_comparison_details.json"
    rerank_comparison_summary_path: str = "experiments/results/retrieval_evaluation_rerank_comparison_summary.json"
    upgrade_details_path: str = "experiments/results/retrieval_evaluation_upgrade_details.json"
    upgrade_summary_path: str = "experiments/results/retrieval_evaluation_upgrade_summary.json"
    embedding_comparison_details_path: str = "experiments/results/embedding_model_comparison_details.json"
    embedding_comparison_summary_path: str = "experiments/results/embedding_model_comparison_summary.json"
    candidate_pool_details_path: str = "experiments/results/candidate_pool_details.json"
    candidate_pool_summary_path: str = "experiments/results/candidate_pool_summary.json"
    medcpt_pool_details_path: str = "experiments/results/medcpt_pool_size_details.json"
    medcpt_pool_summary_path: str = "experiments/results/medcpt_pool_size_summary.json"
    missing_evidence_details_path: str = "experiments/results/missing_evidence_details.json"
    missing_evidence_summary_path: str = "experiments/results/missing_evidence_summary.json"
    expansion_study_details_path: str = "experiments/results/expansion_study_details.json"
    expansion_study_summary_path: str = "experiments/results/expansion_study_summary.json"
    decomposition_study_details_path: str = "experiments/results/decomposition_study_details.json"
    decomposition_study_summary_path: str = "experiments/results/decomposition_study_summary.json"
    objective_failure_details_path: str = "experiments/results/objective_failure_details.json"
    objective_failure_summary_path: str = "experiments/results/objective_failure_summary.json"
    recovery_matrix_details_path: str = "experiments/results/recovery_matrix_details.json"
    recovery_matrix_summary_path: str = "experiments/results/recovery_matrix_summary.json"
    sparse_study_details_path: str = "experiments/results/sparse_study_details.json"
    sparse_study_summary_path: str = "experiments/results/sparse_study_summary.json"
    diagnostics_dir: str = "experiments/results/diagnostics"
    data_quality_path: str = "experiments/results/diagnostics/data_quality_report.json"
    gold_quality_path: str = "experiments/results/diagnostics/evaluation_gold_quality.json"
    index_health_path: str = "experiments/results/diagnostics/index_health_report.json"
    retrieval_diagnostics_path: str = "experiments/results/diagnostics/retrieval_diagnostics.json"
    live_evaluation_path: str = "experiments/results/diagnostics/live_evaluation.jsonl"
    query_trace_dir: str = "experiments/results/diagnostics/traces"
    n_answer_queries: int = 20
    n_manual_review: int = 3
    judge_model_name: str = "gpt-4o-mini"
    judge_temperature: float = 0.0
    judge_max_tokens: int = 400
    answer_queries_path: str = "experiments/configs/answer_evaluation_20.json"
    answer_details_path: str = "experiments/results/answer_evaluation_details.json"
    answer_summary_path: str = "experiments/results/answer_evaluation_summary.json"
    manual_review_path: str = "experiments/results/answer_evaluation_manual_review.md"
    manual_review_json_path: str = "experiments/results/answer_evaluation_manual_review.json"


EVALUATION = EvaluationConfig()
