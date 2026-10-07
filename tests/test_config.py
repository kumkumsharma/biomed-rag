from src.config import DATASET, EVALUATION, GENERATION, RERANKER, RETRIEVAL, UI, DatasetConfig, EvaluationConfig, GenerationConfig, RerankerConfig, RetrievalConfig, UIConfig, reranker_config_from_env


def test_dataset_config_points_at_rag_mini_bioasq():
    assert DATASET.name == "rag-datasets/rag-mini-bioasq"
    assert DATASET.corpus_config == "text-corpus"
    assert DATASET.corpus_split == "passages"
    assert DATASET.qa_config == "question-answer-passages"
    assert DATASET.qa_split == "test"


def test_dataset_config_is_replaceable_without_secrets():
    custom = DatasetConfig(name="local/mock-bioasq", corpus_split="passages[:2]")
    assert custom.name == "local/mock-bioasq"
    assert DATASET.name == "rag-datasets/rag-mini-bioasq"


def test_retrieval_defaults_exist_for_later_stages():
    assert isinstance(RETRIEVAL, RetrievalConfig)
    assert RETRIEVAL.bm25_top_k == 10
    assert RETRIEVAL.dense_top_k == 10
    assert RETRIEVAL.hybrid_top_k == 10
    assert RETRIEVAL.dense_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert RETRIEVAL.rrf_k == 60


def test_generation_defaults_do_not_include_secrets():
    assert isinstance(GENERATION, GenerationConfig)
    assert GENERATION.provider == "openai"
    assert GENERATION.model_name == "gpt-4o-mini"
    assert GENERATION.min_evidence_count == 1
    assert GENERATION.evidence_top_k == 5
    assert GENERATION.temperature == 0.0
    assert GENERATION.query_expansion_enabled is True
    assert GENERATION.query_expansion_count == 3
    assert GENERATION.query_expansion_model_name == "gpt-4o-mini"


def test_ui_defaults_use_the_full_passage_corpus():
    assert isinstance(UI, UIConfig)
    assert UI.corpus_split == "passages"
    assert UI.page_title


def test_evaluation_defaults_use_a_fixed_100_query_set():
    assert isinstance(EVALUATION, EvaluationConfig)
    assert EVALUATION.n_queries == 100
    assert EVALUATION.top_k == 10
    assert EVALUATION.queries_path == "experiments/configs/evaluation_100.json"
    assert EVALUATION.n_answer_queries == 20
    assert EVALUATION.n_manual_review == 3
    assert EVALUATION.judge_model_name == "gpt-4o-mini"
    assert EVALUATION.judge_temperature == 0.0
    assert EVALUATION.hybrid_qe_details_path.endswith("retrieval_evaluation_hybrid_qe_details.json")
    assert EVALUATION.rerank_details_path.endswith("retrieval_evaluation_rerank_details.json")
    assert EVALUATION.rerank_summary_path.endswith("retrieval_evaluation_rerank_summary.json")
    assert EVALUATION.rerank_comparison_details_path.endswith(
        "retrieval_evaluation_rerank_comparison_details.json"
    )
    assert EVALUATION.rerank_comparison_summary_path.endswith(
        "retrieval_evaluation_rerank_comparison_summary.json"
    )
    assert EVALUATION.upgrade_details_path.endswith("retrieval_evaluation_upgrade_details.json")
    assert EVALUATION.upgrade_summary_path.endswith("retrieval_evaluation_upgrade_summary.json")
    assert EVALUATION.embedding_comparison_summary_path.endswith("embedding_model_comparison_summary.json")
    assert EVALUATION.candidate_pool_summary_path.endswith("candidate_pool_summary.json")
    assert EVALUATION.medcpt_pool_summary_path.endswith("medcpt_pool_size_summary.json")
    assert EVALUATION.missing_evidence_summary_path.endswith("missing_evidence_summary.json")
    assert EVALUATION.objective_failure_summary_path.endswith("objective_failure_summary.json")
    assert EVALUATION.recovery_matrix_summary_path.endswith("recovery_matrix_summary.json")
    assert EVALUATION.sparse_study_summary_path.endswith("sparse_study_summary.json")
    assert EVALUATION.sparse_study_details_path.endswith("sparse_study_details.json")
    assert EVALUATION.data_quality_path.endswith("data_quality_report.json")
    assert EVALUATION.gold_quality_path.endswith("evaluation_gold_quality.json")
    assert EVALUATION.index_health_path.endswith("index_health_report.json")
    assert EVALUATION.retrieval_diagnostics_path.endswith("retrieval_diagnostics.json")


def test_reranker_defaults_keep_the_baseline_pipeline():
    cfg = RerankerConfig()
    assert cfg.enabled is False
    assert cfg.model_name == "ncbi/MedCPT-Cross-Encoder"
    assert cfg.candidate_pool_size == 50
    assert cfg.final_top_k == 10
    assert cfg.device == "auto"
    assert isinstance(RERANKER, RerankerConfig)


def test_reranker_config_from_env_enables_the_upgraded_pipeline(monkeypatch):
    monkeypatch.setenv("BIOMED_RERANKER_ENABLED", "1")
    monkeypatch.setenv("BIOMED_RERANKER_CANDIDATE_POOL_SIZE", "25")
    monkeypatch.setenv("BIOMED_RERANKER_FINAL_TOP_K", "8")
    monkeypatch.setenv("BIOMED_RERANKER_DEVICE", "cpu")
    cfg = reranker_config_from_env()
    assert cfg.enabled is True
    assert cfg.candidate_pool_size == 25
    assert cfg.final_top_k == 8
    assert cfg.device == "cpu"
    assert cfg.model_name == "ncbi/MedCPT-Cross-Encoder"
