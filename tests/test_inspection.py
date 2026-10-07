from src.generation.answer_generator import GROUNDED_PROMPT_TEMPLATE
from src.generation.query_expansion import EXPANSION_PROMPT
from src.inspection.config_view import runtime_configuration
from src.inspection.decisions import architecture_decisions
from src.inspection.prompts import llm_prompt_catalog, prompt_by_name
from src.inspection.status import experimental_feature_status


def test_prompt_catalog_uses_real_templates_and_marks_deterministic_components():
    catalog = {item["component"]: item for item in llm_prompt_catalog()}
    assert catalog["answer_generation"]["template"] == GROUNDED_PROMPT_TEMPLATE
    assert "{question}" in catalog["answer_generation"]["template"]
    assert catalog["llm_query_expansion"]["template"] == EXPANSION_PROMPT
    assert catalog["deterministic_query_expansion"]["deterministic"] is True
    assert catalog["deterministic_query_expansion"]["template"] is None
    assert catalog["query_decomposition"]["template"] is None
    assert catalog["llm_judge"]["template"]
    judge = prompt_by_name("LLM Judge (offline)")
    assert judge is not None
    assert "correctness" in judge["template"]


def test_runtime_configuration_is_read_only_snapshot():
    cfg = runtime_configuration()
    assert cfg["retrieval"]["rrf_k"] == 60
    assert cfg["retrieval"]["dense_baseline_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg["retrieval"]["minilm_dimension"] == 384
    assert cfg["retrieval"]["sparse_enabled"] is False
    assert cfg["retrieval"]["sparse_status"] == "EXPERIMENTAL"
    assert cfg["reranking"]["model_name"] == "ncbi/MedCPT-Cross-Encoder"
    assert "normalization_enabled" in cfg["query_processing"]
    assert cfg["evaluation"]["query_count"] == 100


def test_architecture_decisions_include_measured_tradeoffs():
    records = {item["component"]: item for item in architecture_decisions()}
    assert "MiniLM" in records["MiniLM dense encoder"]["decision"]
    assert "384" in records["MiniLM dense encoder"]["why_chosen"]
    assert "0.5082" in records["MedCPT cross-encoder"]["measured_result"]
    assert "default OFF" in records["Query expansion"]["decision"]
    assert "lexical mismatch" in records["Failure analysis methodology"]["why_chosen"].lower()
    sparse_adr = records["Third first-stage retriever (learned sparse)"]
    assert "86.5%" in sparse_adr["why_chosen"]
    assert "dual miss" in sparse_adr["why_chosen"].lower()
    assert "do not enable" in sparse_adr["decision"].lower()


def test_feature_status_distinguishes_baseline_experimental_and_disabled():
    rows = experimental_feature_status()
    by_name = {item["name"]: item for item in rows}
    assert by_name["Hybrid BM25 + Dense + RRF"]["lane"] == "BASELINE"
    assert by_name["Deterministic query expansion"]["lane"] == "EXPERIMENTAL"
    assert by_name["Deterministic query expansion"]["default"].startswith("OFF")
    assert by_name["Biomedical dense embeddings"]["default"].startswith("OFF")
    assert by_name["Learned sparse first-stage (SPLADE)"]["lane"] == "EXPERIMENTAL"
    assert by_name["Learned sparse first-stage (SPLADE)"]["runtime"] == "DISABLED"
    assert by_name["Learned sparse first-stage (SPLADE)"]["default"].startswith("OFF")
    assert "NOT EVALUATED" in by_name["Biomedical dense embeddings"]["benchmark"] or "embedding_model_comparison" in by_name["Biomedical dense embeddings"]["benchmark"]
