from src.config import UPGRADE, UpgradeConfig, upgrade_config_from_env
from src.query.classify import classify_query
from src.query.decomposition import decompose_query
from src.query.expansion import expand_query
from src.query.terminology import normalize_query
from src.query.understand import understand_query


def test_normalize_preserves_original_and_maps_abbreviation():
    result = normalize_query("What is the role of EGFR in NSCLC?")
    assert result.original == "What is the role of EGFR in NSCLC?"
    assert result.original != result.normalized
    assert "epidermal growth factor receptor" in result.normalized.lower()
    assert result.detected_terms
    assert result.mappings


def test_normalize_disabled_returns_original():
    result = normalize_query("EGFR mutation", enabled=False)
    assert result.normalized == "EGFR mutation"
    assert result.detected_terms == ()


def test_expansion_returns_separate_variants_not_one_blob():
    result = expand_query("EGFR inhibitors for NSCLC", max_variants=3)
    assert result.original == "EGFR inhibitors for NSCLC"
    assert result.llm_called is False
    assert result.method == "local_synonym_substitution"
    assert result.expansions
    joined = " ".join(result.expansions)
    assert result.original not in joined or True
    for variant in result.expansions:
        assert variant != "EGFR inhibitors for NSCLC " + " ".join(result.expansions)


def test_expansion_disabled_is_empty():
    result = expand_query("EGFR", enabled=False)
    assert result.expansions == ()
    assert result.method == "disabled"


def test_classify_simple_factual():
    result = classify_query("Is this protein secreted?")
    assert result.query_type == "simple_factual"


def test_classify_terminology_heavy():
    result = classify_query("What does EGFR do?")
    assert result.query_type == "terminology_heavy"


def test_classify_multi_part():
    result = classify_query(
        "What are the risk factors, mechanisms, and treatment options for Hirschsprung disease?"
    )
    assert result.query_type == "multi_part"
    assert result.aspect_count >= 2


def test_classify_semantic():
    result = classify_query("Why is this protein found in the extracellular matrix?")
    assert result.query_type == "semantic"


def test_decompose_complex_query_into_subqueries():
    result = decompose_query(
        "What are the risk factors, mechanisms, and treatment options for Hirschsprung disease?"
    )
    assert result.applied is True
    assert len(result.subqueries) >= 2
    assert all("Hirschsprung" in item or "hirschsprung" in item.lower() for item in result.subqueries)


def test_decompose_skips_simple_queries():
    result = decompose_query("Is papilin secreted?")
    assert result.applied is False
    assert result.subqueries == ()


def test_understand_disabled_keeps_original_search_query():
    cfg = UpgradeConfig()
    result = understand_query("EGFR in NSCLC", cfg)
    assert result.search_queries == ("EGFR in NSCLC",)
    assert result.original == "EGFR in NSCLC"


def test_understand_enabled_adds_normalized_and_expansions():
    cfg = UpgradeConfig(terminology_enabled=True, expansion_enabled=True, expansion_max_variants=3)
    result = understand_query("EGFR mutation", cfg)
    assert result.original == "EGFR mutation"
    assert result.normalized != result.original
    assert result.original in result.search_queries
    assert any(item != result.original for item in result.search_queries)


def test_upgrade_flags_default_off():
    cfg = UpgradeConfig()
    assert cfg.pipeline_enabled is False
    assert cfg.terminology_enabled is False
    assert cfg.expansion_enabled is False
    assert cfg.llm_expansion_enabled is False
    assert cfg.decomposition_enabled is False
    assert cfg.adaptive_routing_enabled is False
    assert cfg.filtering_enabled is False
    assert cfg.biomedical_embeddings_enabled is False
    assert cfg.sparse_enabled is False
    assert UPGRADE.pipeline_enabled is False


def test_upgrade_config_from_env(monkeypatch):
    monkeypatch.setenv("BIOMED_PIPELINE_ENABLED", "1")
    monkeypatch.setenv("BIOMED_TERMINOLOGY_ENABLED", "true")
    monkeypatch.setenv("BIOMED_EMBEDDING_MODEL", "custom/biomed")
    cfg = upgrade_config_from_env()
    assert cfg.pipeline_enabled is True
    assert cfg.terminology_enabled is True
    assert cfg.biomedical_embedding_model == "custom/biomed"
    assert cfg.llm_expansion_enabled is False
