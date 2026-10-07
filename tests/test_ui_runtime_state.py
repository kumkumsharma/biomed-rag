"""UI runtime-state, telemetry, and presentation consistency tests."""

from __future__ import annotations

from types import SimpleNamespace

from app.ui import (
    evidence_inspector_rows,
    evidence_table_display_rows,
    format_heuristic_score,
    format_judge_score,
    pipeline_stage_cards,
    query_analysis_from_trace,
)
from src.evaluation.claim_coverage import claim_evidence_coverage, split_claims
from src.generation.rag import GroundedQAResult
from src.inspection.model_registry import model_registry
from src.inspection.prompt_inspector import non_llm_prompt_notes, prompt_inspector_tabs
from src.inspection.runtime_state import build_runtime_state, infer_medcpt_executed
from src.retrieval.hybrid import RetrievalResult
from src.retrieval.reranker import RerankedResult
from src.validation.citations import CitationValidation, validate_citations


def _result(*, expand: bool = False, medcpt: bool = True, n_evidence: int = 5) -> GroundedQAResult:
    evidence = []
    for index in range(1, n_evidence + 1):
        pid = 17250500 + index
        if medcpt:
            evidence.append(
                RerankedResult(
                    passage_id=pid,
                    score=10.0 - index,
                    text=f"passage {pid} explains cerebral hemiatrophy.",
                    rrf_score=0.03,
                    reranker_score=10.0 - index,
                    bm25_rank=index + 2,
                    dense_rank=index,
                    rrf_rank=index,
                    final_rank=index,
                    pmid=pid,
                )
            )
        else:
            evidence.append(RetrievalResult(passage_id=pid, score=0.02, text=f"passage {pid}"))
    answer = (
        "Cerebral hemiatrophy is volume loss of one hemisphere "
        f"[PASSAGE_ID: {evidence[0].passage_id}]. "
        "Dyke-Davidoff-Masson syndrome is a related presentation "
        f"[PASSAGE_ID: {evidence[1].passage_id}]. "
        "Some unsupported extra claim about an unlisted biomarker."
    )
    citations = validate_citations(answer, [item.passage_id for item in evidence])
    trace = {
        "original_query": "What is cerebral hemiatrophy?",
        "normalized_query": "What is cerebral hemiatrophy?",
        "candidate_k": 50,
        "rrf_k": 60,
        "bm25_ms": 172.0,
        "dense_ms": 85.0,
        "rrf_ms": 0.1,
        "rerank_ms": 4521.0 if medcpt else None,
        "reranker_enabled": medcpt,
        "reranker_model_name": "ncbi/MedCPT-Cross-Encoder" if medcpt else None,
        "reranker_candidate_pool_size": 50 if medcpt else None,
        "rerank_candidate_count": 50 if medcpt else None,
        "rerank_returned": 10 if medcpt else None,
        "generation_ms": 5266.0,
        "citation_ms": 0.1,
        "query_expansion_ms": 0.0,
        "expansion_llm_called": expand,
        "n_search_queries": 4 if expand else 1,
        "bm25_ranks": {item.passage_id: item.bm25_rank for item in evidence if getattr(item, "bm25_rank", None)},
        "dense_ranks": {item.passage_id: item.dense_rank for item in evidence if getattr(item, "dense_rank", None)},
        "rrf_ranks": {item.passage_id: index for index, item in enumerate(evidence, start=1)},
        "dense_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    }
    if not medcpt:
        trace["bm25_ranks"] = {item.passage_id: index for index, item in enumerate(evidence, start=1)}
        trace["dense_ranks"] = {item.passage_id: index for index, item in enumerate(evidence, start=1)}
    return GroundedQAResult(
        question="What is cerebral hemiatrophy?",
        answer=answer,
        evidence=tuple(evidence),
        citations=citations,
        insufficient_evidence=False,
        llm_called=True,
        prompt="prompt",
        expansions=("definition of cerebral hemiatrophy",) if expand else (),
        query_expansion_used=expand,
        trace=trace,
    )


def test_runtime_state_is_consistent_when_medcpt_ran():
    result = _result(medcpt=True, expand=False)
    state = build_runtime_state(
        result,
        expand_requested=False,
        rerank_requested=True,
        total_latency_ms=12650.0,
        live_quality={"evaluated": True, "evaluator_type": "heuristic", "llm_calls": 0, "latency_ms": 0.0},
    )
    assert state["consistent"]
    assert state["medcpt_executed"] is True
    assert state["deep_reranking_status"] == "ON"
    assert state["stage_by_key"]["medcpt"]["status"] == "enabled/active"
    assert state["query_expansion_status"] == "OFF"
    assert state["evidence_count"] == 5
    assert state["stage_by_key"]["bm25"]["output_count"] > 0
    assert state["stage_by_key"]["dense"]["output_count"] > 0
    assert state["stage_by_key"]["rrf"]["output_count"] > 0
    assert state["stage_by_key"]["evidence"]["output_count"] == 5
    assert state["llm_judge_calls"] == 0
    assert state["evaluator_label"] == "Heuristic evaluator"
    assert abs(state["latency"]["total_ms"] - 12650.0) < 1e-6
    assert state["final_score_equals_medcpt"] is True
    assert state["ids_identical"] is True


def test_medcpt_active_even_if_config_flag_missing_when_scores_exist():
    result = _result(medcpt=True)
    result.trace["reranker_enabled"] = False
    assert infer_medcpt_executed(result.trace, result.evidence, rerank_requested=None) is True
    state = build_runtime_state(result, expand_requested=False, rerank_requested=None, total_latency_ms=1.0)
    assert state["medcpt_executed"] is True
    assert "disabled" not in state["stage_by_key"]["medcpt"]["status"]


def test_query_expansion_state_follows_actual_execution():
    off = build_runtime_state(_result(expand=False), expand_requested=False, total_latency_ms=1)
    on = build_runtime_state(_result(expand=True), expand_requested=True, total_latency_ms=1)
    assert off["query_expansion_executed"] is False
    assert off["query_expansion_status"] == "OFF"
    assert on["query_expansion_executed"] is True
    assert on["query_expansion_status"] == "ON"


def test_pipeline_stage_cards_do_not_report_zero_outputs_when_evidence_exists():
    result = _result()
    result.trace["bm25_results"] = []
    result.trace["dense_results"] = []
    result.trace["fusion_results"] = []
    result.trace["final_evidence"] = []
    result.trace["reranker_results"] = []
    cards = pipeline_stage_cards(result.trace, evidence_count=len(result.evidence))
    by_name = {card["stage"]: card for card in cards}
    assert by_name["BM25 + Dense"]["output_count"] > 0
    assert by_name["Fusion"]["output_count"] > 0
    assert by_name["MedCPT"]["enabled"] is True
    assert by_name["MedCPT"]["output_count"] > 0
    assert by_name["Final Evidence"]["output_count"] == 5
    assert by_name["MedCPT"]["decision"] == "enabled/active"
    assert "disabled" not in by_name["Terminology Normalization"]["decision"]
    assert "Normalization applied: no change" in by_name["Terminology Normalization"]["decision"]


def test_passage_id_pmid_are_not_split_when_identical():
    rows = evidence_inspector_rows(_result().evidence)
    display = evidence_table_display_rows(rows)
    assert "Passage ID / PMID" in display[0]
    assert "Passage ID" not in display[0]
    assert "PMID" not in display[0]
    assert str(display[0]["Passage ID / PMID"]) == str(rows[0]["passage_id"]) == str(rows[0]["pmid"])


def test_heuristic_scores_are_zero_to_one_not_over_five():
    assert format_heuristic_score(1.0) == "1.00"
    assert "/5" not in format_heuristic_score(1.0)
    assert format_judge_score(4.1) == "4.10/5"


def test_query_understanding_unclassified_has_reason():
    analysis = query_analysis_from_trace(
        {"original_query": "???", "query_classification": {"query_type": "unclassified"}},
        original="???",
    )
    assert analysis["query_type"] == "unclassified"
    assert "recognized category" in analysis["query_type_reason"]


def test_query_understanding_definition_intent():
    understanding_trace = {
        "original_query": "What is cerebral hemiatrophy?",
        "normalized_query": "What is cerebral hemiatrophy?",
    }
    analysis = query_analysis_from_trace(understanding_trace, original="What is cerebral hemiatrophy?")
    assert analysis["query_type"] == "simple_factual"
    assert "Definition" in analysis["query_type_label"]
    assert analysis["normalization_status"] == "Normalization applied: no change"
    assert "heuristic" in analysis["classifier_mechanism"].lower()


def test_claim_coverage_maps_supported_and_unsupported_claims():
    result = _result()
    coverage = claim_evidence_coverage(result.answer, result.evidence, result.citations)
    assert coverage["total_claims"] >= 2
    assert coverage["supported_claims"] >= 1
    assert 0.0 <= coverage["claim_support_rate"] <= 1.0
    assert split_claims(result.answer)


def test_prompt_registry_only_includes_real_llm_prompts():
    tabs = {item["tab"] for item in prompt_inspector_tabs()}
    assert tabs == {"Query Expansion", "Answer Generation", "LLM Judge"}
    notes = {item["component"] for item in non_llm_prompt_notes()}
    assert "Query Classification" in notes
    assert "Terminology Normalization" in notes


def test_model_registry_keeps_experimental_inactive():
    registry = model_registry(medcpt_active=True, expansion_active=False)
    experimental = {item["id"]: item for item in registry["experimental"]}
    assert experimental["splade"]["active"] is False
    assert experimental["biomedbert-reranker"]["lane"] == "Experimental"
    production = {item["id"]: item for item in registry["production"]}
    assert production["medcpt-ce"]["active"] is True
    assert production["minilm"]["name"] == "sentence-transformers/all-MiniLM-L6-v2"


def test_latency_percentiles_use_history():
    result = _result()
    state = build_runtime_state(
        result,
        expand_requested=False,
        rerank_requested=True,
        total_latency_ms=100.0,
        latency_history_ms=[100.0, 200.0, 300.0],
    )
    assert state["latency"]["p50_ms"] == 200.0
    assert state["latency"]["p99_ms"] == 300.0
    assert state["cost"]["generation_usd"] is None


def test_generation_does_not_invent_runtime_objects():
    dummy = SimpleNamespace(
        question="q",
        answer="a",
        evidence=(),
        citations=CitationValidation((), (), (), False, True),
        insufficient_evidence=True,
        llm_called=False,
        expansions=(),
        query_expansion_used=False,
        trace={},
    )
    state = build_runtime_state(dummy, expand_requested=False, rerank_requested=False, total_latency_ms=0)
    assert state["medcpt_executed"] is False
    assert state["stage_by_key"]["medcpt"]["status"] == "disabled/not executed"
