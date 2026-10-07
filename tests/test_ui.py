from pathlib import Path

from app.ui import (
    METRIC_GLOSSARY,
    answer_inspection_fields,
    answer_review_summary_counts,
    answer_row_by_id,
    build_request_trace,
    citation_status_messages,
    classify_ask_error,
    format_judge_score,
    format_latency_display,
    index_retrieval_details,
    load_json_result,
    load_manual_review_examples,
    per_query_retrieval_rows,
    pipeline_caption,
    public_error_message,
    quality_latency_points,
    query_id_from_choice,
    redact_secrets,
    retrieval_query_choices,
    retrieval_table_rows,
    rerank_comparison_table_rows,
    rerank_experiment_notes,
    rerank_gain_table_rows,
    selected_configuration_notes,
)
from src.config import EVALUATION, GENERATION, RETRIEVAL
from src.generation.rag import GroundedQAResult
from src.retrieval.hybrid import RetrievalResult
from src.validation.citations import CitationValidation

ROOT = Path(__file__).resolve().parents[1]


def test_ui_module_imports():
    import app.ui as ui

    assert callable(ui.main)
    assert callable(ui.build_pipeline)
    assert callable(ui.load_pipeline)


def test_redact_secrets_strips_credential_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-value")
    message = redact_secrets("Authorization Bearer sk-secret-value failed")
    assert "sk-secret-value" not in message
    assert "[redacted]" in message


def test_public_error_message_does_not_include_api_keys(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "super-secret")
    error = public_error_message(RuntimeError("token=super-secret"), stage="llm")
    assert "super-secret" not in error
    assert error.startswith("The language-model call failed.")


def test_classify_ask_error_detects_llm_failures():
    assert classify_ask_error(RuntimeError("Set OPENAI_API_KEY to call the language model.")) == "llm"
    assert classify_ask_error(RuntimeError("index lookup failed")) == "retrieval"


def test_citation_status_shows_invalid_ids():
    citations = CitationValidation(
        cited_ids=("1", "999"),
        valid_ids=(1,),
        invalid_ids=("999",),
        has_citations=True,
        missing_citations=False,
    )
    messages = citation_status_messages(citations, insufficient_evidence=False)
    assert messages[0][0] == "error"
    assert "999" in messages[0][1]


def test_citation_status_is_quiet_for_insufficient_evidence():
    citations = CitationValidation(
        cited_ids=(),
        valid_ids=(),
        invalid_ids=(),
        has_citations=False,
        missing_citations=True,
    )
    assert citation_status_messages(citations, insufficient_evidence=True) == []


def _sample_result(*, expand: bool = True, llm_called: bool = True) -> GroundedQAResult:
    citations = CitationValidation(
        cited_ids=("1",),
        valid_ids=(1,),
        invalid_ids=(),
        has_citations=True,
        missing_citations=False,
    )
    evidence = (
        RetrievalResult(passage_id=1, score=0.1, text="a"),
        RetrievalResult(passage_id=2, score=0.05, text="b"),
    )
    expansions = ("alt 1", "alt 2", "alt 3") if expand else ()
    return GroundedQAResult(
        question="Is papilin secreted?",
        answer="Yes [PASSAGE_ID: 1]",
        evidence=evidence,
        citations=citations,
        insufficient_evidence=False,
        llm_called=llm_called,
        prompt="",
        expansions=expansions,
        query_expansion_used=expand,
    )


def test_request_overview_has_no_duplicate_live_metrics():
    payload = build_request_trace(_sample_result(), expand=True, total_latency_ms=123.4)
    summary = payload["summary"]
    joined = " ".join(summary.keys()) + " " + " ".join(str(v) for v in summary.values())
    assert "Recall@" not in joined
    assert "MRR" not in joined
    assert "nDCG" not in joined
    assert summary["Retrieval strategy"] == "Hybrid + Query Expansion"
    assert summary["Total latency"] == "123.4 ms"
    assert summary["BM25 searches"] == "4"
    assert summary["Dense searches"] == "4"
    assert summary["RRF fusion count"] == "1"
    assert summary["Evidence returned"] == "2"
    assert summary["LLM calls"] == "2"
    assert summary["Valid/invalid citations"].startswith("Valid")
    assert summary["QE cost"] == (
        f"${GENERATION.query_expansion_estimated_cost_per_call_usd:.4f}"
    )
    assert "Query expansions" not in summary
    assert "Latency for this query" not in joined


def test_request_overview_omits_qe_cost_when_expansion_is_off():
    payload = build_request_trace(_sample_result(expand=False), expand=False, total_latency_ms=10)
    summary = payload["summary"]
    assert summary["Retrieval strategy"] == "Hybrid RRF"
    assert summary["LLM calls"] == "1"
    assert summary["BM25 searches"] == "1"
    assert "QE cost" not in summary


def test_retrieval_table_rows_read_saved_comparison_file():
    payload = load_json_result(ROOT / EVALUATION.comparison_path)
    rows = retrieval_table_rows(payload)
    names = [row["Configuration"] for row in rows]
    assert names == ["BM25", "Dense", "Hybrid RRF", "Hybrid + Query Expansion"]
    qe = rows[-1]
    assert qe["Recall@5"] == payload["configurations"][-1]["recall_at_5"]
    assert qe["Cost (USD)"] == 0.01


def test_selected_configuration_notes_describe_measured_tradeoff():
    payload = load_json_result(ROOT / EVALUATION.comparison_path)
    notes = " ".join(selected_configuration_notes(payload["configurations"]))
    assert "Selected configuration: Hybrid + Query Expansion." in notes
    assert "Highest Recall@5" in notes
    assert "Recall@10" in notes
    assert "nDCG@10" in notes
    assert "Hybrid RRF has slightly higher MRR@10" in notes
    assert "Hybrid ≈ 64 ms" in notes
    assert "1,675 ms" in notes


def test_answer_summary_exposes_saved_judge_model_and_scores():
    summary = load_json_result(ROOT / EVALUATION.answer_summary_path)
    assert summary["n_queries"] == 20
    assert summary["judge_model_name"] == EVALUATION.judge_model_name
    assert summary["judge_model_name"] == "gpt-4o-mini"
    assert summary["average_correctness"] == 4.1
    assert summary["average_groundedness"] == 4.1
    assert summary["average_context_relevance"] == 4.45


def test_manual_review_examples_use_saved_answer_details():
    details = load_json_result(ROOT / EVALUATION.answer_details_path)
    success = load_manual_review_examples(details, (0, 10, 19))
    failures = load_manual_review_examples(details, (5, 9, 15))
    assert [item["query_id"] for item in success] == [0, 10, 19]
    assert [item["query_id"] for item in failures] == [5, 9, 15]
    query9 = failures[1]
    assert "DVL-3" in query9["note"]
    assert "Citation validation is valid" in query9["note"]
    query5 = failures[0]
    assert "insufficient-evidence" in query5["note"]
    assert "metformin" in query5["note"]
    assert query5["citation_validation"] == "Invalid"


def _flatten_trace(payload: dict) -> str:
    parts = [str(payload.get("judge_note", ""))]
    parts.extend(str(value) for value in (payload.get("summary") or {}).values())
    for stage in payload.get("stages") or []:
        parts.append(str(stage.get("title", "")))
        parts.extend(f"{key} {value}" for key, value in (stage.get("fields") or {}).items())
    return " ".join(parts)


def test_request_trace_shows_pipeline_stages_without_offline_metrics():
    payload = build_request_trace(_sample_result(), expand=True, total_latency_ms=6191.2)
    titles = [stage["title"] for stage in payload["stages"]]
    assert titles == [
        "1. Query Expansion",
        "2. BM25 Retrieval",
        "3. Dense Retrieval",
        "4. Hybrid / RRF Fusion",
        "5. Answer Generation",
        "6. Citation Validation",
        "7. LLM Judge",
    ]
    assert payload["summary"]["LLM calls"] == "2"
    assert payload["summary"]["Evidence returned"] == "2"
    assert payload["summary"]["Valid/invalid citations"].startswith("Valid")
    assert payload["summary"]["Total latency"] == "6,191.2 ms"
    joined = _flatten_trace(payload)
    assert "Recall@" not in joined
    assert "MRR" not in joined
    assert "nDCG" not in joined
    assert "LIVE ANSWER QUALITY" in payload["judge_note"]
    assert "Evaluation tab" in payload["judge_note"]
    expansion = payload["stages"][0]["fields"]
    assert expansion["Model"] == GENERATION.query_expansion_model_name
    assert expansion["Expansions generated"] == 3
    assert "Calls" not in expansion
    assert "Latency" not in expansion
    bm25 = payload["stages"][1]["fields"]
    assert bm25["Top-k"] == GENERATION.evidence_top_k
    assert "Number of queries searched" not in bm25
    dense = payload["stages"][2]["fields"]
    assert dense["Embedding model"] == RETRIEVAL.dense_model_name
    assert "Number of queries searched" not in dense
    hybrid = payload["stages"][3]["fields"]
    assert hybrid["Fusion method"] == "Reciprocal Rank Fusion"
    assert hybrid["RRF k"] == RETRIEVAL.rrf_k
    assert "Final evidence count" not in hybrid
    generation = payload["stages"][4]["fields"]
    assert generation["Model"] == GENERATION.model_name
    assert "LLM call count" not in generation
    citations = payload["stages"][5]["fields"]
    assert "Citation status" not in citations
    assert "Number of valid citations" not in citations
    judge = payload["stages"][6]["fields"]
    assert judge["Status"] == "Not evaluated"
    assert judge["Model"] == EVALUATION.judge_model_name


def test_request_trace_includes_measured_stage_latencies_when_present():
    result = _sample_result()
    object.__setattr__(
        result,
        "trace",
        {
            "query_expansion_ms": 10.0,
            "bm25_ms": 20.0,
            "dense_ms": 30.0,
            "rrf_ms": 5.0,
            "generation_ms": 40.0,
            "citation_ms": 1.0,
            "expansion_llm_called": True,
            "n_search_queries": 4,
            "n_rankings": 8,
            "candidate_k": 5,
            "rrf_k": 60,
            "embedding_dim": 384,
            "dense_model_name": RETRIEVAL.dense_model_name,
        },
    )
    payload = build_request_trace(result, expand=True, total_latency_ms=100.0)
    assert payload["stages"][0]["fields"]["Latency"] == "10.0 ms"
    assert payload["stages"][1]["fields"]["Latency"] == "20.0 ms"
    assert payload["stages"][2]["fields"]["Latency"] == "30.0 ms"
    assert payload["stages"][2]["fields"]["Embedding dimension"] == 384
    assert payload["stages"][3]["fields"]["Number of input rankings"] == 8
    assert payload["stages"][3]["fields"]["Latency"] == "5.0 ms"
    assert payload["stages"][4]["fields"]["Latency"] == "40.0 ms"
    assert payload["stages"][5]["fields"]["Latency"] == "1.0 ms"


def test_request_trace_includes_live_judge_latency_without_correctness():
    payload = build_request_trace(
        _sample_result(),
        expand=True,
        total_latency_ms=100.0,
        live_quality={
            "evaluated": True,
            "status": "Evaluated",
            "groundedness": 4,
            "context_relevance": 5,
            "model_name": EVALUATION.judge_model_name,
            "latency_ms": 250.0,
            "llm_calls": 1,
        },
    )
    judge = payload["stages"][6]
    assert judge["title"] == "7. LLM Judge"
    assert judge["fields"]["Latency"] == "250.0 ms"
    assert judge["fields"]["Calls"] == 1
    assert judge["fields"]["Model"] == EVALUATION.judge_model_name
    assert "Correctness" not in judge["fields"]
    assert payload["summary"]["LLM calls"] == "2"


def test_request_trace_without_expansion_has_one_search_query_and_no_judge():
    payload = build_request_trace(_sample_result(expand=False), expand=False, total_latency_ms=50)
    assert payload["summary"]["LLM calls"] == "1"
    assert payload["stages"][0]["fields"]["Status"] == "Skipped"
    assert payload["summary"]["BM25 searches"] == "1"
    assert payload["stages"][6]["title"] == "7. LLM Judge"
    assert payload["stages"][6]["fields"]["Status"] == "Not evaluated"


def test_pipeline_caption_lists_live_explorer_stages():
    caption = pipeline_caption(reranker_enabled=False)
    assert caption == (
        "Question → Query Expansion → BM25 + Dense → RRF Fusion → "
        "Evidence → Grounded LLM → Citation Validation → LLM Judge"
    )


def test_pipeline_caption_inserts_medcpt_when_reranker_enabled():
    caption = pipeline_caption(reranker_enabled=True)
    assert "RRF Fusion → MedCPT Rerank → Evidence" in caption


def test_request_trace_inserts_rerank_stage_when_enabled():
    result = _sample_result()
    object.__setattr__(
        result,
        "trace",
        {
            "reranker_enabled": True,
            "reranker_model_name": "ncbi/MedCPT-Cross-Encoder",
            "reranker_device": "cpu",
            "reranker_candidate_pool_size": 50,
            "rerank_candidate_count": 12,
            "rerank_ms": 15.0,
            "generation_ms": 40.0,
            "citation_ms": 1.0,
            "expansion_llm_called": True,
            "n_search_queries": 4,
        },
    )
    payload = build_request_trace(result, expand=True, total_latency_ms=100.0)
    titles = [stage["title"] for stage in payload["stages"]]
    assert titles == [
        "1. Query Expansion",
        "2. BM25 Retrieval",
        "3. Dense Retrieval",
        "4. Hybrid / RRF Fusion",
        "5. MedCPT Cross-Encoder",
        "6. Answer Generation",
        "7. Citation Validation",
        "8. LLM Judge",
    ]
    rerank = payload["stages"][4]["fields"]
    assert rerank["Model"] == "ncbi/MedCPT-Cross-Encoder"
    assert rerank["Candidate pool"] == 50
    assert rerank["Latency"] == "15.0 ms"


def test_request_summary_does_not_repeat_trace_stage_counts():
    payload = build_request_trace(_sample_result(), expand=True, total_latency_ms=6191.2)
    summary = payload["summary"]
    assert summary["Total latency"] == "6,191.2 ms"
    assert summary["LLM calls"] == "2"
    assert summary["Evidence returned"] == "2"
    assert summary["Valid/invalid citations"].startswith("Valid")
    assert summary["BM25 searches"] == "4"
    assert summary["Dense searches"] == "4"
    assert summary["RRF fusion count"] == "1"
    assert "QE cost" in summary
    assert "usage" not in payload
    stage_text = " ".join(
        f"{key} {value}"
        for stage in payload["stages"]
        for key, value in (stage.get("fields") or {}).items()
    )
    assert "Number of queries searched" not in stage_text
    assert "LLM call count" not in stage_text
    assert "Final evidence count" not in stage_text
    assert "Exact Match" not in stage_text
    assert "Hallucination" not in stage_text


def test_judge_scores_are_formatted_from_saved_json():
    summary = load_json_result(ROOT / EVALUATION.answer_summary_path)
    assert format_judge_score(summary["average_correctness"]) == "4.10/5"
    assert format_judge_score(summary["average_groundedness"]) == "4.10/5"
    assert format_judge_score(summary["average_context_relevance"]) == "4.45/5"


def test_quality_latency_points_use_saved_comparison_values():
    payload = load_json_result(ROOT / EVALUATION.comparison_path)
    points = quality_latency_points(payload["configurations"])
    by_name = {item["Configuration"]: item for item in points}
    assert set(by_name) == {"BM25", "Dense", "Hybrid RRF", "Hybrid + Query Expansion"}
    qe = by_name["Hybrid + Query Expansion"]
    assert qe["Recall@10"] == payload["configurations"][-1]["recall_at_10"]
    assert qe["Avg latency (ms)"] == payload["configurations"][-1]["avg_latency_ms"]


def test_retrieval_query_inspector_reads_saved_detail_files():
    details = load_json_result(ROOT / EVALUATION.details_path)
    hybrid_qe = load_json_result(ROOT / EVALUATION.hybrid_qe_details_path)
    indexed = index_retrieval_details(details, hybrid_qe)
    assert 0 in indexed
    assert set(indexed[0]) == {"BM25", "Dense", "Hybrid RRF", "Hybrid + Query Expansion"}
    labels = retrieval_query_choices(indexed)
    assert query_id_from_choice(labels[0]) == 0
    rows = per_query_retrieval_rows(indexed[0])
    names = [row["Configuration"] for row in rows]
    assert names == ["BM25", "Dense", "Hybrid RRF", "Hybrid + Query Expansion"]
    for row in rows:
        assert "Recall@5" in row
        assert "Recall@10" in row
        assert "MRR@10" in row
        assert "nDCG@10" in row
        assert "Latency (ms)" in row
        assert "Retrieved passage IDs" in row
    qe = indexed[0]["Hybrid + Query Expansion"]
    assert qe["recall_at_5"] == rows[-1]["Recall@5"]


def test_answer_inspection_and_review_counts_use_saved_json():
    details = load_json_result(ROOT / EVALUATION.answer_details_path)
    counts = answer_review_summary_counts(details)
    assert counts["Successful manual-review examples"] == 3
    assert counts["Limitation / failure examples"] == 3
    assert counts["Citation-valid cases"] == 19
    assert counts["Missing / invalid citation cases"] == 1
    assert counts["Insufficient-evidence cases"] >= 1
    row = answer_row_by_id(details, 9)
    fields = answer_inspection_fields(row or {})
    assert fields["Query ID"] == 9
    assert "DVL" in fields["Reference answer"]
    assert fields["Citation validation"] == "Valid"
    assert fields["Correctness"] == 2
    assert "Judge reason" in fields
    glossary = " ".join(f"{name} {meaning}" for name, meaning in METRIC_GLOSSARY)
    assert "Whether relevant evidence appears in the top 5." in glossary
    assert "Candidate Recall@50" in glossary
    assert "Rerank recovery@10" in glossary
    assert "Exact Match" not in glossary
    assert "Token F1" not in glossary
    assert "Hallucination" not in glossary
    assert "RAGAS" not in glossary


def test_rerank_comparison_ui_helpers_show_gain_and_do_not_hide_regressions():
    payload = {
        "configurations": [
            {
                "name": "Baseline",
                "recall_at_5": 0.5,
                "recall_at_10": 0.4,
                "mrr_at_10": 0.8,
                "ndcg_at_10": 0.6,
                "avg_latency_ms": 10.0,
            },
            {
                "name": "Hybrid@50",
                "recall_at_5": 0.5,
                "recall_at_10": 0.4,
                "mrr_at_10": 0.8,
                "ndcg_at_10": 0.6,
                "avg_latency_ms": 20.0,
                "candidate_recall_at_50": 0.9,
                "candidate_hit_at_50": 1.0,
            },
            {
                "name": "+ MedCPT",
                "recall_at_5": 0.4,
                "recall_at_10": 0.3,
                "mrr_at_10": 0.7,
                "ndcg_at_10": 0.5,
                "avg_latency_ms": 40.0,
                "candidate_recall_at_50": 0.9,
                "candidate_hit_at_50": 1.0,
            },
        ],
        "candidate_recall_at_50": 0.9,
        "candidate_hit_at_50": 1.0,
        "rerank_gain": {
            "recall_at_5": {"baseline": 0.5, "reranked": 0.4, "absolute": -0.1, "relative": -0.2},
            "recall_at_10": {"baseline": 0.4, "reranked": 0.3, "absolute": -0.1, "relative": -0.25},
            "mrr_at_10": {"baseline": 0.8, "reranked": 0.7, "absolute": -0.1, "relative": -0.125},
            "ndcg_at_10": {"baseline": 0.6, "reranked": 0.5, "absolute": -0.1, "relative": -1 / 6},
        },
        "latency": {"baseline_avg_ms": 10.0, "hybrid_avg_ms": 20.0, "total_reranked_avg_ms": 40.0},
        "findings": ["MedCPT did not improve ranking quality versus Hybrid RRF Top-10 on this set."],
    }
    matrix = rerank_comparison_table_rows(payload)
    assert matrix[1]["Metric"] == "Recall@10"
    assert matrix[1]["+ MedCPT"] == 0.3
    assert matrix[1]["Baseline"] == 0.4
    gains = rerank_gain_table_rows(payload)
    assert gains[0]["Absolute change"] == -0.1
    notes = rerank_experiment_notes(payload)
    assert "did not improve" in notes[0]


def test_pipeline_stage_cards_and_evidence_inspector():
    from app.ui import evidence_inspector_rows, pipeline_stage_cards, query_analysis_from_trace

    trace = {
        "original_query": "EGFR mutation",
        "normalized_query": "epidermal growth factor receptor mutation",
        "query_classification": {"query_type": "terminology_heavy"},
        "expansion_variants": ["epidermal growth factor receptor mutation"],
        "subqueries": [],
        "bm25_results": ["A"],
        "dense_results": ["B"],
        "fusion_results": [{"passage_id": "A", "bm25_rank": 1, "dense_rank": 2, "rrf_score": 0.02, "rrf_rank": 1}],
        "filtering_results": {"before_count": 2, "after_count": 1, "removed_count": 1, "decisions": [{"passage_id": "A", "reason": "kept"}]},
        "reranker_enabled": False,
        "final_evidence": [{"passage_id": "A", "score": 0.02, "pmid": "A", "retrieval_stage": "hybrid"}],
        "stage_timings": {"bm25_ms": 1.0, "dense_ms": 2.0, "rrf_ms": 0.5, "total_ms": 4.0},
        "retrieval_strategy": {"strategy": "fixed_hybrid"},
        "understanding": {"terminology": {"detected_terms": ["egfr"], "mappings": [{"from": "egfr", "to": "epidermal growth factor receptor"}]}},
        "search_queries": ["EGFR mutation"],
    }
    cards = pipeline_stage_cards(trace)
    names = [card["stage"] for card in cards]
    assert names[0] == "Original Query"
    assert names[-1] == "Final Evidence"
    assert "MedCPT" in names
    analysis = query_analysis_from_trace(trace)
    assert analysis["query_type"] == "terminology_heavy"
    assert analysis["normalized_query"].startswith("epidermal")
    rows = evidence_inspector_rows(trace["final_evidence"], trace)
    assert rows[0]["passage_id"] == "A"
    assert rows[0]["pmid"] == "A"
    assert rows[0]["bm25_rank"] == 1


def test_upgrade_experiment_table_and_existing_files():
    from app.ui import existing_result_files, failure_analysis_table_rows, upgrade_experiment_table_rows

    payload = {
        "configurations": [
            {"name": "A Existing Hybrid@10", "recall_at_5": 0.3, "recall_at_10": 0.4, "recall_at_50": 0.6, "mrr_at_10": 0.8, "ndcg_at_10": 0.5, "candidate_recall_at_50": 0.6, "avg_latency_ms": 10}
        ]
    }
    rows = upgrade_experiment_table_rows(payload)
    assert rows[0]["Configuration"] == "A Existing Hybrid@10"
    assert rows[0]["Recall@10"] == 0.4
    available = existing_result_files(ROOT)
    assert "Baseline Hybrid" in available
    failures = failure_analysis_table_rows(
        {
            "rows": [
                {"query_id": 1, "question": "q", "configuration": "A", "recall_at_10": 0.0, "failure": {"category": "relevant_never_retrieved", "query_type": "simple_factual"}},
                {"query_id": 2, "question": "ok", "configuration": "A", "recall_at_10": 1.0, "failure": {"category": "success"}},
            ]
        }
    )
    assert len(failures) == 1
    assert failures[0]["category"] == "relevant_never_retrieved"


def test_architecture_and_research_helpers_read_repo_docs():
    from app.ui import architecture_document_text, research_document_sections

    text = architecture_document_text()
    assert "Hybrid" in text
    names = [name for name, _body in research_document_sections()]
    assert "README.md" in names
    assert "architecture_essentials.md" in names


def test_inspector_ui_helpers_use_real_prompts_and_status_lanes():
    from app.ui import feature_status_table_rows, missing_evidence_table_rows, candidate_pool_table_rows
    from src.generation.answer_generator import GROUNDED_PROMPT_TEMPLATE
    from src.inspection.prompts import llm_prompt_catalog

    catalog = llm_prompt_catalog()
    answer = next(item for item in catalog if item["component"] == "answer_generation")
    assert answer["template"] == GROUNDED_PROMPT_TEMPLATE
    decomp = next(item for item in catalog if item["component"] == "query_decomposition")
    assert decomp["template"] is None
    rows = feature_status_table_rows()
    lanes = {item["Feature"]: item["Lane"] for item in rows}
    assert lanes["Hybrid BM25 + Dense + RRF"] == "BASELINE"
    assert lanes["Deterministic query expansion"] == "EXPERIMENTAL"
    assert lanes["Learned sparse first-stage (SPLADE)"] == "EXPERIMENTAL"
    miss = missing_evidence_table_rows(
        {"table": [{"failure_category": "lexical mismatch", "count": 2, "percentage": 0.5, "potential_remedy": "x"}]}
    )
    assert miss[0]["Failure category"] == "lexical mismatch"
    curve = candidate_pool_table_rows({"curve": [{"k": 50, "recall_at_k": 0.62, "mrr_at_10": 0.8, "ndcg_at_10": 0.5, "avg_latency_ms": 66}]})
    assert curve[0]["K"] == 50
    from app.ui import objective_failure_table_rows, recovery_matrix_table_rows

    obj = objective_failure_table_rows(
        {
            "objective_failure_counts": {"BOTH_RETRIEVERS_MISS": 10, "FUSION_CUTOFF": 2},
            "objective_failure_percent_of_misses": {"BOTH_RETRIEVERS_MISS": 0.8, "FUSION_CUTOFF": 0.2},
        }
    )
    assert obj[0]["Objective category"] == "BOTH_RETRIEVERS_MISS"
    rec = recovery_matrix_table_rows({"n_hybrid50_failures": 10, "counts": {"never_recovered": 7, "BM25": 3}})
    assert rec[0]["Count"] in {7, 3}
    from app.ui import sparse_fusion_comparison_rows, sparse_inspector_rows, sparse_study_metric_rows

    sparse_payload = {
        "sparse": {"model": "naver/splade-v3-distilbert"},
        "latency": {"sparse_avg_search_ms": 12.0},
        "full_retrieval_metrics": [{"name": "Learned sparse", "recall_at_10": 0.1, "recall_at_50": 0.2, "mrr_at_10": 0.3, "ndcg_at_10": 0.2, "avg_latency_ms": 12}],
        "fusion": [
            {"name": "A BM25 + MiniLM", "recall_at_10": 0.4, "recall_at_50": 0.6, "mrr_at_10": 0.8, "ndcg_at_10": 0.5, "avg_latency_ms": 20},
            {"name": "B BM25 + MiniLM + Sparse", "recall_at_10": 0.41, "recall_at_50": 0.62, "mrr_at_10": 0.81, "ndcg_at_10": 0.51, "avg_latency_ms": 30},
        ],
        "fusion_dual_miss": [
            {"name": "A BM25 + MiniLM", "dual_miss_hit_at_50": 0, "dual_miss_hit_at_100": 0},
            {"name": "B BM25 + MiniLM + Sparse", "dual_miss_hit_at_50": 5, "dual_miss_hit_at_100": 12},
        ],
    }
    inspector = {row["Field"]: row["Value"] for row in sparse_inspector_rows(sparse_payload)}
    assert inspector["Status"].startswith("EXPERIMENTAL")
    assert inspector["Nature"].startswith("sparse")
    fusion = sparse_fusion_comparison_rows(sparse_payload)
    assert fusion[0]["Status"] == "BASELINE"
    assert fusion[1]["Status"] == "EXPERIMENTAL"
    assert fusion[1]["Dual-miss@100"] == 12
    assert sparse_study_metric_rows(sparse_payload)[0]["Configuration"] == "Learned sparse"


def test_format_latency_display_includes_milliseconds_and_seconds():
    assert format_latency_display(3133.3) == "3,133.3 ms (3.13 s)"
    assert format_latency_display(None) == "—"


