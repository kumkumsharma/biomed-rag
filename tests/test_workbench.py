"""Workbench dataset, leaderboard, comparison, traces, and registries."""

from __future__ import annotations

from app.ui import NAV_GROUPS
from src.config import RERANKER, RETRIEVAL, UI
from src.inspection.runtime_state import build_runtime_state
from src.workbench.dataset import DATASET_ID, evaluation_dataset
from src.workbench.display import not_measured
from src.workbench.experiments import (
    compare_experiments,
    dataset_consistency_report,
    experiment_by_id,
    leaderboard_query_counts,
    leaderboard_table_rows,
    load_leaderboard,
    load_per_query_results,
    per_query_diff,
    promotion_packet,
)
from src.workbench.failures import WORKBENCH_FAILURE_CATEGORIES, failure_analysis, query_failure_rows, retrieval_funnel
from src.workbench.overview import overview_payload
from src.workbench.registry import evaluator_registry, llm_judge_spec, prompt_registry, workbench_model_registry
from src.workbench.traces import TRACE_STAGES, filter_traces, stored_traces, trace_timeline


def test_production_retrieval_defaults_unchanged():
    assert RERANKER.enabled is False
    assert RERANKER.model_name == "ncbi/MedCPT-Cross-Encoder"
    assert RERANKER.candidate_pool_size == 50
    assert RERANKER.final_top_k == 10
    assert RETRIEVAL.dense_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert RETRIEVAL.rrf_k == 60


def test_evaluation_dataset_is_the_fixed_100():
    dataset = evaluation_dataset()
    assert dataset["id"] == DATASET_ID
    assert dataset["n_queries"] == 100
    assert dataset["n_gold_ids"] == 1142
    assert dataset["reusable"] is True


def test_leaderboard_reuses_stored_100_query_experiments():
    rows = load_leaderboard()
    by_id = {item["id"]: item for item in rows}
    required = ("bm25", "dense-minilm", "hybrid", "hybrid-qe", "hybrid-medcpt")
    for experiment_id in required:
        assert experiment_id in by_id, experiment_id
        assert by_id[experiment_id]["n_queries"] == 100
        assert by_id[experiment_id]["dataset_id"] == DATASET_ID
    assert abs(by_id["bm25"]["recall_at_10"] - 0.4185) < 1e-3
    assert abs(by_id["dense-minilm"]["recall_at_10"] - 0.3432) < 1e-3
    assert abs(by_id["hybrid"]["recall_at_10"] - 0.4286) < 1e-3
    assert abs(by_id["hybrid-qe"]["recall_at_10"] - 0.4452) < 1e-3
    assert abs(by_id["hybrid-medcpt"]["recall_at_10"] - 0.5082) < 1e-3
    assert by_id["hybrid-medcpt"]["production"] is True
    assert by_id["hybrid-medcpt"]["production_label"] == "Yes"
    assert by_id["bm25"]["production_label"] == "No"
    assert by_id["hybrid-medcpt"]["n_queries"] == 100
    counts = leaderboard_query_counts(rows)
    assert all(value == 100 for value in counts.values())


def test_leaderboard_includes_stored_reranker_experiments_only():
    names = {item["name"] for item in load_leaderboard()}
    assert "Hybrid + MedCPT" in names
    assert "Hybrid + candidate pool 100" in names
    assert "Hybrid + MedCPT bi-encoder" in names
    assert "Hybrid + PubMedBERT" in names
    assert "Hybrid + BiomedBERT reranker" in names
    assert "Hybrid + SPLADE + MedCPT" in names
    assert "Invented system" not in names


def test_leaderboard_does_not_invent_missing_percentiles_or_cost():
    pubmed = experiment_by_id("hybrid-pubmedbert")
    assert pubmed is not None
    assert pubmed["p95_latency_ms"] is None
    assert pubmed["cost_per_query_usd"] is None
    assert pubmed["cost_note"] == "Not measured"
    assert not_measured(None) == "Not measured"
    medcpt = experiment_by_id("hybrid-medcpt")
    assert medcpt is not None
    assert medcpt["p95_latency_ms"] is not None
    assert medcpt["recall_at_1"] is not None
    assert medcpt["recall_at_3"] is not None
    bm25 = experiment_by_id("bm25")
    assert bm25 is not None
    assert bm25["recall_at_1"] is None


def test_dataset_consistency_all_experiments_on_100():
    report = dataset_consistency_report()
    assert report["expected_n_queries"] == 100
    assert report["consistent"] is True
    assert report["mismatches"] == []
    assert report["n_experiments"] >= 5


def test_leaderboard_table_aggregation():
    table = leaderboard_table_rows()
    columns = {
        "Experiment",
        "Dataset",
        "Queries",
        "Candidate Recall@50",
        "Recall@1",
        "Recall@3",
        "Recall@5",
        "Recall@10",
        "MRR@10",
        "nDCG@10",
        "Rerank Recovery",
        "Mean latency",
        "P50 latency",
        "P95 latency",
        "Cost/query",
    }
    assert columns.issubset(table[0])
    assert {row["Queries"] for row in table} == {100}


def test_experiment_comparison_uses_neutral_language():
    comparison = compare_experiments("hybrid", "hybrid-medcpt")
    assert comparison["same_dataset"] is True
    assert comparison["same_query_count"] is True
    rec = comparison["metrics"]["recall_at_10"]
    assert rec["delta"] > 0
    joined = " ".join(comparison["statements"])
    assert "Recall@10 increased by 7.96 percentage points" in joined
    assert "better" not in joined.lower()
    assert "winner" not in joined.lower()
    assert "improved" not in joined.lower()


def test_per_query_diff_identifies_helped_and_hurt():
    diff = per_query_diff("hybrid", "hybrid-medcpt")
    assert len(diff) == 100
    assert any(item["helped"] for item in diff)
    first = diff[0]
    assert "query_id" in first
    assert "question" in first
    assert "a_result" in first
    assert "b_result" in first
    assert "recall_change" in first
    assert "rank_change" in first
    assert "latency_change_ms" in first
    hybrid = load_per_query_results("hybrid")
    medcpt = load_per_query_results("hybrid-medcpt")
    assert len(hybrid) == 100
    assert len(medcpt) == 100


def test_trace_consistency_and_filters():
    traces = stored_traces()
    assert traces
    ids = {item["id"] for item in traces}
    assert "query_5" in ids
    filtered = filter_traces(traces, query="metformin")
    assert any(item["id"] == "query_5" for item in filtered)
    payload = next(item["payload"] for item in traces if item["id"] == "query_5")
    stages = trace_timeline(payload)
    names = [item["stage"] for item in stages]
    assert names == list(TRACE_STAGES)
    for stage in stages:
        assert "input" in stage
        assert "output" in stage
        assert "latency_ms" in stage
        assert "model" in stage
        assert "status" in stage


def test_evaluator_registry_identifies_heuristic_not_judge_model():
    names = [item["name"] for item in evaluator_registry()]
    types = {item["type"] for item in evaluator_registry()}
    assert "Heuristic evaluator" in names
    assert "Judge model" not in names
    assert "Heuristic evaluator" in types
    assert "LLM-as-judge" in types
    assert "Code evaluator" in types
    assert "Human evaluator" in types
    heuristic = next(item for item in evaluator_registry() if item["name"] == "Heuristic evaluator")
    assert heuristic["critical_path"] is False
    spec = llm_judge_spec()
    assert spec["enabled_on_ask_path"] is False


def test_prompt_registry_only_existing_prompts():
    names = {item["name"] for item in prompt_registry()}
    assert names == {"Query Expansion", "Answer Generation", "LLM Judge"}


def test_model_registry_separates_production_and_experimental():
    registry = workbench_model_registry()
    prod = {item["id"] for item in registry["production"]}
    exp = {item["id"] for item in registry["experimental"]}
    assert prod == {"bm25", "minilm", "medcpt-ce", "gpt-4o-mini"}
    assert "splade" in exp
    assert "pubmedbert" in exp
    assert "medcpt-query" in exp
    assert "biomedbert-reranker" in exp
    assert registry["retired"] == []
    packet = promotion_packet("hybrid-pubmedbert")
    assert packet["production"] is False
    assert packet["auto_promote"] is False


def test_failure_categories_are_complete_and_mapped():
    analysis = failure_analysis()
    names = [item["category"] for item in analysis["categories"]]
    assert tuple(names) == WORKBENCH_FAILURE_CATEGORIES
    retrieval = next(item for item in analysis["categories"] if item["category"] == "RETRIEVAL")
    gold = next(item for item in analysis["categories"] if item["category"] == "GOLD_LABEL")
    data = next(item for item in analysis["categories"] if item["category"] == "DATA_QUALITY")
    assert retrieval["count"] == 385 + 22 + 37
    assert gold["count"] == 338
    assert data["count"] == 12220
    failed = query_failure_rows()
    assert failed
    assert {item["category"] for item in failed} <= set(WORKBENCH_FAILURE_CATEGORIES)


def test_retrieval_funnel_distinguishes_id_and_query_level():
    funnel = retrieval_funnel()
    assert funnel["id_level"][0]["stage"] == "Gold IDs"
    assert funnel["query_level"][0]["stage"] == "Queries"
    assert "ID-level" in funnel["distinction"]
    assert "Query-level" in funnel["distinction"]


def test_overview_uses_stored_production_metrics():
    payload = overview_payload()
    assert payload["dataset"]["n_queries"] == 100
    assert payload["production"]["id"] == "hybrid-medcpt"
    assert abs(payload["production"]["recall_at_10"] - 0.5082) < 1e-3


def test_navigation_is_the_workbench_information_architecture():
    assert UI.page_title == "RAG Evaluation Workbench"
    assert list(NAV_GROUPS) == [
        "Overview",
        "Playground",
        "Evaluations",
        "Experiments",
        "Traces",
        "Data",
        "System",
    ]
    assert NAV_GROUPS["Evaluations"] == ("Datasets", "Benchmark Runs", "Metrics", "Failure Analysis", "Regression")
    assert NAV_GROUPS["Experiments"] == ("Experiment List", "Compare", "Per-query Diff")
    assert NAV_GROUPS["Traces"] == ("Trace Explorer", "Trace Detail")
    assert NAV_GROUPS["System"] == ("Models", "Prompts", "Architecture", "Observability")


def test_runtime_state_consistency_still_holds():
    from tests.test_ui_runtime_state import _result

    state = build_runtime_state(
        _result(medcpt=True, expand=False),
        expand_requested=False,
        rerank_requested=True,
        total_latency_ms=100.0,
    )
    assert state["medcpt_executed"] is True
    assert state["query_expansion_executed"] is False
    assert state["consistent"] is True
