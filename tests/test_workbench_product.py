"""Productization: comparison, provenance, regression, production lanes, coverage."""

from __future__ import annotations

import inspect

from app.ui import load_pipeline
from src.config import RERANKER
from src.workbench.configuration import (
    configuration_diff,
    display_provenance,
    experiment_configuration,
    provenance,
    run_lineage,
)
from src.workbench.experiments import compare_experiments, per_query_diff
from src.workbench.instrumentation import instrumentation_coverage
from src.workbench.production import PRODUCTION, production_runtime_report, resolve_rerank_enabled
from src.workbench.registry import prompt_registry
from src.workbench.regression import add_to_regression_set, load_regression_set, run_regression, save_regression_set


def test_experiment_comparison_includes_core_metrics_and_neutral_deltas():
    comparison = compare_experiments("hybrid", "hybrid-medcpt")
    assert comparison["same_dataset"] is True
    assert comparison["query_count_baseline"] == 100
    assert comparison["query_count_candidate"] == 100
    rec = comparison["metrics"]["recall_at_10"]
    assert abs(rec["delta"] - 0.0796) < 1e-3
    joined = " ".join(comparison["statements"])
    assert "Recall@10 increased by 7.96 percentage points" in joined
    assert "better" not in joined.lower()
    assert "Mean latency increased" in joined
    assert "recall_at_10" in comparison["metrics"]
    assert "mrr_at_10" in comparison["metrics"]
    assert "ndcg_at_10" in comparison["metrics"]
    assert "mean_latency_ms" in comparison["metrics"]


def test_configuration_diff_marks_changed_and_unchanged():
    rows = {item["field"]: item for item in configuration_diff("hybrid", "hybrid-medcpt")}
    assert rows["retriever"]["changed"] is False
    assert rows["dense_model_id"]["changed"] is False
    assert rows["fusion"]["changed"] is False
    assert rows["reranker_id"]["changed"] is True
    assert rows["reranker_id"]["baseline"] is None
    assert rows["reranker_id"]["candidate"] == "medcpt-ce"
    assert rows["candidate_pool"]["changed"] is True
    assert rows["candidate_pool"]["baseline"] == 10
    assert rows["candidate_pool"]["candidate"] == 50
    assert rows["query_expansion"]["changed"] is False
    assert rows["generation_model_id"]["changed"] is False
    assert {item["component"] for item in configuration_diff("hybrid", "hybrid-medcpt")} == {
        "Retriever",
        "Dense model",
        "Fusion",
        "RRF k",
        "Candidate pool",
        "Reranker",
        "Query expansion",
        "Generation model",
        "Prompt version",
        "Dataset",
        "Dataset version",
    }
    assert rows["rrf_k"]["status"] == "UNCHANGED"
    assert rows["prompt_version"]["status"] == "NOT RECORDED"
    assert rows["reranker_id"]["status"] == "CHANGED"
    assert rows["reranker_id"]["candidate_display"] == "ncbi/MedCPT-Cross-Encoder"


def test_experiment_records_use_model_and_prompt_ids():
    config = experiment_configuration("hybrid-qe")
    assert "minilm" in config["model_ids"]
    assert "gpt-4o-mini" in config["model_ids"]
    assert config["prompt_ids"] == ["query_expansion"]
    registry_ids = {item["id"] for item in prompt_registry()}
    assert "query_expansion" in registry_ids
    assert "answer_generation" in registry_ids
    medcpt = experiment_configuration("hybrid-medcpt")
    assert medcpt["prompt_ids"] == []
    assert "medcpt-ce" in medcpt["model_ids"]


def test_provenance_does_not_invent_timestamp_or_git():
    record = provenance("hybrid-medcpt")
    assert record["experiment_id"] == "hybrid-medcpt"
    assert record["query_count"] == 100
    assert record["dataset_version"] == "1.0"
    assert "rerank_comparison_summary.json" in record["source_artifact"]
    assert record["timestamp"] is None
    assert record["code_version"] is None
    assert record["prompt_version"] is None
    shown = display_provenance(record)
    assert shown["Timestamp"] == "Not recorded"
    assert shown["Code version"] == "Not recorded"
    assert shown["Prompt version"] == "Not recorded"
    lineage = run_lineage("hybrid-medcpt")
    assert [item["stage"] for item in lineage] == [
        "Dataset",
        "Dataset version",
        "Code version",
        "Models",
        "Prompts",
        "Retrieval configuration",
        "Evaluation",
        "Results",
    ]
    assert lineage[2]["value"] == "Not recorded"


def test_regression_dataset_is_separate_and_does_not_invent_fixes(tmp_path):
    save_regression_set([], root=tmp_path)
    added = add_to_regression_set(
        query_id=5,
        question="Does metformin interfere thyroxine absorption?",
        gold_ids=[26191653],
        failure_category="GOLD_LABEL",
        source_experiment="hybrid-medcpt",
        diagnosis="stored Top-10 miss",
        date_added="2026-10-02",
        root=tmp_path,
    )
    assert added["query_id"] == 5
    items = load_regression_set(root=tmp_path)
    assert len(items) == 1
    result = run_regression("hybrid-medcpt", root=tmp_path)
    assert result["counts"]["passed"] + result["counts"]["failed"] + result["counts"]["not_recorded"] == 1
    for item in result["fixed_failures"]:
        assert item["source_recall_at_10"] == 0
        assert item["candidate_recall_at_10"] > 0
    hybrid = run_regression("hybrid", root=tmp_path)
    assert hybrid["counts"]["fixed_failures"] == len(hybrid["fixed_failures"])
    assert hybrid["counts"]["fixed_failures"] == 0


def test_regression_run_labels_fixed_only_from_stored_recall(tmp_path):
    diff = per_query_diff("hybrid", "hybrid-medcpt")
    improved = next(
        item
        for item in diff
        if item["helped"] and (item["a_result"] or 0) == 0 and (item["b_result"] or 0) > 0
    )
    save_regression_set([], root=tmp_path)
    add_to_regression_set(
        query_id=int(improved["query_id"]),
        question=str(improved["question"]),
        gold_ids=list(improved["gold_ids"]),
        failure_category=improved.get("failure_category"),
        source_experiment="hybrid",
        diagnosis="stored Hybrid Top-10 miss",
        date_added="2026-10-02",
        root=tmp_path,
    )
    medcpt = run_regression("hybrid-medcpt", root=tmp_path)
    assert medcpt["counts"]["fixed_failures"] == 1
    assert medcpt["fixed_failures"][0]["source_recall_at_10"] == 0
    assert medcpt["fixed_failures"][0]["candidate_recall_at_10"] > 0
    hybrid = run_regression("hybrid", root=tmp_path)
    assert hybrid["counts"]["fixed_failures"] == 0
    assert hybrid["counts"]["failed"] == 1


def test_production_configuration_lanes_do_not_enable_library_medcpt():
    assert RERANKER.enabled is False
    assert PRODUCTION.reranker_enabled is True
    assert resolve_rerank_enabled(lane="library") is False
    assert resolve_rerank_enabled(lane="production") is True
    assert resolve_rerank_enabled(override=False, lane="playground") is False
    assert resolve_rerank_enabled(override=True, lane="playground") is True
    assert resolve_rerank_enabled(lane="experiment") is False
    report = production_runtime_report()
    assert report["can_accidentally_run_without_medcpt"] is True
    assert report["ranking_behavior_changed"] is False
    assert report["library_default"]["RERANKER.enabled"] is False
    signature = inspect.signature(load_pipeline)
    assert signature.parameters["rerank_enabled"].default is None
    assert signature.parameters["lane"].default == "library"


def test_instrumentation_coverage_never_fabricates():
    rows = {item["signal"]: item for item in instrumentation_coverage()}
    assert rows["Token usage"]["status"] == "Not measured"
    assert rows["Git SHA"]["status"] == "Not measured"
    assert rows["Latency"]["status"] == "Measured"
    assert rows["P50/P95/P99"]["status"] in {"Measured", "Partially available"}
    assert rows["Cost"]["status"] in {"Not measured", "Partially available", "Measured"}
    assert rows["Prompt version"]["status"] == "Not measured"
    assert rows["Model version"]["status"] == "Partially available"


def test_per_query_diff_uses_stored_candidate_and_final_ranks():
    diff = per_query_diff("hybrid", "hybrid-medcpt")
    assert len(diff) == 100
    sample = next(item for item in diff if item["query_id"] == 0)
    assert sample["gold_ids"]
    assert sample["a_final_ids"]
    assert sample["b_final_ids"]
    assert sample["a_candidate_ids"] is None
    assert sample["b_candidate_ids"] is not None
    assert "Candidate-pool IDs not recorded for experiment A" in sample["stage_changes"]
    assert "failure_category" in sample
