"""Workbench package: dataset, experiments, traces, evaluation, failure analysis."""

from src.workbench.dataset import evaluation_dataset
from src.workbench.experiments import (
    compare_experiments,
    dataset_consistency_report,
    leaderboard_query_counts,
    leaderboard_table_rows,
    load_leaderboard,
    per_query_diff,
    promotion_packet,
)
from src.workbench.failures import failure_analysis, query_failure_rows, retrieval_funnel
from src.workbench.overview import overview_payload
from src.workbench.registry import evaluator_registry, llm_judge_spec, prompt_registry, workbench_model_registry
from src.workbench.traces import filter_traces, stored_traces, trace_timeline

__all__ = [
    "compare_experiments",
    "dataset_consistency_report",
    "evaluation_dataset",
    "evaluator_registry",
    "failure_analysis",
    "filter_traces",
    "leaderboard_query_counts",
    "leaderboard_table_rows",
    "llm_judge_spec",
    "load_leaderboard",
    "overview_payload",
    "per_query_diff",
    "prompt_registry",
    "promotion_packet",
    "query_failure_rows",
    "retrieval_funnel",
    "stored_traces",
    "trace_timeline",
    "workbench_model_registry",
]
