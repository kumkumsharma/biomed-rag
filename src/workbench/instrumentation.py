"""Instrumentation coverage. Missing signals stay Not measured."""

from __future__ import annotations

from typing import Any

from src.workbench.experiments import experiment_by_id, load_leaderboard


def _status(*, measured: bool, partial: bool = False) -> str:
    if measured and partial:
        return "Partially available"
    if measured:
        return "Measured"
    return "Not measured"


def instrumentation_coverage(*, experiment_id: str = "hybrid-medcpt", root: Any = None) -> list[dict[str, Any]]:
    item = experiment_by_id(experiment_id, root=root) or {}
    board = load_leaderboard(root=root)
    any_cost = any(row.get("cost_per_query_usd") is not None for row in board)
    any_p95 = item.get("p95_latency_ms") is not None
    any_p50 = item.get("p50_latency_ms") is not None
    any_mean = item.get("mean_latency_ms") is not None
    percentile_complete = bool(any_p50 and any_p95 and item.get("p99_latency_ms") is not None)
    return [
        {
            "signal": "Latency",
            "status": _status(measured=any_mean),
            "detail": item.get("mean_latency_ms"),
        },
        {
            "signal": "P50/P95/P99",
            "status": _status(measured=bool(any_p50 or any_p95), partial=not percentile_complete),
            "detail": {
                "p50": item.get("p50_latency_ms"),
                "p95": item.get("p95_latency_ms"),
                "p99": item.get("p99_latency_ms"),
            },
        },
        {
            "signal": "Token usage",
            "status": "Not measured",
            "detail": None,
        },
        {
            "signal": "Cost",
            "status": _status(measured=any_cost, partial=item.get("cost_per_query_usd") is None),
            "detail": item.get("cost_per_query_usd"),
            "note": item.get("cost_note"),
        },
        {
            "signal": "Git SHA",
            "status": "Not measured",
            "detail": item.get("git_version"),
        },
        {
            "signal": "Prompt version",
            "status": "Not measured",
            "detail": item.get("prompt_versions"),
            "note": "Historical JSON does not store prompt versions. Current registry versions exist in code.",
        },
        {
            "signal": "Model version",
            "status": "Partially available",
            "detail": "Model IDs are attached from the registry; Hugging Face revisions are not recorded.",
        },
    ]
