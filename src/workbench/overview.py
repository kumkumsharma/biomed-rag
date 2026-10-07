"""Overview facts for the workbench home. All numbers come from stored files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.workbench.dataset import evaluation_dataset
from src.workbench.experiments import load_leaderboard
from src.workbench.failures import failure_analysis

ROOT = Path(__file__).resolve().parents[2]


def overview_payload(*, root: Path | None = None) -> dict[str, Any]:
    dataset = evaluation_dataset(root=root)
    board = load_leaderboard(root=root)
    production = next((item for item in board if item.get("production")), None)
    latest = board[-1] if board else None
    regressions = [
        item
        for item in board
        if item.get("promotion") == "rejected" and item.get("recall_at_10") is not None
    ]
    worst = None
    if production and regressions:
        worst = min(
            regressions,
            key=lambda item: float(item["recall_at_10"]) - float(production["recall_at_10"]),
        )
    failures = failure_analysis(root=root)
    top_failure = max(
        [c for c in failures["categories"] if c["category"] in {"RETRIEVAL", "GOLD_LABEL", "RERANKING", "FUSION"}],
        key=lambda item: item.get("count") or 0,
        default=None,
    )
    return {
        "dataset": dataset,
        "production": production,
        "latest_experiment": latest,
        "latest_regression": worst,
        "latest_failure": top_failure,
        "n_experiments": len(board),
        "production_config": (
            "BM25 + all-MiniLM-L6-v2 + RRF k=60 → Top 50 → MedCPT Cross-Encoder → Top 10 → gpt-4o-mini"
        ),
    }
