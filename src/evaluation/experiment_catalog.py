"""Catalog of completed retrieval experiments. Does not overwrite result files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

EXPERIMENTS: tuple[dict[str, Any], ...] = (
    {
        "id": "baseline",
        "name": "Production baseline",
        "date": "2026-09",
        "changed_variable": "None (BM25 + MiniLM + RRF k=60 → Top 50 → MedCPT → Top 10)",
        "stayed_fixed": "dataset, 100 queries, gold labels, RRF k=60, pool 50, MedCPT CE",
        "metrics": {
            "candidate_recall_at_50": 0.6243,
            "recall_at_10": 0.5082,
            "mrr_at_10": 0.8587,
            "ndcg_at_10": 0.6706,
            "latency_ms": 3133.3,
            "pool_hit_rate": 0.99,
            "rerank_recovery_at_10": 1.0,
        },
        "status": "production",
        "conclusion": "Keep this stack. Later one-variable tests did not beat MedCPT@10.",
        "result_file": "experiments/results/retrieval_evaluation_rerank_comparison_summary.json",
    },
    {
        "id": "splade",
        "name": "SPLADE learned sparse",
        "date": "2026-10",
        "changed_variable": "Added experimental SPLADE first-stage (not fused into production)",
        "stayed_fixed": "BM25, MiniLM, RRF, MedCPT",
        "metrics": {"note": "Dual-miss recovery experiment; not enabled in production."},
        "status": "do not enable",
        "conclusion": "Not justified for production dual-miss recovery.",
        "result_file": "experiments/results/sparse_study_summary.json",
    },
    {
        "id": "pool_100",
        "name": "Hybrid pool 50 → 100",
        "date": "2026-10",
        "changed_variable": "Candidate pool size 50 → 100",
        "stayed_fixed": "BM25, MiniLM, RRF k=60, MedCPT, final top-10",
        "metrics": {"recall_at_10": 0.5047, "candidate_recall_at_50": 0.6494, "latency_ms": 6421.0},
        "status": "rejected",
        "conclusion": "Larger pool raised ceiling slightly but MedCPT Recall@10 worsened and latency doubled.",
        "result_file": "experiments/results/retrieval_evaluation_rerank_comparison_pool100_summary.json",
    },
    {
        "id": "medcpt_biencoder",
        "name": "MiniLM → MedCPT bi-encoder",
        "date": "2026-10",
        "changed_variable": "Dense encoder MiniLM → MedCPT Query+Article",
        "stayed_fixed": "BM25, RRF, pool 50, MedCPT CE, top-10",
        "metrics": {"candidate_recall_at_50": 0.6465, "recall_at_10": 0.5062},
        "status": "rejected",
        "conclusion": "Do not replace MiniLM. Rec@10 slightly worse; index time much higher.",
        "result_file": "experiments/results/retrieval_evaluation_rerank_comparison_medcpt_biencoder_summary.json",
    },
    {
        "id": "query_expansion",
        "name": "Medical Query Expansion",
        "date": "2026-10",
        "changed_variable": "gpt-4o-mini QE before BM25+MiniLM",
        "stayed_fixed": "MiniLM, RRF, pool 50, MedCPT CE, no SPLADE",
        "metrics": {
            "candidate_recall_at_50": 0.6275,
            "recall_at_10": 0.5055,
            "mrr_at_10": 0.8587,
            "ndcg_at_10": 0.6682,
            "latency_ms": 3446.0,
        },
        "status": "rejected",
        "conclusion": "Tiny Rec@50 gain did not survive MedCPT; Rec@10 fell. Leave QE off production.",
        "result_file": "experiments/results/retrieval_evaluation_rerank_comparison_qe_summary.json",
    },
    {
        "id": "biomedbert_reranker",
        "name": "MedCPT CE → BiomedBERT reranker",
        "date": "2026-10",
        "changed_variable": "Reranker model only: NeuML/biomedbert-base-reranker",
        "stayed_fixed": "BM25, MiniLM, RRF, pool 50, no QE",
        "metrics": {
            "candidate_recall_at_50": 0.6243,
            "recall_at_10": 0.4942,
            "mrr_at_10": 0.8067,
            "ndcg_at_10": 0.6374,
            "latency_ms": 3215.0,
            "rerank_recovery_at_10": 0.9798,
        },
        "status": "rejected",
        "conclusion": "Same pool, worse Top-10. Keep MedCPT Cross-Encoder.",
        "result_file": "experiments/results/retrieval_evaluation_rerank_comparison_biomedbert_summary.json",
    },
)


def load_experiment_catalog(*, root: Path | None = None) -> list[dict[str, Any]]:
    base = root or ROOT
    rows = []
    for item in EXPERIMENTS:
        path = base / item["result_file"]
        entry = dict(item)
        entry["result_exists"] = path.exists()
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                entry["file_n_queries"] = payload.get("n_queries")
            except json.JSONDecodeError:
                entry["file_n_queries"] = None
        rows.append(entry)
    return rows


def _direction(candidate: Any, baseline: Any, *, label: str) -> str | None:
    if candidate is None or baseline is None:
        return None
    try:
        left = float(candidate)
        right = float(baseline)
    except (TypeError, ValueError):
        return None
    if left > right:
        return f"{label} ↑"
    if left < right:
        return f"{label} ↓"
    return f"{label} ="


def experiment_lab_rows(catalog: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """One-variable experiment cards. Does not rewrite stored metric values."""

    items = catalog or load_experiment_catalog()
    baseline = next((item for item in items if item.get("id") == "baseline"), {})
    baseline_metrics = dict(baseline.get("metrics") or {})
    rows = []
    for item in items:
        metrics = dict(item.get("metrics") or {})
        rec10 = metrics.get("recall_at_10")
        rec50 = metrics.get("candidate_recall_at_50", metrics.get("recall_at_50"))
        directions = []
        if item.get("id") != "baseline":
            for value, key, label in (
                (rec50, "candidate_recall_at_50", "Candidate recall"),
                (rec10, "recall_at_10", "Final recall"),
            ):
                note = _direction(value, baseline_metrics.get(key), label=label)
                if note:
                    directions.append(note)
        rows.append(
            {
                "Experiment": item["name"],
                "Date": item.get("date"),
                "Changed variable": item["changed_variable"],
                "Baseline": baseline.get("name") or "Production baseline",
                "Candidate": item["name"],
                "Recall@5": metrics.get("recall_at_5"),
                "Recall@10": rec10,
                "Recall@50": rec50,
                "MRR": metrics.get("mrr_at_10"),
                "nDCG": metrics.get("ndcg_at_10"),
                "Latency": metrics.get("latency_ms"),
                "Result": " · ".join(directions) if directions else ("Production reference" if item.get("id") == "baseline" else "See conclusion"),
                "Conclusion": item.get("conclusion"),
                "Status": item.get("status"),
                "one_variable": True,
                "result_file": item.get("result_file"),
            }
        )
    return rows


def experiment_table_rows(catalog: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    rows = []
    for item in catalog or load_experiment_catalog():
        metrics = item.get("metrics") or {}
        rows.append(
            {
                "Experiment": item["name"],
                "Changed variable": item["changed_variable"],
                "Recall@10": metrics.get("recall_at_10"),
                "MRR": metrics.get("mrr_at_10"),
                "nDCG": metrics.get("ndcg_at_10"),
                "Candidate Recall@50": metrics.get("candidate_recall_at_50"),
                "Latency": metrics.get("latency_ms"),
                "Status": item["status"],
                "Conclusion": item["conclusion"],
                "Result file": item["result_file"],
            }
        )
    return rows
