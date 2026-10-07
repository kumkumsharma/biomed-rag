"""Feature status labels for the UI: BASELINE / EXPERIMENTAL / ENABLED / DISABLED / NOT EVALUATED."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import EVALUATION, GENERATION, RERANKER, UPGRADE

ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _upgrade_row(name_contains: str) -> dict[str, Any] | None:
    payload = _load(ROOT / EVALUATION.upgrade_summary_path)
    if not payload:
        return None
    for item in payload.get("configurations") or []:
        if name_contains.lower() in str(item.get("name", "")).lower():
            return item
    return None


def _embedding_benchmark(payload: dict[str, Any] | None) -> str:
    if not payload:
        return "NOT EVALUATED until the embedding comparison JSON exists"
    rows = {item.get("name"): item for item in payload.get("configurations") or []}
    mini = next((item for name, item in rows.items() if "MiniLM" in str(name)), None)
    bio = next((item for name, item in rows.items() if "Biomedical" in str(name)), None)
    if mini and bio:
        return (
            f"Measured vs MiniLM on same 100 queries: "
            f"Recall@10 {bio['recall_at_10']:.4f} vs {mini['recall_at_10']:.4f}; "
            f"MRR@10 {bio['mrr_at_10']:.4f} vs {mini['mrr_at_10']:.4f}. "
            f"Default remains MiniLM (index cost). See embedding_model_comparison_summary.json"
        )
    return "See embedding_model_comparison_summary.json"


def experimental_feature_status() -> list[dict[str, Any]]:
    expansion = _upgrade_row("Query expansion")
    decomposition = _upgrade_row("Query decomposition")
    adaptive = _upgrade_row("Adaptive")
    filtering = _upgrade_row("Candidate filtering")
    full = _upgrade_row("Full upgraded")
    medcpt = _upgrade_row("MedCPT")
    embedding = _load(ROOT / EVALUATION.embedding_comparison_summary_path)
    sparse = _load(ROOT / EVALUATION.sparse_study_summary_path)
    if sparse and not sparse.get("skipped"):
        rec = sparse.get("dual_miss_recovery") or {}
        sparse_bench = (
            f"EXPERIMENTAL. Sparse Recovery@100 {rec.get('hit_at_100')}/{rec.get('denominator')} "
            f"dual-miss IDs; sparse-only among dual misses (not PubMedBERT)="
            f"{sparse.get('sparse_only_among_dual_misses_not_pubmedbert', sparse.get('sparse_only_recovery_among_dual_misses'))}. "
            "Not production."
        )
    elif sparse and sparse.get("skipped"):
        sparse_bench = f"EXPERIMENTAL. Skipped: {sparse.get('reason')}. Not production."
    else:
        sparse_bench = "EXPERIMENTAL. NOT EVALUATED until sparse_study_summary.json exists. Not production."
    return [
        {
            "name": "Hybrid BM25 + Dense + RRF",
            "lane": "BASELINE",
            "runtime": "ENABLED",
            "default": "ON",
            "benchmark": "Recall@10=0.4286, MRR@10=0.8105, nDCG@10=0.5655 (~66 ms)",
        },
        {
            "name": "MedCPT rerank",
            "lane": "EXPERIMENTAL",
            "runtime": "ENABLED" if RERANKER.enabled else "DISABLED",
            "default": "OFF",
            "benchmark": (
                f"Improved ranking vs Hybrid@10 (Recall@10={medcpt['recall_at_10']:.4f})"
                if medcpt
                else "Improved ranking vs Hybrid@10 (Recall@10=0.5082) at ~3208 ms total"
            ),
        },
        {
            "name": "Deterministic query expansion",
            "lane": "EXPERIMENTAL",
            "runtime": "ENABLED" if UPGRADE.expansion_enabled else "DISABLED",
            "default": "OFF",
            "benchmark": (
                f"Slight degradation (Recall@10={expansion['recall_at_10']:.4f})"
                if expansion
                else "Slight degradation vs Hybrid@10 in upgrade experiment E"
            ),
        },
        {
            "name": "LLM query expansion",
            "lane": "EXPERIMENTAL",
            "runtime": "ENABLED" if GENERATION.query_expansion_enabled else "DISABLED",
            "default": "ON in Explorer checkbox; not used in the 100-query retrieval benchmark unless --hybrid-qe",
            "benchmark": "Separate Hybrid+QE JSON if present; not the deterministic expander.",
        },
        {
            "name": "Query decomposition",
            "lane": "EXPERIMENTAL",
            "runtime": "ENABLED" if UPGRADE.decomposition_enabled else "DISABLED",
            "default": "OFF; only multi-part queries when enabled",
            "benchmark": (
                f"Small improvement (Recall@10={decomposition['recall_at_10']:.4f})"
                if decomposition
                else "Small improvement in upgrade experiment F"
            ),
        },
        {
            "name": "Adaptive routing",
            "lane": "EXPERIMENTAL",
            "runtime": "ENABLED" if UPGRADE.adaptive_routing_enabled else "DISABLED",
            "default": "OFF",
            "benchmark": (
                f"Slight degradation (Recall@10={adaptive['recall_at_10']:.4f})"
                if adaptive
                else "Slight degradation in upgrade experiment H"
            ),
        },
        {
            "name": "Candidate filtering",
            "lane": "EXPERIMENTAL",
            "runtime": "ENABLED" if UPGRADE.filtering_enabled else "DISABLED",
            "default": "OFF",
            "benchmark": (
                "Almost no candidates removed; zero relevant passages dropped"
                if filtering
                else "Almost no effect in upgrade experiment I"
            ),
        },
        {
            "name": "Biomedical dense embeddings",
            "lane": "EXPERIMENTAL",
            "runtime": "ENABLED" if UPGRADE.biomedical_embeddings_enabled else "DISABLED",
            "default": "OFF; MiniLM remains production default",
            "benchmark": _embedding_benchmark(embedding),
        },
        {
            "name": "Learned sparse first-stage (SPLADE)",
            "lane": "EXPERIMENTAL",
            "runtime": "ENABLED" if UPGRADE.sparse_enabled else "DISABLED",
            "default": "OFF; production remains BM25 + MiniLM + RRF",
            "benchmark": sparse_bench,
        },
        {
            "name": "Full upgraded pipeline",
            "lane": "EXPERIMENTAL",
            "runtime": "ENABLED" if UPGRADE.pipeline_enabled else "DISABLED",
            "default": "OFF",
            "benchmark": (
                f"Recall@10={full['recall_at_10']:.4f} vs MedCPT-only 0.5082"
                if full
                else "Slightly worse than MedCPT-only because candidate recall dropped"
            ),
        },
    ]
