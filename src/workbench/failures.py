"""Failure categories and retrieval funnel from stored diagnostics. No invented counts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import EVALUATION

ROOT = Path(__file__).resolve().parents[2]

WORKBENCH_FAILURE_CATEGORIES = (
    "DATA_QUALITY",
    "GOLD_LABEL",
    "RETRIEVAL",
    "FUSION",
    "RERANKING",
    "GENERATION",
    "CITATION",
    "EVALUATION",
    "UNKNOWN",
)

_OBJECTIVE_TO_WORKBENCH = {
    "NOT_IN_CORPUS": "GOLD_LABEL",
    "BM25_ONLY_MISS": "RETRIEVAL",
    "DENSE_ONLY_MISS": "RETRIEVAL",
    "BOTH_RETRIEVERS_MISS": "RETRIEVAL",
    "FUSION_CUTOFF": "FUSION",
    "TOP50_RETRIEVED": "RERANKING",
}


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def map_objective_category(objective: str) -> str:
    return _OBJECTIVE_TO_WORKBENCH.get(str(objective), "UNKNOWN")


def failure_analysis(*, root: Path | None = None) -> dict[str, Any]:
    base = root or ROOT
    objective = _load(base / EVALUATION.objective_failure_summary_path)
    gold = _load(base / EVALUATION.gold_quality_path)
    quality = _load(base / EVALUATION.data_quality_path)
    diag = _load(base / EVALUATION.retrieval_diagnostics_path)
    details = _load(base / EVALUATION.objective_failure_details_path)
    summary = dict((diag or {}).get("summary") or {})

    groups: dict[str, dict[str, Any]] = {
        name: {"category": name, "count": 0, "unit": "gold_ids", "examples": [], "source": None}
        for name in WORKBENCH_FAILURE_CATEGORIES
    }
    groups["DATA_QUALITY"].update(
        {
            "count": quality.get("nan_text") or 0,
            "unit": "passages",
            "source": EVALUATION.data_quality_path,
            "note": f'{quality.get("nan_text")} literal \"nan\" passages remain in the index.',
        }
    )
    groups["GOLD_LABEL"].update(
        {
            "count": gold.get("invalid_gold_ids") or 0,
            "unit": "gold_ids",
            "source": EVALUATION.gold_quality_path,
            "note": f'{gold.get("queries_where_all_gold_ids_are_invalid")} queries have only invalid gold IDs.',
        }
    )
    obj_counts = dict(objective.get("objective_failure_counts") or {})
    groups["RETRIEVAL"]["count"] = (
        int(obj_counts.get("BOTH_RETRIEVERS_MISS") or 0)
        + int(obj_counts.get("BM25_ONLY_MISS") or 0)
        + int(obj_counts.get("DENSE_ONLY_MISS") or 0)
    )
    groups["RETRIEVAL"]["unit"] = "gold_ids"
    groups["RETRIEVAL"]["source"] = EVALUATION.objective_failure_summary_path
    groups["FUSION"]["count"] = int(obj_counts.get("FUSION_CUTOFF") or 0)
    groups["FUSION"]["unit"] = "gold_ids"
    groups["FUSION"]["source"] = EVALUATION.objective_failure_summary_path
    groups["RERANKING"]["count"] = int(summary.get("gold_present_in_top50_but_lost_by_reranking") or 0)
    groups["RERANKING"]["unit"] = "gold_ids"
    groups["RERANKING"]["source"] = EVALUATION.retrieval_diagnostics_path
    groups["RERANKING"]["note"] = "Gold IDs in Hybrid@50 but absent from MedCPT Top-10."

    answer_details = _load(base / EVALUATION.answer_details_path)
    answer_rows = list(answer_details.get("rows") or [])
    if answer_rows:
        n_insufficient = sum(1 for row in answer_rows if row.get("insufficient_evidence"))
        n_invalid_cite = sum(
            1
            for row in answer_rows
            if not row.get("insufficient_evidence")
            and not ((row.get("citation_validation") or {}).get("is_valid"))
        )
        groups["GENERATION"].update(
            {
                "count": n_insufficient,
                "unit": "queries",
                "source": EVALUATION.answer_details_path,
                "note": "20-query answer sample, not the 100-query retrieval set.",
            }
        )
        groups["CITATION"].update(
            {
                "count": n_invalid_cite,
                "unit": "queries",
                "source": EVALUATION.answer_details_path,
                "note": "Invalid or missing citations in the 20-query answer sample.",
            }
        )

    examples: dict[str, list[dict[str, Any]]] = {name: [] for name in WORKBENCH_FAILURE_CATEGORIES}
    for item in (details.get("failure_rows") or [])[:80]:
        mapped = map_objective_category(item.get("objective_category") or "")
        if len(examples[mapped]) < 5:
            examples[mapped].append(
                {
                    "query_id": item.get("query_id"),
                    "gold_passage_id": item.get("gold_passage_id"),
                    "question": item.get("question"),
                    "objective_category": item.get("objective_category"),
                    "workbench_category": mapped,
                }
            )
    for name, group in groups.items():
        group["examples"] = examples.get(name) or []

    total_misses = sum(int(v or 0) for v in obj_counts.values())
    return {
        "categories": list(groups.values()),
        "objective_counts": obj_counts,
        "n_hybrid50_failures": objective.get("n_hybrid50_failures"),
        "n_queries": objective.get("n_queries") or 100,
        "total_objective_misses": total_misses,
        "mapping_note": (
            "Workbench categories wrap stored objective rank labels. "
            "They are diagnostics, not proven causal claims."
        ),
        "query_failures": query_failure_rows(root=base),
    }


def query_failure_rows(*, root: Path | None = None) -> list[dict[str, Any]]:
    """Classify 100-query Top-10 misses. Categories are diagnostic, not proven causes."""

    base = root or ROOT
    gold = _load(base / EVALUATION.gold_quality_path)
    diag = _load(base / EVALUATION.retrieval_diagnostics_path)
    details = _load(base / EVALUATION.rerank_comparison_details_path)
    gold_by_q = {int(item["query_id"]): item for item in gold.get("queries") or [] if "query_id" in item}
    diag_by_q = {int(item["query_id"]): item for item in diag.get("queries") or [] if "query_id" in item}
    failed: list[dict[str, Any]] = []
    for row in details.get("rows") or []:
        reranked = row.get("reranked") or {}
        recall = reranked.get("recall_at_10")
        if recall is None or float(recall) > 0:
            continue
        qid = int(row["query_id"])
        gold_meta = gold_by_q.get(qid) or {}
        diagnostic = diag_by_q.get(qid) or {}
        gold_passages = list(diagnostic.get("gold") or [])
        gold_ids = set(row.get("relevant_passage_ids") or [])
        pool_ids = set((row.get("hybrid_50") or {}).get("retrieved_passage_ids") or [])
        in_pool = bool(gold_ids & pool_ids) or any(item.get("in_top50") for item in gold_passages)
        bm25_any = any(item.get("bm25_found") for item in gold_passages)
        dense_any = any(item.get("dense_found") for item in gold_passages)
        if gold_meta.get("all_gold_invalid"):
            category = "GOLD_LABEL"
        elif in_pool:
            category = "RERANKING"
        elif not bm25_any and not dense_any:
            category = "RETRIEVAL"
        elif bm25_any or dense_any:
            category = "FUSION"
        else:
            category = "UNKNOWN"
        failed.append(
            {
                "query_id": qid,
                "question": row.get("question"),
                "category": category,
                "recall_at_10": recall,
                "in_candidate_pool": in_pool,
                "n_gold": gold_meta.get("n_gold") or len(gold_ids),
                "n_invalid_gold": gold_meta.get("n_invalid_gold"),
                "gold_ids": list(gold_ids),
            }
        )
    return failed


def retrieval_funnel(*, root: Path | None = None) -> dict[str, Any]:
    base = root or ROOT
    diag = _load(base / EVALUATION.retrieval_diagnostics_path)
    gold = _load(base / EVALUATION.gold_quality_path)
    summary = dict((diag or {}).get("summary") or {})
    hits = dict(summary.get("query_hits") or {})
    n_gold = gold.get("total_gold_ids") or summary.get("n_gold_ids")
    n_queries = summary.get("n_queries") or 100
    counts = dict((summary.get("signal_recovery") or {}).get("counts") or {})
    return {
        "id_level": [
            {"stage": "Gold IDs", "count": n_gold, "note": "All labelled relevant IDs, including unusable text"},
            {
                "stage": "BM25 (inspect@k)",
                "count": int(counts.get("BM25 only") or 0) + int(counts.get("Both BM25 and Dense") or 0),
                "note": "ID-level: gold IDs found by BM25 at inspect k",
            },
            {
                "stage": "Dense (inspect@k)",
                "count": int(counts.get("Dense only") or 0) + int(counts.get("Both BM25 and Dense") or 0),
                "note": "ID-level: gold IDs found by Dense at inspect k",
            },
            {
                "stage": "RRF candidate pool",
                "rate": summary.get("candidate_recall_at_50") or summary.get("rrf_recall_at_50"),
                "note": "ID-level Candidate Recall@50",
            },
            {
                "stage": "MedCPT reranker Top-10",
                "rate": summary.get("rerank_recall_at_10"),
                "note": "ID-level Recall@10 after MedCPT",
            },
            {
                "stage": "Final Top-K",
                "rate": (diag.get("raw_original_benchmark") or {}).get("recall_at_10")
                or summary.get("rerank_recall_at_10"),
                "note": "Same as MedCPT Recall@10 for the production path",
            },
        ],
        "query_level": [
            {"stage": "Queries", "count": n_queries},
            {
                "stage": "BM25 finds ≥1 gold",
                "count": hits.get("queries_where_BM25_finds_at_least_one_gold"),
            },
            {
                "stage": "Dense finds ≥1 gold",
                "count": hits.get("queries_where_Dense_finds_at_least_one_gold"),
            },
            {
                "stage": "Pool hit (Hybrid@50)",
                "rate": summary.get("candidate_hit_at_50")
                or (diag.get("raw_original_benchmark") or {}).get("candidate_hit_at_50"),
                "note": "Query-level: at least one gold in the Top-50 pool",
            },
            {
                "stage": "Rerank recovery@10",
                "rate": summary.get("rerank_recovery_at_10"),
                "note": "When a gold is in the pool, whether MedCPT keeps a gold in Top-10",
            },
        ],
        "distinction": (
            "ID-level rates average over gold passage IDs. "
            "Query-level rates average over the 100 questions."
        ),
        "raw_vs_valid_gold": {
            "raw_recall_at_10": (diag.get("raw_original_benchmark") or {}).get("recall_at_10"),
            "valid_gold_recall_at_10": (diag.get("valid_gold_diagnostic_view") or {}).get("recall_at_10"),
            "note": "Valid-gold does not replace the raw benchmark.",
        },
    }
