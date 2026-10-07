"""Categorize where retrieval failed for a query."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from src.evaluation.metrics import candidate_hit_at_k, passage_id_key
from src.query.classify import classify_query

FAILURE_CATEGORIES = (
    "relevant_never_retrieved",
    "ranked_too_low",
    "terminology_mismatch",
    "semantic_mismatch",
    "query_too_complex",
    "decomposition_failure",
    "filtering_removed_relevant",
    "reranker_failure",
    "insufficient_evidence",
)


def _id_set(values: Iterable[Any] | None) -> set[str]:
    return {passage_id_key(item) for item in (values or ()) if item is not None}


def categorize_retrieval_failure(
    *,
    question: str,
    relevant_ids: Sequence[Any],
    pool_ids: Sequence[Any],
    filtered_ids: Sequence[Any] | None = None,
    final_ids: Sequence[Any],
    query_type: str | None = None,
    decomposition_applied: bool = False,
    reranked: bool = False,
    pool_k: int = 50,
    final_k: int = 10,
) -> dict[str, Any]:
    """Return a primary failure category plus supporting labels.

    A successful top-k hit is reported as category ``success``.
    """

    relevant = _id_set(relevant_ids)
    pool = _id_set(list(pool_ids)[:pool_k])
    kept = _id_set(filtered_ids if filtered_ids is not None else pool_ids[:pool_k])
    final = _id_set(list(final_ids)[:final_k])
    query_type = query_type or classify_query(question).query_type
    labels: list[str] = []

    if not relevant:
        return {
            "category": "insufficient_evidence",
            "labels": ["insufficient_evidence"],
            "query_type": query_type,
            "relevant_in_pool": False,
            "relevant_after_filter": False,
            "relevant_in_final": False,
        }

    relevant_in_final = bool(relevant & final)
    relevant_in_pool = bool(relevant & pool)
    relevant_after_filter = bool(relevant & kept)

    if relevant_in_final:
        return {
            "category": "success",
            "labels": [],
            "query_type": query_type,
            "relevant_in_pool": relevant_in_pool,
            "relevant_after_filter": relevant_after_filter,
            "relevant_in_final": True,
        }

    if relevant_in_pool and not relevant_after_filter:
        labels.append("filtering_removed_relevant")
        primary = "filtering_removed_relevant"
    elif relevant_after_filter or relevant_in_pool:
        if reranked:
            labels.append("reranker_failure")
            primary = "reranker_failure"
        else:
            labels.append("ranked_too_low")
            primary = "ranked_too_low"
    else:
        labels.append("relevant_never_retrieved")
        if query_type == "terminology_heavy":
            labels.append("terminology_mismatch")
            primary = "terminology_mismatch"
        elif query_type == "semantic":
            labels.append("semantic_mismatch")
            primary = "semantic_mismatch"
        elif query_type == "multi_part":
            labels.append("query_too_complex")
            primary = "decomposition_failure" if decomposition_applied else "query_too_complex"
            if decomposition_applied:
                labels.append("decomposition_failure")
        else:
            primary = "relevant_never_retrieved"

    return {
        "category": primary,
        "labels": labels,
        "query_type": query_type,
        "relevant_in_pool": relevant_in_pool,
        "relevant_after_filter": relevant_after_filter,
        "relevant_in_final": relevant_in_final,
        "candidate_hit_at_pool": candidate_hit_at_k(pool_ids, relevant_ids, pool_k),
    }


def summarize_failures(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = {name: 0 for name in FAILURE_CATEGORIES}
    counts["success"] = 0
    for row in rows:
        category = str(row.get("category") or "relevant_never_retrieved")
        counts[category] = counts.get(category, 0) + 1
    return {
        "n_queries": len(rows),
        "n_failures": sum(1 for row in rows if row.get("category") not in {None, "success"}),
        "counts": counts,
    }
