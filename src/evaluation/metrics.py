"""Retrieval metrics for the fixed-query BioASQ evaluation.

Relevance is binary: a retrieved passage is relevant if its ID appears in
the question's golden relevant_passage_ids.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def passage_id_key(value: Any) -> str:
    """Stable comparison key so int and string forms of an ID match."""

    return str(value).strip()


def _relevant_set(relevant_ids: Iterable[Any]) -> set[str]:
    return {passage_id_key(item) for item in relevant_ids}


def recall_at_k(
    retrieved_ids: Sequence[Any],
    relevant_ids: Iterable[Any],
    k: int,
) -> float:
    """Fraction of the golden relevant set found in the top-k retrieved IDs."""

    if k < 1:
        return 0.0
    relevant = _relevant_set(relevant_ids)
    if not relevant:
        return 0.0
    retrieved = {passage_id_key(item) for item in retrieved_ids[:k]}
    return len(retrieved & relevant) / len(relevant)


def mrr_at_k(
    retrieved_ids: Sequence[Any],
    relevant_ids: Iterable[Any],
    k: int = 10,
) -> float:
    """Reciprocal rank of the first relevant passage within the top k."""

    if k < 1:
        return 0.0
    relevant = _relevant_set(relevant_ids)
    if not relevant:
        return 0.0
    for rank, passage_id in enumerate(retrieved_ids[:k], start=1):
        if passage_id_key(passage_id) in relevant:
            return 1.0 / rank
    return 0.0


def _dcg(relevances: Sequence[float], k: int) -> float:
    total = 0.0
    for rank, relevance in enumerate(relevances[:k], start=1):
        total += float(relevance) / math.log2(rank + 1)
    return total


def ndcg_at_k(
    retrieved_ids: Sequence[Any],
    relevant_ids: Iterable[Any],
    k: int = 10,
) -> float:
    """nDCG@k with binary relevance from golden passage IDs."""

    if k < 1:
        return 0.0
    relevant = _relevant_set(relevant_ids)
    if not relevant:
        return 0.0
    gains = [1.0 if passage_id_key(item) in relevant else 0.0 for item in retrieved_ids[:k]]
    dcg = _dcg(gains, k)
    ideal = [1.0] * min(len(relevant), k)
    idcg = _dcg(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_retrieval_metrics(
    retrieved_ids: Sequence[Any],
    relevant_ids: Iterable[Any],
) -> dict[str, float]:
    """Return Recall@5, Recall@10, MRR@10, and nDCG@10."""

    return {
        "recall_at_5": recall_at_k(retrieved_ids, relevant_ids, 5),
        "recall_at_10": recall_at_k(retrieved_ids, relevant_ids, 10),
        "mrr_at_10": mrr_at_k(retrieved_ids, relevant_ids, 10),
        "ndcg_at_10": ndcg_at_k(retrieved_ids, relevant_ids, 10),
    }


def candidate_recall_at_k(
    retrieved_ids: Sequence[Any],
    relevant_ids: Iterable[Any],
    k: int = 50,
) -> float:
    """Fraction of golden IDs found in the first-stage candidate pool of size k.

    Uses the existing Recall@k definition; k defaults to the reranker pool size.
    """

    return recall_at_k(retrieved_ids, relevant_ids, k)


def candidate_hit_at_k(
    retrieved_ids: Sequence[Any],
    relevant_ids: Iterable[Any],
    k: int = 50,
) -> bool:
    """True when at least one golden relevant ID appears in the top k."""

    relevant = _relevant_set(relevant_ids)
    if not relevant or k < 1:
        return False
    retrieved = {passage_id_key(item) for item in retrieved_ids[:k]}
    return bool(retrieved & relevant)


def rerank_recovery_at_k(
    pool_ids: Sequence[Any],
    reranked_ids: Sequence[Any],
    relevant_ids: Iterable[Any],
    *,
    pool_k: int = 50,
    final_k: int = 10,
) -> float | None:
    """1.0 if a pooled relevant ID is in the reranked top-k, else 0.0.

    Returns None when the candidate pool contains no relevant passage so the
    query is a retrieval failure, not a ranking failure.
    """

    if not candidate_hit_at_k(pool_ids, relevant_ids, pool_k):
        return None
    relevant = _relevant_set(relevant_ids)
    reranked = {passage_id_key(item) for item in reranked_ids[:final_k]}
    return 1.0 if reranked & relevant else 0.0


def first_relevant_rank(
    retrieved_ids: Sequence[Any],
    relevant_ids: Iterable[Any],
) -> int | None:
    """1-based rank of the best (earliest) relevant passage, or None."""

    relevant = _relevant_set(relevant_ids)
    if not relevant:
        return None
    for rank, passage_id in enumerate(retrieved_ids, start=1):
        if passage_id_key(passage_id) in relevant:
            return rank
    return None


def rank_of_id(retrieved_ids: Sequence[Any], passage_id: Any) -> int | None:
    """1-based rank of a passage ID, or None if it is missing."""

    key = passage_id_key(passage_id)
    for rank, item in enumerate(retrieved_ids, start=1):
        if passage_id_key(item) == key:
            return rank
    return None


def rank_delta(original_rank: int | None, final_rank: int | None) -> int | None:
    """Positive when the passage moved to a better (lower) rank."""

    if original_rank is None or final_rank is None:
        return None
    return int(original_rank) - int(final_rank)


def relative_change(new: float, old: float) -> float | None:
    """(new - old) / old. None when the baseline is zero so relative change is undefined."""

    if old == 0:
        return None
    return (new - old) / old


def metric_gain(reranked: Mapping[str, float], baseline: Mapping[str, float]) -> dict[str, dict[str, float | None]]:
    """Absolute and relative change for the four ranking metrics. C minus A."""

    keys = ("recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10")
    gains: dict[str, dict[str, float | None]] = {}
    for key in keys:
        new = float(reranked[key])
        old = float(baseline[key])
        gains[key] = {
            "baseline": old,
            "reranked": new,
            "absolute": new - old,
            "relative": relative_change(new, old),
        }
    return gains


def candidate_ceiling_gap(candidate_recall_at_50: float, reranked_recall_at_10: float) -> float:
    """How much pool recall is not realized in the reranked top-10."""

    return float(candidate_recall_at_50) - float(reranked_recall_at_10)
