"""Hybrid retrieval via Reciprocal Rank Fusion of BM25 and dense results.

RRF is used instead of summing raw scores because BM25 scores and cosine
similarities are on different scales. Each ranking contributes
1 / (rrf_k + rank), with ranks starting at 1. The default rrf_k is 60.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from src.config import RETRIEVAL
from src.data.preprocessing import Passage


@dataclass(frozen=True)
class RetrievalResult:
    """One hybrid hit with the original passage ID, RRF score, and text."""

    passage_id: Any
    score: float
    text: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RankedRetriever(Protocol):
    """Minimal search interface expected from BM25 and dense retrievers."""

    def search(self, query: str, top_k: int | None = None) -> Sequence[Any]: ...


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Any]],
    *,
    rrf_k: int = 60,
) -> list[tuple[Any, float]]:
    """Combine rankings with Reciprocal Rank Fusion.

    RRF_score(d) = sum over rankings of 1 / (rrf_k + rank(d))
    Ranks start at 1. A passage missing from a ranking contributes nothing
    from that ranking. Duplicate IDs inside one ranking keep the first rank.
    """

    if rrf_k < 0:
        raise ValueError("rrf_k must be >= 0")

    scores: dict[Any, float] = {}
    for ranking in rankings:
        seen: set[Any] = set()
        for rank, item in enumerate(ranking, start=1):
            passage_id = getattr(item, "passage_id")
            if passage_id in seen:
                continue
            seen.add(passage_id)
            scores[passage_id] = scores.get(passage_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class HybridRetriever:
    """Fuse existing BM25 and dense retrievers with RRF."""

    def __init__(
        self,
        passages: Sequence[Passage],
        *,
        bm25_retriever: RankedRetriever,
        dense_retriever: RankedRetriever,
        top_k: int | None = None,
        rrf_k: int | None = None,
        candidate_k: int | None = None,
    ) -> None:
        if passages is None:
            raise ValueError("Cannot build a hybrid retriever without a passage collection")
        passages = list(passages)
        if not passages:
            raise ValueError("Cannot build a hybrid retriever from an empty passage collection")
        if bm25_retriever is None or dense_retriever is None:
            raise ValueError("HybridRetriever requires existing BM25 and dense retrievers")

        self._passages = passages
        self._texts = {passage.passage_id: passage.text for passage in passages}
        self._bm25 = bm25_retriever
        self._dense = dense_retriever
        self.top_k = RETRIEVAL.hybrid_top_k if top_k is None else top_k
        self.rrf_k = RETRIEVAL.rrf_k if rrf_k is None else rrf_k
        self.candidate_k = candidate_k
        self.expansion_mode = "equal"
        self.last_trace: dict[str, Any] = {}

    def search(
        self,
        query: str,
        top_k: int | None = None,
        expansions: Sequence[str] | None = None,
        expansion_mode: str | None = None,
    ) -> list[RetrievalResult]:
        """Return top-k passages fused from BM25 and dense rankings.

        When expansions are provided, BM25 and dense are queried with the
        original query first, then each expansion. All rankings are fused
        with the existing RRF implementation. The original query is not
        replaced.
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        limit = self.top_k if top_k is None else top_k
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            self.last_trace = _retrieval_trace(self.rrf_k)
            return []
        if not query.strip():
            self.last_trace = _retrieval_trace(self.rrf_k)
            return []

        queries = _search_queries(query, expansions)
        mode = (expansion_mode or self.expansion_mode or "equal").strip().lower()
        candidate_k = self.candidate_k if self.candidate_k is not None else limit
        rankings = []
        bm25_ms = 0.0
        dense_ms = 0.0
        for item in queries:
            started = time.perf_counter()
            rankings.append(list(self._bm25.search(item, top_k=candidate_k)))
            bm25_ms += (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            rankings.append(list(self._dense.search(item, top_k=candidate_k)))
            dense_ms += (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        if mode == "priority" and len(queries) > 1:
            original_fused = reciprocal_rank_fusion(rankings[:2], rrf_k=self.rrf_k)
            extra_fused = reciprocal_rank_fusion(rankings[2:], rrf_k=self.rrf_k) if len(rankings) > 2 else []
            fused = _priority_merge(original_fused, extra_fused)
        else:
            fused = reciprocal_rank_fusion(rankings, rrf_k=self.rrf_k)
        bm25_ranks = _id_ranks(rankings[0]) if rankings else {}
        dense_ranks = _id_ranks(rankings[1]) if len(rankings) > 1 else {}
        rrf_ranks = {passage_id: rank for rank, (passage_id, _score) in enumerate(fused, start=1)}

        results: list[RetrievalResult] = []
        for passage_id, score in fused:
            if passage_id not in self._texts:
                continue
            results.append(
                RetrievalResult(
                    passage_id=passage_id,
                    score=float(score),
                    text=self._texts[passage_id],
                )
            )
            if len(results) == limit:
                break
        rrf_ms = (time.perf_counter() - started) * 1000.0
        self.last_trace = _retrieval_trace(
            self.rrf_k,
            n_queries=len(queries),
            n_rankings=len(rankings),
            candidate_k=candidate_k,
            bm25_ms=bm25_ms,
            dense_ms=dense_ms,
            rrf_ms=rrf_ms,
            bm25_ranks=bm25_ranks,
            dense_ranks=dense_ranks,
            rrf_ranks=rrf_ranks,
            expansion_mode=mode if expansions else "equal",
        )
        return results


def _id_ranks(ranking: Sequence[Any]) -> dict[Any, int]:
    """First-occurrence 1-based ranks from one BM25 or dense ranking."""

    ranks: dict[Any, int] = {}
    for rank, item in enumerate(ranking, start=1):
        passage_id = getattr(item, "passage_id", None)
        if passage_id is None or passage_id in ranks:
            continue
        ranks[passage_id] = rank
    return ranks


def _retrieval_trace(
    rrf_k: int,
    *,
    n_queries: int = 0,
    n_rankings: int = 0,
    candidate_k: int | None = None,
    bm25_ms: float | None = None,
    dense_ms: float | None = None,
    rrf_ms: float | None = None,
    bm25_ranks: dict[Any, int] | None = None,
    dense_ranks: dict[Any, int] | None = None,
    rrf_ranks: dict[Any, int] | None = None,
    expansion_mode: str | None = None,
) -> dict[str, Any]:
    return {
        "n_queries": n_queries,
        "n_rankings": n_rankings,
        "candidate_k": candidate_k,
        "rrf_k": rrf_k,
        "bm25_ms": bm25_ms,
        "dense_ms": dense_ms,
        "rrf_ms": rrf_ms,
        "bm25_ranks": dict(bm25_ranks or {}),
        "dense_ranks": dict(dense_ranks or {}),
        "rrf_ranks": dict(rrf_ranks or {}),
        "expansion_mode": expansion_mode or "equal",
    }


def _search_queries(query: str, expansions: Sequence[str] | None) -> list[str]:
    queries = [query]
    seen = {query.strip().lower()}
    if not expansions:
        return queries
    for item in expansions:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if not stripped:
            continue
        key = stripped.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(stripped)
    return queries


def _priority_merge(
    original: Sequence[tuple[Any, float]],
    extra: Sequence[tuple[Any, float]],
) -> list[tuple[Any, float]]:
    """Keep original-query ranking first; append expansion-only IDs."""

    merged: list[tuple[Any, float]] = []
    seen: set[Any] = set()
    for passage_id, score in original:
        if passage_id in seen:
            continue
        seen.add(passage_id)
        merged.append((passage_id, float(score)))
    for passage_id, score in extra:
        if passage_id in seen:
            continue
        seen.add(passage_id)
        merged.append((passage_id, float(score)))
    return merged


def rrf_fuse_retrievers(
    retrievers: Sequence[RankedRetriever],
    query: str,
    *,
    texts: Mapping[Any, str],
    top_k: int,
    rrf_k: int = 60,
    candidate_k: int | None = None,
) -> list[RetrievalResult]:
    """RRF over an arbitrary list of ranked retrievers. Does not change HybridRetriever defaults."""

    pool = candidate_k if candidate_k is not None else top_k
    rankings = [list(retriever.search(query, top_k=pool)) for retriever in retrievers]
    fused = reciprocal_rank_fusion(rankings, rrf_k=rrf_k)
    results: list[RetrievalResult] = []
    for passage_id, score in fused:
        if passage_id not in texts:
            continue
        results.append(RetrievalResult(passage_id=passage_id, score=float(score), text=texts[passage_id]))
        if len(results) == top_k:
            break
    return results
