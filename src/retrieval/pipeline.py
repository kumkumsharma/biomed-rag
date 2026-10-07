"""Multi-stage biomedical retrieval wrapping existing BM25, dense, RRF, and MedCPT.

Default configuration forwards to Hybrid-equivalent BM25 + dense + RRF so the
baseline remains available. Optional stages are feature-flagged.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.config import RERANKER, UPGRADE, UpgradeConfig
from src.query.understand import understand_query
from src.retrieval.filter import filter_candidates
from src.retrieval.hybrid import reciprocal_rank_fusion
from src.retrieval.reranker import CrossEncoderReranker, PairScorer, get_shared_scorer, resolve_device
from src.retrieval.router import RetrievalPlan, route_query
from src.retrieval.traces import build_query_trace


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (bytes, str, bool)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


@dataclass(frozen=True)
class StageHit:
    """Drop-in retrieval hit plus pipeline provenance."""

    passage_id: Any
    score: float
    text: str
    pmid: Any = None
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rrf_score: float | None = None
    rrf_rank: int | None = None
    reranker_score: float | None = None
    final_rank: int | None = None
    filter_decision: str | None = None
    source_queries: tuple[str, ...] = ()
    retrieval_stage: str = "hybrid"

    def as_dict(self) -> dict[str, Any]:
        return {
            "passage_id": _jsonable(self.passage_id),
            "score": self.score,
            "text": self.text,
            "pmid": _jsonable(self.pmid if self.pmid is not None else self.passage_id),
            "bm25_rank": self.bm25_rank,
            "dense_rank": self.dense_rank,
            "rrf_score": self.rrf_score,
            "rrf_rank": self.rrf_rank,
            "reranker_score": self.reranker_score,
            "final_rank": self.final_rank,
            "filter_decision": self.filter_decision,
            "source_queries": list(self.source_queries),
            "retrieval_stage": self.retrieval_stage,
        }


class StagedRetriever:
    """Configurable BM25 → dense → RRF → filter → MedCPT pipeline."""

    def __init__(
        self,
        *,
        bm25_retriever: Any,
        dense_retriever: Any,
        texts: dict[Any, str] | None = None,
        passages: Sequence[Any] | None = None,
        config: UpgradeConfig | None = None,
        reranker: CrossEncoderReranker | None = None,
        scorer: PairScorer | None = None,
        rerank_enabled: bool | None = None,
        rrf_k: int = 60,
    ) -> None:
        if bm25_retriever is None or dense_retriever is None:
            raise ValueError("StagedRetriever requires BM25 and dense retrievers")
        self._bm25 = bm25_retriever
        self._dense = dense_retriever
        merged: dict[Any, str] = {}
        if passages is not None:
            merged.update({getattr(item, "passage_id"): getattr(item, "text", "") for item in passages})
        if texts:
            merged.update(texts)
        self._texts = merged
        self.config = config or UPGRADE
        self.rrf_k = rrf_k
        self.rerank_enabled = RERANKER.enabled if rerank_enabled is None else rerank_enabled
        if reranker is not None:
            self._reranker = reranker
        elif self.rerank_enabled:
            self._reranker = CrossEncoderReranker(scorer=scorer or get_shared_scorer())
        else:
            self._reranker = None
        self.last_trace: dict[str, Any] = {}

    def preload(self) -> None:
        if self._reranker is not None:
            self._reranker.preload()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        expansions: Sequence[str] | None = None,
    ) -> list[StageHit]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        limit = self.config.final_top_k if top_k is None else top_k
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or not query.strip():
            self.last_trace = empty_trace(query)
            return []

        started_all = time.perf_counter()
        understanding = understand_query(query, self.config)
        plan = route_query(query, config=self.config, understanding=understanding)
        search_queries = _search_queries(understanding, plan, expansions)
        pool = plan.candidate_pool_size if self.config.pipeline_enabled else limit
        if self.rerank_enabled:
            pool = max(pool, RERANKER.candidate_pool_size)

        bm25_rankings: list[list[Any]] = []
        dense_rankings: list[list[Any]] = []
        per_query: list[dict[str, Any]] = []
        bm25_ms = 0.0
        dense_ms = 0.0
        source_queries: dict[Any, list[str]] = {}
        for item in search_queries:
            bm25_hits: list[Any] = []
            dense_hits: list[Any] = []
            if plan.use_bm25:
                started = time.perf_counter()
                bm25_hits = list(self._bm25.search(item, top_k=plan.bm25_candidate_k if self.config.pipeline_enabled else pool))
                bm25_ms += (time.perf_counter() - started) * 1000.0
                bm25_rankings.append(bm25_hits)
                _note_sources(source_queries, bm25_hits, item)
            if plan.use_dense:
                started = time.perf_counter()
                dense_hits = list(self._dense.search(item, top_k=plan.dense_candidate_k if self.config.pipeline_enabled else pool))
                dense_ms += (time.perf_counter() - started) * 1000.0
                dense_rankings.append(dense_hits)
                _note_sources(source_queries, dense_hits, item)
            per_query.append(
                {
                    "query": item,
                    "bm25_ids": [_jsonable(getattr(hit, "passage_id", None)) for hit in bm25_hits],
                    "dense_ids": [_jsonable(getattr(hit, "passage_id", None)) for hit in dense_hits],
                }
            )

        rankings: list[list[Any]] = []
        rankings.extend(bm25_rankings)
        rankings.extend(dense_rankings)
        started = time.perf_counter()
        fused = reciprocal_rank_fusion(rankings, rrf_k=self.rrf_k) if rankings else []
        rrf_ms = (time.perf_counter() - started) * 1000.0
        fused_hits = _hits_from_fusion(fused, self._texts, source_queries, bm25_rankings, dense_rankings, pool)

        filtered = filter_candidates(
            query,
            fused_hits,
            config=self.config,
            enabled=self.config.filtering_enabled if self.config.pipeline_enabled else False,
        )
        candidates = list(filtered.kept)
        rerank_ms = None
        if self.rerank_enabled and self._reranker is not None:
            started = time.perf_counter()
            final = list(
                self._reranker.rerank(
                    understanding.original,
                    candidates,
                    top_k=limit,
                    bm25_ranks=_first_ranks(bm25_rankings),
                    dense_ranks=_first_ranks(dense_rankings),
                )
            )
            rerank_ms = (time.perf_counter() - started) * 1000.0
        else:
            final = candidates[:limit]
        annotated = _annotate_final(final, fused_hits, filtered, source_queries)
        total_ms = (time.perf_counter() - started_all) * 1000.0
        scorer = getattr(self._reranker, "_scorer", None) if self._reranker is not None else None
        self.last_trace = {
            "query_id": None,
            "original_query": understanding.original,
            "normalized_query": understanding.normalized,
            "query_classification": understanding.query_class.as_dict(),
            "terminology": understanding.terminology.as_dict(),
            "expansion_variants": list(understanding.expansion.expansions),
            "subqueries": list(understanding.decomposition.subqueries),
            "search_queries": list(search_queries),
            "retrieval_strategy": plan.as_dict(),
            "per_variant_retrieval": per_query,
            "bm25_results": _ids_only(bm25_rankings[0] if bm25_rankings else []),
            "dense_results": _ids_only(dense_rankings[0] if dense_rankings else []),
            "fusion_results": [hit.as_dict() for hit in fused_hits[:pool]],
            "filtering_results": filtered.as_dict(),
            "reranker_results": [item.as_dict() for item in annotated],
            "final_evidence": [item.as_dict() for item in annotated],
            "stage_timings": {
                "bm25_ms": bm25_ms,
                "dense_ms": dense_ms,
                "rrf_ms": rrf_ms,
                "filter_ms": filtered.latency_ms,
                "rerank_ms": rerank_ms,
                "total_ms": total_ms,
            },
            "candidate_pool_size": pool,
            "candidate_count_before_filter": filtered.before_count,
            "candidate_count_after_filter": filtered.after_count,
            "pipeline_enabled": self.config.pipeline_enabled,
            "reranker_enabled": self.rerank_enabled,
            "reranker_model_name": getattr(scorer, "model_name", RERANKER.model_name) if self.rerank_enabled else None,
            "reranker_device": resolve_device(getattr(scorer, "device", RERANKER.device)) if self.rerank_enabled else None,
            "reranker_candidate_pool_size": pool if self.rerank_enabled else None,
            "rerank_candidate_count": len(candidates) if self.rerank_enabled else None,
            "bm25_ranks": _first_ranks(bm25_rankings),
            "dense_ranks": _first_ranks(dense_rankings),
            "bm25_ms": bm25_ms,
            "dense_ms": dense_ms,
            "rrf_ms": rrf_ms,
            "rerank_ms": rerank_ms,
            "filter_ms": filtered.latency_ms,
            "n_queries": len(search_queries),
            "n_rankings": len(rankings),
            "candidate_k": pool,
            "rrf_k": self.rrf_k,
            "understanding": understanding.as_dict(),
        }
        self.last_trace.update(build_query_trace(self.last_trace))
        return annotated


def empty_trace(query: str) -> dict[str, Any]:
    return build_query_trace(
        {
            "original_query": query,
            "normalized_query": query,
            "search_queries": [],
            "stage_timings": {},
            "final_evidence": [],
        }
    )


def _search_queries(understanding, plan: RetrievalPlan, extra: Sequence[str] | None) -> list[str]:
    queries = [understanding.original]
    seen = {understanding.original.strip().lower()}

    def add(item: str) -> None:
        key = item.strip().lower()
        if item.strip() and key not in seen:
            seen.add(key)
            queries.append(item.strip())

    if plan.use_normalized_query:
        add(understanding.terminology.normalized)
        add(understanding.normalized)
    if plan.use_expansions:
        for item in understanding.expansion.expansions:
            add(item)
    if plan.use_subqueries:
        for item in understanding.decomposition.subqueries:
            add(item)
    if not plan.use_normalized_query and not plan.use_expansions and not plan.use_subqueries:
        for item in understanding.search_queries:
            add(item)
    if extra:
        for item in extra:
            if isinstance(item, str):
                add(item)
    return queries


def _note_sources(bucket: dict[Any, list[str]], hits: Sequence[Any], query: str) -> None:
    for hit in hits:
        pid = getattr(hit, "passage_id", None)
        bucket.setdefault(pid, [])
        if query not in bucket[pid]:
            bucket[pid].append(query)


def _first_ranks(rankings: Sequence[Sequence[Any]]) -> dict[Any, int]:
    if not rankings:
        return {}
    ranks: dict[Any, int] = {}
    for rank, item in enumerate(rankings[0], start=1):
        pid = getattr(item, "passage_id", None)
        if pid is not None and pid not in ranks:
            ranks[pid] = rank
    return ranks


def _ids_only(hits: Sequence[Any]) -> list[Any]:
    return [_jsonable(getattr(item, "passage_id", None)) for item in hits]


def _hits_from_fusion(
    fused: Sequence[tuple[Any, float]],
    texts: dict[Any, str],
    source_queries: dict[Any, list[str]],
    bm25_rankings: Sequence[Sequence[Any]],
    dense_rankings: Sequence[Sequence[Any]],
    pool: int,
) -> list[StageHit]:
    bm25_ranks = _first_ranks(bm25_rankings)
    dense_ranks = _first_ranks(dense_rankings)
    hits: list[StageHit] = []
    for rank, (passage_id, score) in enumerate(fused, start=1):
        text = texts.get(passage_id, "")
        if not text:
            for ranking in (*bm25_rankings, *dense_rankings):
                for item in ranking:
                    if getattr(item, "passage_id", None) == passage_id:
                        text = getattr(item, "text", "") or text
                        break
        hits.append(
            StageHit(
                passage_id=passage_id,
                score=float(score),
                text=text,
                pmid=passage_id,
                bm25_rank=bm25_ranks.get(passage_id),
                dense_rank=dense_ranks.get(passage_id),
                rrf_score=float(score),
                rrf_rank=rank,
                source_queries=tuple(source_queries.get(passage_id) or ()),
                retrieval_stage="rrf",
            )
        )
        if len(hits) == pool:
            break
    return hits


def _annotate_final(
    final: Sequence[Any],
    fused_hits: Sequence[StageHit],
    filtered: Any,
    source_queries: dict[Any, list[str]],
) -> list[StageHit]:
    fused_by_id = {hit.passage_id: hit for hit in fused_hits}
    decisions = {item.passage_id: item.reason for item in filtered.decisions}
    annotated: list[StageHit] = []
    for rank, item in enumerate(final, start=1):
        pid = getattr(item, "passage_id")
        base = fused_by_id.get(pid)
        annotated.append(
            StageHit(
                passage_id=pid,
                score=float(getattr(item, "score", 0.0)),
                text=getattr(item, "text", "") or (base.text if base else ""),
                pmid=getattr(item, "pmid", pid),
                bm25_rank=getattr(item, "bm25_rank", base.bm25_rank if base else None),
                dense_rank=getattr(item, "dense_rank", base.dense_rank if base else None),
                rrf_score=getattr(item, "rrf_score", base.rrf_score if base else getattr(item, "score", None)),
                rrf_rank=getattr(item, "rrf_rank", base.rrf_rank if base else None),
                reranker_score=getattr(item, "reranker_score", None),
                final_rank=rank,
                filter_decision=decisions.get(pid),
                source_queries=tuple(source_queries.get(pid) or (base.source_queries if base else ())),
                retrieval_stage="rerank" if getattr(item, "reranker_score", None) is not None else "hybrid",
            )
        )
    return annotated
