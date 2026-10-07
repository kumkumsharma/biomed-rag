"""Deterministic adaptive retrieval router."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import UPGRADE, UpgradeConfig
from src.query.classify import QueryClass, classify_query
from src.query.understand import QueryUnderstanding, understand_query


@dataclass(frozen=True)
class RetrievalPlan:
    query_type: str
    strategy: str
    use_bm25: bool
    use_dense: bool
    use_normalized_query: bool
    use_expansions: bool
    use_subqueries: bool
    bm25_candidate_k: int
    dense_candidate_k: int
    candidate_pool_size: int
    reasons: tuple[str, ...]
    components: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type,
            "strategy": self.strategy,
            "use_bm25": self.use_bm25,
            "use_dense": self.use_dense,
            "use_normalized_query": self.use_normalized_query,
            "use_expansions": self.use_expansions,
            "use_subqueries": self.use_subqueries,
            "bm25_candidate_k": self.bm25_candidate_k,
            "dense_candidate_k": self.dense_candidate_k,
            "candidate_pool_size": self.candidate_pool_size,
            "reasons": list(self.reasons),
            "components": list(self.components),
        }


def route_query(
    query: str,
    *,
    config: UpgradeConfig | None = None,
    understanding: QueryUnderstanding | None = None,
) -> RetrievalPlan:
    cfg = config or UPGRADE
    pool = cfg.candidate_pool_size
    if not cfg.adaptive_routing_enabled:
        return RetrievalPlan(
            query_type=(understanding.query_class.query_type if understanding else classify_query(query).query_type),
            strategy="fixed_hybrid",
            use_bm25=True,
            use_dense=True,
            use_normalized_query=False,
            use_expansions=False,
            use_subqueries=False,
            bm25_candidate_k=pool,
            dense_candidate_k=pool,
            candidate_pool_size=pool,
            reasons=("adaptive routing disabled",),
            components=("bm25", "dense", "rrf"),
        )
    classification = understanding.query_class if understanding else classify_query(query)
    return plan_for_class(classification, cfg)


def plan_for_class(classification: QueryClass, config: UpgradeConfig) -> RetrievalPlan:
    pool = config.candidate_pool_size
    query_type = classification.query_type
    if query_type == "terminology_heavy":
        return RetrievalPlan(
            query_type=query_type,
            strategy="lexical_normalized_hybrid",
            use_bm25=True,
            use_dense=True,
            use_normalized_query=True,
            use_expansions=config.expansion_enabled,
            use_subqueries=False,
            bm25_candidate_k=max(pool, 80),
            dense_candidate_k=pool,
            candidate_pool_size=max(pool, 80),
            reasons=classification.reasons,
            components=("terminology", "bm25", "dense", "rrf"),
        )
    if query_type == "semantic":
        return RetrievalPlan(
            query_type=query_type,
            strategy="dense_emphasized_hybrid",
            use_bm25=True,
            use_dense=True,
            use_normalized_query=config.terminology_enabled,
            use_expansions=config.expansion_enabled,
            use_subqueries=False,
            bm25_candidate_k=pool,
            dense_candidate_k=max(pool, 80),
            candidate_pool_size=max(pool, 80),
            reasons=classification.reasons,
            components=("dense", "bm25", "rrf"),
        )
    if query_type == "multi_part":
        return RetrievalPlan(
            query_type=query_type,
            strategy="decompose_and_fuse",
            use_bm25=True,
            use_dense=True,
            use_normalized_query=config.terminology_enabled,
            use_expansions=config.expansion_enabled,
            use_subqueries=True,
            bm25_candidate_k=pool,
            dense_candidate_k=pool,
            candidate_pool_size=pool,
            reasons=classification.reasons,
            components=("decomposition", "bm25", "dense", "rrf"),
        )
    if query_type in {"entity_heavy", "broad_literature"}:
        return RetrievalPlan(
            query_type=query_type,
            strategy="hybrid_larger_pool",
            use_bm25=True,
            use_dense=True,
            use_normalized_query=config.terminology_enabled,
            use_expansions=config.expansion_enabled,
            use_subqueries=False,
            bm25_candidate_k=max(pool, 80),
            dense_candidate_k=max(pool, 80),
            candidate_pool_size=max(pool, 80),
            reasons=classification.reasons,
            components=("bm25", "dense", "rrf"),
        )
    return RetrievalPlan(
        query_type=query_type,
        strategy="fixed_hybrid",
        use_bm25=True,
        use_dense=True,
        use_normalized_query=config.terminology_enabled,
        use_expansions=False,
        use_subqueries=False,
        bm25_candidate_k=pool,
        dense_candidate_k=pool,
        candidate_pool_size=pool,
        reasons=classification.reasons,
        components=("bm25", "dense", "rrf"),
    )
