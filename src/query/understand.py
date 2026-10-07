"""Orchestrate classification, terminology, expansion, and decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import UPGRADE, UpgradeConfig
from src.query.classify import QueryClass, classify_query
from src.query.decomposition import Decomposition, decompose_query
from src.query.expansion import DeterministicExpansion, expand_query
from src.query.terminology import TerminologyResult, normalize_query


@dataclass(frozen=True)
class QueryUnderstanding:
    original: str
    normalized: str
    query_class: QueryClass
    terminology: TerminologyResult
    expansion: DeterministicExpansion
    decomposition: Decomposition
    search_queries: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "query_class": self.query_class.as_dict(),
            "terminology": self.terminology.as_dict(),
            "expansion": self.expansion.as_dict(),
            "decomposition": self.decomposition.as_dict(),
            "search_queries": list(self.search_queries),
        }


def understand_query(query: str, config: UpgradeConfig | None = None) -> QueryUnderstanding:
    """Build a query plan. Disabled stages leave the original query untouched."""

    if not isinstance(query, str):
        raise TypeError("query must be a string")
    cfg = config or UPGRADE
    query_class = classify_query(query)
    terminology = normalize_query(query, enabled=True)
    expansion = expand_query(
        query,
        enabled=cfg.expansion_enabled,
        max_variants=cfg.expansion_max_variants,
        normalized=terminology.normalized,
    )
    decomposition = decompose_query(
        query,
        enabled=cfg.decomposition_enabled,
        min_aspects=cfg.decomposition_min_aspects,
    )
    searches: list[str] = [query]
    seen = {query.strip().lower()}

    def _add(item: str) -> None:
        stripped = item.strip()
        key = stripped.lower()
        if stripped and key not in seen:
            seen.add(key)
            searches.append(stripped)

    if cfg.terminology_enabled:
        _add(terminology.normalized)
    if cfg.expansion_enabled:
        for variant in expansion.expansions:
            _add(variant)
    if cfg.decomposition_enabled and decomposition.applied:
        for subquery in decomposition.subqueries:
            _add(subquery)
    return QueryUnderstanding(
        original=query,
        normalized=terminology.normalized if cfg.terminology_enabled else query,
        query_class=query_class,
        terminology=terminology,
        expansion=expansion,
        decomposition=decomposition,
        search_queries=tuple(searches),
    )
