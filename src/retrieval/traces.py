"""Serializable retrieval traces. No LangSmith; JSON only."""

from __future__ import annotations

from typing import Any


TRACE_KEYS = (
    "query_id",
    "original_query",
    "normalized_query",
    "query_classification",
    "expansion_variants",
    "subqueries",
    "search_queries",
    "retrieval_strategy",
    "bm25_results",
    "dense_results",
    "fusion_results",
    "filtering_results",
    "reranker_results",
    "final_evidence",
    "stage_timings",
    "final_answer",
    "citations",
    "grounding_validation",
)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "as_dict"):
        return _jsonable(value.as_dict())
    if hasattr(value, "item") and not isinstance(value, (bytes, str, bool)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return str(value)
    return str(value)


def build_query_trace(
    retrieval_trace: dict[str, Any] | None = None,
    *,
    query_id: Any = None,
    final_answer: str | None = None,
    citations: Any = None,
    grounding_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize a retrieval last_trace into a JSON-serializable query trace."""

    source = dict(retrieval_trace or {})
    citations_payload = _jsonable(citations) if citations is not None else source.get("citations")
    trace = {
        "query_id": _jsonable(query_id if query_id is not None else source.get("query_id")),
        "original_query": source.get("original_query"),
        "normalized_query": source.get("normalized_query"),
        "query_classification": _jsonable(source.get("query_classification") or {}),
        "expansion_variants": list(source.get("expansion_variants") or []),
        "subqueries": list(source.get("subqueries") or []),
        "search_queries": list(source.get("search_queries") or []),
        "retrieval_strategy": _jsonable(source.get("retrieval_strategy") or {}),
        "bm25_results": _jsonable(source.get("bm25_results") or []),
        "dense_results": _jsonable(source.get("dense_results") or []),
        "fusion_results": _jsonable(source.get("fusion_results") or []),
        "filtering_results": _jsonable(source.get("filtering_results") or {}),
        "reranker_results": _jsonable(source.get("reranker_results") or []),
        "final_evidence": _jsonable(source.get("final_evidence") or []),
        "stage_timings": dict(source.get("stage_timings") or {}),
        "final_answer": final_answer if final_answer is not None else source.get("final_answer"),
        "citations": citations_payload,
        "grounding_validation": _jsonable(grounding_validation or source.get("grounding_validation") or {}),
        "pipeline_enabled": bool(source.get("pipeline_enabled")),
        "reranker_enabled": bool(source.get("reranker_enabled")),
        "candidate_pool_size": source.get("candidate_pool_size"),
        "candidate_count_before_filter": source.get("candidate_count_before_filter"),
        "candidate_count_after_filter": source.get("candidate_count_after_filter"),
    }
    return trace
