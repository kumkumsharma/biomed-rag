"""Authoritative runtime pipeline state for UI and telemetry.

Every Ask/Trace/Observability panel should read this object rather than
inferring enabled/disabled status from config defaults or unrelated widgets.
Does not change retrieval ranking.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.config import EVALUATION, GENERATION, RERANKER, RETRIEVAL
from src.query.classify import ENTITY_RE, QUERY_TYPES, classify_query
from src.query.terminology import detect_terms
from src.validation.citations import CitationValidation

INTENT_LABELS = {
    "simple_factual": "Definition / factual",
    "terminology_heavy": "Terminology-heavy",
    "semantic": "Semantic / explanatory",
    "multi_part": "Multi-part",
    "entity_heavy": "Entity-heavy",
    "broad_literature": "Broad / literature",
}
RECOGNIZED_QUERY_TYPES = frozenset(QUERY_TYPES)
CLASSIFIER_MECHANISM = (
    "Heuristic / deterministic classifier (`src/query/classify.py`). No LLM prompt."
)
TERMINOLOGY_MECHANISM = (
    "Heuristic / deterministic terminology map (`src/query/terminology.py`). No LLM prompt."
)
RANKING_COLUMN_HELP = {
    "Final rank": "Final position after the complete retrieval/reranking pipeline.",
    "Passage ID / PMID": (
        "Identifier of the retrieved passage. In this corpus passage IDs and PMIDs "
        "are identical; they can be split later if chunking creates distinct IDs."
    ),
    "BM25 rank": "Position assigned by BM25.",
    "Dense rank": "Position assigned by dense retrieval.",
    "RRF rank": "Position after BM25 + dense Reciprocal Rank Fusion.",
    "RRF score": "Fusion score combining the lexical and dense rankings. Not a probability.",
    "MedCPT score": "Cross-encoder relevance score for query + passage. Not a probability.",
    "Final score": "Score used by the final ranking stage.",
}


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    if number is None:
        return None
    return int(number)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Mapping):
        return list(value.keys())
    try:
        return list(value)
    except TypeError:
        return []


def _hit_id(item: Any) -> Any:
    if isinstance(item, Mapping):
        return item.get("passage_id", item.get("pmid"))
    return getattr(item, "passage_id", getattr(item, "pmid", item))


def _hit_field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def passage_id_pmid(item: Any) -> str:
    """Single display identifier. Split later if passage_id and PMID diverge."""

    pid = _hit_id(item)
    pmid = _hit_field(item, "pmid", pid)
    if pmid in {None, ""}:
        pmid = pid
    if pid in {None, ""}:
        return str(pmid)
    if str(pid) == str(pmid):
        return str(pid)
    return f"{pid} / {pmid}"


def ids_are_identical(evidence: Sequence[Any]) -> bool:
    for item in evidence:
        pid = _hit_id(item)
        pmid = _hit_field(item, "pmid", pid)
        if pmid not in {None, "", pid} and str(pmid) != str(pid):
            return False
    return True


def infer_medcpt_executed(
    trace: Mapping[str, Any] | None,
    evidence: Sequence[Any] | None = None,
    *,
    rerank_requested: bool | None = None,
) -> bool:
    """True when MedCPT actually scored this request."""

    payload = _mapping(trace)
    if payload.get("reranker_enabled") is True:
        return True
    if _as_float(payload.get("rerank_ms")) not in {None, 0.0}:
        return True
    if _as_int(payload.get("rerank_returned")) not in {None, 0}:
        return True
    if payload.get("reranker_model_name") and _as_int(payload.get("rerank_candidate_count")):
        return True
    for item in evidence or []:
        score = _hit_field(item, "reranker_score")
        if score is not None:
            return True
        stage = str(_hit_field(item, "retrieval_stage") or "")
        if stage.lower() in {"rerank", "medcpt"}:
            return True
    if rerank_requested is True and payload.get("reranker_enabled") is not False:
        if payload.get("rerank_candidate_count") or payload.get("reranker_candidate_pool_size"):
            return True
    return False


def infer_query_expansion_executed(
    *,
    expand_requested: bool,
    query_expansion_used: bool,
    expansions: Sequence[str] | None,
    trace: Mapping[str, Any] | None,
) -> bool:
    payload = _mapping(trace)
    if query_expansion_used or _sequence(expansions):
        return True
    if payload.get("expansion_llm_called"):
        return True
    if _sequence(payload.get("expansion_variants")):
        return True
    return bool(expand_requested and query_expansion_used)


def _count_rank_map(value: Any) -> int:
    if isinstance(value, Mapping):
        return len(value)
    return len(_sequence(value))


def _first_positive(*values: Any) -> int:
    for value in values:
        if isinstance(value, Mapping) and value:
            return len(value)
        if isinstance(value, (list, tuple)) and value:
            return len(value)
        number = _as_int(value)
        if number is not None and number > 0:
            return number
    return 0


def _extract_definition_entity(query: str) -> str | None:
    import re

    match = re.match(
        r"^(?:what is|what's|whats|define|definition of)\s+(.+?)\s*\??$",
        query.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    entity = match.group(1).strip(" .")
    return entity or None


def build_query_understanding(
    question: str,
    trace: Mapping[str, Any] | None = None,
    *,
    expansions: Sequence[str] | None = None,
) -> dict[str, Any]:
    payload = _mapping(trace)
    original = str(payload.get("original_query") or question or "")
    classification = _mapping(payload.get("query_classification"))
    if not classification:
        classification = _mapping((_mapping(payload.get("understanding")).get("query_class")))
    classification_source = "trace"
    if not classification.get("query_type"):
        classification = classify_query(original).as_dict() if original else {}
        classification_source = "display_heuristic"
    query_type = str(classification.get("query_type") or "")
    recognized = query_type in RECOGNIZED_QUERY_TYPES
    if recognized:
        intent_label = INTENT_LABELS[query_type]
        intent_reason = "; ".join(str(item) for item in _sequence(classification.get("reasons"))) or (
            "short factual query" if query_type == "simple_factual" else query_type
        )
    else:
        intent_label = "Unclassified"
        intent_reason = "classifier did not produce a recognized category"
        query_type = "unclassified"

    terminology = _mapping(_mapping(payload.get("understanding")).get("terminology"))
    medical_terms = [
        str(item)
        for item in (
            _sequence(terminology.get("detected_terms")) or detect_terms(original)
        )
    ]
    entities: list[str] = []
    definition_entity = _extract_definition_entity(original)
    if definition_entity:
        entities.append(definition_entity)
    for token in ENTITY_RE.findall(original):
        if token not in entities:
            entities.append(token)
    for term in medical_terms:
        if term not in entities and term.lower() not in {item.lower() for item in entities}:
            entities.append(term)

    normalized = str(payload.get("normalized_query") or original)
    mappings = _sequence(terminology.get("mappings"))
    changed = bool(normalized.strip() and original.strip() and normalized.strip() != original.strip())
    if changed:
        normalization_status = "Normalization applied: query rewritten"
        normalization_note = normalized
    elif mappings:
        normalization_status = "Normalization applied: no change"
        normalization_note = "Mappings were detected but the search string was left unchanged."
    else:
        normalization_status = "Normalization applied: no change"
        normalization_note = "No terminology rewrite was required for this query."

    rewrites = [str(item) for item in _sequence(expansions) if str(item).strip()]
    if not rewrites:
        rewrites = [str(item) for item in _sequence(payload.get("expansion_variants")) if str(item).strip()]

    return {
        "original_query": original,
        "intent": intent_label,
        "intent_code": query_type,
        "intent_reason": intent_reason,
        "intent_recognized": recognized,
        "classifier_mechanism": CLASSIFIER_MECHANISM,
        "classification_source": classification_source,
        "entities": entities,
        "medical_terms": medical_terms,
        "normalized_query": normalized,
        "normalization_status": normalization_status,
        "normalization_note": normalization_note,
        "normalization_changed": changed,
        "terminology_mechanism": TERMINOLOGY_MECHANISM,
        "rewrites": rewrites,
        "subqueries": [str(item) for item in _sequence(payload.get("subqueries"))],
    }


def _stage(
    key: str,
    *,
    label: str,
    executed: bool,
    output_count: int,
    latency_ms: Any = None,
    model: str | None = None,
    detail: str = "",
    status_override: str | None = None,
) -> dict[str, Any]:
    if status_override:
        status = status_override
    else:
        status = "enabled/active" if executed else "disabled/not executed"
    return {
        "key": key,
        "label": label,
        "executed": executed,
        "status": status,
        "output_count": int(output_count),
        "latency_ms": _as_float(latency_ms),
        "model": model,
        "detail": detail,
    }


def latency_percentiles(values: Sequence[float]) -> dict[str, float | None]:
    numbers = sorted(float(item) for item in values if item is not None)
    if not numbers:
        return {"p50_ms": None, "p95_ms": None, "p99_ms": None, "n": 0}

    def _pct(p: float) -> float:
        index = min(len(numbers) - 1, max(0, int(round((p / 100.0) * (len(numbers) - 1)))))
        return numbers[index]

    return {
        "p50_ms": _pct(50),
        "p95_ms": _pct(95),
        "p99_ms": _pct(99),
        "n": len(numbers),
    }


def build_runtime_state(
    result: Any,
    *,
    expand_requested: bool,
    rerank_requested: bool | None = None,
    total_latency_ms: float = 0.0,
    live_quality: Mapping[str, Any] | None = None,
    latency_history_ms: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Single source of truth for the current query's pipeline."""

    trace = _mapping(getattr(result, "trace", None))
    evidence = list(getattr(result, "evidence", ()) or ())
    citations: CitationValidation | None = getattr(result, "citations", None)
    expansions = tuple(getattr(result, "expansions", ()) or ())
    question = str(getattr(result, "question", "") or "")
    if not trace.get("final_evidence") and evidence:
        trace["final_evidence"] = [
            {
                "passage_id": _hit_id(item),
                "pmid": _hit_field(item, "pmid", _hit_id(item)),
                "score": _hit_field(item, "score"),
                "reranker_score": _hit_field(item, "reranker_score"),
                "bm25_rank": _hit_field(item, "bm25_rank"),
                "dense_rank": _hit_field(item, "dense_rank"),
                "rrf_rank": _hit_field(item, "rrf_rank"),
                "rrf_score": _hit_field(item, "rrf_score"),
                "final_rank": _hit_field(item, "final_rank"),
                "retrieval_stage": _hit_field(item, "retrieval_stage"),
            }
            for item in evidence
        ]

    medcpt_ran = infer_medcpt_executed(trace, evidence, rerank_requested=rerank_requested)
    expansion_ran = infer_query_expansion_executed(
        expand_requested=expand_requested,
        query_expansion_used=bool(getattr(result, "query_expansion_used", False)),
        expansions=expansions,
        trace=trace,
    )
    timings = _mapping(trace.get("stage_timings"))
    candidate_k = _as_int(trace.get("candidate_k"))
    pool = _first_positive(
        trace.get("rerank_candidate_count"),
        trace.get("reranker_candidate_pool_size"),
        candidate_k,
        _count_rank_map(trace.get("rrf_ranks")),
        _count_rank_map(trace.get("bm25_ranks")),
    )
    bm25_n = _first_positive(
        _sequence(trace.get("bm25_results")),
        _count_rank_map(trace.get("bm25_ranks")),
        pool,
        len(evidence),
    )
    dense_n = _first_positive(
        _sequence(trace.get("dense_results")),
        _count_rank_map(trace.get("dense_ranks")),
        pool,
        len(evidence),
    )
    rrf_n = _first_positive(
        _sequence(trace.get("fusion_results")),
        _count_rank_map(trace.get("rrf_ranks")),
        pool,
        len(evidence),
    )
    rerank_in = _first_positive(trace.get("rerank_candidate_count"), pool, rrf_n)
    rerank_out = _first_positive(
        _sequence(trace.get("reranker_results")),
        trace.get("rerank_returned"),
        len(evidence) if medcpt_ran else 0,
    )
    evidence_n = _first_positive(_sequence(trace.get("final_evidence")), len(evidence))
    if evidence_n and bm25_n == 0:
        bm25_n = evidence_n
    if evidence_n and dense_n == 0:
        dense_n = evidence_n
    if evidence_n and rrf_n == 0:
        rrf_n = evidence_n

    generation_ran = bool(getattr(result, "llm_called", False)) or bool(
        str(getattr(result, "answer", "") or "").strip()
    )
    citation_ran = citations is not None
    citation_valid = bool(citations.is_valid) if citations is not None else False

    medcpt_model = None
    if medcpt_ran:
        medcpt_model = str(trace.get("reranker_model_name") or RERANKER.model_name)
    dense_model = str(trace.get("dense_model_name") or RETRIEVAL.dense_model_name)
    expansion_model = GENERATION.query_expansion_model_name if expansion_ran else None
    generation_model = GENERATION.model_name if generation_ran else None

    quality = _mapping(live_quality)
    judge_calls = _as_int(quality.get("llm_calls")) or 0
    evaluator_type = str(quality.get("evaluator_type") or quality.get("model_name") or "")
    heuristic = evaluator_type in {"heuristic", "heuristic_live_evaluator"} or judge_calls == 0
    if heuristic and quality.get("evaluated"):
        evaluator_label = "Heuristic evaluator"
        evaluator_type = "heuristic"
    elif quality.get("evaluated"):
        evaluator_label = str(quality.get("model_name") or EVALUATION.judge_model_name)
        evaluator_type = "llm_judge"
    else:
        evaluator_label = "Not evaluated"
        evaluator_type = "none"

    qe_calls = 1 if trace.get("expansion_llm_called") else (1 if expansion_ran and expand_requested else 0)
    qe_cost = (
        qe_calls * GENERATION.query_expansion_estimated_cost_per_call_usd if qe_calls else None
    )
    generation_cost = None
    total_cost = qe_cost

    n_queries = _as_int(trace.get("n_search_queries")) or _as_int(trace.get("n_queries"))
    if n_queries is None:
        n_queries = 1 + len(expansions) if expansion_ran else 1

    stages = [
        _stage("query", label="Query", executed=True, output_count=1, detail=question[:120]),
        _stage(
            "understanding",
            label="Query Understanding",
            executed=True,
            output_count=1,
            detail=build_query_understanding(question, trace, expansions=expansions)["intent"],
        ),
        _stage(
            "bm25",
            label="BM25",
            executed=bm25_n > 0,
            output_count=bm25_n,
            latency_ms=timings.get("bm25_ms") or trace.get("bm25_ms"),
            model=None,
            detail=f"{bm25_n} candidates",
        ),
        _stage(
            "dense",
            label="Dense",
            executed=dense_n > 0,
            output_count=dense_n,
            latency_ms=timings.get("dense_ms") or trace.get("dense_ms"),
            model=dense_model,
            detail=f"{dense_n} candidates",
        ),
        _stage(
            "rrf",
            label="RRF Fusion",
            executed=rrf_n > 0,
            output_count=rrf_n,
            latency_ms=timings.get("rrf_ms") or trace.get("rrf_ms"),
            model=f"RRF k={trace.get('rrf_k') or RETRIEVAL.rrf_k}",
            detail=f"Top-{rrf_n} candidates",
        ),
        _stage(
            "medcpt",
            label="MedCPT Reranker",
            executed=medcpt_ran,
            output_count=rerank_out if medcpt_ran else 0,
            latency_ms=timings.get("rerank_ms") or trace.get("rerank_ms"),
            model=medcpt_model,
            detail=(f"{rerank_in} → {rerank_out}" if medcpt_ran else "not executed"),
        ),
        _stage(
            "evidence",
            label="Final Evidence",
            executed=evidence_n > 0,
            output_count=evidence_n,
            detail=f"{evidence_n} passages",
        ),
        _stage(
            "generation",
            label="Grounded Answer",
            executed=generation_ran,
            output_count=1 if str(getattr(result, "answer", "") or "").strip() else 0,
            latency_ms=trace.get("generation_ms"),
            model=generation_model,
            detail="1 answer" if generation_ran else "not generated",
        ),
        _stage(
            "citations",
            label="Citation Validation",
            executed=citation_ran,
            output_count=len(getattr(citations, "cited_ids", ()) or ()) if citations else 0,
            latency_ms=trace.get("citation_ms"),
            detail="valid" if citation_valid else "invalid or missing",
            status_override=("valid" if citation_valid else "invalid or missing") if citation_ran else None,
        ),
        _stage(
            "evaluation",
            label="Evaluation",
            executed=bool(quality.get("evaluated")),
            output_count=1 if quality.get("evaluated") else 0,
            latency_ms=quality.get("latency_ms"),
            model=evaluator_label,
            detail=evaluator_label,
        ),
    ]
    by_key = {item["key"]: item for item in stages}
    understanding = build_query_understanding(question, trace, expansions=expansions)

    final_equals_medcpt = False
    if evidence and medcpt_ran:
        comparable = []
        for item in evidence:
            final_score = _as_float(_hit_field(item, "score"))
            medcpt_score = _as_float(_hit_field(item, "reranker_score"))
            if final_score is not None and medcpt_score is not None:
                comparable.append(abs(final_score - medcpt_score) < 1e-9)
        final_equals_medcpt = bool(comparable) and all(comparable)

    inconsistencies = []
    if medcpt_ran and by_key["medcpt"]["status"] not in {"enabled/active"}:
        inconsistencies.append("MedCPT executed but status is not enabled/active")
    if not medcpt_ran and "disabled" not in by_key["medcpt"]["status"]:
        inconsistencies.append("MedCPT did not execute but status is not disabled")
    if evidence_n > 0:
        for key in ("bm25", "dense", "rrf"):
            if by_key[key]["output_count"] == 0:
                inconsistencies.append(f"{key} output_count is 0 while evidence exists")

    return {
        "question": question,
        "expand_requested": bool(expand_requested),
        "rerank_requested": bool(rerank_requested) if rerank_requested is not None else medcpt_ran,
        "medcpt_executed": medcpt_ran,
        "query_expansion_executed": expansion_ran,
        "query_expansion_status": "ON" if expansion_ran else "OFF",
        "deep_reranking_status": "ON" if medcpt_ran else "OFF",
        "understanding": understanding,
        "stages": stages,
        "stage_by_key": by_key,
        "trace": trace,
        "evidence_count": evidence_n,
        "candidate_pool": rerank_in if medcpt_ran else rrf_n,
        "candidates_reranked": rerank_in if medcpt_ran else 0,
        "bm25_searches": n_queries,
        "dense_searches": n_queries,
        "rrf_operations": 1 if rrf_n else 0,
        "llm_calls": (1 if trace.get("expansion_llm_called") else 0)
        + (1 if getattr(result, "llm_called", False) else 0),
        "llm_judge_calls": judge_calls,
        "evaluator_type": evaluator_type,
        "evaluator_label": evaluator_label,
        "live_quality": quality,
        "citations_valid": citation_valid,
        "citation_status": "Valid" if citation_valid else "Invalid",
        "insufficient_evidence": bool(getattr(result, "insufficient_evidence", False)),
        "ids_identical": ids_are_identical(evidence),
        "final_score_equals_medcpt": final_equals_medcpt,
        "models": {
            "bm25": "BM25 (lexical, no neural model)",
            "dense": dense_model,
            "rrf": f"RRF k={trace.get('rrf_k') or RETRIEVAL.rrf_k}",
            "medcpt": medcpt_model,
            "generation": generation_model,
            "expansion": expansion_model,
        },
        "latency": {
            "total_ms": float(total_latency_ms),
            "query_expansion_ms": _as_float(trace.get("query_expansion_ms")),
            "bm25_ms": _as_float(timings.get("bm25_ms") or trace.get("bm25_ms")),
            "dense_ms": _as_float(timings.get("dense_ms") or trace.get("dense_ms")),
            "rrf_ms": _as_float(timings.get("rrf_ms") or trace.get("rrf_ms")),
            "medcpt_ms": _as_float(timings.get("rerank_ms") or trace.get("rerank_ms")) if medcpt_ran else None,
            "generation_ms": _as_float(trace.get("generation_ms")),
            "citation_ms": _as_float(trace.get("citation_ms")),
            **latency_percentiles(latency_history_ms or ()),
        },
        "cost": {
            "query_expansion_usd": qe_cost,
            "generation_usd": generation_cost,
            "total_estimated_usd": total_cost,
            "generation_note": "Not measured (no token usage is recorded by the generation client).",
        },
        "inconsistencies": inconsistencies,
        "consistent": not inconsistencies,
    }


def medcpt_status_labels(state: Mapping[str, Any]) -> set[str]:
    executed = bool(state.get("medcpt_executed"))
    return {"enabled/active", "ON"} if executed else {"disabled/not executed", "OFF"}
