"""Thin RAG orchestration: hybrid retrieval → evidence → grounded LLM → citations."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from src.config import GENERATION
from src.generation.answer_generator import AnswerGenerator, GeneratedAnswer, LLMClient
from src.generation.query_expansion import QueryExpander
from src.retrieval.hybrid import RetrievalResult
from src.retrieval.traces import build_query_trace
from src.validation.citations import CitationValidation, validate_citations


def select_evidence(
    hits: Sequence[Any],
    *,
    top_k: int | None = None,
) -> list[Any]:
    """Keep the top ranked hybrid hits as LLM evidence."""

    limit = GENERATION.evidence_top_k if top_k is None else top_k
    if limit < 1:
        return []
    return list(hits[:limit])


@dataclass(frozen=True)
class GroundedQAResult:
    question: str
    answer: str
    evidence: tuple[RetrievalResult, ...]
    citations: CitationValidation
    insufficient_evidence: bool
    llm_called: bool
    prompt: str
    expansions: tuple[str, ...] = ()
    query_expansion_used: bool = False
    trace: dict[str, Any] = field(default_factory=dict)


class GroundedQA:
    """Connect hybrid retrieval to grounded generation and citation checks."""

    def __init__(
        self,
        retriever: Any,
        *,
        generator: AnswerGenerator | None = None,
        llm: LLMClient | None = None,
        expander: QueryExpander | None = None,
        evidence_top_k: int | None = None,
        min_evidence_count: int | None = None,
        query_expansion_enabled: bool | None = None,
    ) -> None:
        if retriever is None:
            raise ValueError("GroundedQA requires a retriever")
        self._retriever = retriever
        self._generator = generator or AnswerGenerator(llm=llm)
        self._expander = expander
        self.evidence_top_k = (
            GENERATION.evidence_top_k if evidence_top_k is None else evidence_top_k
        )
        self.min_evidence_count = (
            GENERATION.min_evidence_count if min_evidence_count is None else min_evidence_count
        )
        self.query_expansion_enabled = (
            expander is not None
            if query_expansion_enabled is None
            else query_expansion_enabled
        )

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        expand: bool | None = None,
    ) -> GroundedQAResult:
        original = question
        expansions: tuple[str, ...] = ()
        expansion_llm_called = False
        query_expansion_ms: float | None = None
        use_expansion = self.query_expansion_enabled if expand is None else expand
        if use_expansion and self._expander is not None:
            started = time.perf_counter()
            expanded = self._expander.expand(original)
            query_expansion_ms = (time.perf_counter() - started) * 1000.0
            expansions = expanded.expansions
            expansion_llm_called = expanded.llm_called
        started = time.perf_counter()
        hits = self._retriever.search(
            original,
            top_k=top_k or self.evidence_top_k,
            expansions=expansions if expansions else None,
        )
        evidence = select_evidence(hits, top_k=self.evidence_top_k)
        retrieval_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        generated = self._generate(original, evidence)
        generation_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        citations = validate_citations(
            generated.answer,
            [item.passage_id for item in evidence],
        )
        citation_ms = (time.perf_counter() - started) * 1000.0
        retrieval_trace = dict(getattr(self._retriever, "last_trace", {}) or {})
        citations_payload = citations.as_dict() if hasattr(citations, "as_dict") else {
            "cited_ids": list(citations.cited_ids),
            "valid_ids": list(citations.valid_ids),
            "invalid_ids": list(citations.invalid_ids),
            "has_citations": citations.has_citations,
            "missing_citations": citations.missing_citations,
            "is_valid": citations.is_valid,
        }
        query_trace = build_query_trace(
            retrieval_trace,
            final_answer=generated.answer,
            citations=citations_payload,
            grounding_validation={
                "insufficient_evidence": generated.insufficient_evidence,
                "llm_called": generated.llm_called,
                "citation_valid": citations.is_valid,
            },
        )
        return GroundedQAResult(
            question=original,
            answer=generated.answer,
            evidence=tuple(evidence),
            citations=citations,
            insufficient_evidence=generated.insufficient_evidence,
            llm_called=generated.llm_called,
            prompt=generated.prompt,
            expansions=expansions,
            query_expansion_used=bool(use_expansion and self._expander is not None),
            trace=_pipeline_trace(
                query_expansion_ms=query_expansion_ms,
                generation_ms=generation_ms,
                citation_ms=citation_ms,
                retrieval_ms=retrieval_ms,
                expansion_llm_called=expansion_llm_called,
                retrieval_trace=retrieval_trace,
                query_trace=query_trace,
                dense=getattr(self._retriever, "_dense", None),
            ),
        )

    def _generate(
        self,
        question: str,
        evidence: Sequence[Any],
    ) -> GeneratedAnswer:
        if len(evidence) < self.min_evidence_count:
            # Application-level guard: do not call the LLM without enough evidence.
            return self._generator.generate(question, [])
        return self._generator.generate(question, evidence)


def _embedding_dim(dense: Any) -> int | None:
    embeddings = getattr(dense, "_embeddings", None)
    shape = getattr(embeddings, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    return int(shape[1])


def _pipeline_trace(
    *,
    query_expansion_ms: float | None,
    generation_ms: float,
    citation_ms: float,
    retrieval_ms: float,
    expansion_llm_called: bool,
    retrieval_trace: dict[str, Any],
    query_trace: dict[str, Any],
    dense: Any,
) -> dict[str, Any]:
    merged = dict(query_trace)
    merged.update(
        {
            "query_expansion_ms": query_expansion_ms,
            "generation_ms": generation_ms,
            "citation_ms": citation_ms,
            "retrieval_ms": retrieval_ms,
            "expansion_llm_called": expansion_llm_called,
            "bm25_ms": retrieval_trace.get("bm25_ms"),
            "dense_ms": retrieval_trace.get("dense_ms"),
            "rrf_ms": retrieval_trace.get("rrf_ms"),
            "filter_ms": retrieval_trace.get("filter_ms"),
            "n_search_queries": retrieval_trace.get("n_queries"),
            "n_rankings": retrieval_trace.get("n_rankings"),
            "candidate_k": retrieval_trace.get("candidate_k"),
            "rrf_k": retrieval_trace.get("rrf_k"),
            "embedding_dim": _embedding_dim(dense),
            "dense_model_name": getattr(dense, "model_name", None),
            "reranker_enabled": bool(retrieval_trace.get("reranker_enabled")),
            "rerank_ms": retrieval_trace.get("rerank_ms"),
            "reranker_model_name": retrieval_trace.get("reranker_model_name"),
            "reranker_device": retrieval_trace.get("reranker_device"),
            "reranker_candidate_pool_size": retrieval_trace.get("reranker_candidate_pool_size"),
            "rerank_candidate_count": retrieval_trace.get("rerank_candidate_count"),
            "original_query": retrieval_trace.get("original_query") or merged.get("original_query"),
            "normalized_query": retrieval_trace.get("normalized_query") or merged.get("normalized_query"),
            "query_classification": retrieval_trace.get("query_classification") or merged.get("query_classification"),
            "expansion_variants": retrieval_trace.get("expansion_variants") or merged.get("expansion_variants"),
            "subqueries": retrieval_trace.get("subqueries") or merged.get("subqueries"),
            "retrieval_strategy": retrieval_trace.get("retrieval_strategy") or merged.get("retrieval_strategy"),
            "stage_timings": retrieval_trace.get("stage_timings") or merged.get("stage_timings"),
            "filtering_results": retrieval_trace.get("filtering_results") or merged.get("filtering_results"),
            "final_evidence": retrieval_trace.get("final_evidence") or merged.get("final_evidence"),
            "understanding": retrieval_trace.get("understanding"),
        }
    )
    return merged
