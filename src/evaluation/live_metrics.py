"""Deterministic live-query metrics. Does not block generation.

Faithfulness / relevancy here are overlap heuristics plus optional reuse of
the existing LLM judge. Citation correctness uses src.validation.citations.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.evaluation.metrics import passage_id_key, recall_at_k
from src.validation.citations import CitationValidation, validate_citations

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {item.lower() for item in _TOKEN_RE.findall(text or "")}


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [part for part in parts if part.strip()]


def overlap_ratio(left: str, right: str) -> float:
    a = _tokens(left)
    b = _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a)


def faithfulness_score(answer: str, evidence_texts: Sequence[str]) -> float | None:
    if not (answer or "").strip() or not evidence_texts:
        return None
    combined = " ".join(str(item) for item in evidence_texts)
    sentences = _sentences(answer)
    if not sentences:
        return overlap_ratio(answer, combined)
    supported = sum(1 for sentence in sentences if overlap_ratio(sentence, combined) >= 0.2)
    return supported / len(sentences)


def answer_relevancy_score(question: str, answer: str) -> float | None:
    if not (question or "").strip() or not (answer or "").strip():
        return None
    return overlap_ratio(question, answer)


def context_precision_score(question: str, evidence_texts: Sequence[str]) -> float | None:
    if not evidence_texts:
        return None
    hits = sum(1 for text in evidence_texts if overlap_ratio(question, text) >= 0.08)
    return hits / len(evidence_texts)


def context_recall_score(
    retrieved_ids: Sequence[Any],
    gold_ids: Sequence[Any] | None,
    *,
    k: int | None = None,
) -> float | None:
    if not gold_ids:
        return None
    limit = len(retrieved_ids) if k is None else k
    return recall_at_k(retrieved_ids, gold_ids, max(1, limit))


def citation_correctness_score(citations: CitationValidation) -> float | None:
    if citations.missing_citations:
        return 0.0 if citations.cited_ids else None
    if not citations.cited_ids:
        return None
    return 1.0 if citations.is_valid else 0.0


def evaluate_live_query(
    *,
    question: str,
    answer: str,
    evidence: Sequence[Any],
    citations: CitationValidation | None = None,
    gold_ids: Sequence[Any] | None = None,
    insufficient_evidence: bool = False,
    latency_ms: float | None = None,
    stage_timings: dict[str, Any] | None = None,
    token_usage: dict[str, Any] | None = None,
    estimated_cost_usd: float | None = None,
    judge: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one live query without calling an LLM."""

    texts = [str(getattr(item, "text", "") or "") for item in evidence]
    ids = [getattr(item, "passage_id", None) for item in evidence]
    citation_result = citations or validate_citations(answer, ids)
    record = {
        "question": question,
        "generated_answer": answer,
        "document_ids": ids,
        "retrieved_context_count": len(evidence),
        "selected_context_count": len(evidence),
        "insufficient_evidence": insufficient_evidence,
        "latency_ms": latency_ms,
        "stage_timings": dict(stage_timings or {}),
        "token_usage": dict(token_usage or {}),
        "estimated_cost_usd": estimated_cost_usd,
        "citations": {
            "cited_ids": list(citation_result.cited_ids),
            "valid_ids": list(citation_result.valid_ids),
            "invalid_ids": list(citation_result.invalid_ids),
            "is_valid": citation_result.is_valid,
            "missing_citations": citation_result.missing_citations,
        },
        "metrics": {
            "faithfulness": None if insufficient_evidence else faithfulness_score(answer, texts),
            "answer_relevancy": None if insufficient_evidence else answer_relevancy_score(question, answer),
            "context_recall": context_recall_score(ids, gold_ids),
            "context_precision": context_precision_score(question, texts),
            "citation_correctness": citation_correctness_score(citation_result),
        },
        "judge": judge,
        "source": "heuristic_live_evaluator",
        "blocks_user_response": False,
        "created_at_unix": time.time(),
    }
    return record


def append_live_evaluation(record: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    return output


def load_live_evaluations(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    rows = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
