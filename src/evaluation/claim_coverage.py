"""Claim → evidence coverage. Evaluation metric only; not used in generation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from src.validation.citations import CITATION_PATTERN, CitationValidation

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {item.lower() for item in _TOKEN_RE.findall(text or "")}


def _overlap(left: str, right: str) -> float:
    a = _tokens(left)
    b = _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a)


def strip_citations(text: str) -> str:
    return CITATION_PATTERN.sub("", text or "").strip()


def split_claims(answer: str) -> list[str]:
    """Split a generated answer into claim-like sentences."""

    cleaned = strip_citations(answer)
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    claims = []
    for part in parts:
        text = re.sub(r"\s+", " ", part).strip()
        if len(_tokens(text)) < 4:
            continue
        claims.append(text)
    if not claims and cleaned:
        claims.append(re.sub(r"\s+", " ", cleaned).strip())
    return claims


def _passage_id(item: Any) -> str:
    if isinstance(item, dict):
        value = item.get("passage_id", item.get("pmid"))
    else:
        value = getattr(item, "passage_id", getattr(item, "pmid", None))
    return str(value)


def _passage_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("text") or "")
    return str(getattr(item, "text", "") or "")


def cited_ids_for_claim(claim: str, original_answer: str, valid_ids: Sequence[Any]) -> list[str]:
    """Best-effort: citations that appear in the same original sentence as this claim."""

    valid = {str(item) for item in valid_ids}
    sentences = re.split(r"(?<=[.!?])\s+", original_answer or "")
    found: list[str] = []
    claim_core = strip_citations(claim)
    for sentence in sentences:
        if claim_core and claim_core[:40] in strip_citations(sentence):
            for match in CITATION_PATTERN.findall(sentence):
                key = str(match)
                if key in valid and key not in found:
                    found.append(key)
    return found


def claim_evidence_coverage(
    answer: str,
    evidence: Sequence[Any],
    citations: CitationValidation | None = None,
    *,
    overlap_threshold: float = 0.2,
) -> dict[str, Any]:
    """Map each claim to supporting evidence IDs using overlap + citation IDs."""

    claims = split_claims(answer)
    texts = {_passage_id(item): _passage_text(item) for item in evidence}
    valid_ids = list(citations.valid_ids) if citations is not None else []
    rows = []
    supported = 0
    for index, claim in enumerate(claims, start=1):
        cited = cited_ids_for_claim(claim, answer, valid_ids)
        supporting: list[str] = []
        for pid in cited:
            text = texts.get(pid, "")
            if text and _overlap(claim, text) >= overlap_threshold:
                supporting.append(pid)
            elif pid in texts:
                supporting.append(pid)
        if not supporting:
            ranked = sorted(
                texts.items(),
                key=lambda item: _overlap(claim, item[1]),
                reverse=True,
            )
            if ranked and _overlap(claim, ranked[0][1]) >= overlap_threshold:
                supporting.append(ranked[0][0])
        is_supported = bool(supporting)
        if is_supported:
            supported += 1
        rows.append(
            {
                "claim_index": index,
                "claim": claim,
                "supported": is_supported,
                "passage_ids": supporting,
            }
        )
    total = len(rows)
    rate = (supported / total) if total else None
    return {
        "claims": rows,
        "supported_claims": supported,
        "total_claims": total,
        "claim_support_rate": rate,
        "summary": (
            f"Supported claims: {supported}/{total}"
            if total
            else "No claim sentences were extracted from the answer."
        ),
    }
