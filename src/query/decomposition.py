"""Heuristic decomposition of multi-part biomedical questions. No LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.query.classify import ASPECT_TERMS, classify_query, count_aspects

_ASPECT_PATTERN = re.compile(
    r"(risk factors?|mechanisms?|treatments?|therapies?|therapy|"
    r"diagnosis|symptoms?|causes?|prognosis|prevention|"
    r"pathophysiology|epidemiology|management|complications?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Decomposition:
    original: str
    subqueries: tuple[str, ...]
    applied: bool
    method: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "subqueries": list(self.subqueries),
            "applied": self.applied,
            "method": self.method,
        }


def topic_focus(query: str) -> str:
    """Best-effort remaining topic after stripping listed aspects."""

    cleaned = _ASPECT_PATTERN.sub(" ", query)
    cleaned = re.sub(r"\b(what|are|the|and|or|of|for|in|to|options?)\b", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[,?/]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return cleaned or query.strip()


def decompose_query(query: str, *, enabled: bool = True, min_aspects: int = 2) -> Decomposition:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not enabled:
        return Decomposition(query, (), False, "disabled")
    classification = classify_query(query)
    if classification.query_type != "multi_part" or count_aspects(query) < min_aspects:
        return Decomposition(query, (), False, "not_complex")
    focus = topic_focus(query)
    aspects = []
    lowered = query.lower()
    for term in ASPECT_TERMS:
        if term in lowered and term not in aspects:
            # keep the longer phrase when both "risk factor" and "risk factors" match
            if any(term in existing and term != existing for existing in aspects):
                continue
            aspects.append(term)
    subqueries = tuple(f"{aspect} of {focus}" if focus else aspect for aspect in aspects)
    unique: list[str] = []
    seen: set[str] = set()
    for item in subqueries:
        key = item.strip().lower()
        if not item.strip() or key in seen or key == query.strip().lower():
            continue
        seen.add(key)
        unique.append(item.strip())
    return Decomposition(
        original=query,
        subqueries=tuple(unique),
        applied=len(unique) >= min_aspects,
        method="aspect_split",
    )
