"""Heuristic query classification. No LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.query.terminology import detect_terms

ASPECT_TERMS = (
    "risk factor",
    "risk factors",
    "mechanism",
    "mechanisms",
    "treatment",
    "treatments",
    "therapy",
    "therapies",
    "diagnosis",
    "symptom",
    "symptoms",
    "cause",
    "causes",
    "prognosis",
    "prevention",
    "pathophysiology",
    "epidemiology",
    "management",
    "complication",
    "complications",
)

SEMANTIC_CUES = ("how", "why", "relationship", "associated with", "compared to", "versus", "difference")
ENTITY_RE = re.compile(r"\b[A-Z]{2,}[A-Z0-9-]*\b")


QUERY_TYPES = (
    "simple_factual",
    "terminology_heavy",
    "semantic",
    "multi_part",
    "entity_heavy",
    "broad_literature",
)


@dataclass(frozen=True)
class QueryClass:
    query_type: str
    reasons: tuple[str, ...]
    aspect_count: int
    abbreviation_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type,
            "reasons": list(self.reasons),
            "aspect_count": self.aspect_count,
            "abbreviation_count": self.abbreviation_count,
        }


def count_aspects(query: str) -> int:
    lowered = query.lower()
    return sum(1 for term in ASPECT_TERMS if term in lowered)


def classify_query(query: str) -> QueryClass:
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    reasons: list[str] = []
    lowered = query.lower()
    aspects = count_aspects(query)
    abbreviations = detect_terms(query)
    entities = ENTITY_RE.findall(query)
    tokens = re.findall(r"[A-Za-z0-9]+", query)
    if aspects >= 2 or ("," in query and aspects >= 1):
        reasons.append("multiple clinical aspects")
        return QueryClass("multi_part", tuple(reasons), aspects, len(abbreviations))
    if abbreviations:
        reasons.append("biomedical abbreviations or synonyms detected")
        return QueryClass("terminology_heavy", tuple(reasons), aspects, len(abbreviations))
    if any(cue in lowered for cue in SEMANTIC_CUES) or lowered.startswith(("how ", "why ")):
        reasons.append("semantic / explanatory phrasing")
        return QueryClass("semantic", tuple(reasons), aspects, len(abbreviations))
    if len(entities) >= 2:
        reasons.append("multiple capitalized entity tokens")
        return QueryClass("entity_heavy", tuple(reasons), aspects, len(abbreviations))
    if len(tokens) >= 18 or "review" in lowered or "literature" in lowered:
        reasons.append("broad or literature-style wording")
        return QueryClass("broad_literature", tuple(reasons), aspects, len(abbreviations))
    reasons.append("short factual query")
    return QueryClass("simple_factual", tuple(reasons), aspects, len(abbreviations))
