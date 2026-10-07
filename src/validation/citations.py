"""Deterministic citation extraction and validation.

The LLM is not asked to check its own citations. Cited IDs are parsed from
the answer text and compared with the evidence that was actually supplied.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

CITATION_PATTERN = re.compile(r"\[PASSAGE_ID:\s*([^\]]+?)\]")


@dataclass(frozen=True)
class CitationValidation:
    """Result of checking cited IDs against supplied evidence IDs."""

    cited_ids: tuple[Any, ...]
    valid_ids: tuple[Any, ...]
    invalid_ids: tuple[Any, ...]
    has_citations: bool
    missing_citations: bool

    @property
    def is_valid(self) -> bool:
        return self.has_citations and not self.invalid_ids


def format_citation(passage_id: Any) -> str:
    """Return the canonical citation token for a passage ID."""

    return f"[PASSAGE_ID: {passage_id}]"


def normalize_passage_id(passage_id: Any) -> str:
    return str(passage_id).strip()


def extract_cited_passage_ids(answer: str) -> list[str]:
    """Extract cited IDs in order of appearance, preserving duplicates."""

    if not answer:
        return []
    return [match.strip() for match in CITATION_PATTERN.findall(answer)]


def _unique(values: Iterable[Any]) -> tuple[Any, ...]:
    seen: set[str] = set()
    unique: list[Any] = []
    for value in values:
        key = normalize_passage_id(value)
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return tuple(unique)


def validate_citations(
    answer: str,
    evidence_ids: Iterable[Any],
) -> CitationValidation:
    """Check that every cited ID was present in the supplied evidence."""

    evidence_by_key = {
        normalize_passage_id(passage_id): passage_id for passage_id in evidence_ids
    }
    extracted = extract_cited_passage_ids(answer)
    unique_cited = _unique(extracted)

    valid: list[Any] = []
    invalid: list[str] = []
    for cited in unique_cited:
        key = normalize_passage_id(cited)
        if key in evidence_by_key:
            valid.append(evidence_by_key[key])
        else:
            invalid.append(cited)

    has_citations = bool(unique_cited)
    return CitationValidation(
        cited_ids=unique_cited,
        valid_ids=tuple(valid),
        invalid_ids=tuple(invalid),
        has_citations=has_citations,
        missing_citations=not has_citations,
    )
