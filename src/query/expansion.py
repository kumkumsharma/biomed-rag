"""Deterministic local query expansion. LLM expansion stays optional and off."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.query.terminology import TERM_MAP, detect_terms, normalize_query


@dataclass(frozen=True)
class DeterministicExpansion:
    original: str
    normalized: str
    expansions: tuple[str, ...]
    method: str
    llm_called: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "normalized": self.normalized,
            "expansions": list(self.expansions),
            "method": self.method,
            "llm_called": self.llm_called,
        }


def expand_query(
    query: str,
    *,
    enabled: bool = True,
    max_variants: int = 4,
    normalized: str | None = None,
) -> DeterministicExpansion:
    """Return unique search variants without concatenating them into one string."""

    if not isinstance(query, str):
        raise TypeError("query must be a string")
    terms = normalize_query(query, enabled=True)
    normalized_query = normalized if normalized is not None else terms.normalized
    if not enabled or not query.strip():
        return DeterministicExpansion(
            original=query,
            normalized=normalized_query,
            expansions=(),
            method="disabled",
        )
    variants: list[str] = []
    seen = {query.strip().lower()}
    if normalized_query.strip() and normalized_query.strip().lower() not in seen:
        variants.append(normalized_query.strip())
        seen.add(normalized_query.strip().lower())
    for surface in detect_terms(query):
        for synonym in TERM_MAP.get(surface) or ():
            pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(surface)}(?![A-Za-z0-9])", re.IGNORECASE)
            match = pattern.search(query)
            if match:
                candidate = query[: match.start()] + synonym + query[match.end() :]
            else:
                candidate = query
            if candidate.strip().lower() not in seen:
                variants.append(candidate.strip())
                seen.add(candidate.strip().lower())
            if len(variants) >= max_variants:
                break
        if len(variants) >= max_variants:
            break
    return DeterministicExpansion(
        original=query,
        normalized=normalized_query,
        expansions=tuple(variants[:max_variants]),
        method="local_synonym_substitution",
    )
