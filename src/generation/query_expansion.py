"""Optional LLM query expansion for biomedical search.

The original query is never rewritten. The expander only proposes a small
set of alternate search queries. Retrieval still uses the original query
together with those expansions.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.config import GENERATION
from src.generation.answer_generator import LLMClient, OpenAICompatibleClient

EXPANSION_PROMPT = """You generate alternative biomedical search queries.

Keep the original query unchanged. Do not answer the question.
Propose {count} short, search-oriented paraphrases or closely related
biomedical phrasings that could retrieve relevant passages.

Return JSON only:
{{"expansions": ["query 1", "query 2", "query 3"]}}

Original query:
{query}
"""


@dataclass(frozen=True)
class ExpandedQuery:
    """Original query plus alternate search strings. Original is unchanged."""

    original: str
    expansions: tuple[str, ...]
    llm_called: bool
    prompt: str

    @property
    def all_queries(self) -> tuple[str, ...]:
        return (self.original, *self.expansions)


def combine_original_and_expansions(
    original: str,
    expansions: Sequence[str] | None = None,
) -> list[str]:
    """Return original first, then unique non-empty expansions."""

    if not isinstance(original, str):
        raise TypeError("original query must be a string")
    queries = [original]
    seen = {original.strip().lower()}
    if not expansions:
        return queries
    for item in expansions:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if not stripped:
            continue
        key = stripped.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(stripped)
    return queries


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None


def parse_expansion_response(
    text: str,
    *,
    original: str,
    count: int,
) -> tuple[str, ...]:
    payload = _extract_json_object(text or "")
    if payload is None:
        return ()
    raw = payload.get("expansions", [])
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return ()
    alternates = combine_original_and_expansions(original, [str(item) for item in raw])[1:]
    return tuple(alternates[:count])


def build_expansion_prompt(query: str, *, count: int) -> str:
    return EXPANSION_PROMPT.replace("{count}", str(count)).replace("{query}", query)


class QueryExpander:
    """Generate alternate search queries. Disable to skip the LLM."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        *,
        enabled: bool | None = None,
        count: int | None = None,
        model_name: str | None = None,
    ) -> None:
        self._llm = llm
        self.enabled = GENERATION.query_expansion_enabled if enabled is None else enabled
        self.count = GENERATION.query_expansion_count if count is None else count
        self.model_name = model_name or GENERATION.query_expansion_model_name

    def expand(self, query: str) -> ExpandedQuery:
        """Return the original query plus a small set of alternates."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")
        original = query
        if not self.enabled or self.count < 1 or not original.strip():
            return ExpandedQuery(original=original, expansions=(), llm_called=False, prompt="")

        prompt = build_expansion_prompt(original, count=self.count)
        raw = self._client().complete(prompt)
        expansions = parse_expansion_response(raw, original=original, count=self.count)
        return ExpandedQuery(
            original=original,
            expansions=expansions,
            llm_called=True,
            prompt=prompt,
        )

    def _client(self) -> LLMClient:
        if self._llm is None:
            self._llm = OpenAICompatibleClient(
                model_name=self.model_name,
                temperature=GENERATION.temperature,
                max_tokens=200,
            )
        return self._llm


class ExpandingHybridRetriever:
    """Run Hybrid RRF over the original query plus expansions.

    BM25 and dense indexes are reused from the injected HybridRetriever.
    """

    def __init__(
        self,
        hybrid,
        expander: QueryExpander,
        *,
        estimated_cost_per_call_usd: float | None = None,
    ) -> None:
        if hybrid is None:
            raise ValueError("ExpandingHybridRetriever requires a HybridRetriever")
        if expander is None:
            raise ValueError("ExpandingHybridRetriever requires a QueryExpander")
        self._hybrid = hybrid
        self._expander = expander
        self.estimated_cost_per_call_usd = (
            GENERATION.query_expansion_estimated_cost_per_call_usd
            if estimated_cost_per_call_usd is None
            else estimated_cost_per_call_usd
        )
        self.last_expansion: ExpandedQuery | None = None
        self.last_llm_calls = 0
        self.last_cost_usd = 0.0
        self.last_trace: dict[str, Any] = {}
        self.last_expansion_ms = 0.0

    def search(self, query: str, top_k: int | None = None):
        started = time.perf_counter()
        expansion = self._expander.expand(query)
        self.last_expansion_ms = (time.perf_counter() - started) * 1000.0
        self.last_expansion = expansion
        self.last_llm_calls = 1 if expansion.llm_called else 0
        self.last_cost_usd = (
            self.estimated_cost_per_call_usd if expansion.llm_called else 0.0
        )
        hits = self._hybrid.search(
            expansion.original,
            top_k=top_k,
            expansions=expansion.expansions,
        )
        inner = dict(getattr(self._hybrid, "last_trace", {}) or {})
        inner["expansion_ms"] = self.last_expansion_ms
        inner["n_expansions"] = len(expansion.expansions)
        self.last_trace = inner
        return hits
