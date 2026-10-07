"""Conservative candidate filter before expensive cross-encoder reranking."""

from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from src.config import UPGRADE, UpgradeConfig
from src.retrieval.bm25 import tokenize

_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class FilterDecision:
    passage_id: Any
    kept: bool
    reason: str


@dataclass(frozen=True)
class FilterResult:
    kept: tuple[Any, ...]
    removed: tuple[Any, ...]
    decisions: tuple[FilterDecision, ...]
    before_count: int
    after_count: int
    latency_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "before_count": self.before_count,
            "after_count": self.after_count,
            "removed_count": len(self.removed),
            "kept_passage_ids": [_jsonable(getattr(item, "passage_id", None)) for item in self.kept],
            "removed_passage_ids": [_jsonable(getattr(item, "passage_id", item)) for item in self.removed],
            "decisions": [
                {"passage_id": _jsonable(item.passage_id), "kept": item.kept, "reason": item.reason}
                for item in self.decisions
            ],
            "latency_ms": self.latency_ms,
        }


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (bytes, str, bool)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def _normalize_text(text: str) -> str:
    return _SPACE_RE.sub(" ", (text or "").strip().lower())


def _token_overlap(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    passage_tokens = set(tokenize(text))
    if not query_tokens or not passage_tokens:
        return 0.0
    return len(query_tokens & passage_tokens) / len(query_tokens)


def filter_candidates(
    query: str,
    candidates: Sequence[Any],
    *,
    config: UpgradeConfig | None = None,
    enabled: bool | None = None,
) -> FilterResult:
    """Drop empty/duplicate passages. Overlap filtering is off unless configured.

    Conservative by default so candidate recall is preserved.
    """

    cfg = config or UPGRADE
    use = cfg.filtering_enabled if enabled is None else enabled
    items = list(candidates)
    started = time.perf_counter()
    if not use:
        decisions = tuple(FilterDecision(getattr(item, "passage_id", None), True, "disabled") for item in items)
        latency = (time.perf_counter() - started) * 1000.0
        return FilterResult(tuple(items), (), decisions, len(items), len(items), latency)

    kept: list[Any] = []
    removed: list[Any] = []
    decisions: list[FilterDecision] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    for item in items:
        passage_id = getattr(item, "passage_id", None)
        text = getattr(item, "text", "") or ""
        pid_key = str(passage_id)
        text_key = hashlib.sha1(_normalize_text(text).encode("utf-8")).hexdigest()
        if cfg.filter_drop_empty and not text.strip():
            removed.append(item)
            decisions.append(FilterDecision(passage_id, False, "empty_text"))
            continue
        if pid_key in seen_ids:
            removed.append(item)
            decisions.append(FilterDecision(passage_id, False, "duplicate_id"))
            continue
        if cfg.filter_drop_duplicates and text_key in seen_text:
            removed.append(item)
            decisions.append(FilterDecision(passage_id, False, "duplicate_text"))
            continue
        if cfg.filter_min_token_overlap > 0 and _token_overlap(query, text) < cfg.filter_min_token_overlap:
            removed.append(item)
            decisions.append(FilterDecision(passage_id, False, "low_token_overlap"))
            continue
        seen_ids.add(pid_key)
        seen_text.add(text_key)
        kept.append(item)
        decisions.append(FilterDecision(passage_id, True, "kept"))
    latency = (time.perf_counter() - started) * 1000.0
    if not kept and items:
        # Never empty the pool entirely; that would hide all evidence.
        first = items[0]
        kept = [first]
        decisions = tuple(
            FilterDecision(getattr(item, "passage_id", None), item is first, "recall_guard" if item is first else "dropped_then_restored")
            for item in items
        )
        removed = [item for item in items if item is not first]
    return FilterResult(tuple(kept), tuple(removed), tuple(decisions), len(items), len(kept), latency)
