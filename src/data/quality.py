"""Corpus data-quality classification. Does not silently delete documents.

Invalid rows are labelled and counted. Production indexing still receives the
original passage list unless a caller explicitly filters by ``usable``.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.data.preprocessing import Passage, preserve_id
from src.evaluation.metrics import passage_id_key

MIN_USABLE_CHARS = 20
PREPROCESSING_PROFILE = {
    "retrieval_unit": "Passage-level retrieval unit; no additional chunking.",
    "chunking": False,
    "chunk_size": None,
    "chunk_overlap": None,
    "passages_used_directly": True,
    "text_normalization": "None. Source passage text is stored as-is.",
    "whitespace_normalization": "None at index time. BM25 tokenizes with [A-Za-z0-9]+ lowercase.",
    "truncation": False,
    "filtering": "Classification only. Unusable rows are reported, not dropped from production.",
    "id_preservation": "Original passage IDs are preserved without remapping.",
}


def is_nan_value(value: Any) -> bool:
    """True for float NaN or the literal string 'nan' (any case, stripped)."""

    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        if value != value:  # numpy-style NaN
            return True
    except Exception:
        pass
    if isinstance(value, str) and value.strip().lower() == "nan":
        return True
    return False


def is_empty_text(value: Any) -> bool:
    return isinstance(value, str) and value == ""


def is_whitespace_only(value: Any) -> bool:
    return isinstance(value, str) and value != "" and not value.strip()


def is_extremely_short(value: Any, *, min_chars: int = MIN_USABLE_CHARS) -> bool:
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if not stripped or stripped.lower() == "nan":
        return False
    return len(stripped) < min_chars


def classify_text(value: Any, *, min_chars: int = MIN_USABLE_CHARS) -> str:
    """Return one text-status label for a raw or indexed passage body."""

    if value is None:
        return "missing_text"
    if is_nan_value(value):
        return "nan_text"
    if not isinstance(value, str):
        return "invalid_non_string_text"
    if is_empty_text(value):
        return "empty_text"
    if is_whitespace_only(value):
        return "whitespace_only"
    if is_extremely_short(value, min_chars=min_chars):
        return "extremely_short"
    return "usable"


def is_usable_text(value: Any, *, min_chars: int = MIN_USABLE_CHARS) -> bool:
    return classify_text(value, min_chars=min_chars) == "usable"


def _length_stats(lengths: Sequence[int]) -> dict[str, float | int | None]:
    if not lengths:
        return {
            "average_text_length": None,
            "median_text_length": None,
            "p95_text_length": None,
            "minimum_text_length": None,
            "maximum_text_length": None,
        }
    ordered = sorted(lengths)
    p95_index = min(len(ordered) - 1, max(0, int(math.ceil(0.95 * len(ordered)) - 1)))
    return {
        "average_text_length": float(statistics.fmean(ordered)),
        "median_text_length": float(statistics.median(ordered)),
        "p95_text_length": int(ordered[p95_index]),
        "minimum_text_length": int(ordered[0]),
        "maximum_text_length": int(ordered[-1]),
    }


@dataclass(frozen=True)
class DocumentIssue:
    passage_id: Any
    reasons: tuple[str, ...]
    text_length: int | None
    text_status: str


def inspect_documents(
    records: Iterable[Mapping[str, Any] | Passage],
    *,
    id_field: str = "id",
    text_field: str = "passage",
    min_chars: int = MIN_USABLE_CHARS,
) -> dict[str, Any]:
    """Classify every document. Original IDs are kept on rejected rows."""

    issues: list[DocumentIssue] = []
    seen_ids: dict[str, int] = {}
    text_to_ids: dict[str, list[Any]] = defaultdict(list)
    usable: list[Passage] = []
    all_passages: list[Passage] = []
    counts: Counter[str] = Counter()
    lengths: list[int] = []

    for index, record in enumerate(records):
        if isinstance(record, Passage):
            raw_id = record.passage_id
            raw_text = record.text
        else:
            raw_id = record.get(id_field) if id_field in record else None
            raw_text = record.get(text_field) if text_field in record else None
        reasons: list[str] = []
        if raw_id is None:
            reasons.append("missing_id")
            counts["missing_ids"] += 1
        id_key = passage_id_key(raw_id) if raw_id is not None else f"<missing:{index}>"
        if raw_id is not None:
            seen_ids[id_key] = seen_ids.get(id_key, 0) + 1
            if seen_ids[id_key] > 1:
                reasons.append("duplicate_id")
                counts["duplicate_ids"] += 1
        text_status = classify_text(raw_text, min_chars=min_chars)
        if text_status != "usable":
            reasons.append(text_status)
            counts[text_status] += 1
        text_str = raw_text if isinstance(raw_text, str) else ("" if raw_text is None else str(raw_text))
        length = len(text_str) if isinstance(raw_text, str) else None
        if isinstance(raw_text, str):
            lengths.append(len(raw_text))
            text_to_ids[raw_text].append(preserve_id(raw_id) if raw_id is not None else id_key)
        passage = Passage(
            passage_id=preserve_id(raw_id) if raw_id is not None else id_key,
            text=text_str if isinstance(raw_text, str) else "",
        )
        all_passages.append(passage)
        if reasons:
            issues.append(
                DocumentIssue(
                    passage_id=passage.passage_id,
                    reasons=tuple(reasons),
                    text_length=length,
                    text_status=text_status,
                )
            )
        if text_status == "usable" and "missing_id" not in reasons and "duplicate_id" not in reasons:
            usable.append(passage)
            counts["valid_documents"] += 1

    duplicate_text_groups = {
        text: ids for text, ids in text_to_ids.items() if len(ids) > 1 and text.strip() and text.strip().lower() != "nan"
    }
    counts["duplicate_texts"] = sum(len(ids) - 1 for ids in duplicate_text_groups.values())
    counts["total_documents"] = len(all_passages)
    rejected = counts["total_documents"] - counts["valid_documents"]
    report = {
        "total_documents": counts["total_documents"],
        "valid_documents": counts["valid_documents"],
        "missing_text": counts["missing_text"],
        "nan_text": counts["nan_text"],
        "empty_text": counts["empty_text"],
        "whitespace_only": counts["whitespace_only"],
        "extremely_short": counts["extremely_short"],
        "invalid_non_string_text": counts["invalid_non_string_text"],
        "missing_ids": counts["missing_ids"],
        "duplicate_ids": counts["duplicate_ids"],
        "duplicate_texts": counts["duplicate_texts"],
        "rejected_documents": rejected,
        "silently_deleted": False,
        "production_index_still_includes_invalid_rows": True,
        "min_usable_chars": min_chars,
        "preprocessing": dict(PREPROCESSING_PROFILE),
        "statistics": {
            "document_count": counts["total_documents"],
            "rejected_document_count": rejected,
            **_length_stats(lengths),
        },
        "rejected_sample": [
            {
                "passage_id": item.passage_id,
                "reasons": list(item.reasons),
                "text_status": item.text_status,
                "text_length": item.text_length,
            }
            for item in issues[:50]
        ],
        "n_rejected_rows_listed": min(50, len(issues)),
        "n_rejected_rows": len(issues),
        "n_duplicate_text_groups": len(duplicate_text_groups),
    }
    return {
        "report": report,
        "usable_passages": usable,
        "all_passages": all_passages,
        "issues": issues,
    }


def write_data_quality_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(report), indent=2) + "\n", encoding="utf-8")
    return output
