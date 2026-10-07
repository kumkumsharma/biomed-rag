"""Validate evaluation gold IDs against the indexed corpus. Does not edit labels."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.data.preprocessing import Passage
from src.data.quality import classify_text, is_usable_text
from src.evaluation.dataset import EvaluationQuery
from src.evaluation.metrics import passage_id_key


def gold_id_status(
    gold_id: Any,
    by_id: Mapping[str, Passage],
) -> dict[str, Any]:
    key = passage_id_key(gold_id)
    passage = by_id.get(key)
    if passage is None:
        return {
            "gold_id": gold_id,
            "exists": False,
            "usable_text": False,
            "literal_nan": False,
            "empty": False,
            "whitespace_only": False,
            "extremely_short": False,
            "text_status": "missing_id",
            "text_length": None,
            "valid": False,
        }
    status = classify_text(passage.text)
    return {
        "gold_id": gold_id,
        "exists": True,
        "usable_text": status == "usable",
        "literal_nan": status == "nan_text",
        "empty": status == "empty_text",
        "whitespace_only": status == "whitespace_only",
        "extremely_short": status == "extremely_short",
        "text_status": status,
        "text_length": len(passage.text or ""),
        "valid": is_usable_text(passage.text),
    }


def evaluate_gold_quality(
    queries: Sequence[EvaluationQuery],
    passages: Sequence[Passage],
) -> dict[str, Any]:
    by_id = {passage_id_key(item.passage_id): item for item in passages}
    gold_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    for query in queries:
        statuses = [gold_id_status(gold_id, by_id) for gold_id in query.relevant_passage_ids]
        gold_rows.extend(statuses)
        n_invalid = sum(1 for item in statuses if not item["valid"])
        query_rows.append(
            {
                "query_id": query.question_id,
                "question": query.question,
                "n_gold": len(statuses),
                "n_valid_gold": len(statuses) - n_invalid,
                "n_invalid_gold": n_invalid,
                "all_gold_invalid": bool(statuses) and n_invalid == len(statuses),
                "has_invalid_gold": n_invalid > 0,
            }
        )
    n_gold = len(gold_rows)
    n_valid = sum(1 for item in gold_rows if item["valid"])
    n_invalid = n_gold - n_valid
    return {
        "total_gold_ids": n_gold,
        "valid_gold_ids": n_valid,
        "invalid_gold_ids": n_invalid,
        "queries_with_at_least_one_invalid_gold": sum(1 for item in query_rows if item["has_invalid_gold"]),
        "queries_where_all_gold_ids_are_invalid": sum(1 for item in query_rows if item["all_gold_invalid"]),
        "n_queries": len(query_rows),
        "literal_nan_gold_ids": sum(1 for item in gold_rows if item["literal_nan"]),
        "missing_gold_ids": sum(1 for item in gold_rows if not item["exists"]),
        "empty_gold_ids": sum(1 for item in gold_rows if item["empty"]),
        "historical_metrics_unchanged": True,
        "note": (
            "Invalid gold IDs are flagged only. The raw/original 100-query benchmark "
            "still uses every labelled ID."
        ),
        "queries": query_rows,
        "gold_sample": [item for item in gold_rows if not item["valid"]][:80],
    }


def write_gold_quality_report(report: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(report), indent=2, default=str) + "\n", encoding="utf-8")
    return output
