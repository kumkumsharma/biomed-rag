"""Fixed 100-query evaluation set from the BioASQ golden QA split.

Selection is deterministic: questions with at least one relevant passage ID,
sorted by numeric question ID, first 100. The chosen IDs are saved so later
runs reuse the same queries.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.config import DATASET, EVALUATION
from src.data.loader import GoldenQA, load_golden_qa


SELECTION_METHOD = "first_100_nonempty_relevant_sorted_by_numeric_question_id"


@dataclass(frozen=True)
class EvaluationQuery:
    """One frozen evaluation question with original IDs preserved."""

    question_id: Any
    question: str
    relevant_passage_ids: tuple[Any, ...]


def _sort_key(record: GoldenQA | EvaluationQuery) -> tuple[int, int | str, str]:
    question_id = record.question_id
    if isinstance(question_id, bool):
        return (1, str(question_id), record.question)
    if isinstance(question_id, int):
        return (0, question_id, record.question)
    try:
        return (0, int(str(question_id).strip()), record.question)
    except (TypeError, ValueError):
        return (1, str(question_id), record.question)


def select_fixed_queries(
    records: Sequence[GoldenQA],
    *,
    n: int = 100,
) -> list[EvaluationQuery]:
    """Select n queries deterministically from golden QA records."""

    if n < 1:
        raise ValueError("n must be >= 1")
    eligible = [record for record in records if record.relevant_passage_ids]
    eligible.sort(key=_sort_key)
    if len(eligible) < n:
        raise ValueError(
            f"Need {n} questions with relevant passage IDs, found {len(eligible)}"
        )
    selected = eligible[:n]
    return [
        EvaluationQuery(
            question_id=item.question_id,
            question=item.question,
            relevant_passage_ids=tuple(item.relevant_passage_ids),
        )
        for item in selected
    ]


def evaluation_queries_to_json(queries: Sequence[EvaluationQuery]) -> dict[str, Any]:
    return {
        "dataset": DATASET.name,
        "qa_config": DATASET.qa_config,
        "qa_split": DATASET.qa_split,
        "n_queries": len(queries),
        "selection_method": SELECTION_METHOD,
        "query_ids": [query.question_id for query in queries],
        "queries": [
            {
                "question_id": query.question_id,
                "question": query.question,
                "relevant_passage_ids": list(query.relevant_passage_ids),
            }
            for query in queries
        ],
    }


def save_evaluation_queries(queries: Sequence[EvaluationQuery], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = evaluation_queries_to_json(queries)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def load_evaluation_queries(path: str | Path) -> list[EvaluationQuery]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    queries = []
    for item in payload["queries"]:
        queries.append(
            EvaluationQuery(
                question_id=item["question_id"],
                question=item["question"],
                relevant_passage_ids=tuple(item["relevant_passage_ids"]),
            )
        )
    return queries


def load_or_create_fixed_queries(
    *,
    n: int | None = None,
    path: str | Path | None = None,
    records: Sequence[GoldenQA] | None = None,
) -> list[EvaluationQuery]:
    """Load the saved 100-query set, or select and save it once."""

    n = EVALUATION.n_queries if n is None else n
    destination = Path(path or EVALUATION.queries_path)
    if destination.exists():
        queries = load_evaluation_queries(destination)
        if len(queries) != n:
            raise ValueError(
                f"{destination} has {len(queries)} queries, expected {n}"
            )
        return queries

    golden = list(records) if records is not None else load_golden_qa()
    queries = select_fixed_queries(golden, n=n)
    save_evaluation_queries(queries, destination)
    return queries
