"""Load the rag-mini-bioasq corpus and golden QA data from Hugging Face."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, load_dataset

from src.config import DATASET, DatasetConfig
from src.data.preprocessing import Passage, parse_relevant_passage_ids, prepare_corpus, preserve_id


@dataclass(frozen=True)
class GoldenQA:
    """One question-answer example with original relevant passage IDs."""

    question_id: Any
    question: str
    answer: str
    relevant_passage_ids: tuple[Any, ...]


def load_corpus_dataset(
    config: DatasetConfig | None = None,
    split: str | None = None,
) -> Dataset:
    """Load the unmodified passage corpus from Hugging Face."""

    cfg = config or DATASET
    return load_dataset(cfg.name, cfg.corpus_config, split=split or cfg.corpus_split)


def load_qa_dataset(
    config: DatasetConfig | None = None,
    split: str | None = None,
) -> Dataset:
    """Load the unmodified question-answer/golden split from Hugging Face."""

    cfg = config or DATASET
    return load_dataset(cfg.name, cfg.qa_config, split=split or cfg.qa_split)


def get_passage_collection(
    config: DatasetConfig | None = None,
    split: str | None = None,
    records: Any | None = None,
) -> list[Passage]:
    """Return a retrieval-ready passage collection with original IDs.

    Does not modify the source dataset. Each item keeps `passage_id` and `text`.
    """

    cfg = config or DATASET
    if records is None:
        records = load_corpus_dataset(config=cfg, split=split)
    return prepare_corpus(
        records,
        id_field=cfg.passage_id_field,
        text_field=cfg.passage_text_field,
    )


def load_golden_qa(
    config: DatasetConfig | None = None,
    split: str | None = None,
    records: Any | None = None,
) -> list[GoldenQA]:
    """Return golden QA records with original relevant passage IDs preserved."""

    cfg = config or DATASET
    if records is None:
        records = load_qa_dataset(config=cfg, split=split)
    return [_to_golden_qa(record, cfg) for record in records]


def _to_golden_qa(record: Mapping[str, Any], cfg: DatasetConfig) -> GoldenQA:
    return GoldenQA(
        question_id=preserve_id(record[cfg.question_id_field]),
        question=record[cfg.question_field],
        answer=record[cfg.answer_field],
        relevant_passage_ids=tuple(
            parse_relevant_passage_ids(record[cfg.relevant_passage_ids_field])
        ),
    )
