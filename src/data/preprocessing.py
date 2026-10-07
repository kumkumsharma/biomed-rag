"""Prepare passages for later retrieval without changing source IDs or text."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Passage:
    """One corpus passage with its original identifier."""

    passage_id: Any
    text: str


def preserve_id(value: Any) -> Any:
    """Return the original passage ID value without remapping it.

    Numpy scalar IDs are converted to their Python equivalents so callers get
    a stable Python value with the same numeric/string identity.
    """

    if isinstance(value, (str, bytes, bool)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, AttributeError):
            return value
    return value


def parse_relevant_passage_ids(value: Any) -> list[Any]:
    """Normalize relevant passage IDs while preserving each original ID.

    The Hugging Face QA split may store IDs as a list, a sequence, or a
    string representation of a list. This does not rewrite IDs.
    """

    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return []
        value = ast.literal_eval(stripped)
    if isinstance(value, Mapping):
        raise TypeError("relevant passage IDs must be a sequence, not a mapping")
    if isinstance(value, (list, tuple)):
        return [preserve_id(item) for item in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return [preserve_id(item) for item in tolist()]
    return [preserve_id(value)]


def prepare_passage(
    record: Mapping[str, Any],
    *,
    id_field: str = "id",
    text_field: str = "passage",
) -> Passage:
    """Build a retrieval-ready passage from a source record.

    The original passage ID and passage text are preserved exactly.
    """

    if id_field not in record:
        raise KeyError(f"passage record is missing ID field {id_field!r}")
    if text_field not in record:
        raise KeyError(f"passage record is missing text field {text_field!r}")

    text = record[text_field]
    if text is None:
        text = ""

    return Passage(passage_id=preserve_id(record[id_field]), text=str(text) if not isinstance(text, str) else text)


def prepare_corpus(
    records: Iterable[Mapping[str, Any]],
    *,
    id_field: str = "id",
    text_field: str = "passage",
) -> list[Passage]:
    """Prepare a passage collection for BM25 and dense retrieval."""

    return [
        prepare_passage(record, id_field=id_field, text_field=text_field)
        for record in records
    ]
