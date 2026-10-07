"""BM25 lexical retrieval.

BM25 is the lexical retriever for this project. It is independently callable
and preserves original passage IDs from the prepared corpus.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from src.config import RETRIEVAL
from src.data.preprocessing import Passage

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class RetrievalResult:
    """One BM25 hit with the original passage ID, score, and text."""

    passage_id: Any
    score: float
    text: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer for BM25 indexing and queries."""

    if not text:
        return []
    return [token.lower() for token in _TOKEN_RE.findall(text)]


class BM25Retriever:
    """Lexical retriever over a prepared passage collection."""

    def __init__(
        self,
        passages: Sequence[Passage],
        *,
        top_k: int | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if passages is None:
            raise ValueError("Cannot build a BM25 index without a passage collection")
        passages = list(passages)
        if not passages:
            raise ValueError("Cannot build a BM25 index from an empty passage collection")

        self._passages = passages
        self.top_k = RETRIEVAL.bm25_top_k if top_k is None else top_k
        self._bm25 = BM25Okapi(
            [tokenize(passage.text) for passage in passages],
            k1=k1,
            b=b,
        )

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Return top-k passages for a text query, highest BM25 score first."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        limit = self.top_k if top_k is None else top_k
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = np.asarray(self._bm25.get_scores(tokens), dtype=float)
        ranked = np.argsort(-scores, kind="mergesort")[: min(limit, len(self._passages))]
        return [
            RetrievalResult(
                passage_id=self._passages[int(index)].passage_id,
                score=float(scores[int(index)]),
                text=self._passages[int(index)].text,
            )
            for index in ranked
        ]
