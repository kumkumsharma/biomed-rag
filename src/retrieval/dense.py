"""Dense retrieval with a local sentence-transformer bi-encoder.

Dense retrieval is independently callable from BM25. It embeds passages and
queries with the same model and ranks by cosine similarity while preserving
original passage IDs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np

from src.config import RETRIEVAL
from src.data.preprocessing import Passage


@dataclass(frozen=True)
class RetrievalResult:
    """One dense hit with the original passage ID, similarity score, and text."""

    passage_id: Any
    score: float
    text: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Embedder(Protocol):
    """Embedding interface so the sentence-transformer call stays isolated."""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _embed_texts(embedder: Embedder, texts: Sequence[str], *, queries: bool) -> Any:
    """Use query/document encoders when the embedder is asymmetric (MedCPT)."""

    if queries:
        encode_queries = getattr(embedder, "encode_queries", None)
        if callable(encode_queries):
            return encode_queries(texts)
    else:
        encode_documents = getattr(embedder, "encode_documents", None)
        if callable(encode_documents):
            return encode_documents(texts)
    return embedder.encode(texts)


class SentenceTransformerEmbedder:
    """Local sentence-transformer encoder used for both passages and queries."""

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or RETRIEVAL.dense_model_name
        self._model = SentenceTransformer(self.model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=float,
        )


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def _as_embedding_matrix(encoded: Any, n_texts: int) -> np.ndarray:
    matrix = np.asarray(encoded, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.shape[0] != n_texts:
        raise ValueError(
            f"embedder returned {matrix.shape[0]} vectors for {n_texts} texts"
        )
    return _l2_normalize(matrix)


def _load_or_encode(
    passages: Sequence[Passage],
    *,
    embedder: Embedder,
    model_name: str,
    cache_dir: str | None,
) -> np.ndarray:
    from src.retrieval.embedding_cache import cache_file, load_cached_embeddings, save_cached_embeddings

    n_texts = len(passages)
    if cache_dir:
        path = cache_file(cache_dir, model_name, [passage.passage_id for passage in passages])
        cached = load_cached_embeddings(path)
        if cached is not None and cached.shape[0] == n_texts:
            return _l2_normalize(np.asarray(cached, dtype=float))
        matrix = _as_embedding_matrix(
            _embed_texts(embedder, [passage.text for passage in passages], queries=False),
            n_texts=n_texts,
        )
        save_cached_embeddings(path, matrix, model_name=model_name)
        return matrix
    return _as_embedding_matrix(
        _embed_texts(embedder, [passage.text for passage in passages], queries=False),
        n_texts=n_texts,
    )


class DenseRetriever:
    """In-memory dense retriever over a prepared passage collection."""

    def __init__(
        self,
        passages: Sequence[Passage],
        *,
        embedder: Embedder | None = None,
        model_name: str | None = None,
        top_k: int | None = None,
        cache_dir: str | None = None,
    ) -> None:
        if passages is None:
            raise ValueError("Cannot build a dense index without a passage collection")
        passages = list(passages)
        if not passages:
            raise ValueError("Cannot build a dense index from an empty passage collection")

        self._passages = passages
        self.top_k = RETRIEVAL.dense_top_k if top_k is None else top_k
        self.model_name = model_name or RETRIEVAL.dense_model_name
        self._embedder = embedder or SentenceTransformerEmbedder(self.model_name)
        self._embeddings = _load_or_encode(
            passages,
            embedder=self._embedder,
            model_name=self.model_name,
            cache_dir=cache_dir,
        )

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Return top-k passages for a text query, highest cosine similarity first."""

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        limit = self.top_k if top_k is None else top_k
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            return []
        if not query.strip():
            return []

        query_vector = _as_embedding_matrix(
            _embed_texts(self._embedder, [query], queries=True),
            n_texts=1,
        )
        scores = self._embeddings @ query_vector.T
        scores = np.asarray(scores, dtype=float).reshape(-1)
        ranked = np.argsort(-scores, kind="mergesort")[: min(limit, len(self._passages))]
        return [
            RetrievalResult(
                passage_id=self._passages[int(index)].passage_id,
                score=float(scores[int(index)]),
                text=self._passages[int(index)].text,
            )
            for index in ranked
        ]
