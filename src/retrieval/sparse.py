"""Learned sparse first-stage retrieval (SPLADE-style).

Independent of BM25. Production Hybrid does not use this module unless an
experiment explicitly constructs SparseRetriever. Default experimental flag is off.

The encoder sees only query text and passage text. It does not receive gold
IDs, relevance labels, or evaluation answers.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy import sparse as sp

from src.config import RETRIEVAL, UPGRADE
from src.data.preprocessing import Passage
from src.retrieval.embedding_cache import cache_file
from src.retrieval.sparse_models import DEFAULT_SPARSE_MODEL


@dataclass(frozen=True)
class RetrievalResult:
    """One sparse hit. Passage ID is preserved; pmid mirrors the corpus ID."""

    passage_id: Any
    score: float
    text: str
    pmid: Any = None
    rank: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SparseEmbedder(Protocol):
    """Maps texts to a scipy CSR matrix of shape (n_texts, vocab_size)."""

    vocab_size: int
    model_name: str

    def encode(self, texts: Sequence[str]) -> sp.csr_matrix: ...


def _as_csr(encoded: Any, n_texts: int) -> sp.csr_matrix:
    if sp.issparse(encoded):
        matrix = encoded.tocsr().astype(np.float32)
        if matrix.shape[0] != n_texts:
            raise ValueError(f"sparse embedder returned {matrix.shape[0]} rows for {n_texts} texts")
        return matrix
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]
    if torch is not None and hasattr(encoded, "is_sparse"):
        tensor = encoded
        if getattr(tensor, "is_sparse", False) or getattr(tensor, "is_sparse_csr", False):
            if getattr(tensor, "is_sparse_csr", False):
                tensor = tensor.to_sparse_coo()
            tensor = tensor.coalesce().cpu()
            indices = tensor.indices().numpy()
            values = tensor.values().detach().float().numpy()
            matrix = sp.coo_matrix(
                (values, (indices[0], indices[1])),
                shape=tuple(int(item) for item in tensor.shape),
                dtype=np.float32,
            ).tocsr()
        else:
            matrix = sp.csr_matrix(tensor.detach().cpu().numpy(), dtype=np.float32)
        if matrix.shape[0] != n_texts:
            raise ValueError(f"sparse embedder returned {matrix.shape[0]} rows for {n_texts} texts")
        return matrix
    array = np.asarray(encoded, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.shape[0] != n_texts:
        raise ValueError(f"sparse embedder returned {array.shape[0]} rows for {n_texts} texts")
    return sp.csr_matrix(array)


class SentenceTransformerSparseEmbedder:
    """Local SPLADE SparseEncoder. Isolated so tests can inject a fake embedder."""

    def __init__(self, model_name: str | None = None, *, max_length: int = 256, batch_size: int = 16) -> None:
        from sentence_transformers import SparseEncoder

        self.model_name = model_name or DEFAULT_SPARSE_MODEL
        self.batch_size = batch_size
        self._model = SparseEncoder(self.model_name)
        try:
            self._model.max_seq_length = max_length
        except Exception:  # noqa: BLE001
            pass
        dim = getattr(self._model, "sparse_embedding_dimension", None)
        if not dim:
            getter = getattr(self._model, "get_embedding_dimension", None) or getattr(
                self._model, "get_sentence_embedding_dimension", None
            )
            dim = getter() if callable(getter) else 30522
        self.vocab_size = int(dim or 30522)

    def encode(self, texts: Sequence[str]) -> sp.csr_matrix:
        encoded = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_tensor=True,
            convert_to_sparse_tensor=True,
            show_progress_bar=False,
        )
        return _as_csr(encoded, n_texts=len(texts))


def _load_or_encode(
    passages: Sequence[Passage],
    *,
    embedder: SparseEmbedder,
    model_name: str,
    cache_dir: str | None,
    batch_size: int,
) -> sp.csr_matrix:
    n_texts = len(passages)
    path = None
    if cache_dir:
        path = cache_file(cache_dir, "sparse::" + model_name, [passage.passage_id for passage in passages])
        csr_path = path.with_suffix(".csr.npz")
        if csr_path.exists():
            loaded = sp.load_npz(csr_path)
            if loaded.shape[0] == n_texts:
                return loaded.tocsr()
    chunks = []
    texts = [passage.text for passage in passages]
    for start in range(0, n_texts, max(1, batch_size)):
        chunk = list(texts[start : start + batch_size])
        chunks.append(_as_csr(embedder.encode(chunk), n_texts=len(chunk)))
        if start == 0 or (start // batch_size) % 20 == 0:
            print(f"  sparse encode {min(start + batch_size, n_texts)}/{n_texts}", flush=True)
    matrix = sp.vstack(chunks).tocsr() if chunks else sp.csr_matrix((n_texts, getattr(embedder, "vocab_size", 1)))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        sp.save_npz(path.with_suffix(".csr.npz"), matrix)
        path.with_suffix(".json").write_text(
            json.dumps({"model_name": model_name, "shape": list(matrix.shape), "nnz": int(matrix.nnz)}),
            encoding="utf-8",
        )
    return matrix


class SparseRetriever:
    """In-memory learned-sparse retriever over a prepared passage collection."""

    def __init__(
        self,
        passages: Sequence[Passage],
        *,
        embedder: SparseEmbedder | None = None,
        model_name: str | None = None,
        top_k: int | None = None,
        cache_dir: str | None = None,
        batch_size: int = 16,
        max_length: int = 256,
    ) -> None:
        if passages is None:
            raise ValueError("Cannot build a sparse index without a passage collection")
        passages = list(passages)
        if not passages:
            raise ValueError("Cannot build a sparse index from an empty passage collection")

        self._passages = passages
        self.top_k = RETRIEVAL.bm25_top_k if top_k is None else top_k
        self.model_name = model_name or UPGRADE.sparse_model_name or DEFAULT_SPARSE_MODEL
        self._embedder = embedder or SentenceTransformerSparseEmbedder(
            self.model_name, max_length=max_length, batch_size=batch_size
        )
        if embedder is not None and model_name is None:
            self.model_name = getattr(embedder, "model_name", self.model_name)
        started = time.perf_counter()
        self._matrix = _load_or_encode(
            passages,
            embedder=self._embedder,
            model_name=self.model_name,
            cache_dir=cache_dir,
            batch_size=batch_size,
        )
        self.index_build_seconds = time.perf_counter() - started
        self.vocab_size = int(self._matrix.shape[1])
        self.nnz = int(self._matrix.nnz)
        self.last_trace: dict[str, Any] = {}

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        limit = self.top_k if top_k is None else top_k
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            self.last_trace = {"n_hits": 0, "model_name": self.model_name}
            return []
        if not query.strip():
            self.last_trace = {"n_hits": 0, "model_name": self.model_name, "empty_query": True}
            return []
        started = time.perf_counter()
        query_vec = _as_csr(self._embedder.encode([query]), n_texts=1)
        scores = np.asarray(self._matrix.dot(query_vec.T).todense(), dtype=float).reshape(-1)
        ranked = np.argsort(-scores, kind="mergesort")[: min(limit, len(self._passages))]
        results = []
        for rank, index in enumerate(ranked, start=1):
            passage = self._passages[int(index)]
            results.append(
                RetrievalResult(
                    passage_id=passage.passage_id,
                    score=float(scores[int(index)]),
                    text=passage.text,
                    pmid=passage.passage_id,
                    rank=rank,
                )
            )
        self.last_trace = {
            "model_name": self.model_name,
            "retrieval_type": "learned_sparse",
            "n_hits": len(results),
            "top_k": limit,
            "vocab_size": self.vocab_size,
            "nnz": self.nnz,
            "query_nnz": int(query_vec.nnz),
            "search_ms": (time.perf_counter() - started) * 1000.0,
            "index_build_seconds": self.index_build_seconds,
        }
        return results

    def index_nbytes(self) -> int:
        matrix = self._matrix.tocsr()
        return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
