"""Dense and BM25 index alignment checks. Read-only."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.data.preprocessing import Passage
from src.evaluation.metrics import passage_id_key


def _nan_count(matrix: np.ndarray) -> int:
    if matrix.size == 0:
        return 0
    return int(np.isnan(matrix).sum())


def inspect_index_health(
    passages: Sequence[Passage],
    *,
    bm25: Any,
    dense: Any,
) -> dict[str, Any]:
    """Verify corpus → embeddings → BM25 index alignment."""

    n_passages = len(passages)
    embeddings = getattr(dense, "_embeddings", None)
    if embeddings is None:
        embedding_count = 0
        dimension = None
        nan_embeddings = None
        shape = None
    else:
        matrix = np.asarray(embeddings)
        shape = list(matrix.shape)
        embedding_count = int(matrix.shape[0]) if matrix.ndim >= 1 else 0
        dimension = int(matrix.shape[1]) if matrix.ndim >= 2 else None
        nan_embeddings = _nan_count(matrix)

    bm25_passages = list(getattr(bm25, "_passages", []) or [])
    dense_passages = list(getattr(dense, "_passages", []) or [])
    bm25_okapi = getattr(bm25, "_bm25", None)
    bm25_corpus_size = getattr(bm25_okapi, "corpus_size", None)
    if bm25_corpus_size is None and bm25_okapi is not None:
        doc_len = getattr(bm25_okapi, "doc_len", None)
        bm25_corpus_size = len(doc_len) if doc_len is not None else None

    passage_ids = [passage_id_key(item.passage_id) for item in passages]
    bm25_ids = [passage_id_key(item.passage_id) for item in bm25_passages]
    dense_ids = [passage_id_key(item.passage_id) for item in dense_passages]
    duplicate_ids = len(passage_ids) - len(set(passage_ids))
    aligned = (
        n_passages == len(bm25_passages) == len(dense_passages) == embedding_count
        and passage_ids == bm25_ids == dense_ids
        and (bm25_corpus_size is None or int(bm25_corpus_size) == n_passages)
        and (nan_embeddings in {0, None})
    )
    return {
        "corpus_document_count": n_passages,
        "bm25_document_count": len(bm25_passages),
        "bm25_corpus_size": None if bm25_corpus_size is None else int(bm25_corpus_size),
        "embedding_count": embedding_count,
        "embedding_dimension": dimension,
        "embedding_shape": shape,
        "missing_embeddings": max(0, n_passages - embedding_count),
        "nan_embeddings": nan_embeddings,
        "duplicate_ids": duplicate_ids,
        "id_embedding_alignment": passage_ids == dense_ids and n_passages == embedding_count,
        "id_bm25_alignment": passage_ids == bm25_ids and (
            bm25_corpus_size is None or int(bm25_corpus_size) == n_passages
        ),
        "dense_model_name": getattr(dense, "model_name", None),
        "aligned": aligned,
        "index_size": {
            "passages": n_passages,
            "bm25_docs": len(bm25_passages),
            "dense_docs": len(dense_passages),
            "embeddings": embedding_count,
        },
        "pipeline": "corpus → preprocessing (identity) → BM25 tokens + MiniLM embeddings → in-memory index",
    }


def write_index_health_report(report: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    return output
