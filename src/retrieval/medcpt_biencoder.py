"""MedCPT Query + Article bi-encoder. Experimental first-stage dense swap.

Not the production MiniLM encoder. Queries use ncbi/MedCPT-Query-Encoder;
passages use ncbi/MedCPT-Article-Encoder. The two towers are not interchangeable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.retrieval.reranker import resolve_device

MEDCPT_QUERY_ENCODER = "ncbi/MedCPT-Query-Encoder"
MEDCPT_ARTICLE_ENCODER = "ncbi/MedCPT-Article-Encoder"
MEDCPT_QUERY_MAX_LENGTH = 64
MEDCPT_ARTICLE_MAX_LENGTH = 512


class MedCPTBiEncoder:
    """Asymmetric MedCPT embedder. Isolated so tests can inject a fake."""

    def __init__(
        self,
        *,
        query_model_name: str = MEDCPT_QUERY_ENCODER,
        article_model_name: str = MEDCPT_ARTICLE_ENCODER,
        device: str | None = None,
        batch_size: int = 16,
        query_max_length: int = MEDCPT_QUERY_MAX_LENGTH,
        article_max_length: int = MEDCPT_ARTICLE_MAX_LENGTH,
    ) -> None:
        self.query_model_name = query_model_name
        self.article_model_name = article_model_name
        self.model_name = f"{query_model_name}+{article_model_name}"
        self.device = resolve_device(device)
        self.batch_size = max(1, int(batch_size))
        self.query_max_length = query_max_length
        self.article_max_length = article_max_length
        self._query: tuple[Any, Any] | None = None
        self._article: tuple[Any, Any] | None = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Symmetric fallback: documents. Prefer encode_queries / encode_documents."""

        return self.encode_documents(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        tokenizer, model = self._load_query()
        return self._encode(texts, tokenizer, model, max_length=self.query_max_length)

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        tokenizer, model = self._load_article()
        return self._encode(texts, tokenizer, model, max_length=self.article_max_length)

    def _load_query(self) -> tuple[Any, Any]:
        if self._query is None:
            self._query = self._load_tower(self.query_model_name)
        return self._query

    def _load_article(self) -> tuple[Any, Any]:
        if self._article is None:
            self._article = self._load_tower(self.article_model_name)
        return self._article

    def _load_tower(self, model_name: str) -> tuple[Any, Any]:
        from transformers import AutoModel, AutoTokenizer

        print(f"Loading MedCPT encoder {model_name} on {self.device}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(self.device)
        model.eval()
        return tokenizer, model

    def _encode(
        self,
        texts: Sequence[str],
        tokenizer: Any,
        model: Any,
        *,
        max_length: int,
    ) -> np.ndarray:
        import torch

        ordered = [text if isinstance(text, str) else str(text) for text in texts]
        if not ordered:
            hidden = int(getattr(model.config, "hidden_size", 768))
            return np.zeros((0, hidden), dtype=float)
        chunks: list[np.ndarray] = []
        total = len(ordered)
        for start in range(0, total, self.batch_size):
            batch = ordered[start : start + self.batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                cls = model(**encoded).last_hidden_state[:, 0, :]
            chunks.append(cls.detach().cpu().numpy().astype(float, copy=False))
            done = min(start + self.batch_size, total)
            if start == 0 or done == total or (start // self.batch_size) % 20 == 0:
                print(f"  medcpt encode {done}/{total}", flush=True)
        return np.vstack(chunks)
