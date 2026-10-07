"""Second-stage biomedical cross-encoder reranking after Hybrid RRF.

The first-stage retriever (typically Hybrid RRF) is unchanged. This module
reranks a candidate pool with a local Hugging Face cross-encoder
(`ncbi/MedCPT-Cross-Encoder` by default). It does not call OpenAI.

The model is loaded once per process and reused.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from src.config import RERANKER, RerankerConfig
from src.retrieval.hybrid import RetrievalResult

_SHARED_SCORERS: dict[tuple[str, str, int, int], "TransformersCrossEncoderScorer"] = {}


class PairScorer(Protocol):
    """Scores (query, passage) pairs. Injected in tests to avoid model downloads."""

    def score_pairs(self, query: str, texts: Sequence[str]) -> Sequence[float]: ...


@dataclass(frozen=True)
class RerankedResult(RetrievalResult):
    """Hybrid hit plus reranker metadata. Drop-in for RetrievalResult."""

    rrf_score: float | None = None
    reranker_score: float | None = None
    bm25_rank: int | None = None
    dense_rank: int | None = None
    rrf_rank: int | None = None
    final_rank: int | None = None
    pmid: Any = None


def resolve_device(device: str | None = None) -> str:
    """Return CUDA when available, otherwise CPU. Honors an explicit device."""

    requested = (device or "auto").strip().lower()
    if requested and requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def logits_to_scores(logits: Sequence[Any]) -> list[float]:
    """Convert cross-encoder logits to one relevance score per pair.

    MedCPT typically emits shape ``[batch, 1]``. Two-class classifiers use the
    positive (index 1) logit. A flat vector is used as-is.
    """

    rows = list(logits)
    if not rows:
        return []
    first = rows[0]
    if isinstance(first, (int, float)):
        return [float(item) for item in rows]
    scores: list[float] = []
    for row in rows:
        values = list(row)
        if not values:
            scores.append(0.0)
        elif len(values) == 1:
            scores.append(float(values[0]))
        else:
            scores.append(float(values[1]))
    return scores


class TransformersCrossEncoderScorer:
    """Local Hugging Face sequence-classification scorer. Loads the model once."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> None:
        config = RERANKER
        self.model_name = model_name or config.model_name
        self.device = resolve_device(device if device is not None else config.device)
        self.batch_size = config.batch_size if batch_size is None else batch_size
        self.max_length = config.max_length if max_length is None else max_length
        self._tokenizer: Any = None
        self._model: Any = None
        self._score_cache: dict[tuple[str, str], float] = {}

    def ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    def score_pairs(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        self.ensure_loaded()
        import torch

        query_text = query if isinstance(query, str) else str(query)
        ordered_texts = [text if isinstance(text, str) else str(text) for text in texts]
        scores: list[float | None] = [None] * len(ordered_texts)
        missing: list[int] = []
        for index, text in enumerate(ordered_texts):
            cached = self._score_cache.get((query_text, text))
            if cached is None:
                missing.append(index)
            else:
                scores[index] = cached

        batch_size = max(1, int(self.batch_size))
        for start in range(0, len(missing), batch_size):
            batch_indexes = missing[start : start + batch_size]
            batch = [ordered_texts[index] for index in batch_indexes]
            pairs = [(query_text, text) for text in batch]
            encoded = self._tokenizer(
                pairs,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = self._model(**encoded).logits.detach().cpu()
            nested = logits.tolist()
            if logits.ndim == 1:
                batch_scores = [float(item) for item in nested]
            else:
                batch_scores = logits_to_scores(nested)
            for index, score in zip(batch_indexes, batch_scores):
                scores[index] = float(score)
                if len(self._score_cache) < 4096:
                    self._score_cache[(query_text, ordered_texts[index])] = float(score)
        return [float(item) if item is not None else 0.0 for item in scores]


def get_shared_scorer(config: RerankerConfig | None = None) -> TransformersCrossEncoderScorer:
    """Reuse one Transformers scorer per (model, device, batch, max_length)."""

    cfg = config or RERANKER
    device = resolve_device(cfg.device)
    key = (cfg.model_name, device, int(cfg.batch_size), int(cfg.max_length))
    scorer = _SHARED_SCORERS.get(key)
    if scorer is None:
        scorer = TransformersCrossEncoderScorer(
            model_name=cfg.model_name,
            device=device,
            batch_size=cfg.batch_size,
            max_length=cfg.max_length,
        )
        _SHARED_SCORERS[key] = scorer
    return scorer


class CrossEncoderReranker:
    """Rerank retrieval candidates with a local cross-encoder."""

    def __init__(self, scorer: PairScorer) -> None:
        if scorer is None:
            raise ValueError("CrossEncoderReranker requires a pair scorer")
        self._scorer = scorer

    def preload(self) -> None:
        ensure = getattr(self._scorer, "ensure_loaded", None)
        if callable(ensure):
            ensure()

    def rerank(
        self,
        query: str,
        candidates: Sequence[Any],
        top_k: int | None = None,
        *,
        bm25_ranks: dict[Any, int] | None = None,
        dense_ranks: dict[Any, int] | None = None,
    ) -> list[RerankedResult]:
        """Return candidates ordered by cross-encoder score.

        ``top_k`` is the final cut. If it is larger than the candidate list,
        every candidate is returned. Ties keep the original (RRF) order.
        """

        items = list(candidates)
        if not items:
            return []
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        limit = len(items) if top_k is None else top_k
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            return []

        texts = [getattr(item, "text", "") for item in items]
        scores = list(self._scorer.score_pairs(query, texts))
        if len(scores) != len(items):
            raise ValueError("pair scorer must return one score per candidate")

        ranked_indexes = sorted(
            range(len(items)),
            key=lambda index: (-float(scores[index]), index),
        )[:limit]
        bm25_ranks = bm25_ranks or {}
        dense_ranks = dense_ranks or {}
        results: list[RerankedResult] = []
        for final_rank, index in enumerate(ranked_indexes, start=1):
            item = items[index]
            passage_id = getattr(item, "passage_id")
            rrf_score = float(getattr(item, "rrf_score", getattr(item, "score", 0.0)))
            pmid = getattr(item, "pmid", passage_id)
            results.append(
                RerankedResult(
                    passage_id=passage_id,
                    score=float(scores[index]),
                    text=getattr(item, "text", ""),
                    rrf_score=rrf_score,
                    reranker_score=float(scores[index]),
                    bm25_rank=_rank_lookup(bm25_ranks, passage_id, item, "bm25_rank"),
                    dense_rank=_rank_lookup(dense_ranks, passage_id, item, "dense_rank"),
                    rrf_rank=index + 1,
                    final_rank=final_rank,
                    pmid=pmid,
                )
            )
        return results


class RerankingRetriever:
    """Wrap a first-stage retriever with optional cross-encoder reranking.

    Disabled mode forwards ``search`` unchanged so the baseline BM25 + Dense +
    RRF pipeline remains available for A/B evaluation.
    """

    def __init__(
        self,
        retriever: Any,
        *,
        config: RerankerConfig | None = None,
        scorer: PairScorer | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        if retriever is None:
            raise ValueError("RerankingRetriever requires a first-stage retriever")
        self._retriever = retriever
        self.config = config or RERANKER
        if reranker is not None:
            self._reranker = reranker
        elif self.config.enabled:
            self._reranker = CrossEncoderReranker(scorer=scorer or get_shared_scorer(self.config))
        else:
            self._reranker = None
        self.last_trace: dict[str, Any] = {}

    @property
    def _dense(self) -> Any:
        dense = getattr(self._retriever, "_dense", None)
        if dense is not None:
            return dense
        nested = getattr(self._retriever, "_hybrid", None)
        return getattr(nested, "_dense", None)

    def preload(self) -> None:
        if self._reranker is not None:
            self._reranker.preload()

    def search(
        self,
        query: str,
        top_k: int | None = None,
        expansions: Sequence[str] | None = None,
    ) -> list[Any]:
        limit = self.config.final_top_k if top_k is None else top_k
        inner_top_k = self.config.candidate_pool_size if self.config.enabled else limit
        started = time.perf_counter()
        candidates = _search_inner(self._retriever, query, inner_top_k, expansions)
        inner_ms = (time.perf_counter() - started) * 1000.0
        inner_trace = _inner_trace(self._retriever)
        if not self.config.enabled or self._reranker is None:
            self.last_trace = _rerank_trace(
                inner_trace,
                enabled=False,
                config=self.config,
                inner_ms=inner_ms,
                candidate_count=len(candidates),
            )
            return list(candidates)

        started = time.perf_counter()
        ranked = self._reranker.rerank(
            query,
            candidates,
            top_k=limit,
            bm25_ranks=inner_trace.get("bm25_ranks") or {},
            dense_ranks=inner_trace.get("dense_ranks") or {},
        )
        rerank_ms = (time.perf_counter() - started) * 1000.0
        self.last_trace = _rerank_trace(
            inner_trace,
            enabled=True,
            config=self.config,
            inner_ms=inner_ms,
            rerank_ms=rerank_ms,
            candidate_count=len(candidates),
            returned=len(ranked),
        )
        return ranked


def wrap_retriever(
    retriever: Any,
    *,
    config: RerankerConfig | None = None,
    scorer: PairScorer | None = None,
) -> RerankingRetriever:
    """Always wrap so callers can toggle reranking without changing wiring."""

    return RerankingRetriever(retriever, config=config, scorer=scorer)


def _search_inner(
    retriever: Any,
    query: str,
    top_k: int | None,
    expansions: Sequence[str] | None,
) -> list[Any]:
    if expansions:
        try:
            return list(retriever.search(query, top_k=top_k, expansions=expansions))
        except TypeError:
            return list(retriever.search(query, top_k=top_k))
    return list(retriever.search(query, top_k=top_k))


def _inner_trace(retriever: Any) -> dict[str, Any]:
    trace = dict(getattr(retriever, "last_trace", {}) or {})
    if trace:
        return trace
    nested = getattr(retriever, "_hybrid", None)
    return dict(getattr(nested, "last_trace", {}) or {})


def _rank_lookup(
    ranks: dict[Any, int],
    passage_id: Any,
    item: Any,
    field: str,
) -> int | None:
    if passage_id in ranks:
        return int(ranks[passage_id])
    value = getattr(item, field, None)
    return int(value) if value is not None else None


def _rerank_trace(
    inner_trace: dict[str, Any],
    *,
    enabled: bool,
    config: RerankerConfig,
    inner_ms: float,
    candidate_count: int,
    rerank_ms: float | None = None,
    returned: int | None = None,
) -> dict[str, Any]:
    trace = dict(inner_trace)
    trace.update(
        {
            "reranker_enabled": enabled,
            "reranker_model_name": config.model_name if enabled else None,
            "reranker_device": resolve_device(config.device) if enabled else None,
            "reranker_candidate_pool_size": config.candidate_pool_size if enabled else None,
            "reranker_final_top_k": config.final_top_k,
            "rerank_ms": rerank_ms,
            "rerank_candidate_count": candidate_count,
            "rerank_returned": returned if enabled else candidate_count,
            "first_stage_ms": inner_ms,
        }
    )
    return trace
