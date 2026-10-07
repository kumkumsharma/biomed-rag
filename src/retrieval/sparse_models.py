"""Learned sparse retrieval candidates. Not used in production unless flagged."""

from __future__ import annotations

from typing import Any

from src.retrieval.embedding_models import huggingface_model_cached

DEFAULT_SPARSE_MODEL = "naver/splade-v3-distilbert"

SPARSE_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "model": "naver/splade-v3-distilbert",
        "source": "Hugging Face / Naver (Sentence Transformers SparseEncoder)",
        "task": "Learned sparse first-stage retrieval (SPLADE)",
        "biomedical_relevance": (
            "Trained on MS MARCO, not BioASQ. Not domain-adapted. Chosen because it "
            "produces sparse term-weight vectors that can match latent terms BM25 never sees, "
            "and DistilBERT is small enough to encode 40k passages locally overnight."
        ),
        "local_inference": "CPU or CUDA; DistilBERT MLM (~67M params); no API key",
        "expected_output": "30522-d sparse vector (WordPiece vocab weights, ReLU + log saturation)",
        "produces_sparse_term_weights": True,
        "indexing": "CSR document-term matrix; reusable on-disk cache",
        "license": "CC BY-NC-SA 4.0 (Naver SPLADE-v3). Experimental/research use only.",
        "expected_memory_runtime": (
            "Index ~tens of MB sparse CSR for 40k docs. Encode time on CPU similar to "
            "or faster than BERT-base PubMedBERT (~minutes, not hours)."
        ),
        "preferred": True,
    },
    {
        "model": "naver/splade-v3",
        "source": "Hugging Face / Naver",
        "task": "Learned sparse first-stage retrieval (SPLADE-v3 BERT-base)",
        "biomedical_relevance": "MS MARCO, not BioASQ. Higher MS MARCO quality, slower CPU encode.",
        "local_inference": "BERT-base MLM (~110M); heavier than DistilBERT",
        "expected_output": "30522-d sparse term weights",
        "produces_sparse_term_weights": True,
        "indexing": "CSR cache",
        "license": "CC BY-NC-SA 4.0",
        "expected_memory_runtime": "Slower than DistilBERT; encode cost similar to PubMedBERT (~5–10 min CPU).",
        "preferred": False,
    },
    {
        "model": "naver/splade-cocondenser-ensembledistil",
        "source": "Hugging Face / Naver",
        "task": "SPLADE++ EnsembleDistil",
        "biomedical_relevance": "MS MARCO DistilBERT; widely used SPLADE checkpoint.",
        "local_inference": "DistilBERT; older checkpoint than v3-distilbert",
        "expected_output": "sparse term weights over WordPiece vocab",
        "produces_sparse_term_weights": True,
        "indexing": "CSR cache",
        "license": "CC BY-NC-SA 4.0",
        "expected_memory_runtime": "Similar to splade-v3-distilbert",
        "preferred": False,
    },
    {
        "model": "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
        "source": "Hugging Face / OpenSearch",
        "task": "Learned sparse document encoding (neural sparse)",
        "biomedical_relevance": "General web/MS MARCO style, not BioASQ. Query encoder is separate (doc-only weights).",
        "local_inference": "Would need a matching query encoder; more plumbing than SPLADE",
        "expected_output": "sparse term weights",
        "produces_sparse_term_weights": True,
        "indexing": "inverted index or CSR",
        "license": "Apache-2.0",
        "expected_memory_runtime": "Similar DistilBERT cost, but two-tower setup is extra integration risk overnight.",
        "preferred": False,
        "rejected_reason": "Asymmetric query/doc encoders add integration risk versus a single SPLADE encoder.",
    },
    {
        "model": "(none found) BioSPLADE / PubMed-adapted SPLADE",
        "source": "Literature search; no drop-in Sentence-Transformers SparseEncoder checkpoint",
        "task": "Domain-adapted learned sparse retrieval",
        "biomedical_relevance": "Would be preferable if a local, licensed, HF-loadable checkpoint existed.",
        "local_inference": "n/a",
        "expected_output": "n/a",
        "produces_sparse_term_weights": True,
        "indexing": "n/a",
        "license": "n/a",
        "expected_memory_runtime": "n/a",
        "preferred": False,
        "rejected_reason": "No practical local biomedical SPLADE checkpoint was available to evaluate overnight.",
    },
)


def try_resolve_sparse_model(*, prefer: str | None = None, allow_download: bool = True) -> dict[str, Any]:
    ordered: list[str] = []
    if prefer:
        ordered.append(prefer)
    for item in SPARSE_CANDIDATES:
        name = str(item["model"])
        if name not in ordered:
            ordered.append(name)
    checked = []
    for name in ordered:
        cached = huggingface_model_cached(name)
        checked.append({"model": name, "cached": cached})
        if cached:
            return {"model": name, "available": True, "source": "cache", "checked": checked}
        if not allow_download:
            continue
        try:
            from sentence_transformers import SparseEncoder

            print(f"Attempting to load sparse encoder {name}...", flush=True)
            SparseEncoder(name)
            return {"model": name, "available": True, "source": "download", "checked": checked}
        except Exception as exc:  # noqa: BLE001
            checked[-1]["error"] = str(exc)[:300]
            print(f"  unavailable: {name}: {exc}", flush=True)
    return {"model": None, "available": False, "source": None, "checked": checked}
