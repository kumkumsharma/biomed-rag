"""Query expansion pattern.

This example shows the production interface used by
`src/generation/query_expansion.py`.

The original query is retained. A small set of biomedical search
alternates is generated, then BM25 and dense retrieval run over the
original plus expansions. Existing Hybrid RRF fuses those rankings.

When to refer to this example:
- implementing or modifying query expansion
- wiring expansions into HybridRetriever.search
- disabling expansion without changing BM25/dense/hybrid

This example uses a fake LLM. Production expansion uses the configured
chat model. Final answers are still generated only from selected evidence.

Production code must still follow `PRD.md` and `architecture.md`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocessing import prepare_corpus
from src.generation.query_expansion import ExpandingHybridRetriever, QueryExpander
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever


class FakeLLM:
    def complete(self, prompt: str) -> str:
        return '{"expansions": ["papilin extracellular matrix protein"]}'


def main() -> None:
    passages = prepare_corpus(
        [
            {"id": 1001, "passage": "Papilin is a secreted extracellular matrix protein."},
            {"id": 42, "passage": "Insulin regulates blood glucose."},
        ]
    )
    hybrid = HybridRetriever(
        passages,
        bm25_retriever=BM25Retriever(passages),
        dense_retriever=DenseRetriever(passages, embedder=_KeywordEmbedder()),
    )
    retriever = ExpandingHybridRetriever(
        hybrid,
        QueryExpander(llm=FakeLLM(), enabled=True, count=3),
    )
    original = "Is papilin secreted?"
    results = retriever.search(original, top_k=2)
    expansion = retriever.last_expansion
    print(
        {
            "original": expansion.original if expansion else original,
            "expansions": list(expansion.expansions) if expansion else [],
            "hits": [result.as_dict() for result in results],
        }
    )


class _KeywordEmbedder:
    def encode(self, texts):
        import numpy as np

        vocab = ("papilin", "secreted", "insulin", "glucose")
        return np.asarray(
            [[1.0 if token in text.lower() else 0.0 for token in vocab] for text in texts],
            dtype=float,
        )


if __name__ == "__main__":
    main()
