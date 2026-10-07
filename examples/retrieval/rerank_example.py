"""Cross-encoder reranking pattern.

This example shows the production interface used by
`src/retrieval/reranker.py`. Reranking is a second stage after Hybrid RRF.
It does not replace BM25, dense, or RRF, and it does not call OpenAI.

When to refer to this example:
- wrapping HybridRetriever with RerankingRetriever
- toggling baseline (RRF only) vs upgraded (RRF + MedCPT) mode
- inspecting reranker_score alongside existing RRF scores

Production code must still follow `PRD.md` and `architecture.md`.
If this example conflicts with those documents, follow the PRD and
architecture.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RerankerConfig
from src.data.preprocessing import prepare_corpus
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import wrap_retriever


class FakeScorer:
    """Stand-in so this example does not download MedCPT."""

    def score_pairs(self, query: str, texts):
        needle = query.lower()
        return [float(text.lower().count(needle.split()[0]) if text else 0) for text in texts]


def main() -> None:
    passages = prepare_corpus(
        [
            {
                "id": 20598273,
                "passage": "Hirschsprung disease is a congenital intestinal disorder.",
            },
            {
                "id": 42,
                "passage": "Insulin regulates blood glucose in diabetes mellitus.",
            },
        ]
    )
    bm25 = BM25Retriever(passages)
    dense = DenseRetriever(passages)
    hybrid = HybridRetriever(
        passages,
        bm25_retriever=bm25,
        dense_retriever=dense,
    )

    baseline = wrap_retriever(
        hybrid,
        config=RerankerConfig(enabled=False),
        scorer=FakeScorer(),
    )
    upgraded = wrap_retriever(
        hybrid,
        config=RerankerConfig(
            enabled=True,
            candidate_pool_size=10,
            final_top_k=2,
            model_name="ncbi/MedCPT-Cross-Encoder",
            device="cpu",
        ),
        scorer=FakeScorer(),
    )

    print("baseline (BM25 + Dense + RRF)")
    for result in baseline.search("Hirschsprung disease", top_k=2):
        print(result.as_dict())

    print("upgraded (BM25 + Dense + RRF + rerank)")
    for result in upgraded.search("Hirschsprung disease", top_k=2):
        print(result.as_dict())


if __name__ == "__main__":
    main()
