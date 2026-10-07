"""Hybrid retrieval pattern.

This example shows the production interface used by
`src/retrieval/hybrid.py`. Hybrid retrieval combines BM25 (lexical) and
dense (embedding) rankings with Reciprocal Rank Fusion. It is not a
second lexical retriever.

When to refer to this example:
- implementing or modifying hybrid retrieval
- wiring HybridRetriever to existing BM25 and Dense retrievers
- checking RRF fusion and the result contract: passage_id, score, text

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

from src.data.preprocessing import prepare_corpus
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever


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
    retriever = HybridRetriever(
        passages,
        bm25_retriever=bm25,
        dense_retriever=dense,
    )
    results = retriever.search("Hirschsprung disease", top_k=2)

    for result in results:
        print(result.as_dict())


if __name__ == "__main__":
    main()
