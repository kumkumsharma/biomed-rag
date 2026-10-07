"""BM25 lexical retrieval pattern.

This example shows the production interface used by
`src/retrieval/bm25.py`. It is a reference for naming, result shape,
and independent BM25 usage. It is not a second retrieval system.

When to refer to this example:
- implementing or modifying BM25 / lexical retrieval
- wiring BM25 to the prepared passage collection
- checking the result contract: passage_id, score, text

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
    retriever = BM25Retriever(passages)
    results = retriever.search("Hirschsprung disease", top_k=2)

    for result in results:
        print(result.as_dict())


if __name__ == "__main__":
    main()
