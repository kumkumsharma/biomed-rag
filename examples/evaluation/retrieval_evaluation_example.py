"""Offline retrieval evaluation pattern.

This example shows the production interface used by
`src/evaluation/run_evaluation.py`. Every retrieval configuration is scored
on the same fixed query set. Metrics are Recall@5, Recall@10, MRR@10, and
nDCG@10, using each question's golden relevant_passage_ids.

When to refer to this example:
- implementing or modifying the offline evaluation runner
- checking that BM25, Dense, and Hybrid RRF use the same queries
- writing evaluation result files

This example uses a tiny synthetic corpus so it is fast to run. Production
evaluation uses the BioASQ golden QA split, a saved set of exactly 100
query IDs, and the full passage corpus. Do not evaluate against the UI
200-passage demo subset.

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
from src.evaluation.dataset import EvaluationQuery
from src.evaluation.run_evaluation import (
    build_evaluation_retrievers,
    evaluate_configurations,
    format_summary_table,
)
from src.retrieval.dense import SentenceTransformerEmbedder


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
            {
                "id": 1001,
                "passage": "Papilin is a secreted extracellular matrix protein.",
            },
        ]
    )
    queries = [
        EvaluationQuery(1, "Hirschsprung disease", (20598273,)),
        EvaluationQuery(2, "secreted papilin", (1001,)),
    ]
    retrievers, _timings = build_evaluation_retrievers(
        passages,
        top_k=10,
        embedder=SentenceTransformerEmbedder(),
    )
    _rows, summaries = evaluate_configurations(retrievers, queries, top_k=10)
    print(format_summary_table(summaries))
    print("Production: python -m src.evaluation.run_evaluation")


if __name__ == "__main__":
    main()
