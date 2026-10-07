"""Grounded answer generation pattern.

This example shows the production interface used by
`src/generation/answer_generator.py` and `src/generation/rag.py`.

It demonstrates:
- answering only from supplied evidence
- citing passage IDs as [PASSAGE_ID: <id>]
- deterministic citation validation, not LLM self-checks
- skipping the LLM when retrieval returns no evidence

When to refer to this example:
- implementing or modifying grounded answer generation
- wiring HybridRetriever to AnswerGenerator and CitationValidator
- checking insufficient-evidence behaviour

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
from src.generation.answer_generator import AnswerGenerator
from src.generation.rag import GroundedQA
from src.retrieval.hybrid import HybridRetriever
from src.validation.citations import validate_citations


class FakeLLM:
    def complete(self, prompt: str) -> str:
        return "Papilin is a secreted protein. [PASSAGE_ID: 1001]"


class FakeRetriever:
    def __init__(self, ranking):
        self.ranking = ranking

    def search(self, query: str, top_k: int | None = None):
        return list(self.ranking[:top_k]) if top_k is not None else list(self.ranking)


def main() -> None:
    passages = prepare_corpus(
        [
            {"id": 1001, "passage": "Papilin is a secreted extracellular matrix protein."},
            {"id": 42, "passage": "Insulin regulates blood glucose."},
        ]
    )
    hit = type("Hit", (), {"passage_id": 1001, "score": 1.0, "text": passages[0].text})()
    hybrid = HybridRetriever(
        passages,
        bm25_retriever=FakeRetriever([hit]),
        dense_retriever=FakeRetriever([hit]),
    )
    qa = GroundedQA(hybrid, generator=AnswerGenerator(llm=FakeLLM()))
    result = qa.ask("Is papilin secreted?")
    citations = validate_citations(
        result.answer,
        [item.passage_id for item in result.evidence],
    )
    print(
        {
            "answer": result.answer,
            "insufficient_evidence": result.insufficient_evidence,
            "valid_ids": citations.valid_ids,
            "invalid_ids": citations.invalid_ids,
        }
    )


if __name__ == "__main__":
    main()
