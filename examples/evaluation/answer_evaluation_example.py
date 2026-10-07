"""Final-answer evaluation pattern.

This example shows the production interface used by
`src/evaluation/answer_evaluation.py`.

Retrieval evaluation scores BM25 / Dense / Hybrid rankings. Final-answer
evaluation is different: Hybrid RRF retrieves evidence, GroundedQA writes
an answer, citations are checked deterministically, and a separate LLM
judge scores correctness, groundedness, and context relevance.

The judge does not validate citations. Citation validity is determined by
`src/validation/citations.py`.

When to refer to this example:
- implementing or modifying the LLM judge
- wiring GroundedQA into answer-quality evaluation
- writing the 20-query answer-evaluation sample

This example uses fake LLMs so it is safe to run without an API key.
Production evaluation uses Hybrid RRF over the full corpus, the first 20
queries of the fixed 100-query set, and real generation + judge models.

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
from src.evaluation.answer_evaluation import (
    AnswerEvaluationQuery,
    LLMJudge,
    evaluate_one_answer,
    summarize_answer_rows,
)
from src.generation.answer_generator import AnswerGenerator
from src.generation.rag import GroundedQA
from src.retrieval.hybrid import HybridRetriever


class FakeHit:
    def __init__(self, passage_id, score, text):
        self.passage_id = passage_id
        self.score = score
        self.text = text


class FakeRetriever:
    def __init__(self, ranking):
        self.ranking = ranking

    def search(self, query, top_k=None):
        return list(self.ranking[:top_k] if top_k is not None else self.ranking)


class FakeLLM:
    def __init__(self, response):
        self.response = response

    def complete(self, prompt):
        return self.response


def main() -> None:
    passages = prepare_corpus(
        [{"id": 1001, "passage": "Papilin is a secreted extracellular matrix protein."}]
    )
    hit = FakeHit(1001, 1.0, passages[0].text)
    hybrid = HybridRetriever(
        passages,
        bm25_retriever=FakeRetriever([hit]),
        dense_retriever=FakeRetriever([hit]),
    )
    qa = GroundedQA(
        hybrid,
        generator=AnswerGenerator(
            llm=FakeLLM("Papilin is secreted. [PASSAGE_ID: 1001]")
        ),
    )
    judge = LLMJudge(
        llm=FakeLLM(
            '{"correctness": 4, "groundedness": 5, "context_relevance": 5, "reason": "supported"}'
        )
    )
    query = AnswerEvaluationQuery(
        question_id=1,
        question="Is papilin secreted?",
        reference_answer="Yes, papilin is secreted.",
        relevant_passage_ids=(1001,),
    )
    row = evaluate_one_answer(query, qa, judge, top_k=1)
    print(summarize_answer_rows([row]))
    print("Production: python -m src.evaluation.answer_evaluation")


if __name__ == "__main__":
    main()
