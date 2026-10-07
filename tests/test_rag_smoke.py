from dataclasses import dataclass

from src.data.preprocessing import prepare_corpus
from src.generation.answer_generator import INSUFFICIENT_EVIDENCE_ANSWER, AnswerGenerator
from src.generation.rag import GroundedQA
from src.retrieval.hybrid import HybridRetriever
from src.validation.citations import format_citation


@dataclass(frozen=True)
class FakeHit:
    passage_id: object
    score: float
    text: str


class FakeRetriever:
    def __init__(self, ranking: list[FakeHit]):
        self.ranking = ranking
        self.calls: list[tuple[str, int | None]] = []

    def search(self, query: str, top_k: int | None = None) -> list[FakeHit]:
        self.calls.append((query, top_k))
        if top_k is None:
            return list(self.ranking)
        return list(self.ranking[:top_k])


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def _passages():
    return prepare_corpus(
        [
            {"id": 1001, "passage": "Papilin is a secreted extracellular matrix protein."},
            {"id": 42, "passage": "Insulin regulates blood glucose."},
        ]
    )


def test_question_to_hybrid_to_answer_to_citation_validation():
    passages = _passages()
    texts = {passage.passage_id: passage.text for passage in passages}
    bm25 = FakeRetriever([FakeHit(1001, 2.0, texts[1001])])
    dense = FakeRetriever([FakeHit(1001, 0.9, texts[1001]), FakeHit(42, 0.1, texts[42])])
    hybrid = HybridRetriever(passages, bm25_retriever=bm25, dense_retriever=dense)
    llm = FakeLLM("Papilin is secreted. [PASSAGE_ID: 1001]")

    result = GroundedQA(
        hybrid,
        generator=AnswerGenerator(llm=llm),
        evidence_top_k=2,
    ).ask("Is papilin secreted?")

    assert bm25.calls
    assert dense.calls
    assert result.llm_called is True
    assert result.insufficient_evidence is False
    assert result.answer == "Papilin is secreted. [PASSAGE_ID: 1001]"
    assert [item.passage_id for item in result.evidence][0] == 1001
    assert format_citation(1001) in result.prompt
    assert "Is papilin secreted?" in result.prompt
    assert result.citations.is_valid is True
    assert result.citations.valid_ids == (1001,)
    assert result.citations.invalid_ids == ()
    assert result.trace["generation_ms"] >= 0
    assert result.trace["citation_ms"] >= 0
    assert result.trace["bm25_ms"] is not None
    assert result.trace["dense_ms"] is not None
    assert result.trace["rrf_ms"] is not None
    assert result.trace["query_expansion_ms"] is None
    assert result.trace["expansion_llm_called"] is False


def test_no_retrieved_passages_skips_llm_with_insufficient_evidence():
    passages = _passages()
    hybrid = HybridRetriever(
        passages,
        bm25_retriever=FakeRetriever([]),
        dense_retriever=FakeRetriever([]),
    )
    llm = FakeLLM("should not be used")

    result = GroundedQA(hybrid, generator=AnswerGenerator(llm=llm)).ask(
        "What is an unanswerable question?"
    )

    assert llm.prompts == []
    assert result.llm_called is False
    assert result.insufficient_evidence is True
    assert result.evidence == ()
    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.citations.missing_citations is True
