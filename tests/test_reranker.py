from dataclasses import dataclass, replace

import pytest

from src.config import RerankerConfig
from src.data.preprocessing import prepare_corpus
from src.generation.answer_generator import AnswerGenerator
from src.generation.rag import GroundedQA
from src.retrieval.hybrid import HybridRetriever, RetrievalResult
from src.retrieval.reranker import (
    CrossEncoderReranker,
    RerankedResult,
    RerankingRetriever,
    TransformersCrossEncoderScorer,
    get_shared_scorer,
    logits_to_scores,
    resolve_device,
    wrap_retriever,
)
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


class FakeScorer:
    def __init__(self, scores_by_text: dict[str, float]):
        self.scores_by_text = scores_by_text
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def score_pairs(self, query: str, texts):
        self.calls.append((query, tuple(texts)))
        return [float(self.scores_by_text[text]) for text in texts]


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
            {"id": "A", "passage": "Hirschsprung disease of the intestine."},
            {"id": "B", "passage": "Insulin regulates blood glucose."},
            {"id": "C", "passage": "EGFR ligands include EGF."},
            {"id": "D", "passage": "Papilin is a secreted protein."},
        ]
    )


def _hybrid():
    passages = _passages()
    texts = {passage.passage_id: passage.text for passage in passages}
    bm25 = FakeRetriever(
        [
            FakeHit("A", 3.0, texts["A"]),
            FakeHit("B", 2.0, texts["B"]),
            FakeHit("C", 1.0, texts["C"]),
        ]
    )
    dense = FakeRetriever(
        [
            FakeHit("B", 0.9, texts["B"]),
            FakeHit("D", 0.8, texts["D"]),
            FakeHit("A", 0.1, texts["A"]),
        ]
    )
    retriever = HybridRetriever(
        passages,
        bm25_retriever=bm25,
        dense_retriever=dense,
        rrf_k=60,
    )
    return retriever, bm25, dense, texts


def _enabled_config(**overrides) -> RerankerConfig:
    values = dict(
        enabled=True,
        candidate_pool_size=50,
        final_top_k=10,
        model_name="ncbi/MedCPT-Cross-Encoder",
        device="cpu",
    )
    values.update(overrides)
    return RerankerConfig(**values)


def test_cross_encoder_reranker_requires_a_scorer():
    with pytest.raises(ValueError):
        CrossEncoderReranker(scorer=None)  # type: ignore[arg-type]


def test_transformers_scorer_does_not_load_model_on_init():
    scorer = TransformersCrossEncoderScorer(
        model_name="ncbi/MedCPT-Cross-Encoder",
        device="cpu",
    )
    assert scorer.model_name == "ncbi/MedCPT-Cross-Encoder"
    assert scorer.device == "cpu"
    assert scorer._model is None
    assert scorer._tokenizer is None


def test_shared_scorer_reuses_the_same_instance():
    config = RerankerConfig(model_name="unit-test/medcpt", device="cpu", batch_size=8)
    first = get_shared_scorer(config)
    second = get_shared_scorer(config)
    assert first is second
    assert first._model is None


def test_resolve_device_honors_an_explicit_choice():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("CUDA") == "cuda"


def test_logits_to_scores_handles_single_and_two_class_outputs():
    assert logits_to_scores([]) == []
    assert logits_to_scores([1.25, -0.5]) == [1.25, -0.5]
    assert logits_to_scores([[2.0], [0.1]]) == [2.0, 0.1]
    assert logits_to_scores([[0.1, 4.0], [0.9, 0.2]]) == [4.0, 0.2]


def test_rerank_orders_candidates_by_score_and_keeps_rrf_ties():
    texts = {passage.passage_id: passage.text for passage in _passages()}
    candidates = [
        RetrievalResult("A", 0.03, texts["A"]),
        RetrievalResult("B", 0.04, texts["B"]),
        RetrievalResult("C", 0.01, texts["C"]),
    ]
    scorer = FakeScorer({texts["A"]: 1.0, texts["B"]: 5.0, texts["C"]: 2.0})
    ranked = CrossEncoderReranker(scorer).rerank("insulin", candidates, top_k=3)

    assert [item.passage_id for item in ranked] == ["B", "C", "A"]
    assert [item.final_rank for item in ranked] == [1, 2, 3]
    assert ranked[0].reranker_score == 5.0
    assert ranked[0].score == 5.0
    assert ranked[0].rrf_score == 0.04
    assert ranked[0].rrf_rank == 2
    assert scorer.calls[0][0] == "insulin"


def test_rerank_top_k_truncates_the_resorted_list():
    texts = {passage.passage_id: passage.text for passage in _passages()}
    candidates = [
        RetrievalResult("A", 0.2, texts["A"]),
        RetrievalResult("B", 0.1, texts["B"]),
        RetrievalResult("C", 0.05, texts["C"]),
    ]
    scorer = FakeScorer({texts["A"]: 1.0, texts["B"]: 3.0, texts["C"]: 2.0})
    ranked = CrossEncoderReranker(scorer).rerank("q", candidates, top_k=2)
    assert [item.passage_id for item in ranked] == ["B", "C"]
    assert len(ranked) == 2


def test_rerank_empty_candidate_list_returns_empty_without_scoring():
    scorer = FakeScorer({})
    assert CrossEncoderReranker(scorer).rerank("q", [], top_k=5) == []
    assert scorer.calls == []


def test_rerank_top_k_larger_than_available_returns_all():
    texts = {passage.passage_id: passage.text for passage in _passages()}
    candidates = [
        RetrievalResult("A", 0.2, texts["A"]),
        RetrievalResult("B", 0.1, texts["B"]),
    ]
    scorer = FakeScorer({texts["A"]: 1.0, texts["B"]: 2.0})
    ranked = CrossEncoderReranker(scorer).rerank("q", candidates, top_k=50)
    assert [item.passage_id for item in ranked] == ["B", "A"]
    assert len(ranked) == 2


def test_rerank_preserves_passage_metadata_and_adds_inspectable_fields():
    candidate = RetrievalResult(passage_id=20598273, score=0.12, text="Hirschsprung disease.")
    scorer = FakeScorer({"Hirschsprung disease.": 9.5})
    ranked = CrossEncoderReranker(scorer).rerank(
        "Hirschsprung",
        [candidate],
        top_k=1,
        bm25_ranks={20598273: 3},
        dense_ranks={20598273: 1},
    )
    item = ranked[0]
    payload = item.as_dict()
    assert item.passage_id == 20598273
    assert item.pmid == 20598273
    assert item.text == "Hirschsprung disease."
    assert item.bm25_rank == 3
    assert item.dense_rank == 1
    assert item.rrf_rank == 1
    assert item.final_rank == 1
    assert item.rrf_score == 0.12
    assert item.reranker_score == 9.5
    assert payload["passage_id"] == 20598273
    assert payload["rrf_score"] == 0.12
    assert payload["reranker_score"] == 9.5
    assert payload["pmid"] == 20598273
    assert isinstance(item, RetrievalResult)


def test_rerank_is_deterministic_for_the_same_inputs():
    texts = {passage.passage_id: passage.text for passage in _passages()}
    candidates = [
        RetrievalResult("A", 0.2, texts["A"]),
        RetrievalResult("B", 0.2, texts["B"]),
        RetrievalResult("C", 0.2, texts["C"]),
    ]
    scorer = FakeScorer({texts["A"]: 1.0, texts["B"]: 1.0, texts["C"]: 0.5})
    reranker = CrossEncoderReranker(scorer)
    first = [item.passage_id for item in reranker.rerank("q", candidates, top_k=3)]
    second = [item.passage_id for item in reranker.rerank("q", candidates, top_k=3)]
    assert first == second == ["A", "B", "C"]


def test_disabled_reranker_forwards_first_stage_results_unchanged():
    hybrid, bm25, dense, _ = _hybrid()
    scorer = FakeScorer({})
    wrapped = wrap_retriever(
        hybrid,
        config=_enabled_config(enabled=False, candidate_pool_size=50, final_top_k=2),
        scorer=scorer,
    )
    baseline = hybrid.search("query", top_k=2)
    reranked = wrapped.search("query", top_k=2)
    assert [item.passage_id for item in reranked] == [item.passage_id for item in baseline]
    assert scorer.calls == []
    assert wrapped.last_trace["reranker_enabled"] is False
    assert bm25.calls[-1][1] == 2
    assert dense.calls[-1][1] == 2


def test_enabled_reranker_requests_the_candidate_pool_then_cuts_to_top_k():
    hybrid, bm25, _, texts = _hybrid()
    scorer = FakeScorer(
        {
            texts["A"]: 0.1,
            texts["B"]: 0.2,
            texts["C"]: 9.0,
            texts["D"]: 0.3,
        }
    )
    wrapped = wrap_retriever(
        hybrid,
        config=_enabled_config(candidate_pool_size=50, final_top_k=2),
        scorer=scorer,
    )
    results = wrapped.search("query", top_k=2)
    assert bm25.calls[-1][1] == 50
    assert [item.passage_id for item in results] == ["C", "D"]
    assert len(results) == 2
    assert wrapped.last_trace["reranker_enabled"] is True
    assert wrapped.last_trace["rerank_ms"] is not None
    assert wrapped.last_trace["reranker_model_name"] == "ncbi/MedCPT-Cross-Encoder"


def test_candidate_pool_larger_than_available_candidates_reranks_all():
    hybrid, _, _, texts = _hybrid()
    scorer = FakeScorer(
        {texts["A"]: 1.0, texts["B"]: 2.0, texts["C"]: 3.0, texts["D"]: 4.0}
    )
    wrapped = wrap_retriever(
        hybrid,
        config=_enabled_config(candidate_pool_size=50, final_top_k=10),
        scorer=scorer,
    )
    results = wrapped.search("query", top_k=10)
    assert len(results) == 4
    assert [item.passage_id for item in results] == ["D", "C", "B", "A"]


def test_reranking_retriever_exposes_dense_retriever_for_pipeline_trace():
    hybrid, _, dense, texts = _hybrid()
    wrapped = wrap_retriever(
        hybrid,
        config=_enabled_config(),
        scorer=FakeScorer({text: 0.0 for text in texts.values()}),
    )
    assert wrapped._dense is dense


def test_reranker_module_does_not_call_openai():
    import inspect

    import src.retrieval.reranker as module

    source = inspect.getsource(module)
    assert "import openai" not in source
    assert "from openai" not in source


def test_bm25_dense_rrf_rerank_to_grounded_evidence():
    hybrid, bm25, dense, texts = _hybrid()
    scorer = FakeScorer(
        {
            texts["A"]: 0.2,
            texts["B"]: 0.1,
            texts["C"]: 8.0,
            texts["D"]: 0.3,
        }
    )
    wrapped = RerankingRetriever(
        hybrid,
        config=_enabled_config(candidate_pool_size=50, final_top_k=2),
        scorer=scorer,
    )
    llm = FakeLLM("EGFR ligands include EGF. [PASSAGE_ID: C]")
    result = GroundedQA(
        wrapped,
        generator=AnswerGenerator(llm=llm),
        evidence_top_k=2,
    ).ask("What ligands does EGFR include?")

    assert bm25.calls
    assert dense.calls
    assert result.insufficient_evidence is False
    assert result.llm_called is True
    assert [item.passage_id for item in result.evidence] == ["C", "D"]
    assert result.evidence[0].reranker_score == 8.0
    assert result.evidence[0].rrf_score is not None
    assert result.evidence[0].bm25_rank == 3
    assert result.evidence[0].final_rank == 1
    assert result.evidence[0].pmid == "C"
    assert result.citations.is_valid is True
    assert result.citations.valid_ids == ("C",)
    assert format_citation("C") in result.prompt
    assert result.trace["reranker_enabled"] is True
    assert result.trace["rerank_ms"] is not None
    assert result.trace["bm25_ms"] is not None
    assert result.trace["rrf_ms"] is not None


def test_reranked_result_keeps_baseline_score_fields():
    item = RerankedResult(
        passage_id="A",
        score=1.5,
        text="hello",
        rrf_score=0.03,
        reranker_score=1.5,
        bm25_rank=1,
        dense_rank=2,
        rrf_rank=3,
        final_rank=1,
        pmid="A",
    )
    assert item.score == 1.5
    assert item.rrf_score == 0.03
    assert replace(_enabled_config(), enabled=False).enabled is False
