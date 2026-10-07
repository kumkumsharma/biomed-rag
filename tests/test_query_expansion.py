from src.data.preprocessing import prepare_corpus
from src.evaluation.dataset import EvaluationQuery
from src.evaluation.run_evaluation import run_hybrid_qe_evaluation
from src.generation.answer_generator import AnswerGenerator
from src.generation.query_expansion import (
    ExpandingHybridRetriever,
    QueryExpander,
    combine_original_and_expansions,
    parse_expansion_response,
)
from src.generation.rag import GroundedQA
from src.retrieval.hybrid import HybridRetriever
from src.validation.citations import format_citation


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class FakeHit:
    def __init__(self, passage_id, score, text):
        self.passage_id = passage_id
        self.score = score
        self.text = text


class FakeRetriever:
    def __init__(self, ranking):
        self.ranking = ranking
        self.calls: list[tuple[str, int | None]] = []

    def search(self, query: str, top_k: int | None = None):
        self.calls.append((query, top_k))
        if top_k is None:
            return list(self.ranking)
        return list(self.ranking[:top_k])


def _passages():
    return prepare_corpus(
        [
            {"id": 1001, "passage": "Papilin is a secreted extracellular matrix protein."},
            {"id": 42, "passage": "Insulin regulates blood glucose."},
        ]
    )


def test_original_query_is_never_rewritten():
    original = "Is papilin secreted?"
    expander = QueryExpander(
        llm=FakeLLM('{"expansions": ["papilin secretion", "Is papilin secreted?"]}'),
        enabled=True,
        count=3,
    )
    result = expander.expand(original)
    assert result.original == original
    assert original in result.all_queries
    assert result.all_queries[0] == original
    assert "papilin secretion" in result.expansions
    assert original not in result.expansions


def test_disabled_expander_does_not_call_the_llm():
    llm = FakeLLM('{"expansions": ["should not appear"]}')
    result = QueryExpander(llm=llm, enabled=False).expand("Is papilin secreted?")
    assert llm.prompts == []
    assert result.original == "Is papilin secreted?"
    assert result.expansions == ()
    assert result.llm_called is False


def test_parse_expansion_response_drops_blanks_and_duplicates():
    expansions = parse_expansion_response(
        '{"expansions": ["Papilin secretion", " ", "is papilin secreted?", "ECM protein papilin"]}',
        original="Is papilin secreted?",
        count=3,
    )
    assert expansions == ("Papilin secretion", "ECM protein papilin")


def test_combine_original_and_expansions_keeps_original_first():
    combined = combine_original_and_expansions(
        "original",
        ["alt", "original", "ALT", "other"],
    )
    assert combined == ["original", "alt", "other"]


def test_expanding_hybrid_records_passage_ids_scores_and_expansions():
    passages = _passages()
    hit = FakeHit(1001, 1.0, passages[0].text)
    hybrid = HybridRetriever(
        passages,
        bm25_retriever=FakeRetriever([hit]),
        dense_retriever=FakeRetriever([hit]),
    )
    expander = QueryExpander(
        llm=FakeLLM('{"expansions": ["papilin extracellular matrix"]}'),
        enabled=True,
        count=3,
    )
    retriever = ExpandingHybridRetriever(hybrid, expander)
    results = retriever.search("Is papilin secreted?", top_k=2)

    assert retriever.last_expansion is not None
    assert retriever.last_expansion.original == "Is papilin secreted?"
    assert retriever.last_expansion.expansions == ("papilin extracellular matrix",)
    assert retriever.last_llm_calls == 1
    assert retriever.last_cost_usd > 0
    assert results[0].passage_id == 1001
    assert isinstance(results[0].score, float)
    assert retriever.last_trace.get("n_expansions") == 1


def test_grounded_qa_keeps_original_query_and_can_disable_expansion():
    passages = _passages()
    hit = FakeHit(1001, 1.0, passages[0].text)
    hybrid = HybridRetriever(
        passages,
        bm25_retriever=FakeRetriever([hit]),
        dense_retriever=FakeRetriever([hit]),
    )
    expander_llm = FakeLLM('{"expansions": ["papilin secretion assay"]}')
    answer_llm = FakeLLM("Papilin is secreted. [PASSAGE_ID: 1001]")
    qa = GroundedQA(
        hybrid,
        generator=AnswerGenerator(llm=answer_llm),
        expander=QueryExpander(llm=expander_llm, enabled=True, count=3),
        evidence_top_k=1,
    )
    expanded = qa.ask("Is papilin secreted?")
    disabled = qa.ask("Is papilin secreted?", expand=False)

    assert expanded.question == "Is papilin secreted?"
    assert expanded.expansions == ("papilin secretion assay",)
    assert expanded.query_expansion_used is True
    assert format_citation(1001) in expanded.prompt
    assert "papilin secretion assay" not in expanded.prompt
    assert disabled.expansions == ()
    assert disabled.query_expansion_used is False
    assert expander_llm.prompts  # used only when expand is on


def test_hybrid_qe_evaluation_uses_the_same_fixed_queries(tmp_path):
    passages = _passages()
    hit = FakeHit(1001, 1.0, passages[0].text)
    hybrid = HybridRetriever(
        passages,
        bm25_retriever=FakeRetriever([hit]),
        dense_retriever=FakeRetriever([hit]),
    )
    expander = QueryExpander(
        llm=FakeLLM('{"expansions": ["papilin secreted protein"]}'),
        enabled=True,
        count=1,
    )
    queries = [
        EvaluationQuery(1, "Is papilin secreted?", (1001,)),
        EvaluationQuery(2, "Is papilin secreted?", (1001,)),
    ]
    result = run_hybrid_qe_evaluation(
        passages=passages,
        queries=queries,
        retrievers={
            "Hybrid + Query Expansion": ExpandingHybridRetriever(hybrid, expander)
        },
        queries_path=tmp_path / "evaluation_100.json",
        details_path=tmp_path / "qe_details.json",
        summary_path=tmp_path / "qe_summary.json",
        comparison_path=tmp_path / "comparison.json",
        existing_summary_path=tmp_path / "missing_summary.json",
    )
    assert [row.query_id for row in result["rows"]] == [1, 2]
    assert result["rows"][0].configuration == "Hybrid + Query Expansion"
    assert result["rows"][0].expansions == ("papilin secreted protein",)
    assert result["rows"][0].retrieved_passage_ids[0] == 1001
    assert result["summaries"][0]["llm_calls"] == 2
    assert result["details_path"].exists()
    assert "query_expansion" in result
    assert result["query_expansion"]["original_query_retained"] is True
