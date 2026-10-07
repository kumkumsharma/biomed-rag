from dataclasses import dataclass

import pytest

from src.data.preprocessing import prepare_corpus
from src.retrieval.hybrid import HybridRetriever, RetrievalResult, reciprocal_rank_fusion


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


def _toy_passages():
    return prepare_corpus(
        [
            {"id": "A", "passage": "Hirschsprung disease of the intestine."},
            {"id": "B", "passage": "Insulin regulates blood glucose."},
            {"id": "C", "passage": "EGFR ligands include EGF."},
            {"id": "D", "passage": "Papilin is a secreted protein."},
        ]
    )


def _hybrid():
    passages = _toy_passages()
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


def test_hybrid_retriever_can_be_constructed():
    retriever, _, _, _ = _hybrid()
    assert isinstance(retriever, HybridRetriever)


def test_hybrid_calls_bm25_and_dense():
    retriever, bm25, dense, _ = _hybrid()
    retriever.search("Hirschsprung disease", top_k=3)

    assert bm25.calls == [("Hirschsprung disease", 3)]
    assert dense.calls == [("Hirschsprung disease", 3)]


def test_hybrid_combines_both_rankings_without_duplicate_ids():
    results = _hybrid()[0].search("Hirschsprung disease", top_k=4)
    ids = [result.passage_id for result in results]

    assert set(ids) == {"A", "B", "C", "D"}
    assert len(ids) == len(set(ids))


def test_document_in_both_rankings_gets_both_rrf_contributions():
    fused = reciprocal_rank_fusion(
        [
            [FakeHit("A", 1.0, "a"), FakeHit("B", 0.5, "b")],
            [FakeHit("B", 0.9, "b"), FakeHit("A", 0.1, "a")],
        ],
        rrf_k=60,
    )
    scores = dict(fused)
    both = 1.0 / 61 + 1.0 / 62
    assert scores["A"] == pytest.approx(both)
    assert scores["B"] == pytest.approx(both)


def test_document_in_only_one_ranking_can_still_appear():
    results = _hybrid()[0].search("query", top_k=4)
    ids = {result.passage_id for result in results}

    assert "C" in ids
    assert "D" in ids


def test_duplicate_ids_inside_one_ranking_use_first_rank_only():
    fused = reciprocal_rank_fusion(
        [[FakeHit("A", 1.0, "a"), FakeHit("A", 0.2, "a")]],
        rrf_k=60,
    )
    assert len(fused) == 1
    assert fused[0][0] == "A"
    assert fused[0][1] == pytest.approx(1.0 / 61)


def test_hybrid_results_are_sorted_by_descending_rrf_score():
    results = _hybrid()[0].search("query", top_k=4)
    scores = [result.score for result in results]

    assert scores == sorted(scores, reverse=True)
    assert results[0].passage_id == "B"
    assert results[1].passage_id == "A"


def test_hybrid_top_k_is_respected():
    results = _hybrid()[0].search("query", top_k=2)
    assert len(results) == 2


def test_hybrid_preserves_original_passage_ids_and_text():
    retriever, _, _, texts = _hybrid()
    results = retriever.search("query", top_k=4)

    assert all(isinstance(result, RetrievalResult) for result in results)
    for result in results:
        assert result.passage_id in texts
        assert result.text == texts[result.passage_id]
        assert result.as_dict()["passage_id"] == result.passage_id
        assert "score" in result.as_dict()
        assert "text" in result.as_dict()


def test_hybrid_ignores_ids_outside_the_passage_collection():
    passages = _toy_passages()
    bm25 = FakeRetriever([FakeHit("missing", 9.0, "not in corpus"), FakeHit("A", 1.0, "x")])
    dense = FakeRetriever([FakeHit("A", 0.5, "x")])
    results = HybridRetriever(
        passages,
        bm25_retriever=bm25,
        dense_retriever=dense,
    ).search("query", top_k=5)

    assert [result.passage_id for result in results] == ["A"]


def test_hybrid_empty_query_returns_no_results():
    retriever, bm25, dense, _ = _hybrid()
    assert retriever.search("") == []
    assert retriever.search("   ") == []
    assert bm25.calls == []
    assert dense.calls == []


def test_hybrid_invalid_top_k_returns_no_results():
    retriever, _, _, _ = _hybrid()
    assert retriever.search("query", top_k=0) == []
    assert retriever.search("query", top_k=-1) == []


def test_hybrid_rejects_non_string_query():
    with pytest.raises(TypeError):
        _hybrid()[0].search(None)  # type: ignore[arg-type]


def test_hybrid_rejects_empty_corpus():
    with pytest.raises(ValueError):
        HybridRetriever([], bm25_retriever=FakeRetriever([]), dense_retriever=FakeRetriever([]))


def test_hybrid_expansions_keep_the_original_query_first():
    retriever, bm25, dense, _ = _hybrid()
    results = retriever.search(
        "Hirschsprung disease",
        top_k=4,
        expansions=["Hirschsprung disease", "congenital intestinal aganglionosis"],
    )

    assert bm25.calls[0] == ("Hirschsprung disease", 4)
    assert dense.calls[0] == ("Hirschsprung disease", 4)
    assert ("congenital intestinal aganglionosis", 4) in bm25.calls
    assert ("congenital intestinal aganglionosis", 4) in dense.calls
    assert [query for query, _top_k in bm25.calls].count("Hirschsprung disease") == 1
    assert results
    assert {result.passage_id for result in results} <= {"A", "B", "C", "D"}


def test_hybrid_search_records_stage_timings_without_changing_ranks():
    retriever, _, _, _ = _hybrid()
    results = retriever.search("Hirschsprung disease", top_k=4)
    ids = [result.passage_id for result in results]
    assert ids[0] == "B"
    assert retriever.last_trace["n_queries"] == 1
    assert retriever.last_trace["n_rankings"] == 2
    assert retriever.last_trace["candidate_k"] == 4
    assert retriever.last_trace["rrf_k"] == 60
    assert retriever.last_trace["bm25_ms"] >= 0
    assert retriever.last_trace["dense_ms"] >= 0
    assert retriever.last_trace["rrf_ms"] >= 0
    assert retriever.last_trace["bm25_ranks"]["A"] == 1
    assert retriever.last_trace["bm25_ranks"]["B"] == 2
    assert retriever.last_trace["dense_ranks"]["B"] == 1
    assert retriever.last_trace["dense_ranks"]["D"] == 2
    assert retriever.last_trace["rrf_ranks"][results[0].passage_id] == 1


def test_rrf_fuse_retrievers_combines_three_rankings():
    from src.retrieval.hybrid import rrf_fuse_retrievers

    texts = {"A": "a", "B": "b", "C": "c"}
    bm25 = FakeRetriever([FakeHit("A", 1.0, "a"), FakeHit("B", 0.5, "b")])
    dense = FakeRetriever([FakeHit("B", 0.9, "b"), FakeHit("A", 0.1, "a")])
    sparse = FakeRetriever([FakeHit("C", 2.0, "c"), FakeHit("B", 1.0, "b")])
    fused = rrf_fuse_retrievers([bm25, dense, sparse], "q", texts=texts, top_k=3, rrf_k=60, candidate_k=3)
    ids = [item.passage_id for item in fused]
    assert ids[0] == "B"
    assert set(ids) == {"A", "B", "C"}


def test_priority_expansion_keeps_original_ranking_first():
    retriever, bm25, dense, _ = _hybrid()
    original = retriever.search("Hirschsprung disease", top_k=4)
    original_ids = [item.passage_id for item in original]
    ranked = retriever.search(
        "Hirschsprung disease",
        top_k=4,
        expansions=["insulin glucose"],
        expansion_mode="priority",
    )
    assert [item.passage_id for item in ranked[: len(original_ids)]] == original_ids
    assert retriever.last_trace["expansion_mode"] == "priority"
    assert bm25.calls[0][0] == "Hirschsprung disease"
