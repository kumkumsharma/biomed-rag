import pytest

from src.data.preprocessing import prepare_corpus
from src.retrieval.bm25 import BM25Retriever, RetrievalResult


def _toy_passages():
    return prepare_corpus(
        [
            {
                "id": 20598273,
                "passage": "Hirschsprung disease is a congenital disorder of the intestine.",
            },
            {
                "id": 42,
                "passage": "Insulin regulates blood glucose in diabetes mellitus.",
            },
            {
                "id": 99,
                "passage": "EGFR ligands include epidermal growth factor and TGF-alpha.",
            },
            {
                "id": 1001,
                "passage": "Papilin is a secreted extracellular matrix protein.",
            },
        ]
    )


def test_bm25_index_can_be_built_from_prepared_passages():
    retriever = BM25Retriever(_toy_passages())
    assert isinstance(retriever, BM25Retriever)


def test_bm25_query_returns_results_with_ids_scores_and_text():
    retriever = BM25Retriever(_toy_passages())
    results = retriever.search("Hirschsprung disease intestine")

    assert results
    assert isinstance(results[0], RetrievalResult)
    assert results[0].passage_id == 20598273
    assert results[0].text.startswith("Hirschsprung disease")
    assert isinstance(results[0].score, float)
    assert results[0].as_dict()["passage_id"] == 20598273
    assert "score" in results[0].as_dict()
    assert "text" in results[0].as_dict()


def test_bm25_top_k_is_respected():
    retriever = BM25Retriever(_toy_passages())
    results = retriever.search("protein factor glucose", top_k=2)

    assert len(results) == 2


def test_bm25_top_k_cannot_exceed_corpus_size():
    retriever = BM25Retriever(_toy_passages())
    results = retriever.search("disease", top_k=50)

    assert len(results) <= 4


def test_bm25_returned_ids_belong_to_indexed_corpus():
    passages = _toy_passages()
    known_ids = {passage.passage_id for passage in passages}
    results = BM25Retriever(passages).search("EGFR insulin papilin Hirschsprung")

    assert results
    assert all(result.passage_id in known_ids for result in results)


def test_bm25_results_are_ordered_by_descending_score():
    results = BM25Retriever(_toy_passages()).search("secreted extracellular papilin")
    scores = [result.score for result in results]

    assert scores == sorted(scores, reverse=True)
    assert results[0].passage_id == 1001


def test_bm25_preserves_original_passage_ids():
    results = BM25Retriever(_toy_passages()).search("diabetes insulin glucose")
    assert results[0].passage_id == 42
    assert results[0].passage_id != 0
    assert results[0].passage_id != 1


def test_bm25_empty_query_returns_no_results():
    retriever = BM25Retriever(_toy_passages())

    assert retriever.search("") == []
    assert retriever.search("   ") == []


def test_bm25_invalid_top_k_returns_no_results():
    retriever = BM25Retriever(_toy_passages())

    assert retriever.search("insulin", top_k=0) == []
    assert retriever.search("insulin", top_k=-3) == []


def test_bm25_rejects_non_string_query():
    retriever = BM25Retriever(_toy_passages())
    with pytest.raises(TypeError):
        retriever.search(None)  # type: ignore[arg-type]


def test_bm25_rejects_empty_corpus():
    with pytest.raises(ValueError):
        BM25Retriever([])
