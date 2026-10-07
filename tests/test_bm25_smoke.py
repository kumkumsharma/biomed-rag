import pytest

from src.data.loader import get_passage_collection
from src.retrieval.bm25 import BM25Retriever


@pytest.fixture(scope="module")
def bioasq_bm25():
    passages = get_passage_collection(split="passages[:200]")
    return BM25Retriever(passages), passages


@pytest.mark.network
def test_bm25_indexes_a_small_real_bioasq_subset(bioasq_bm25):
    retriever, passages = bioasq_bm25
    known_ids = {passage.passage_id for passage in passages}

    assert len(passages) == 200
    results = retriever.search("virus isolated from patients", top_k=5)

    assert len(results) == 5
    assert all(result.passage_id in known_ids for result in results)
    assert all(isinstance(result.score, float) for result in results)
    assert [result.score for result in results] == sorted(
        (result.score for result in results), reverse=True
    )
    assert any(result.text.strip() for result in results)
