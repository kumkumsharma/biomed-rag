import pytest

from src.data.loader import get_passage_collection
from src.retrieval.dense import DenseRetriever


@pytest.fixture(scope="module")
def bioasq_dense():
    passages = get_passage_collection(split="passages[:80]")
    retriever = DenseRetriever(passages)
    return retriever, passages


@pytest.mark.network
def test_dense_indexes_a_small_real_bioasq_subset(bioasq_dense):
    retriever, passages = bioasq_dense
    known_ids = {passage.passage_id for passage in passages}

    assert len(passages) == 80
    results = retriever.search("virus isolated from patients", top_k=5)

    assert len(results) == 5
    assert all(result.passage_id in known_ids for result in results)
    assert all(isinstance(result.score, float) for result in results)
    assert [result.score for result in results] == sorted(
        (result.score for result in results), reverse=True
    )
    assert any(result.text.strip() for result in results)
