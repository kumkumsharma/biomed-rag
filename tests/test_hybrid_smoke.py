import pytest

from src.data.loader import get_passage_collection
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever


@pytest.fixture(scope="module")
def bioasq_hybrid():
    passages = get_passage_collection(split="passages[:80]")
    bm25 = BM25Retriever(passages)
    dense = DenseRetriever(passages)
    hybrid = HybridRetriever(passages, bm25_retriever=bm25, dense_retriever=dense)
    return hybrid, passages


@pytest.mark.network
def test_hybrid_rrf_over_real_bioasq_bm25_and_dense(bioasq_hybrid):
    retriever, passages = bioasq_hybrid
    known_ids = {passage.passage_id for passage in passages}

    assert len(passages) == 80
    results = retriever.search("virus isolated from patients", top_k=5)

    assert len(results) == 5
    assert len({result.passage_id for result in results}) == 5
    assert all(result.passage_id in known_ids for result in results)
    assert all(isinstance(result.score, float) and result.score > 0 for result in results)
    assert [result.score for result in results] == sorted(
        (result.score for result in results), reverse=True
    )
    assert any(result.text.strip() for result in results)
