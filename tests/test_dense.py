import numpy as np
import pytest

from src.data.preprocessing import prepare_corpus
from src.retrieval.dense import DenseRetriever, RetrievalResult


class KeywordEmbedder:
    """Deterministic stand-in so unit tests do not download an embedding model."""

    _vocab = (
        "hirschsprung",
        "intestine",
        "insulin",
        "glucose",
        "diabetes",
        "egfr",
        "ligand",
        "papilin",
        "secreted",
        "protein",
    )

    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([1.0 if token in lowered else 0.0 for token in self._vocab])
        return np.asarray(vectors, dtype=float)


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


def _retriever():
    return DenseRetriever(_toy_passages(), embedder=KeywordEmbedder())


def test_dense_index_can_be_built_from_prepared_passages():
    retriever = _retriever()
    assert isinstance(retriever, DenseRetriever)
    assert retriever._embeddings.shape[0] == 4


def test_dense_query_returns_results_with_ids_scores_and_text():
    results = _retriever().search("Hirschsprung disease intestine")

    assert results
    assert isinstance(results[0], RetrievalResult)
    assert results[0].passage_id == 20598273
    assert results[0].text.startswith("Hirschsprung disease")
    assert isinstance(results[0].score, float)
    assert results[0].as_dict()["passage_id"] == 20598273
    assert "score" in results[0].as_dict()
    assert "text" in results[0].as_dict()


def test_dense_top_k_is_respected():
    results = _retriever().search("protein ligand glucose", top_k=2)
    assert len(results) == 2


def test_dense_top_k_cannot_exceed_corpus_size():
    results = _retriever().search("disease", top_k=50)
    assert len(results) <= 4


def test_dense_returned_ids_belong_to_indexed_corpus():
    passages = _toy_passages()
    known_ids = {passage.passage_id for passage in passages}
    results = DenseRetriever(passages, embedder=KeywordEmbedder()).search(
        "EGFR insulin papilin Hirschsprung"
    )

    assert results
    assert all(result.passage_id in known_ids for result in results)


def test_dense_results_are_ordered_by_descending_similarity():
    results = _retriever().search("secreted extracellular papilin")
    scores = [result.score for result in results]

    assert scores == sorted(scores, reverse=True)
    assert results[0].passage_id == 1001


def test_dense_preserves_original_passage_ids():
    results = _retriever().search("diabetes insulin glucose")
    assert results[0].passage_id == 42
    assert results[0].passage_id != 0
    assert results[0].passage_id != 1


def test_dense_empty_query_returns_no_results():
    retriever = _retriever()
    assert retriever.search("") == []
    assert retriever.search("   ") == []


def test_dense_invalid_top_k_returns_no_results():
    retriever = _retriever()
    assert retriever.search("insulin", top_k=0) == []
    assert retriever.search("insulin", top_k=-3) == []


def test_dense_rejects_non_string_query():
    with pytest.raises(TypeError):
        _retriever().search(None)  # type: ignore[arg-type]


def test_dense_rejects_empty_corpus():
    with pytest.raises(ValueError):
        DenseRetriever([], embedder=KeywordEmbedder())


class AsymmetricKeywordEmbedder:
    """Query and document encoders differ; encode() must not be used."""

    _doc_vocab = ("hirschsprung", "intestine", "insulin", "glucose", "papilin", "secreted")
    _query_vocab = ("hirschsprung", "intestine", "insulin", "glucose", "papilin", "secreted")

    def encode(self, texts):
        raise AssertionError("symmetric encode() should not be called for an asymmetric embedder")

    def encode_documents(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([1.0 if token in lowered else 0.0 for token in self._doc_vocab])
        return np.asarray(vectors, dtype=float)

    def encode_queries(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([1.0 if token in lowered else 0.0 for token in self._query_vocab])
        return np.asarray(vectors, dtype=float)


def test_asymmetric_embedder_uses_query_and_document_encoders():
    retriever = DenseRetriever(_toy_passages(), embedder=AsymmetricKeywordEmbedder())
    results = retriever.search("Hirschsprung disease intestine")
    assert results[0].passage_id == 20598273

