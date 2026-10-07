import pytest

from src.data.loader import get_passage_collection, load_corpus_dataset, load_golden_qa, load_qa_dataset


@pytest.mark.network
def test_huggingface_corpus_and_qa_can_be_loaded():
    corpus = load_corpus_dataset(split="passages[:5]")
    qa = load_qa_dataset(split="test[:5]")

    assert len(corpus) == 5
    assert "id" in corpus[0]
    assert "passage" in corpus[0]
    assert corpus[0]["id"] is not None
    assert isinstance(corpus[0]["passage"], str)

    assert len(qa) == 5
    assert "question" in qa[0]
    assert "answer" in qa[0]
    assert "relevant_passage_ids" in qa[0]


@pytest.mark.network
def test_huggingface_passage_ids_are_preserved_after_prepare():
    collection = get_passage_collection(split="passages[:5]")
    raw = load_corpus_dataset(split="passages[:5]")

    assert [item.passage_id for item in collection] == [row["id"] for row in raw]
    assert [item.text for item in collection] == [row["passage"] for row in raw]


@pytest.mark.network
def test_huggingface_golden_qa_preserves_relevant_ids():
    golden = load_golden_qa(split="test[:3]")
    raw = load_qa_dataset(split="test[:3]")

    assert len(golden) == 3
    assert golden[0].question == raw[0]["question"]
    assert golden[0].relevant_passage_ids
    assert all(isinstance(item, int) for item in golden[0].relevant_passage_ids)
    assert golden[0].relevant_passage_ids == (
        20598273,
        6650562,
        15829955,
        15617541,
        23001136,
        8896569,
        21995290,
        12239580,
        15858239,
    )
