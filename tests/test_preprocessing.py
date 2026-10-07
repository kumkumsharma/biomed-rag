import pytest

from src.data.preprocessing import (
    Passage,
    parse_relevant_passage_ids,
    prepare_corpus,
    prepare_passage,
    preserve_id,
)


def test_prepare_passage_preserves_original_id_and_text():
    text = "Papilin is a secreted extracellular matrix protein.\n"
    passage = prepare_passage({"id": 20598273, "passage": text})

    assert isinstance(passage, Passage)
    assert passage.passage_id == 20598273
    assert type(passage.passage_id) is int
    assert passage.text == text


def test_prepare_passage_does_not_reindex_ids():
    records = [
        {"id": 6650562, "passage": "first"},
        {"id": 20598273, "passage": "second"},
    ]
    corpus = prepare_corpus(records)

    assert [item.passage_id for item in corpus] == [6650562, 20598273]
    assert [item.text for item in corpus] == ["first", "second"]


def test_prepare_passage_preserves_string_ids_exactly():
    passage = prepare_passage({"id": "PMID:20598273", "passage": "text"})
    assert passage.passage_id == "PMID:20598273"


def test_prepare_corpus_uses_configured_field_names():
    corpus = prepare_corpus(
        [{"passage_id": 42, "body": "abc"}],
        id_field="passage_id",
        text_field="body",
    )
    assert corpus[0].passage_id == 42
    assert corpus[0].text == "abc"


def test_parse_relevant_passage_ids_from_list_preserves_values():
    assert parse_relevant_passage_ids([20598273, 6650562]) == [20598273, 6650562]


def test_parse_relevant_passage_ids_from_string_preserves_values():
    assert parse_relevant_passage_ids("[20598273, 6650562]") == [20598273, 6650562]


def test_preserve_id_converts_numpy_scalar_without_changing_value():
    numpy = pytest.importorskip("numpy")
    value = numpy.int64(20598273)
    preserved = preserve_id(value)
    assert preserved == 20598273
    assert type(preserved) is int
