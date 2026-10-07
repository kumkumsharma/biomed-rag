from src.config import DatasetConfig
from src.data.loader import get_passage_collection, load_corpus_dataset, load_golden_qa, load_qa_dataset


def test_load_corpus_dataset_uses_huggingface_config(monkeypatch):
    captured = {}

    def fake_load_dataset(name, config_name, split=None):
        captured["name"] = name
        captured["config_name"] = config_name
        captured["split"] = split
        return [{"id": 20598273, "passage": "Papilin is secreted."}]

    monkeypatch.setattr("src.data.loader.load_dataset", fake_load_dataset)

    records = load_corpus_dataset()

    assert captured == {
        "name": "rag-datasets/rag-mini-bioasq",
        "config_name": "text-corpus",
        "split": "passages",
    }
    assert records[0]["id"] == 20598273
    assert records[0]["passage"] == "Papilin is secreted."


def test_load_qa_dataset_uses_huggingface_config(monkeypatch):
    captured = {}

    def fake_load_dataset(name, config_name, split=None):
        captured["name"] = name
        captured["config_name"] = config_name
        captured["split"] = split
        return [
            {
                "id": 0,
                "question": "Is papilin secreted?",
                "answer": "Yes.",
                "relevant_passage_ids": [20598273],
            }
        ]

    monkeypatch.setattr("src.data.loader.load_dataset", fake_load_dataset)

    records = load_qa_dataset()

    assert captured == {
        "name": "rag-datasets/rag-mini-bioasq",
        "config_name": "question-answer-passages",
        "split": "test",
    }
    assert records[0]["relevant_passage_ids"] == [20598273]


def test_get_passage_collection_preserves_passage_ids(monkeypatch):
    def fake_load_dataset(name, config_name, split=None):
        return [
            {"id": 20598273, "passage": "alpha"},
            {"id": 6650562, "passage": "beta"},
        ]

    monkeypatch.setattr("src.data.loader.load_dataset", fake_load_dataset)

    collection = get_passage_collection()

    assert [item.passage_id for item in collection] == [20598273, 6650562]
    assert [item.text for item in collection] == ["alpha", "beta"]


def test_get_passage_collection_accepts_provided_records():
    records = [{"id": 42, "passage": "local fixture"}]
    collection = get_passage_collection(records=records)
    assert collection[0].passage_id == 42
    assert collection[0].text == "local fixture"


def test_load_golden_qa_preserves_relevant_passage_ids():
    records = [
        {
            "id": 0,
            "question": "Is Hirschsprung disease a mendelian or a multifactorial disorder?",
            "answer": "Both.",
            "relevant_passage_ids": "[20598273, 6650562]",
        }
    ]

    golden = load_golden_qa(records=records)

    assert golden[0].question_id == 0
    assert golden[0].relevant_passage_ids == (20598273, 6650562)
    assert golden[0].question.startswith("Is Hirschsprung")


def test_loader_honours_custom_dataset_config(monkeypatch):
    captured = {}

    def fake_load_dataset(name, config_name, split=None):
        captured["name"] = name
        captured["split"] = split
        return []

    monkeypatch.setattr("src.data.loader.load_dataset", fake_load_dataset)
    config = DatasetConfig(name="mock/bioasq", corpus_split="passages[:3]")

    load_corpus_dataset(config=config)

    assert captured["name"] == "mock/bioasq"
    assert captured["split"] == "passages[:3]"
