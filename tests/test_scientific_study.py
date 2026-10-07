import json
from pathlib import Path

import numpy as np

from src.data.preprocessing import prepare_corpus
from src.evaluation.dataset import EvaluationQuery
from src.evaluation.scientific_study import (
    categorize_missing_evidence,
    run_candidate_pool_experiment,
    run_decomposition_study,
    run_embedding_comparison,
    run_expansion_study,
    run_medcpt_pool_experiment,
    run_missing_evidence_study,
    try_resolve_biomedical_model,
)
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever


class KeywordEmbedder:
    _vocab = ("hirschsprung", "intestine", "insulin", "glucose", "papilin", "secreted", "egfr")

    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([1.0 if token in lowered else 0.0 for token in self._vocab])
        return np.asarray(vectors, dtype=float)


class AltEmbedder:
    """Different encoding so biomedical vs MiniLM can disagree in tests."""

    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "papilin" in lowered else 0.0,
                    1.0 if "insulin" in lowered else 0.0,
                    1.0 if "hirschsprung" in lowered else 0.0,
                ]
            )
        return np.asarray(vectors, dtype=float)


class FakeScorer:
    def score_pairs(self, query, texts):
        return [float(text.lower().count("papilin")) for text in texts]


def _passages():
    return prepare_corpus(
        [
            {"id": 1, "passage": "Hirschsprung disease of the intestine."},
            {"id": 2, "passage": "Insulin regulates blood glucose."},
            {"id": 3, "passage": "Papilin is a secreted protein."},
        ]
    )


def _queries():
    return [
        EvaluationQuery(1, "Hirschsprung intestine", (1,)),
        EvaluationQuery(2, "secreted papilin", (3,)),
    ]


def _hybrid():
    passages = _passages()
    bm25 = BM25Retriever(passages)
    dense = DenseRetriever(passages, embedder=KeywordEmbedder(), model_name="toy")
    hybrid = HybridRetriever(passages, bm25_retriever=bm25, dense_retriever=dense, top_k=3)
    return passages, bm25, dense, hybrid


def test_try_resolve_biomedical_does_not_return_minilm_when_unavailable(monkeypatch):
    import src.evaluation.scientific_study as study

    monkeypatch.setattr(study, "BIOMEDICAL_CANDIDATES", ("not-a-real/model",))
    result = try_resolve_biomedical_model(prefer="also-fake/encoder", allow_download=False)
    assert result["available"] is False
    assert result["model"] is None
    assert result["checked"]
    assert "minilm" not in json.dumps(result).lower()


def test_embedding_comparison_keeps_minilm_and_optional_biomedical(tmp_path):
    result = run_embedding_comparison(
        passages=_passages(),
        queries=_queries(),
        minilm_embedder=KeywordEmbedder(),
        biomedical_embedder=AltEmbedder(),
        biomedical_model_name="toy-biomed",
        allow_download=False,
        details_path=tmp_path / "emb_details.json",
        summary_path=tmp_path / "emb_summary.json",
    )
    names = [item["name"] for item in result["summaries"]]
    assert names[0].startswith("A MiniLM")
    assert names[1].startswith("B Biomedical")
    summary = json.loads(Path(tmp_path / "emb_summary.json").read_text(encoding="utf-8"))
    assert "rows" not in summary
    assert summary["minilm_model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert summary["same_bm25"] is True


def test_candidate_pool_curve_uses_prefixes(tmp_path):
    passages, _bm25, _dense, hybrid = _hybrid()
    result = run_candidate_pool_experiment(
        passages=passages,
        queries=_queries(),
        retrievers={"Hybrid RRF": hybrid},
        details_path=tmp_path / "pool_details.json",
        summary_path=tmp_path / "pool_summary.json",
    )
    ks = [item["k"] for item in result["curve"]]
    assert ks[:3] == [10, 20, 30]
    assert 50 in ks
    recalls = [item["recall_at_k"] for item in result["curve"]]
    assert recalls == sorted(recalls)


def test_medcpt_pool_experiment_with_fake_scorer(tmp_path):
    passages, _bm25, _dense, hybrid = _hybrid()
    result = run_medcpt_pool_experiment(
        passages=passages,
        queries=_queries(),
        hybrid=hybrid,
        scorer=FakeScorer(),
        details_path=tmp_path / "medcpt_details.json",
        summary_path=tmp_path / "medcpt_summary.json",
    )
    pools = [item["candidate_pool"] for item in result["summaries"]]
    assert pools == [20, 30, 50, 75, 100]
    assert all("recall_at_10" in item for item in result["summaries"])


def test_missing_evidence_categories_and_table(tmp_path):
    assert categorize_missing_evidence(
        question="Is this protein secreted?",
        relevant_id=99,
        bm25_rank=None,
        dense_rank=None,
        rrf_rank=None,
        in_corpus=True,
        overlap=0.0,
    ) in {"lexical mismatch", "both BM25 and dense failure"}
    assert (
        categorize_missing_evidence(
            question="q",
            relevant_id=99,
            bm25_rank=None,
            dense_rank=None,
            rrf_rank=None,
            in_corpus=False,
            overlap=0.0,
        )
        == "dataset/relevance issue"
    )
    passages, bm25, dense, hybrid = _hybrid()
    queries = [EvaluationQuery(9, "unrelated zinc transporter", (99,))]
    result = run_missing_evidence_study(
        passages=passages,
        queries=queries,
        bm25=bm25,
        dense=dense,
        hybrid=hybrid,
        details_path=tmp_path / "miss_details.json",
        summary_path=tmp_path / "miss_summary.json",
    )
    assert result["table"]
    assert result["rows"][0]["question"] == "unrelated zinc transporter"
    assert result["rows"][0]["misses"][0]["relevant_passage_id"] == 99
    assert "passage_table" in json.loads(Path(tmp_path / "miss_summary.json").read_text(encoding="utf-8"))


def test_expansion_study_compares_original_equal_and_priority(tmp_path):
    passages, _bm25, _dense, hybrid = _hybrid()
    result = run_expansion_study(
        passages=passages,
        queries=_queries(),
        hybrid=hybrid,
        top_k=3,
        details_path=tmp_path / "exp_details.json",
        summary_path=tmp_path / "exp_summary.json",
    )
    summary = result["summary"]
    assert "original_recall_at_10" in summary
    assert "equal_recall_at_10" in summary
    assert "priority_recall_at_10" in summary
    assert summary["priority_is_default"] is False


def test_decomposition_study_splits_complex_and_simple(tmp_path):
    passages, _bm25, _dense, hybrid = _hybrid()
    queries = [
        EvaluationQuery(
            1,
            "What are the risk factors, mechanisms, and treatment options for Hirschsprung disease?",
            (1,),
        ),
        EvaluationQuery(2, "Is this protein secreted?", (3,)),
    ]
    result = run_decomposition_study(
        passages=passages,
        queries=queries,
        hybrid=hybrid,
        details_path=tmp_path / "dec_details.json",
        summary_path=tmp_path / "dec_summary.json",
    )
    assert result["summary"]["n_decomposed"] == 1
    assert result["summary"]["n_not_decomposed"] == 1
    assert result["summary"]["selective_activation"] is True
