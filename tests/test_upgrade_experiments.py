import json
from pathlib import Path

import numpy as np

from src.config import UpgradeConfig
from src.data.preprocessing import prepare_corpus
from src.evaluation.dataset import EvaluationQuery
from src.evaluation.failure_analysis import categorize_retrieval_failure, summarize_failures
from src.evaluation.upgrade_experiments import evaluate_named_retrievers, run_upgrade_experiments
from src.retrieval.dense import DenseRetriever
from src.retrieval.embedding_cache import cache_file, load_cached_embeddings, save_cached_embeddings
from src.retrieval.embedding_models import MINILM_MODEL, resolve_embedding_model
from src.retrieval.pipeline import StagedRetriever


class KeywordEmbedder:
    _vocab = ("hirschsprung", "intestine", "insulin", "glucose", "papilin", "secreted", "egfr")

    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([1.0 if token in lowered else 0.0 for token in self._vocab])
        return np.asarray(vectors, dtype=float)


class FakeScorer:
    def score_pairs(self, query, texts):
        return [1.0 if "papilin" in text.lower() else 0.1 for text in texts]


def _passages():
    return prepare_corpus(
        [
            {"id": 1, "passage": "Hirschsprung disease of the intestine."},
            {"id": 2, "passage": "Insulin regulates blood glucose."},
            {"id": 3, "passage": "Papilin is a secreted protein."},
        ]
    )


def test_resolve_embedding_model_keeps_minilm_by_default():
    assert resolve_embedding_model(UpgradeConfig()) == MINILM_MODEL


def test_resolve_embedding_model_uses_biomedical_when_enabled():
    cfg = UpgradeConfig(biomedical_embeddings_enabled=True, biomedical_embedding_model="local/biomed")
    assert resolve_embedding_model(cfg) == "local/biomed"


def test_embedding_cache_roundtrip(tmp_path):
    matrix = np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    path = cache_file(tmp_path, "tiny", [1, 2])
    save_cached_embeddings(path, matrix, model_name="tiny")
    loaded = load_cached_embeddings(path)
    assert loaded.shape == (2, 2)
    np.testing.assert_allclose(loaded, matrix)


def test_dense_retriever_uses_cache_dir(tmp_path):
    passages = _passages()
    DenseRetriever(passages, embedder=KeywordEmbedder(), model_name="toy", cache_dir=str(tmp_path))
    files = list(Path(tmp_path).glob("*.npz"))
    assert files
    DenseRetriever(passages, embedder=KeywordEmbedder(), model_name="toy", cache_dir=str(tmp_path))
    assert load_cached_embeddings(files[0]) is not None


def test_failure_never_retrieved():
    result = categorize_retrieval_failure(
        question="Is this protein secreted?",
        relevant_ids=(3,),
        pool_ids=(1, 2),
        final_ids=(1, 2),
    )
    assert result["category"] == "relevant_never_retrieved"
    assert result["relevant_in_pool"] is False


def test_failure_filtering_removed_relevant():
    result = categorize_retrieval_failure(
        question="Is papilin secreted?",
        relevant_ids=(3,),
        pool_ids=(3, 1),
        filtered_ids=(1,),
        final_ids=(1,),
    )
    assert result["category"] == "filtering_removed_relevant"


def test_failure_reranker():
    result = categorize_retrieval_failure(
        question="Is papilin secreted?",
        relevant_ids=(3,),
        pool_ids=(3, 1),
        filtered_ids=(3, 1),
        final_ids=(1,),
        reranked=True,
    )
    assert result["category"] == "reranker_failure"


def test_failure_ranked_too_low_without_rerank():
    result = categorize_retrieval_failure(
        question="Is papilin secreted?",
        relevant_ids=(3,),
        pool_ids=(1, 2, 3),
        final_ids=(1, 2),
        reranked=False,
        final_k=2,
        pool_k=50,
    )
    assert result["category"] == "ranked_too_low"


def test_failure_terminology_mismatch():
    result = categorize_retrieval_failure(
        question="What does EGFR do?",
        relevant_ids=(9,),
        pool_ids=(1,),
        final_ids=(1,),
    )
    assert result["category"] == "terminology_mismatch"


def test_failure_insufficient_evidence():
    result = categorize_retrieval_failure(
        question="anything",
        relevant_ids=(),
        pool_ids=(1,),
        final_ids=(1,),
    )
    assert result["category"] == "insufficient_evidence"


def test_summarize_failures_counts_success():
    summary = summarize_failures(
        [
            {"category": "success"},
            {"category": "relevant_never_retrieved"},
            {"category": "relevant_never_retrieved"},
        ]
    )
    assert summary["n_queries"] == 3
    assert summary["n_failures"] == 2
    assert summary["counts"]["relevant_never_retrieved"] == 2


def test_upgrade_experiment_runner_with_fakes(tmp_path):
    passages = _passages()
    queries = [
        EvaluationQuery(question_id=0, question="secreted papilin", relevant_passage_ids=(3,)),
        EvaluationQuery(question_id=1, question="Hirschsprung intestine", relevant_passage_ids=(1,)),
    ]
    from src.retrieval.bm25 import BM25Retriever

    bm25 = BM25Retriever(passages)
    dense = DenseRetriever(passages, embedder=KeywordEmbedder(), model_name="toy")
    retriever = StagedRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        passages=passages,
        config=UpgradeConfig(pipeline_enabled=True, terminology_enabled=True, expansion_enabled=True),
        rerank_enabled=False,
    )
    reranked = StagedRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        passages=passages,
        config=UpgradeConfig(pipeline_enabled=True, filtering_enabled=True),
        scorer=FakeScorer(),
        rerank_enabled=True,
    )
    result = run_upgrade_experiments(
        passages=passages,
        queries=queries,
        retrievers={
            "A Existing Hybrid@10": retriever,
            "E Query expansion + Hybrid": retriever,
            "I Candidate filtering + MedCPT": reranked,
        },
        details_path=tmp_path / "upgrade_details.json",
        summary_path=tmp_path / "upgrade_summary.json",
    )
    payload = json.loads(Path(tmp_path / "upgrade_summary.json").read_text(encoding="utf-8"))
    names = [item["name"] for item in payload["configurations"]]
    assert "A Existing Hybrid@10" in names
    assert "I Candidate filtering + MedCPT" in names
    assert "rows" not in payload
    details = json.loads(Path(tmp_path / "upgrade_details.json").read_text(encoding="utf-8"))
    assert details["rows"]
    assert result["summaries"]
    assert all("recall_at_10" in item for item in result["summaries"])


def test_evaluate_named_retrievers_records_stage_latency():
    passages = _passages()
    from src.retrieval.bm25 import BM25Retriever

    retriever = StagedRetriever(
        bm25_retriever=BM25Retriever(passages),
        dense_retriever=DenseRetriever(passages, embedder=KeywordEmbedder(), model_name="toy"),
        passages=passages,
        config=UpgradeConfig(pipeline_enabled=True),
        rerank_enabled=False,
    )
    queries = [
        EvaluationQuery(question_id=1, question="papilin", relevant_passage_ids=(3,)),
    ]
    rows, summaries, failures = evaluate_named_retrievers({"A Existing Hybrid@10": retriever}, queries)
    assert rows[0]["stage_timings"]
    assert summaries[0]["avg_latency_ms"] >= 0
    assert failures["n_queries"] == 1
