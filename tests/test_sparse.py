from pathlib import Path

import numpy as np
import pytest
from scipy import sparse as sp

from src.config import EVALUATION, UPGRADE, UpgradeConfig, upgrade_config_from_env
from src.data.preprocessing import prepare_corpus
from src.evaluation.dataset import EvaluationQuery
from src.evaluation.sparse_study import dual_miss_recovery_rate, load_dual_miss_rows, run_sparse_study
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import rrf_fuse_retrievers
from src.retrieval.sparse import SparseRetriever
from src.retrieval.sparse_models import DEFAULT_SPARSE_MODEL, SPARSE_CANDIDATES, try_resolve_sparse_model


class TokenSparseEmbedder:
    vocab = ("hirschsprung", "intestine", "papilin", "secreted", "zinc")
    vocab_size = 5
    model_name = "toy-splade"

    def encode(self, texts):
        rows, cols, data = [], [], []
        for index, text in enumerate(texts):
            lowered = text.lower()
            for col, token in enumerate(self.vocab):
                if token in lowered:
                    rows.append(index)
                    cols.append(col)
                    data.append(1.0)
        return sp.csr_matrix((data, (rows, cols)), shape=(len(texts), self.vocab_size))


class KeywordDenseEmbedder:
    vocab = ("hirschsprung", "intestine", "insulin", "glucose", "papilin", "secreted")

    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([1.0 if token in lowered else 0.0 for token in self.vocab])
        return np.asarray(vectors, dtype=float)


def _passages():
    return prepare_corpus(
        [
            {"id": 1, "passage": "Hirschsprung disease of the intestine."},
            {"id": 2, "passage": "Insulin regulates blood glucose."},
            {"id": 3, "passage": "Papilin is a secreted protein."},
            {"id": 99, "passage": "Unrelated zinc transporter biology."},
        ]
    )


def test_sparse_retriever_ranks_and_preserves_ids():
    passages = _passages()
    retriever = SparseRetriever(passages, embedder=TokenSparseEmbedder(), model_name="toy-splade", cache_dir=None)
    hits = retriever.search("secreted papilin", top_k=3)
    assert hits
    assert hits[0].passage_id == 3
    assert hits[0].pmid == 3
    assert hits[0].rank == 1
    assert hits[0].score > 0
    assert hits[0].as_dict()["passage_id"] == 3
    assert retriever.last_trace["retrieval_type"] == "learned_sparse"
    assert retriever.last_trace["model_name"] == "toy-splade"
    assert retriever.last_trace["n_hits"] == 3
    assert retriever.last_trace["query_nnz"] >= 1
    assert retriever.nnz >= 1
    assert retriever.index_nbytes() > 0


def test_sparse_empty_query_and_invalid_k():
    retriever = SparseRetriever(_passages(), embedder=TokenSparseEmbedder(), cache_dir=None)
    assert retriever.search("   ", top_k=5) == []
    assert retriever.search("papilin", top_k=0) == []
    with pytest.raises(TypeError):
        retriever.search(None)  # type: ignore[arg-type]


def test_sparse_rejects_empty_corpus():
    with pytest.raises(ValueError):
        SparseRetriever([], embedder=TokenSparseEmbedder(), cache_dir=None)


def test_sparse_index_is_reusable_on_disk(tmp_path):
    passages = _passages()
    first = SparseRetriever(
        passages, embedder=TokenSparseEmbedder(), model_name="toy-splade", cache_dir=str(tmp_path)
    )
    assert list(tmp_path.glob("*.csr.npz"))
    second = SparseRetriever(
        passages, embedder=TokenSparseEmbedder(), model_name="toy-splade", cache_dir=str(tmp_path)
    )
    assert first._matrix.shape == second._matrix.shape
    assert first.search("papilin")[0].passage_id == second.search("papilin")[0].passage_id


def test_rrf_fusion_includes_sparse_only_hit():
    passages = _passages()
    bm25 = BM25Retriever(passages, top_k=2)
    sparse = SparseRetriever(passages, embedder=TokenSparseEmbedder(), cache_dir=None, top_k=4)
    texts = {item.passage_id: item.text for item in passages}
    fused = rrf_fuse_retrievers([bm25, sparse], "zinc transporter", texts=texts, top_k=4, candidate_k=4)
    ids = [item.passage_id for item in fused]
    assert 99 in ids


def test_dual_miss_recovery_rate_denominator():
    assert dual_miss_recovery_rate(recovered=21, denominator=385) == 21 / 385
    assert dual_miss_recovery_rate(recovered=0, denominator=0) is None


def test_load_dual_miss_rows_filters_category(tmp_path):
    path = tmp_path / "objective.json"
    path.write_text(
        '{"failure_rows": ['
        '{"objective_category": "BOTH_RETRIEVERS_MISS", "gold_passage_id": 1},'
        '{"objective_category": "FUSION_CUTOFF", "gold_passage_id": 2}'
        "]}",
        encoding="utf-8",
    )
    rows = load_dual_miss_rows(path)
    assert len(rows) == 1
    assert rows[0]["gold_passage_id"] == 1


def test_sparse_study_recovers_dual_miss_without_downloading(tmp_path):
    passages = _passages()
    queries = [
        EvaluationQuery(1, "secreted papilin", (3,)),
        EvaluationQuery(2, "zinc transporter", (99,)),
    ]
    dual = [
        {
            "query_id": 2,
            "question": "zinc transporter",
            "gold_passage_id": 99,
            "objective_category": "BOTH_RETRIEVERS_MISS",
            "present_in_bm25_100": False,
            "present_in_dense_100": False,
        }
    ]
    result = run_sparse_study(
        passages=passages,
        queries=queries,
        sparse_embedder=TokenSparseEmbedder(),
        minilm_embedder=KeywordDenseEmbedder(),
        biomedical_embedder=None,
        allow_download=False,
        include_medcpt=False,
        dual_miss_rows=dual,
        details_path=tmp_path / "sparse_details.json",
        summary_path=tmp_path / "sparse_summary.json",
    )
    summary = result["summary"]
    assert summary["n_dual_misses"] == 1
    assert summary["dual_miss_recovery"]["hit_at_100"] >= 1
    assert 99 in summary["dual_miss_recovery"]["newly_recovered_ids"]
    assert summary["sparse_only_recovery_among_dual_misses"] >= 1
    assert summary["production_defaults_unchanged"] is True
    assert "never passed" in summary["leakage_note"]
    assert Path(tmp_path / "sparse_summary.json").exists()
    assert summary["status"] == "EXPERIMENTAL"
    names = [item["name"] for item in summary["fusion"]]
    assert "A BM25 + MiniLM" in names
    assert "B BM25 + MiniLM + Sparse" in names
    assert summary["medcpt"]["skipped"] is True


def test_sparse_study_skips_when_model_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "src.evaluation.sparse_study.try_resolve_sparse_model",
        lambda **_kwargs: {"model": None, "available": False, "source": None, "checked": []},
    )
    result = run_sparse_study(
        passages=_passages(),
        queries=[EvaluationQuery(1, "papilin", (3,))],
        allow_download=False,
        include_medcpt=False,
        dual_miss_rows=[],
        details_path=tmp_path / "skip_details.json",
        summary_path=tmp_path / "skip_summary.json",
    )
    assert result["skipped"] is True
    assert result["summary"]["status"] == "EXPERIMENTAL"
    assert result["summary"]["production_defaults_unchanged"] is True


def test_try_resolve_sparse_model_does_not_download_when_disabled(monkeypatch):
    monkeypatch.setattr("src.retrieval.sparse_models.huggingface_model_cached", lambda _name: False)
    info = try_resolve_sparse_model(prefer=DEFAULT_SPARSE_MODEL, allow_download=False)
    assert info["available"] is False
    assert info["model"] is None


def test_sparse_candidates_document_learned_sparse_output():
    preferred = next(item for item in SPARSE_CANDIDATES if item.get("preferred"))
    assert preferred["model"] == "naver/splade-v3-distilbert"
    assert preferred["produces_sparse_term_weights"] is True


def test_sparse_defaults_remain_off():
    assert UPGRADE.sparse_enabled is False
    assert UPGRADE.sparse_model_name == "naver/splade-v3-distilbert"
    assert EVALUATION.sparse_study_summary_path.endswith("sparse_study_summary.json")
    assert UpgradeConfig().sparse_enabled is False


def test_sparse_config_from_env(monkeypatch):
    monkeypatch.setenv("BIOMED_SPARSE_ENABLED", "1")
    monkeypatch.setenv("BIOMED_SPARSE_MODEL", "naver/splade-v3")
    cfg = upgrade_config_from_env()
    assert cfg.sparse_enabled is True
    assert cfg.sparse_model_name == "naver/splade-v3"
    assert UPGRADE.sparse_enabled is False
