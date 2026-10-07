import math

import numpy as np

from src.data.preprocessing import Passage
from src.data.quality import classify_text, inspect_documents, is_nan_value
from src.evaluation.dataset import EvaluationQuery
from src.evaluation.experiment_catalog import experiment_table_rows, load_experiment_catalog
from src.evaluation.gold_quality import evaluate_gold_quality
from src.evaluation.index_health import inspect_index_health
from src.evaluation.live_metrics import evaluate_live_query, faithfulness_score
from src.evaluation.pipeline_diagnostic import diagnose_loss, inspect_gold_passages, stage_code
from src.evaluation.retrieval_diagnostics import diagnose_query, summarize_retrieval_diagnostics
from src.validation.citations import validate_citations


class _Hit:
    def __init__(self, passage_id, score, text=""):
        self.passage_id = passage_id
        self.score = score
        self.text = text


class _FakeSearcher:
    def __init__(self, hits):
        self._hits = hits
        self.last_trace = {"bm25_ranks": {1: 1}, "dense_ranks": {1: 2}}

    def search(self, query, top_k=None):
        return list(self._hits[: top_k or len(self._hits)])


class _FakeReranker:
    def rerank(self, query, candidates, top_k=None, **kwargs):
        items = list(candidates)
        if top_k:
            items = items[:top_k]
        return items


def test_nan_detection_covers_float_and_literal():
    assert is_nan_value(float("nan"))
    assert is_nan_value("nan")
    assert is_nan_value("NaN")
    assert not is_nan_value("nanny")
    assert not is_nan_value(None)


def test_empty_and_short_text_classification():
    assert classify_text("") == "empty_text"
    assert classify_text("   ") == "whitespace_only"
    assert classify_text("ab") == "extremely_short"
    assert classify_text(None) == "missing_text"
    assert classify_text(12) == "invalid_non_string_text"
    assert classify_text("Papilin is a secreted extracellular matrix protein.") == "usable"


def test_duplicate_id_and_text_detection():
    records = [
        {"id": 1, "passage": "Papilin is secreted in Drosophila basement membranes during development."},
        {"id": 1, "passage": "Papilin is secreted in Drosophila basement membranes during development."},
        {"id": 2, "passage": "Papilin is secreted in Drosophila basement membranes during development."},
        {"id": 3, "passage": "nan"},
    ]
    report = inspect_documents(records)["report"]
    assert report["total_documents"] == 4
    assert report["duplicate_ids"] == 1
    assert report["duplicate_texts"] >= 1
    assert report["nan_text"] == 1
    assert report["silently_deleted"] is False
    assert report["rejected_documents"] >= 2


def test_gold_label_validation_flags_nan_and_missing():
    passages = [
        Passage(1, "Papilin is a secreted extracellular matrix glycoprotein found in basement membranes."),
        Passage(2, "nan"),
    ]
    queries = [
        EvaluationQuery(0, "Is papilin secreted?", (1, 2, 99)),
        EvaluationQuery(1, "Empty gold only?", (2,)),
    ]
    report = evaluate_gold_quality(queries, passages)
    assert report["total_gold_ids"] == 4
    assert report["valid_gold_ids"] == 1
    assert report["invalid_gold_ids"] == 3
    assert report["queries_with_at_least_one_invalid_gold"] == 2
    assert report["queries_where_all_gold_ids_are_invalid"] == 1
    assert report["historical_metrics_unchanged"] is True


def test_index_health_alignment_and_nan_embeddings():
    passages = [Passage(1, "a"), Passage(2, "b")]

    class BM25:
        def __init__(self):
            self._passages = passages
            self._bm25 = type("Okapi", (), {"corpus_size": 2})()

    class Dense:
        def __init__(self):
            self._passages = passages
            self._embeddings = np.array([[0.1, 0.2], [0.3, math.nan]], dtype=float)
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

    report = inspect_index_health(passages, bm25=BM25(), dense=Dense())
    assert report["corpus_document_count"] == 2
    assert report["embedding_dimension"] == 2
    assert report["nan_embeddings"] == 1
    assert report["aligned"] is False


def test_retrieval_diagnostics_classifies_bm25_only_and_rrf_pushout():
    passages = [Passage(1, "gold papilin secreted"), Passage(9, "noise")]
    query = EvaluationQuery(0, "Is papilin secreted?", (1,))
    bm25 = _FakeSearcher([_Hit(1, 9.0, "gold"), _Hit(9, 1.0, "noise")])
    dense = _FakeSearcher([_Hit(9, 0.9, "noise")])
    hybrid = _FakeSearcher([_Hit(9, 0.03, "noise")])
    row = diagnose_query(
        query,
        passages=passages,
        bm25=bm25,
        dense=dense,
        hybrid=hybrid,
        reranker=_FakeReranker(),
        pool_k=50,
        final_k=10,
        inspect_k=10,
    )
    gold = row["gold"][0]
    assert gold["signal"] == "BM25 only"
    assert gold["rrf_class"] == "found by BM25 but pushed out by RRF"
    summary = summarize_retrieval_diagnostics([row])
    assert summary["signal_recovery"]["counts"]["BM25 only"] == 1


def test_reranker_diagnostic_marks_pool_hit_lost_in_top10():
    passages = [Passage(1, "gold")]
    query = EvaluationQuery(0, "q", (1,))
    hits = [_Hit(2, 0.1, "a")] + [_Hit(1, 0.09, "gold")]
    row = diagnose_query(
        query,
        passages=passages,
        bm25=_FakeSearcher(hits),
        dense=_FakeSearcher(hits),
        hybrid=_FakeSearcher(hits),
        reranker=_FakeReranker(),
        pool_k=50,
        final_k=1,
        inspect_k=10,
    )
    gold = row["gold"][0]
    assert gold["in_top50"] is True
    assert gold["lost_by_reranking"] is True


def test_citation_validation_and_live_metrics():
    citations = validate_citations("Papilin is secreted [PASSAGE_ID: 1].", [1, 2])
    assert citations.is_valid
    evidence = [_Hit(1, 1.0, "Papilin is a secreted extracellular matrix protein.")]
    record = evaluate_live_query(
        question="Is papilin secreted?",
        answer="Papilin is secreted [PASSAGE_ID: 1].",
        evidence=evidence,
        citations=citations,
        gold_ids=[1],
        insufficient_evidence=False,
        latency_ms=12.0,
    )
    assert record["blocks_user_response"] is False
    assert record["metrics"]["citation_correctness"] == 1.0
    assert record["metrics"]["context_recall"] == 1.0
    assert faithfulness_score("Papilin is secreted.", ["Papilin is secreted from cells."]) > 0


def test_trace_stage_codes_for_nan_gold():
    reports = inspect_gold_passages([7], [Passage(7, "nan")])
    assert reports[0]["failure_point"] == "DATA / PARSING"
    assert (
        diagnose_loss(
            gold_exists=True,
            empty_or_nan_text=True,
            in_bm25_pool=False,
            in_dense_pool=False,
            rrf_rank=None,
            medcpt_rank=None,
        )
        == "DATA / PARSING"
    )
    assert stage_code("DATA / PARSING") == "DATA_QUALITY"


def test_experiment_catalog_loads_without_overwriting(tmp_path):
    rows = experiment_table_rows(load_experiment_catalog())
    names = {row["Experiment"] for row in rows}
    assert "Production baseline" in names
    assert "Medical Query Expansion" in names
    assert "MedCPT CE → BiomedBERT reranker" in names
