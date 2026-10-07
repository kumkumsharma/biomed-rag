from dataclasses import dataclass

import numpy as np

from src.data.loader import GoldenQA
from src.data.preprocessing import prepare_corpus
from src.evaluation.dataset import EvaluationQuery, select_fixed_queries
from src.evaluation.run_evaluation import (
    QueryEvaluationRow,
    build_evaluation_retrievers,
    evaluate_configurations,
    evaluate_retriever,
    run_offline_evaluation,
    run_rerank_evaluation,
    save_evaluation_outputs,
    summarize_rows,
)
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.config import RerankerConfig
from src.retrieval.reranker import wrap_retriever


@dataclass(frozen=True)
class FakeHit:
    passage_id: object
    score: float
    text: str = ""


class FakeRetriever:
    def __init__(self, ranking_by_query: dict[str, list[FakeHit]]):
        self.ranking_by_query = ranking_by_query
        self.calls: list[tuple[str, int | None]] = []

    def search(self, query: str, top_k: int | None = None) -> list[FakeHit]:
        self.calls.append((query, top_k))
        hits = self.ranking_by_query[query]
        if top_k is None:
            return list(hits)
        return list(hits[:top_k])


class KeywordEmbedder:
    """Deterministic stand-in so tests do not download an embedding model."""

    _vocab = (
        "hirschsprung",
        "intestine",
        "insulin",
        "glucose",
        "papilin",
        "secreted",
        "protein",
        "egfr",
    )

    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([1.0 if token in lowered else 0.0 for token in self._vocab])
        return np.asarray(vectors, dtype=float)


def _queries():
    return [
        EvaluationQuery(
            question_id=1,
            question="Hirschsprung intestine",
            relevant_passage_ids=(20598273,),
        ),
        EvaluationQuery(
            question_id=2,
            question="secreted papilin",
            relevant_passage_ids=(1001, 42),
        ),
    ]


def test_evaluate_retriever_records_ids_scores_latency_and_zero_cost():
    retriever = FakeRetriever(
        {
            "Hirschsprung intestine": [
                FakeHit(20598273, 3.0),
                FakeHit(99, 1.0),
            ],
            "secreted papilin": [
                FakeHit(7, 2.0),
                FakeHit(1001, 1.5),
            ],
        }
    )
    rows = evaluate_retriever("BM25", retriever, _queries(), top_k=10)

    assert [row.query_id for row in rows] == [1, 2]
    assert rows[0].configuration == "BM25"
    assert rows[0].retrieved_passage_ids == (20598273, 99)
    assert rows[0].retrieved_scores == (3.0, 1.0)
    assert rows[0].recall_at_5 == 1.0
    assert rows[0].mrr_at_10 == 1.0
    assert rows[1].recall_at_5 == 0.5
    assert rows[1].mrr_at_10 == 0.5
    assert all(row.retrieval_cost_usd == 0.0 for row in rows)
    assert all(row.latency_ms >= 0.0 for row in rows)
    assert retriever.calls == [
        ("Hirschsprung intestine", 10),
        ("secreted papilin", 10),
    ]


def test_summarize_rows_averages_metrics_and_reports_median_latency():
    rows = [
        QueryEvaluationRow(
            query_id=1,
            question="q1",
            configuration="Dense",
            retrieved_passage_ids=(1,),
            retrieved_scores=(0.9,),
            latency_ms=10.0,
            recall_at_5=1.0,
            recall_at_10=1.0,
            mrr_at_10=1.0,
            ndcg_at_10=1.0,
            retrieval_cost_usd=0.0,
        ),
        QueryEvaluationRow(
            query_id=2,
            question="q2",
            configuration="Dense",
            retrieved_passage_ids=(9, 2),
            retrieved_scores=(0.8, 0.1),
            latency_ms=30.0,
            recall_at_5=0.0,
            recall_at_10=0.5,
            mrr_at_10=0.5,
            ndcg_at_10=0.5,
            retrieval_cost_usd=0.0,
        ),
    ]
    summary = summarize_rows(rows)
    assert summary["name"] == "Dense"
    assert summary["n_queries"] == 2
    assert summary["recall_at_5"] == 0.5
    assert summary["recall_at_10"] == 0.75
    assert summary["mrr_at_10"] == 0.75
    assert summary["ndcg_at_10"] == 0.75
    assert summary["avg_latency_ms"] == 20.0
    assert summary["median_latency_ms"] == 20.0
    assert summary["retrieval_cost_usd"] == 0.0


def test_save_evaluation_outputs_writes_machine_readable_results(tmp_path):
    rows = evaluate_retriever(
        "BM25",
        FakeRetriever({"q": [FakeHit(1, 1.0)]}),
        [EvaluationQuery(1, "q", (1,))],
        top_k=10,
    )
    details, summary = save_evaluation_outputs(
        rows=rows,
        summaries=[summarize_rows(rows)],
        details_path=tmp_path / "details.json",
        summary_path=tmp_path / "summary.json",
        n_passages=4,
        top_k=10,
        queries_path=tmp_path / "evaluation_100.json",
        index_build_seconds={"BM25_index_build_seconds": 0.01},
    )
    assert details.exists()
    assert summary.exists()
    assert "recall_at_5" in details.read_text(encoding="utf-8")
    assert "median_latency_ms" in summary.read_text(encoding="utf-8")
    assert "BM25_index_build_seconds" in summary.read_text(encoding="utf-8")


def test_evaluate_configurations_uses_the_same_queries_for_every_retriever():
    queries = _queries()
    retrievers = {
        "BM25": FakeRetriever(
            {
                "Hirschsprung intestine": [FakeHit(20598273, 1.0)],
                "secreted papilin": [FakeHit(1001, 1.0)],
            }
        ),
        "Dense": FakeRetriever(
            {
                "Hirschsprung intestine": [FakeHit(99, 0.9)],
                "secreted papilin": [FakeHit(42, 0.8)],
            }
        ),
    }
    rows, summaries = evaluate_configurations(retrievers, queries, top_k=10)
    assert [item["name"] for item in summaries] == ["BM25", "Dense"]
    bm25_ids = [row.query_id for row in rows if row.configuration == "BM25"]
    dense_ids = [row.query_id for row in rows if row.configuration == "Dense"]
    assert bm25_ids == dense_ids == [1, 2]


def test_retrieval_evaluation_smoke_golden_queries_to_saved_results(tmp_path):
    passages = prepare_corpus(
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
                "passage": "EGFR ligands include epidermal growth factor.",
            },
            {
                "id": 1001,
                "passage": "Papilin is a secreted extracellular matrix protein.",
            },
        ]
    )
    golden = [
        GoldenQA(1, "Hirschsprung intestine", "yes", (20598273,)),
        GoldenQA(2, "secreted papilin protein", "yes", (1001,)),
        GoldenQA(3, "skip empty", "no", ()),
    ]
    queries = select_fixed_queries(golden, n=2)
    retrievers, timings = build_evaluation_retrievers(
        passages,
        top_k=10,
        embedder=KeywordEmbedder(),
    )
    assert set(retrievers) == {"BM25", "Dense", "Hybrid RRF"}
    assert isinstance(retrievers["BM25"], BM25Retriever)
    assert isinstance(retrievers["Dense"], DenseRetriever)
    assert isinstance(retrievers["Hybrid RRF"], HybridRetriever)
    assert "Dense_index_build_seconds" in timings

    result = run_offline_evaluation(
        passages=passages,
        queries=queries,
        retrievers=retrievers,
        top_k=10,
        queries_path=tmp_path / "evaluation_100.json",
        details_path=tmp_path / "details.json",
        summary_path=tmp_path / "summary.json",
        index_build_seconds=timings,
    )

    assert result["n_passages"] == 4
    assert result["n_queries"] == 2
    assert result["retrieval_cost_usd"] == 0.0
    names = [item["name"] for item in result["summaries"]]
    assert names == ["BM25", "Dense", "Hybrid RRF"]
    for summary in result["summaries"]:
        assert 0.0 <= summary["recall_at_5"] <= 1.0
        assert 0.0 <= summary["recall_at_10"] <= 1.0
        assert 0.0 <= summary["mrr_at_10"] <= 1.0
        assert 0.0 <= summary["ndcg_at_10"] <= 1.0
        assert summary["retrieval_cost_usd"] == 0.0
    query_ids_by_config = {}
    for row in result["rows"]:
        query_ids_by_config.setdefault(row.configuration, []).append(row.query_id)
        assert row.retrieved_passage_ids
        assert row.retrieval_cost_usd == 0.0
    assert query_ids_by_config["BM25"] == query_ids_by_config["Dense"] == query_ids_by_config["Hybrid RRF"]
    assert result["details_path"].exists()
    assert result["summary_path"].exists()
    assert "Recall@5" in result["table"]


class LengthScorer:
    def score_pairs(self, query: str, texts):
        return [float(len(text)) for text in texts]


def test_rerank_evaluation_writes_separate_files_without_openai(tmp_path):
    passages = prepare_corpus(
        [
            {"id": 20598273, "passage": "Hirschsprung disease of the intestine."},
            {"id": 42, "passage": "Insulin regulates blood glucose."},
            {"id": 1001, "passage": "Papilin is a secreted extracellular matrix protein."},
        ]
    )
    queries = [
        EvaluationQuery(1, "Hirschsprung intestine", (20598273,)),
        EvaluationQuery(2, "secreted papilin", (1001,)),
    ]
    built, timings = build_evaluation_retrievers(
        passages,
        top_k=10,
        embedder=KeywordEmbedder(),
    )
    config = RerankerConfig(
        enabled=True,
        candidate_pool_size=10,
        final_top_k=10,
        model_name="ncbi/MedCPT-Cross-Encoder",
        device="cpu",
    )
    scorer = LengthScorer()
    wrapped = wrap_retriever(built["Hybrid RRF"], config=config, scorer=scorer)
    result = run_rerank_evaluation(
        passages=passages,
        queries=queries,
        retrievers={"Hybrid RRF + MedCPT Rerank": wrapped},
        scorer=scorer,
        config=config,
        top_k=10,
        queries_path=tmp_path / "evaluation_100.json",
        details_path=tmp_path / "rerank_details.json",
        summary_path=tmp_path / "rerank_summary.json",
        index_build_seconds=timings,
    )
    assert result["summaries"][0]["name"] == "Hybrid RRF + MedCPT Rerank"
    assert result["reranker"]["enabled"] is True
    assert result["reranker"]["openai_used"] is False
    assert result["retrieval_cost_usd"] == 0.0
    assert result["details_path"].exists()
    assert result["summary_path"].exists()
    assert result["details_path"].name == "rerank_details.json"
    for row in result["rows"]:
        assert row.retrieved_passage_ids
        assert row.configuration == "Hybrid RRF + MedCPT Rerank"
