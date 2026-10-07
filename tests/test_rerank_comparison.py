from dataclasses import dataclass

from src.config import EVALUATION, RerankerConfig
from src.data.preprocessing import prepare_corpus
from src.evaluation.dataset import EvaluationQuery
from src.evaluation.metrics import compute_retrieval_metrics
from src.evaluation.rerank_comparison import (
    EXPERIMENT_A,
    EXPERIMENT_B,
    EXPERIMENT_C,
    analyze_query_experiments,
    comparison_findings,
    comparison_matrix_rows,
    evaluate_rerank_comparison,
    gain_rows,
    query_inspection_rows,
    run_rerank_comparison_experiment,
    summarize_rerank_comparison,
)
from src.evaluation.run_evaluation import build_evaluation_retrievers
from src.retrieval.hybrid import HybridRetriever, RetrievalResult
from src.retrieval.reranker import CrossEncoderReranker
import numpy as np


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


@dataclass(frozen=True)
class FakeHit:
    passage_id: object
    score: float
    text: str


class FakeHybrid:
    def __init__(self, rankings: dict[str, list[FakeHit]]):
        self.rankings = rankings
        self.calls: list[tuple[str, int | None]] = []
        self.last_trace: dict = {"bm25_ms": 1.5, "dense_ms": 2.5, "rrf_ms": 0.25}

    def search(self, query: str, top_k: int | None = None) -> list[FakeHit]:
        self.calls.append((query, top_k))
        hits = self.rankings[query]
        if top_k is None:
            return list(hits)
        return list(hits[:top_k])


class DictScorer:
    def __init__(self, scores_by_id: dict[object, float]):
        self.scores_by_id = scores_by_id

    def score_pairs(self, query: str, texts):
        return [float(self.scores_by_id.get(text, 0.0)) for text in texts]


def _fillers(prefix: str, count: int) -> list[FakeHit]:
    return [FakeHit(f"{prefix}{i}", 1.0 / (i + 1), f"{prefix}{i} text") for i in range(count)]


def test_analyze_query_separates_retrieval_failure_from_ranking_failure():
    missed = analyze_query_experiments(
        EvaluationQuery(1, "missing", ("GOLD",)),
        baseline_ids=["a", "b"],
        pool_ids=["a", "b", "c"],
        reranked_ids=["a", "b"],
    )
    assert missed["candidate_hit_at_50"] is False
    assert missed["rerank_recovery_at_10"] is None
    assert missed["was_candidate_retrieved"] is False

    recovered = analyze_query_experiments(
        EvaluationQuery(2, "recovered", ("GOLD",)),
        baseline_ids=["a"] + [f"n{i}" for i in range(9)],
        pool_ids=["a"] + [f"n{i}" for i in range(13)] + ["GOLD"],
        reranked_ids=["GOLD"] + [f"n{i}" for i in range(9)],
    )
    assert recovered["candidate_hit_at_50"] is True
    assert recovered["rerank_recovery_at_10"] == 1.0
    assert recovered["best_rrf_rank"] == 15
    assert recovered["best_reranked_rank"] == 1
    assert recovered["rank_delta"] == 14
    assert recovered["was_relevant_passage_promoted"] is True
    assert recovered["rank_movements"][0]["rrf_rank"] == 15
    assert recovered["rank_movements"][0]["reranked_rank"] == 1

    dropped = analyze_query_experiments(
        EvaluationQuery(3, "dropped", ("GOLD",)),
        baseline_ids=["GOLD"] + [f"n{i}" for i in range(9)],
        pool_ids=["GOLD"] + [f"n{i}" for i in range(20)],
        reranked_ids=[f"n{i}" for i in range(10)],
    )
    assert dropped["candidate_hit_at_50"] is True
    assert dropped["rerank_recovery_at_10"] == 0.0
    assert dropped["was_relevant_passage_promoted"] is False
    assert dropped["rank_movements"] == []


def test_summarize_preserves_a_rerank_regression_and_reports_ceiling():
    rows = [
        {
            **analyze_query_experiments(
                EvaluationQuery(1, "q", ("GOLD",)),
                baseline_ids=["GOLD"] + [f"n{i}" for i in range(9)],
                pool_ids=["GOLD"] + [f"n{i}" for i in range(20)],
                reranked_ids=[f"n{i}" for i in range(10)],
            ),
            "latency": {
                "baseline_ms": 10.0,
                "hybrid_ms": 20.0,
                "rerank_ms": 5.0,
                "total_reranked_ms": 25.0,
            },
        }
    ]
    summary = summarize_rerank_comparison(rows)
    baseline = summary["configurations"][0]
    reranked = summary["configurations"][2]
    assert baseline["name"] == EXPERIMENT_A
    assert reranked["name"] == EXPERIMENT_C
    assert reranked["recall_at_10"] < baseline["recall_at_10"]
    assert summary["rerank_gain"]["recall_at_10"]["absolute"] < 0
    assert summary["candidate_ceiling"]["candidate_recall_at_50"] == 1.0
    assert summary["candidate_ceiling"]["gap"] == 1.0
    assert summary["rerank_recovery"]["rerank_recovery_at_10"] == 0.0
    assert summary["latency"]["rerank_avg_ms"] == 5.0
    assert summary["latency"]["total_reranked_avg_ms"] == 25.0
    findings = " ".join(summary["findings"])
    assert "did not improve" in findings.lower() or "worsened" in findings.lower()


def test_comparison_findings_call_out_mixed_gains():
    gain = {
        "recall_at_5": {"absolute": 0.1, "relative": 0.2, "baseline": 0.5, "reranked": 0.6},
        "recall_at_10": {"absolute": -0.05, "relative": -0.1, "baseline": 0.5, "reranked": 0.45},
        "mrr_at_10": {"absolute": 0.0, "relative": 0.0, "baseline": 0.8, "reranked": 0.8},
        "ndcg_at_10": {"absolute": 0.02, "relative": 0.04, "baseline": 0.5, "reranked": 0.52},
    }
    notes = comparison_findings(
        gain,
        {"candidate_recall_at_50": 0.8, "reranked_recall_at_10": 0.45, "gap": 0.35},
        {"rerank_recovery_at_10": 0.5, "n_queries_with_relevant_in_pool": 10},
        {
            "baseline_avg_ms": 10.0,
            "hybrid_avg_ms": 20.0,
            "rerank_avg_ms": 30.0,
            "total_reranked_avg_ms": 50.0,
        },
    )
    joined = " ".join(notes)
    assert "mixed" in joined
    assert "recall_at_5" in joined
    assert "recall_at_10" in joined


def test_three_way_experiment_uses_existing_metrics_and_medcpt_interface():
    recover_gold = FakeHit("GOLD", 0.01, "recover gold")
    drop_gold = FakeHit("GOLD", 0.01, "drop gold")
    recovered_pool = _fillers("r", 14) + [recover_gold] + _fillers("t", 5)
    missed_pool = _fillers("m", 12)
    dropped_pool = [drop_gold] + _fillers("d", 11)
    hybrid = FakeHybrid(
        {
            "promote gold": recovered_pool,
            "never retrieved": missed_pool,
            "drop gold": dropped_pool,
        }
    )
    scorer = DictScorer(
        {
            "recover gold": 9.0,
            "drop gold": 0.0,
            **{hit.text: 0.1 for hit in recovered_pool if hit.passage_id != "GOLD"},
            **{hit.text: 0.1 for hit in missed_pool},
            **{hit.text: 8.0 for hit in dropped_pool if hit.passage_id != "GOLD"},
        }
    )
    queries = [
        EvaluationQuery("q1", "promote gold", ("GOLD",)),
        EvaluationQuery("q2", "never retrieved", ("GOLD",)),
        EvaluationQuery("q3", "drop gold", ("GOLD",)),
    ]
    rows = evaluate_rerank_comparison(
        queries,
        hybrid=hybrid,
        reranker=CrossEncoderReranker(scorer),
        baseline_k=10,
        pool_k=50,
        final_k=10,
    )
    by_question = {row["question"]: row for row in rows}
    assert hybrid.calls[0] == ("promote gold", 10)
    assert hybrid.calls[1] == ("promote gold", 50)
    assert by_question["promote gold"]["baseline"]["recall_at_10"] == 0.0
    assert by_question["promote gold"]["candidate_hit_at_50"] is True
    assert by_question["promote gold"]["reranked"]["recall_at_10"] == 1.0
    assert by_question["promote gold"]["rerank_recovery_at_10"] == 1.0
    assert by_question["never retrieved"]["rerank_recovery_at_10"] is None
    assert by_question["drop gold"]["baseline"]["recall_at_10"] == 1.0
    assert by_question["drop gold"]["reranked"]["recall_at_10"] == 0.0
    assert by_question["drop gold"]["rerank_recovery_at_10"] == 0.0
    assert rows[0]["latency"]["bm25_ms"] == 1.5
    assert rows[0]["latency"]["total_reranked_ms"] == (
        rows[0]["latency"]["hybrid_ms"] + rows[0]["latency"]["rerank_ms"]
    )
    for row in rows:
        expected = compute_retrieval_metrics(
            row["reranked"]["retrieved_passage_ids"],
            row["relevant_passage_ids"],
        )
        assert row["reranked"]["recall_at_5"] == expected["recall_at_5"]
        assert row["reranked"]["recall_at_10"] == expected["recall_at_10"]
        assert row["reranked"]["mrr_at_10"] == expected["mrr_at_10"]
        assert row["reranked"]["ndcg_at_10"] == expected["ndcg_at_10"]

    summary = summarize_rerank_comparison(rows)
    matrix = comparison_matrix_rows(summary)
    metrics = [row["Metric"] for row in matrix]
    assert metrics == [
        "Recall@5",
        "Recall@10",
        "MRR@10",
        "nDCG@10",
        "Candidate Recall@50",
        "Candidate Hit@50",
        "Latency (ms)",
    ]
    assert matrix[0][EXPERIMENT_A] is not None
    assert matrix[4][EXPERIMENT_A] is None
    inspection = query_inspection_rows(rows)
    assert inspection[0]["Was candidate retrieved?"] == "Yes"
    assert inspection[1]["Was candidate retrieved?"] == "No"
    assert "Absolute change" in gain_rows(summary)[0]


def test_run_rerank_comparison_writes_dedicated_files_not_baseline(tmp_path):
    passages = prepare_corpus(
        [
            {"id": 20598273, "passage": "Hirschsprung disease of the intestine."},
            {"id": 42, "passage": "Insulin regulates blood glucose."},
            {"id": 1001, "passage": "Papilin is a secreted extracellular matrix protein."},
            {"id": 99, "passage": "EGFR ligands include epidermal growth factor."},
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
    texts = {passage.passage_id: passage.text for passage in passages}

    class PreferGoldScorer:
        def score_pairs(self, query: str, texts_in):
            scores = []
            for text in texts_in:
                if text == texts[20598273] or text == texts[1001]:
                    scores.append(9.0)
                else:
                    scores.append(0.1)
            return scores

    details = tmp_path / "rerank_comparison_details.json"
    summary = tmp_path / "rerank_comparison_summary.json"
    baseline_details = tmp_path / "retrieval_evaluation_details.json"
    result = run_rerank_comparison_experiment(
        passages=passages,
        queries=queries,
        hybrid=built["Hybrid RRF"],
        scorer=PreferGoldScorer(),
        config=RerankerConfig(enabled=True, candidate_pool_size=50, final_top_k=10, device="cpu"),
        queries_path=tmp_path / "evaluation_100.json",
        details_path=details,
        summary_path=summary,
        index_build_seconds=timings,
    )
    assert result["details_path"] == details
    assert result["summary_path"] == summary
    assert details.exists()
    assert summary.exists()
    assert not baseline_details.exists()
    assert result["summary_path"].name != EVALUATION.details_path.split("/")[-1]
    assert result["reranker"]["openai_used"] is False
    payload = summary.read_text(encoding="utf-8")
    assert EXPERIMENT_A in payload
    assert EXPERIMENT_B in payload
    assert EXPERIMENT_C in payload
    assert "rerank_gain" in payload
    assert isinstance(built["Hybrid RRF"], HybridRetriever)
    assert any(
        isinstance(hit, RetrievalResult)
        for hit in built["Hybrid RRF"].search("papilin", top_k=2)
    )
