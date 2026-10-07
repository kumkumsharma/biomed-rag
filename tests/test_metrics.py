import math

import pytest

from src.evaluation.metrics import (
    candidate_ceiling_gap,
    candidate_hit_at_k,
    candidate_recall_at_k,
    compute_retrieval_metrics,
    first_relevant_rank,
    metric_gain,
    mrr_at_k,
    ndcg_at_k,
    rank_delta,
    rank_of_id,
    recall_at_k,
    relative_change,
    rerank_recovery_at_k,
)


def test_recall_at_5_is_the_fraction_of_relevant_ids_in_the_top_5():
    retrieved = [1, 9, 2, 8, 7, 3]
    relevant = [1, 2, 3]
    assert recall_at_k(retrieved, relevant, 5) == 2 / 3


def test_recall_at_10_counts_relevant_ids_anywhere_in_the_top_10():
    retrieved = [1, 9, 2, 8, 7, 3]
    relevant = [1, 2, 3]
    assert recall_at_k(retrieved, relevant, 10) == 1.0


def test_recall_is_zero_when_nothing_relevant_is_retrieved():
    assert recall_at_k([9, 8, 7, 6, 5], [1, 2], 5) == 0.0


def test_recall_treats_string_and_int_passage_ids_as_the_same_id():
    assert recall_at_k(["1", "2"], [1, 2, 3], 5) == 2 / 3


def test_mrr_at_10_is_the_reciprocal_rank_of_the_first_relevant_hit():
    assert mrr_at_k([9, 1, 2], [1, 2], 10) == 0.5
    assert mrr_at_k([1, 9], [1], 10) == 1.0


def test_mrr_at_10_is_zero_when_the_first_relevant_hit_is_outside_top_10():
    retrieved = list(range(20, 0, -1))
    assert mrr_at_k(retrieved, [1], 10) == 0.0


def test_ndcg_at_10_uses_binary_relevance():
    retrieved = ["A", "X", "B"]
    relevant = ["A", "B"]
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert math.isclose(ndcg_at_k(retrieved, relevant, 10), dcg / idcg)


def test_ndcg_is_one_for_a_perfect_binary_ranking():
    assert ndcg_at_k([1, 2, 9], [1, 2], 10) == 1.0


def test_empty_relevant_set_scores_zero():
    assert recall_at_k([1, 2], [], 5) == 0.0
    assert mrr_at_k([1, 2], [], 10) == 0.0
    assert ndcg_at_k([1, 2], [], 10) == 0.0


def test_compute_retrieval_metrics_returns_the_four_required_scores():
    metrics = compute_retrieval_metrics([1, 9, 2, 8, 7, 3], [1, 2, 3])
    assert set(metrics) == {"recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10"}
    assert metrics["recall_at_5"] == 2 / 3
    assert metrics["recall_at_10"] == 1.0
    assert metrics["mrr_at_10"] == 1.0
    assert metrics["ndcg_at_10"] > 0.0


def test_candidate_recall_at_50_uses_the_existing_recall_definition():
    retrieved = list(range(50))
    retrieved[40] = 100
    relevant = [100, 200]
    assert candidate_recall_at_k(retrieved, relevant, 50) == 0.5
    assert candidate_recall_at_k(retrieved, relevant, 50) == recall_at_k(retrieved, relevant, 50)


def test_candidate_hit_at_50_is_true_when_any_relevant_id_is_in_the_pool():
    assert candidate_hit_at_k(list(range(50)), [49], 50) is True
    assert candidate_hit_at_k(list(range(50)), [99], 50) is False
    assert candidate_hit_at_k([1, 2], [], 50) is False


def test_rerank_recovery_is_undefined_when_the_pool_misses_all_relevant_ids():
    assert (
        rerank_recovery_at_k([9, 8, 7], [1, 2], [1], pool_k=50, final_k=10)
        is None
    )


def test_rerank_recovery_is_one_when_medcpt_keeps_a_pooled_relevant_id():
    pool = [9, 1, 8]
    reranked = [1, 9]
    assert rerank_recovery_at_k(pool, reranked, [1, 2], pool_k=50, final_k=10) == 1.0


def test_rerank_recovery_is_zero_when_the_pool_has_relevant_ids_but_top_10_does_not():
    pool = [1] + list(range(20, 40))
    reranked = list(range(20, 30))
    assert rerank_recovery_at_k(pool, reranked, [1], pool_k=50, final_k=10) == 0.0


def test_rank_movement_records_rrf_rank_final_rank_and_positive_delta_on_promotion():
    assert rank_of_id(["A", "B", "GOLD"], "GOLD") == 3
    assert first_relevant_rank(["A", "GOLD", "B"], ["GOLD"]) == 2
    assert rank_delta(15, 1) == 14
    assert rank_delta(2, 8) == -6
    assert rank_delta(3, None) is None


def test_metric_gain_reports_absolute_and_relative_change_including_regressions():
    baseline = {
        "recall_at_5": 0.50,
        "recall_at_10": 0.40,
        "mrr_at_10": 0.80,
        "ndcg_at_10": 0.60,
    }
    reranked = {
        "recall_at_5": 0.55,
        "recall_at_10": 0.32,
        "mrr_at_10": 0.80,
        "ndcg_at_10": 0.66,
    }
    gain = metric_gain(reranked, baseline)
    assert gain["recall_at_5"]["absolute"] == pytest.approx(0.05)
    assert gain["recall_at_5"]["relative"] == pytest.approx(0.10)
    assert gain["recall_at_10"]["absolute"] == pytest.approx(-0.08)
    assert gain["mrr_at_10"]["absolute"] == 0.0
    assert gain["ndcg_at_10"]["absolute"] == pytest.approx(0.06)
    assert relative_change(0.1, 0.0) is None


def test_candidate_ceiling_gap_is_pool_recall_minus_reranked_recall():
    assert candidate_ceiling_gap(0.80, 0.40) == pytest.approx(0.40)
    assert candidate_ceiling_gap(0.20, 0.35) == pytest.approx(-0.15)
