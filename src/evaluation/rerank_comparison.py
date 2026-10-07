"""Three-way Hybrid vs candidate-pool vs MedCPT rerank evaluation.

Experiment A: Hybrid RRF → Top 10 (baseline)
Experiment B: Hybrid RRF → Top 50 (candidate-pool ceiling)
Experiment C: Hybrid RRF → Top 50 → MedCPT → Top 10

Uses the existing MedCPT CrossEncoderReranker. Does not rewrite baseline
BM25/Dense/Hybrid evaluation files. Ranking metrics keep the existing
Recall@5 / Recall@10 / MRR@10 / nDCG@10 definitions.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import EVALUATION, RERANKER, RerankerConfig
from src.data.loader import get_passage_collection
from src.data.preprocessing import Passage
from src.evaluation.dataset import EvaluationQuery, load_or_create_fixed_queries
from src.evaluation.metrics import (
    candidate_ceiling_gap,
    candidate_hit_at_k,
    candidate_recall_at_k,
    compute_retrieval_metrics,
    first_relevant_rank,
    metric_gain,
    rank_delta,
    rank_of_id,
    rerank_recovery_at_k,
)
from src.evaluation.run_evaluation import build_evaluation_retrievers
from src.retrieval.reranker import CrossEncoderReranker, PairScorer, get_shared_scorer

EXPERIMENT_A = "Baseline"
EXPERIMENT_B = "Hybrid@50"
EXPERIMENT_C = "+ MedCPT"
RANKING_METRICS = ("recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10")


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (bytes, str, bool)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def _hit_ids(hits: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(getattr(item, "passage_id") for item in hits)


def _hit_scores(hits: Sequence[Any]) -> tuple[float, ...]:
    return tuple(float(getattr(item, "score", 0.0)) for item in hits)


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def relevant_rank_movements(
    pool_ids: Sequence[Any],
    reranked_ids: Sequence[Any],
    relevant_ids: Sequence[Any],
) -> list[dict[str, Any]]:
    """RRF vs reranked ranks for golden IDs present in both lists."""

    movements: list[dict[str, Any]] = []
    for passage_id in relevant_ids:
        original = rank_of_id(pool_ids, passage_id)
        final = rank_of_id(reranked_ids, passage_id)
        if original is None or final is None:
            continue
        movements.append(
            {
                "passage_id": _jsonable(passage_id),
                "rrf_rank": original,
                "reranked_rank": final,
                "rank_delta": rank_delta(original, final),
            }
        )
    return movements


def analyze_query_experiments(
    query: EvaluationQuery,
    *,
    baseline_ids: Sequence[Any],
    pool_ids: Sequence[Any],
    reranked_ids: Sequence[Any],
    pool_k: int = 50,
    final_k: int = 10,
) -> dict[str, Any]:
    """Per-query ranking, pool, recovery, and rank-movement records."""

    relevant = tuple(query.relevant_passage_ids)
    baseline_metrics = compute_retrieval_metrics(baseline_ids, relevant)
    pool_metrics = compute_retrieval_metrics(pool_ids, relevant)
    reranked_metrics = compute_retrieval_metrics(reranked_ids, relevant)
    best_rrf_rank = first_relevant_rank(pool_ids, relevant)
    best_reranked_rank = first_relevant_rank(reranked_ids, relevant)
    delta = rank_delta(best_rrf_rank, best_reranked_rank)
    in_pool = candidate_hit_at_k(pool_ids, relevant, pool_k)
    return {
        "query_id": _jsonable(query.question_id),
        "question": query.question,
        "relevant_passage_ids": [_jsonable(item) for item in relevant],
        "baseline": {
            **baseline_metrics,
            "retrieved_passage_ids": [_jsonable(item) for item in baseline_ids],
        },
        "hybrid_50": {
            **pool_metrics,
            "retrieved_passage_ids": [_jsonable(item) for item in pool_ids],
        },
        "reranked": {
            **reranked_metrics,
            "retrieved_passage_ids": [_jsonable(item) for item in reranked_ids],
        },
        "candidate_recall_at_50": candidate_recall_at_k(pool_ids, relevant, pool_k),
        "candidate_hit_at_50": in_pool,
        "rerank_recovery_at_10": rerank_recovery_at_k(
            pool_ids,
            reranked_ids,
            relevant,
            pool_k=pool_k,
            final_k=final_k,
        ),
        "best_rrf_rank": best_rrf_rank,
        "best_reranked_rank": best_reranked_rank,
        "rank_delta": delta,
        "was_candidate_retrieved": in_pool,
        "was_relevant_passage_promoted": bool(
            best_rrf_rank is not None
            and best_reranked_rank is not None
            and best_reranked_rank < best_rrf_rank
        ),
        "rank_movements": relevant_rank_movements(pool_ids, reranked_ids, relevant),
    }


def _configuration_summary(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    metric_key: str,
    *,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    latencies = [float(row[metric_key]) for row in rows]
    payload = {
        "name": name,
        "n_queries": len(rows),
        "recall_at_5": _mean([float(row["recall_at_5"]) for row in rows]),
        "recall_at_10": _mean([float(row["recall_at_10"]) for row in rows]),
        "mrr_at_10": _mean([float(row["mrr_at_10"]) for row in rows]),
        "ndcg_at_10": _mean([float(row["ndcg_at_10"]) for row in rows]),
        "avg_latency_ms": _mean(latencies),
        "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
    }
    if extra:
        payload.update(dict(extra))
    return payload


def summarize_rerank_comparison(
    query_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate A/B/C metrics, gain, ceiling, recovery, and latency."""

    if not query_rows:
        raise ValueError("Cannot summarize an empty rerank comparison")

    baseline_rows = [
        {**row["baseline"], "latency_ms": row["latency"]["baseline_ms"]}
        for row in query_rows
    ]
    pool_rows = [
        {**row["hybrid_50"], "latency_ms": row["latency"]["hybrid_ms"]}
        for row in query_rows
    ]
    reranked_rows = [
        {**row["reranked"], "latency_ms": row["latency"]["total_reranked_ms"]}
        for row in query_rows
    ]
    candidate_recalls = [float(row["candidate_recall_at_50"]) for row in query_rows]
    candidate_hits = [1.0 if row["candidate_hit_at_50"] else 0.0 for row in query_rows]
    recoveries = [
        float(row["rerank_recovery_at_10"])
        for row in query_rows
        if row["rerank_recovery_at_10"] is not None
    ]
    deltas = [
        int(row["rank_delta"])
        for row in query_rows
        if row.get("rank_delta") is not None
    ]
    passage_deltas = [
        int(item["rank_delta"])
        for row in query_rows
        for item in row.get("rank_movements") or []
        if item.get("rank_delta") is not None
    ]
    hybrid_ms = [float(row["latency"]["hybrid_ms"]) for row in query_rows]
    rerank_ms = [float(row["latency"]["rerank_ms"]) for row in query_rows]
    total_ms = [float(row["latency"]["total_reranked_ms"]) for row in query_rows]
    baseline_ms = [float(row["latency"]["baseline_ms"]) for row in query_rows]

    baseline = _configuration_summary(EXPERIMENT_A, baseline_rows, "latency_ms")
    hybrid_50 = _configuration_summary(
        EXPERIMENT_B,
        pool_rows,
        "latency_ms",
        extra={
            "candidate_recall_at_50": _mean(candidate_recalls),
            "candidate_hit_at_50": _mean(candidate_hits),
        },
    )
    reranked = _configuration_summary(
        EXPERIMENT_C,
        reranked_rows,
        "latency_ms",
        extra={
            "candidate_recall_at_50": _mean(candidate_recalls),
            "candidate_hit_at_50": _mean(candidate_hits),
            "avg_hybrid_ms": _mean(hybrid_ms),
            "avg_rerank_ms": _mean(rerank_ms),
        },
    )
    gain = metric_gain(reranked, baseline)
    ceiling_recall_50 = _mean(candidate_recalls)
    ceiling = {
        "candidate_recall_at_50": ceiling_recall_50,
        "reranked_recall_at_10": reranked["recall_at_10"],
        "gap": candidate_ceiling_gap(ceiling_recall_50, reranked["recall_at_10"]),
    }
    recovery = {
        "n_queries_with_relevant_in_pool": len(recoveries),
        "n_queries": len(query_rows),
        "rerank_recovery_at_10": _mean(recoveries) if recoveries else None,
    }
    latency = {
        "baseline_avg_ms": _mean(baseline_ms),
        "hybrid_avg_ms": _mean(hybrid_ms),
        "rerank_avg_ms": _mean(rerank_ms),
        "total_reranked_avg_ms": _mean(total_ms),
        "overhead_vs_baseline_ms": _mean(total_ms) - _mean(baseline_ms),
        "baseline_median_ms": statistics.median(baseline_ms) if baseline_ms else 0.0,
        "hybrid_median_ms": statistics.median(hybrid_ms) if hybrid_ms else 0.0,
        "rerank_median_ms": statistics.median(rerank_ms) if rerank_ms else 0.0,
        "total_reranked_median_ms": statistics.median(total_ms) if total_ms else 0.0,
    }
    return {
        "configurations": [baseline, hybrid_50, reranked],
        "rerank_gain": gain,
        "candidate_ceiling": ceiling,
        "rerank_recovery": recovery,
        "latency": latency,
        "candidate_recall_at_50": ceiling_recall_50,
        "candidate_hit_at_50": _mean(candidate_hits),
        "mean_best_rank_delta": _mean(deltas) if deltas else None,
        "n_queries_with_rank_delta": len(deltas),
        "mean_passage_rank_delta": _mean(passage_deltas) if passage_deltas else None,
        "n_passages_in_pool_and_reranked": len(passage_deltas),
        "findings": comparison_findings(gain, ceiling, recovery, latency),
    }


def comparison_findings(
    gain: Mapping[str, Mapping[str, float | None]],
    ceiling: Mapping[str, Any],
    recovery: Mapping[str, Any],
    latency: Mapping[str, Any],
) -> list[str]:
    """Describe measured A vs C results without assuming MedCPT is better."""

    improved = [key for key in RANKING_METRICS if (gain[key]["absolute"] or 0) > 0]
    worsened = [key for key in RANKING_METRICS if (gain[key]["absolute"] or 0) < 0]
    unchanged = [key for key in RANKING_METRICS if (gain[key]["absolute"] or 0) == 0]
    notes: list[str] = []
    if improved and not worsened:
        notes.append(
            "MedCPT improved all four ranking metrics versus Hybrid RRF Top-10 on this set."
        )
    elif worsened and not improved:
        notes.append(
            "MedCPT did not improve ranking quality versus Hybrid RRF Top-10 on this set."
        )
    elif improved and worsened:
        notes.append(
            "MedCPT is mixed versus Hybrid RRF Top-10: improved "
            + ", ".join(improved)
            + "; worsened "
            + ", ".join(worsened)
            + "."
        )
    else:
        notes.append("MedCPT matched Hybrid RRF Top-10 on the four ranking metrics.")
    if unchanged and (improved or worsened):
        notes.append("Unchanged: " + ", ".join(unchanged) + ".")
    recovery_rate = recovery.get("rerank_recovery_at_10")
    n_pool = recovery.get("n_queries_with_relevant_in_pool", 0)
    if recovery_rate is None:
        notes.append("Rerank recovery is undefined: no query had a relevant passage in the Hybrid top-50 pool.")
    else:
        notes.append(
            f"Rerank recovery@10 is {recovery_rate:.4f} on {n_pool} queries "
            "with at least one relevant passage in the candidate pool."
        )
    notes.append(
        "Candidate ceiling: Recall@50 "
        f"{ceiling['candidate_recall_at_50']:.4f} vs reranked Recall@10 "
        f"{ceiling['reranked_recall_at_10']:.4f} (gap {ceiling['gap']:.4f})."
    )
    notes.append(
        "Latency: baseline "
        f"{latency['baseline_avg_ms']:.1f} ms vs reranked total "
        f"{latency['total_reranked_avg_ms']:.1f} ms "
        f"(hybrid {latency['hybrid_avg_ms']:.1f} ms + MedCPT "
        f"{latency['rerank_avg_ms']:.1f} ms)."
    )
    return notes


def comparison_matrix_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Metric rows with Baseline / Hybrid@50 / + MedCPT columns."""

    by_name = {item.get("name"): item for item in summary.get("configurations") or []}
    baseline = by_name.get(EXPERIMENT_A, {})
    hybrid_50 = by_name.get(EXPERIMENT_B, {})
    reranked = by_name.get(EXPERIMENT_C, {})
    candidate_recall = summary.get("candidate_recall_at_50")
    candidate_hit = summary.get("candidate_hit_at_50")
    latency = summary.get("latency") or {}

    def cell(config: Mapping[str, Any], key: str, fallback: Any = None) -> Any:
        if key in config and config[key] is not None:
            return config[key]
        return fallback

    return [
        {
            "Metric": "Recall@5",
            EXPERIMENT_A: cell(baseline, "recall_at_5"),
            EXPERIMENT_B: cell(hybrid_50, "recall_at_5"),
            EXPERIMENT_C: cell(reranked, "recall_at_5"),
        },
        {
            "Metric": "Recall@10",
            EXPERIMENT_A: cell(baseline, "recall_at_10"),
            EXPERIMENT_B: cell(hybrid_50, "recall_at_10"),
            EXPERIMENT_C: cell(reranked, "recall_at_10"),
        },
        {
            "Metric": "MRR@10",
            EXPERIMENT_A: cell(baseline, "mrr_at_10"),
            EXPERIMENT_B: cell(hybrid_50, "mrr_at_10"),
            EXPERIMENT_C: cell(reranked, "mrr_at_10"),
        },
        {
            "Metric": "nDCG@10",
            EXPERIMENT_A: cell(baseline, "ndcg_at_10"),
            EXPERIMENT_B: cell(hybrid_50, "ndcg_at_10"),
            EXPERIMENT_C: cell(reranked, "ndcg_at_10"),
        },
        {
            "Metric": "Candidate Recall@50",
            EXPERIMENT_A: None,
            EXPERIMENT_B: cell(hybrid_50, "candidate_recall_at_50", candidate_recall),
            EXPERIMENT_C: cell(reranked, "candidate_recall_at_50", candidate_recall),
        },
        {
            "Metric": "Candidate Hit@50",
            EXPERIMENT_A: None,
            EXPERIMENT_B: cell(hybrid_50, "candidate_hit_at_50", candidate_hit),
            EXPERIMENT_C: cell(reranked, "candidate_hit_at_50", candidate_hit),
        },
        {
            "Metric": "Latency (ms)",
            EXPERIMENT_A: cell(baseline, "avg_latency_ms", latency.get("baseline_avg_ms")),
            EXPERIMENT_B: cell(hybrid_50, "avg_latency_ms", latency.get("hybrid_avg_ms")),
            EXPERIMENT_C: cell(reranked, "avg_latency_ms", latency.get("total_reranked_avg_ms")),
        },
    ]


def gain_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    labels = {
        "recall_at_5": "Recall@5",
        "recall_at_10": "Recall@10",
        "mrr_at_10": "MRR@10",
        "ndcg_at_10": "nDCG@10",
    }
    rows = []
    for key, label in labels.items():
        item = (summary.get("rerank_gain") or {}).get(key) or {}
        rows.append(
            {
                "Metric": label,
                "Baseline": item.get("baseline"),
                "+ MedCPT": item.get("reranked"),
                "Absolute change": item.get("absolute"),
                "Relative change": item.get("relative"),
            }
        )
    return rows


def query_inspection_rows(query_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in query_rows:
        relevant = item.get("relevant_passage_ids") or []
        rows.append(
            {
                "Query": item.get("question", ""),
                "Query ID": item.get("query_id"),
                "Relevant passage IDs": ", ".join(str(value) for value in relevant),
                "Best RRF rank": item.get("best_rrf_rank"),
                "Best reranked rank": item.get("best_reranked_rank"),
                "Was candidate retrieved?": "Yes" if item.get("was_candidate_retrieved") else "No",
                "Was relevant passage promoted?": (
                    "Yes" if item.get("was_relevant_passage_promoted") else "No"
                ),
                "Rank delta": item.get("rank_delta"),
            }
        )
    return rows


def _hybrid_trace(retriever: Any) -> dict[str, Any]:
    trace = dict(getattr(retriever, "last_trace", {}) or {})
    measured = {}
    for key in ("bm25_ms", "dense_ms", "rrf_ms"):
        value = trace.get(key)
        if value is not None:
            measured[key] = float(value)
    return measured


def evaluate_rerank_comparison(
    queries: Sequence[EvaluationQuery],
    *,
    hybrid: Any,
    reranker: CrossEncoderReranker,
    baseline_k: int = 10,
    pool_k: int = 50,
    final_k: int = 10,
) -> list[dict[str, Any]]:
    """Run A, B, and C on the same queries. C reranks B's candidate list."""

    rows: list[dict[str, Any]] = []
    for index, query in enumerate(queries, start=1):
        started = time.perf_counter()
        baseline_hits = list(hybrid.search(query.question, top_k=baseline_k))
        baseline_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        pool_hits = list(hybrid.search(query.question, top_k=pool_k))
        hybrid_ms = (time.perf_counter() - started) * 1000.0
        hybrid_parts = _hybrid_trace(hybrid)
        bm25_ranks = dict(getattr(hybrid, "last_trace", {}).get("bm25_ranks") or {})
        dense_ranks = dict(getattr(hybrid, "last_trace", {}).get("dense_ranks") or {})

        started = time.perf_counter()
        reranked_hits = list(
            reranker.rerank(
                query.question,
                pool_hits,
                top_k=final_k,
                bm25_ranks=bm25_ranks,
                dense_ranks=dense_ranks,
            )
        )
        rerank_ms = (time.perf_counter() - started) * 1000.0
        analyzed = analyze_query_experiments(
            query,
            baseline_ids=_hit_ids(baseline_hits),
            pool_ids=_hit_ids(pool_hits),
            reranked_ids=_hit_ids(reranked_hits),
            pool_k=pool_k,
            final_k=final_k,
        )
        analyzed["baseline"]["retrieved_scores"] = list(_hit_scores(baseline_hits))
        analyzed["hybrid_50"]["retrieved_scores"] = list(_hit_scores(pool_hits))
        analyzed["reranked"]["retrieved_scores"] = list(_hit_scores(reranked_hits))
        analyzed["latency"] = {
            "baseline_ms": baseline_ms,
            "hybrid_ms": hybrid_ms,
            "rerank_ms": rerank_ms,
            "total_reranked_ms": hybrid_ms + rerank_ms,
            **hybrid_parts,
        }
        rows.append(analyzed)
        if index == len(queries) or index % 10 == 0:
            print(
                f"  rerank comparison: {index}/{len(queries)} queries",
                flush=True,
            )
    return rows


def run_rerank_comparison_experiment(
    *,
    passages: Sequence[Passage] | None = None,
    queries: Sequence[EvaluationQuery] | None = None,
    hybrid: Any | None = None,
    scorer: PairScorer | None = None,
    reranker: CrossEncoderReranker | None = None,
    config: RerankerConfig | None = None,
    baseline_k: int | None = None,
    pool_k: int | None = None,
    final_k: int | None = None,
    queries_path: str | Path | None = None,
    details_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    index_build_seconds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Run the three-way experiment and write dedicated result files."""

    baseline_k = EVALUATION.top_k if baseline_k is None else baseline_k
    pool_k = RERANKER.candidate_pool_size if pool_k is None else pool_k
    final_k = EVALUATION.top_k if final_k is None else final_k
    queries_path = Path(queries_path or EVALUATION.queries_path)
    details_path = Path(details_path or EVALUATION.rerank_comparison_details_path)
    summary_path = Path(summary_path or EVALUATION.rerank_comparison_summary_path)
    rerank_config = config or RerankerConfig(
        enabled=True,
        candidate_pool_size=pool_k,
        final_top_k=final_k,
        model_name=RERANKER.model_name,
        device=RERANKER.device,
        batch_size=RERANKER.batch_size,
        max_length=RERANKER.max_length,
    )

    if queries is None:
        queries = load_or_create_fixed_queries(path=queries_path)
    if hybrid is None:
        if passages is None:
            print("Loading the full BioASQ passage corpus...", flush=True)
            passages = get_passage_collection()
        print(
            f"Building BM25 and dense indexes once over {len(passages)} passages...",
            flush=True,
        )
        built, timings = build_evaluation_retrievers(passages, top_k=baseline_k)
        index_build_seconds = dict(timings)
        hybrid = built["Hybrid RRF"]
    if reranker is None:
        reranker = CrossEncoderReranker(scorer=scorer or get_shared_scorer(rerank_config))
        reranker.preload()

    n_passages = len(passages) if passages is not None else 0
    print(
        f"Evaluating Hybrid@10 vs Hybrid@{pool_k} vs MedCPT@{final_k} "
        f"on {len(queries)} queries from {queries_path}...",
        flush=True,
    )
    query_rows = evaluate_rerank_comparison(
        queries,
        hybrid=hybrid,
        reranker=reranker,
        baseline_k=baseline_k,
        pool_k=pool_k,
        final_k=final_k,
    )
    summary = summarize_rerank_comparison(query_rows)
    extra = {
        "experiment": {
            "a": f"Hybrid RRF → Top {baseline_k}",
            "b": f"Hybrid RRF → Top {pool_k}",
            "c": f"Hybrid RRF → Top {pool_k} → MedCPT → Top {final_k}",
            "model_name": rerank_config.model_name,
            "openai_used": False,
            "baseline_files_preserved": True,
        },
        "reranker": {
            "enabled": True,
            "model_name": rerank_config.model_name,
            "candidate_pool_size": pool_k,
            "final_top_k": final_k,
            "device": rerank_config.device,
            "openai_used": False,
        },
        "index_build_seconds": dict(index_build_seconds or {}),
        "n_passages": n_passages,
        "queries_path": str(queries_path),
        "query_ids": [_jsonable(query.question_id) for query in queries],
    }
    details_payload = {
        **extra,
        "n_queries": len(query_rows),
        "rows": query_rows,
        "summary": {
            "rerank_gain": summary["rerank_gain"],
            "candidate_ceiling": summary["candidate_ceiling"],
            "rerank_recovery": summary["rerank_recovery"],
            "latency": summary["latency"],
            "findings": summary["findings"],
        },
    }
    summary_payload = {
        **extra,
        "n_queries": len(query_rows),
        "configurations": summary["configurations"],
        "rerank_gain": summary["rerank_gain"],
        "candidate_ceiling": summary["candidate_ceiling"],
        "rerank_recovery": summary["rerank_recovery"],
        "latency": summary["latency"],
        "candidate_recall_at_50": summary["candidate_recall_at_50"],
        "candidate_hit_at_50": summary["candidate_hit_at_50"],
        "mean_best_rank_delta": summary["mean_best_rank_delta"],
        "n_queries_with_rank_delta": summary["n_queries_with_rank_delta"],
        "mean_passage_rank_delta": summary["mean_passage_rank_delta"],
        "n_passages_in_pool_and_reranked": summary["n_passages_in_pool_and_reranked"],
        "findings": summary["findings"],
        "comparison_table": comparison_matrix_rows(summary),
        "gain_table": gain_rows(summary),
        "query_inspection": query_inspection_rows(query_rows),
    }
    details_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text(json.dumps(details_payload, indent=2) + "\n", encoding="utf-8")
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    print("Rerank comparison", flush=True)
    for line in summary["findings"]:
        print(f"  {line}", flush=True)
    print(f"Detailed comparison: {details_path}", flush=True)
    print(f"Summary comparison: {summary_path}", flush=True)
    return {
        "n_passages": n_passages,
        "n_queries": len(query_rows),
        "query_rows": query_rows,
        "summary": summary,
        "details_path": details_path,
        "summary_path": summary_path,
        "queries_path": queries_path,
        "index_build_seconds": dict(index_build_seconds or {}),
        "reranker": extra["reranker"],
        "retrieval_cost_usd": 0.0,
    }
