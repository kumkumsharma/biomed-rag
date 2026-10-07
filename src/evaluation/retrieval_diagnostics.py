"""Per-gold BM25 / MiniLM / RRF / MedCPT diagnostics. Does not change ranking."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.data.preprocessing import Passage
from src.evaluation.dataset import EvaluationQuery
from src.evaluation.gold_quality import gold_id_status
from src.evaluation.metrics import (
    candidate_hit_at_k,
    compute_retrieval_metrics,
    passage_id_key,
    rank_of_id,
    recall_at_k,
    rerank_recovery_at_k,
)
from src.retrieval.reranker import CrossEncoderReranker


def _lookup(hits: Sequence[Any], gold_id: Any) -> tuple[int | None, float | None]:
    rank = rank_of_id([getattr(hit, "passage_id", None) for hit in hits], gold_id)
    if rank is None:
        return None, None
    return rank, float(getattr(hits[rank - 1], "score", 0.0))


def _signal_class(*, bm25_found: bool, dense_found: bool) -> str:
    if bm25_found and dense_found:
        return "Both BM25 and Dense"
    if bm25_found:
        return "BM25 only"
    if dense_found:
        return "Dense only"
    return "Neither BM25 nor Dense"


def _rrf_class(*, bm25_found: bool, dense_found: bool, rrf_found: bool) -> str:
    if rrf_found and bm25_found and dense_found:
        return "found by both and retained"
    if rrf_found and bm25_found:
        return "found by BM25 and retained"
    if rrf_found and dense_found:
        return "found by Dense and retained"
    if bm25_found and not rrf_found:
        return "found by BM25 but pushed out by RRF"
    if dense_found and not rrf_found:
        return "found by Dense but pushed out by RRF"
    if not bm25_found and not dense_found:
        return "found by neither"
    return "found by both and retained" if rrf_found else "found by BM25 but pushed out by RRF"


def diagnose_query(
    query: EvaluationQuery,
    *,
    passages: Sequence[Passage],
    bm25: Any,
    dense: Any,
    hybrid: Any,
    reranker: CrossEncoderReranker | None,
    pool_k: int = 50,
    final_k: int = 10,
    inspect_k: int | None = None,
) -> dict[str, Any]:
    inspect_k = inspect_k or max(pool_k, 200)
    by_id = {passage_id_key(item.passage_id): item for item in passages}
    bm25_hits = list(bm25.search(query.question, top_k=inspect_k))
    dense_hits = list(dense.search(query.question, top_k=inspect_k))
    started = time.perf_counter()
    rrf_hits = list(hybrid.search(query.question, top_k=pool_k))
    rrf_ms = (time.perf_counter() - started) * 1000.0
    trace = dict(getattr(hybrid, "last_trace", {}) or {})
    bm25_pool = list(bm25.search(query.question, top_k=pool_k))
    dense_pool = list(dense.search(query.question, top_k=pool_k))
    rerank_hits: list[Any] = []
    rerank_ms = None
    if reranker is not None and rrf_hits:
        started = time.perf_counter()
        rerank_hits = list(
            reranker.rerank(
                query.question,
                rrf_hits,
                top_k=final_k,
                bm25_ranks=dict(trace.get("bm25_ranks") or {}),
                dense_ranks=dict(trace.get("dense_ranks") or {}),
            )
        )
        rerank_ms = (time.perf_counter() - started) * 1000.0

    gold_rows = []
    for gold_id in query.relevant_passage_ids:
        status = gold_id_status(gold_id, by_id)
        bm25_rank, bm25_score = _lookup(bm25_hits, gold_id)
        dense_rank, dense_score = _lookup(dense_hits, gold_id)
        rrf_rank, rrf_score = _lookup(rrf_hits, gold_id)
        med_rank, med_score = _lookup(rerank_hits, gold_id)
        bm25_found = rank_of_id([hit.passage_id for hit in bm25_pool], gold_id) is not None
        dense_found = rank_of_id([hit.passage_id for hit in dense_pool], gold_id) is not None
        rrf_found = rrf_rank is not None
        gold_rows.append(
            {
                "gold_id": gold_id,
                "valid_text": status["valid"],
                "text_status": status["text_status"],
                "bm25_found": bm25_found,
                "bm25_rank": bm25_rank,
                "bm25_score": bm25_score,
                "dense_found": dense_found,
                "dense_rank": dense_rank,
                "dense_score": dense_score,
                "rrf_found": rrf_found,
                "rrf_rank": rrf_rank,
                "rrf_score": rrf_score,
                "signal": _signal_class(bm25_found=bm25_found, dense_found=dense_found),
                "rrf_class": _rrf_class(
                    bm25_found=bm25_found, dense_found=dense_found, rrf_found=rrf_found
                ),
                "medcpt_rank": med_rank,
                "medcpt_score": med_score,
                "in_top50": rrf_found,
                "in_top10": med_rank is not None and med_rank <= 10,
                "in_top5": med_rank is not None and med_rank <= 5,
                "in_top3": med_rank is not None and med_rank <= 3,
                "in_top1": med_rank is not None and med_rank <= 1,
                "lost_by_reranking": rrf_found and (med_rank is None or med_rank > final_k),
            }
        )

    bm25_ids = [hit.passage_id for hit in bm25_hits]
    dense_ids = [hit.passage_id for hit in dense_hits]
    rrf_ids = [hit.passage_id for hit in rrf_hits]
    rerank_ids = [hit.passage_id for hit in rerank_hits]
    all_gold = query.relevant_passage_ids
    valid_gold = tuple(item["gold_id"] for item in gold_rows if item["valid_text"])
    pool_has_gold = candidate_hit_at_k(rrf_ids, all_gold, pool_k)
    return {
        "query_id": query.question_id,
        "question": query.question,
        "n_gold": len(all_gold),
        "n_valid_gold": len(valid_gold),
        "gold": gold_rows,
        "bm25_recall_at_5": recall_at_k(bm25_ids, all_gold, 5),
        "bm25_recall_at_10": recall_at_k(bm25_ids, all_gold, 10),
        "bm25_recall_at_50": recall_at_k(bm25_ids, all_gold, 50),
        "dense_recall_at_5": recall_at_k(dense_ids, all_gold, 5),
        "dense_recall_at_10": recall_at_k(dense_ids, all_gold, 10),
        "dense_recall_at_50": recall_at_k(dense_ids, all_gold, 50),
        "rrf_recall_at_50": recall_at_k(rrf_ids, all_gold, 50),
        "rerank_recall_at_10": recall_at_k(rerank_ids, all_gold, 10),
        "rerank_recall_at_5": recall_at_k(rerank_ids, all_gold, 5),
        "rerank_recall_at_3": recall_at_k(rerank_ids, all_gold, 3),
        "rerank_recall_at_1": recall_at_k(rerank_ids, all_gold, 1),
        "valid_gold_rrf_recall_at_50": recall_at_k(rrf_ids, valid_gold, 50) if valid_gold else None,
        "valid_gold_rerank_recall_at_10": recall_at_k(rerank_ids, valid_gold, 10) if valid_gold else None,
        "candidate_hit_at_50": pool_has_gold,
        "rerank_recovery_at_10": rerank_recovery_at_k(
            rrf_ids, rerank_ids, all_gold, pool_k=pool_k, final_k=final_k
        ),
        "n_gold_lost_by_reranking": sum(1 for item in gold_rows if item["lost_by_reranking"]),
        "query_bm25_hit": any(item["bm25_found"] for item in gold_rows),
        "query_dense_hit": any(item["dense_found"] for item in gold_rows),
        "query_both_hit": any(item["bm25_found"] and item["dense_found"] for item in gold_rows),
        "query_neither_hit": not any(item["bm25_found"] or item["dense_found"] for item in gold_rows),
        "rrf_ms": rrf_ms,
        "rerank_ms": rerank_ms,
        "raw_metrics": compute_retrieval_metrics(rerank_ids, all_gold),
        "valid_gold_metrics": compute_retrieval_metrics(rerank_ids, valid_gold) if valid_gold else None,
    }


def _mean(values: Sequence[float | None]) -> float | None:
    numbers = [float(item) for item in values if item is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def summarize_retrieval_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    golds = [item for row in rows for item in row.get("gold") or []]
    n_gold = len(golds) or 1
    signal_counts = {
        "BM25 only": sum(1 for item in golds if item["signal"] == "BM25 only"),
        "Dense only": sum(1 for item in golds if item["signal"] == "Dense only"),
        "Both BM25 and Dense": sum(1 for item in golds if item["signal"] == "Both BM25 and Dense"),
        "Neither BM25 nor Dense": sum(1 for item in golds if item["signal"] == "Neither BM25 nor Dense"),
    }
    rrf_counts = {}
    for label in (
        "found by BM25 but pushed out by RRF",
        "found by Dense but pushed out by RRF",
        "found by both and retained",
        "found by neither",
        "found by BM25 and retained",
        "found by Dense and retained",
    ):
        rrf_counts[label] = sum(1 for item in golds if item.get("rrf_class") == label)
    n_queries = len(rows)
    return {
        "n_queries": n_queries,
        "n_gold_ids": len(golds),
        "signal_recovery": {
            "BM25_only": signal_counts["BM25 only"] / n_gold,
            "Dense_only": signal_counts["Dense only"] / n_gold,
            "Both": signal_counts["Both BM25 and Dense"] / n_gold,
            "Neither": signal_counts["Neither BM25 nor Dense"] / n_gold,
            "counts": signal_counts,
        },
        "query_hits": {
            "queries_where_BM25_finds_at_least_one_gold": sum(1 for row in rows if row["query_bm25_hit"]),
            "queries_where_Dense_finds_at_least_one_gold": sum(1 for row in rows if row["query_dense_hit"]),
            "queries_where_both_find_gold": sum(1 for row in rows if row["query_both_hit"]),
            "queries_where_neither_finds_gold": sum(1 for row in rows if row["query_neither_hit"]),
        },
        "bm25_recall_at_5": _mean([row["bm25_recall_at_5"] for row in rows]),
        "bm25_recall_at_10": _mean([row["bm25_recall_at_10"] for row in rows]),
        "bm25_recall_at_50": _mean([row["bm25_recall_at_50"] for row in rows]),
        "dense_recall_at_5": _mean([row["dense_recall_at_5"] for row in rows]),
        "dense_recall_at_10": _mean([row["dense_recall_at_10"] for row in rows]),
        "dense_recall_at_50": _mean([row["dense_recall_at_50"] for row in rows]),
        "rrf_recall_at_50": _mean([row["rrf_recall_at_50"] for row in rows]),
        "candidate_recall_at_50": _mean([row["rrf_recall_at_50"] for row in rows]),
        "rerank_recall_at_10": _mean([row["rerank_recall_at_10"] for row in rows]),
        "rerank_recall_at_5": _mean([row["rerank_recall_at_5"] for row in rows]),
        "rerank_recall_at_3": _mean([row["rerank_recall_at_3"] for row in rows]),
        "rerank_recall_at_1": _mean([row["rerank_recall_at_1"] for row in rows]),
        "rerank_recovery_at_10": _mean(
            [row["rerank_recovery_at_10"] for row in rows if row["rerank_recovery_at_10"] is not None]
        ),
        "gold_present_in_top50_but_lost_by_reranking": sum(
            1 for item in golds if item.get("lost_by_reranking")
        ),
        "rrf_loss_counts": rrf_counts,
        "raw_recall_at_10": _mean([row["raw_metrics"]["recall_at_10"] for row in rows]),
        "valid_gold_recall_at_10": _mean(
            [
                (row["valid_gold_metrics"] or {}).get("recall_at_10")
                for row in rows
                if row.get("n_valid_gold")
            ]
        ),
        "valid_gold_recall_at_50": _mean(
            [row["valid_gold_rrf_recall_at_50"] for row in rows if row.get("n_valid_gold")]
        ),
        "queries_affected_by_invalid_gold": sum(1 for row in rows if row["n_valid_gold"] < row["n_gold"]),
        "invalid_gold_ids": sum(row["n_gold"] - row["n_valid_gold"] for row in rows),
    }


def run_retrieval_diagnostics(
    queries: Sequence[EvaluationQuery],
    *,
    passages: Sequence[Passage],
    bm25: Any,
    dense: Any,
    hybrid: Any,
    reranker: CrossEncoderReranker | None,
    pool_k: int = 50,
    final_k: int = 10,
) -> dict[str, Any]:
    rows = []
    for index, query in enumerate(queries, start=1):
        rows.append(
            diagnose_query(
                query,
                passages=passages,
                bm25=bm25,
                dense=dense,
                hybrid=hybrid,
                reranker=reranker,
                pool_k=pool_k,
                final_k=final_k,
            )
        )
        if index == len(queries) or index % 10 == 0:
            print(f"  retrieval diagnostics: {index}/{len(queries)} queries", flush=True)
    summary = summarize_retrieval_diagnostics(rows)
    return {
        "historical_metrics_unchanged": True,
        "raw_original_benchmark": {
            "recall_at_10": summary["raw_recall_at_10"],
            "candidate_recall_at_50": summary["candidate_recall_at_50"],
            "note": "Includes every labelled gold ID, including unusable 'nan' rows.",
        },
        "valid_gold_diagnostic_view": {
            "recall_at_10": summary["valid_gold_recall_at_10"],
            "candidate_recall_at_50": summary["valid_gold_recall_at_50"],
            "invalid_gold_ids": summary["invalid_gold_ids"],
            "queries_affected": summary["queries_affected_by_invalid_gold"],
            "note": "Excludes gold IDs whose corpus text is unusable. Does not replace the raw benchmark.",
        },
        "summary": summary,
        "queries": rows,
    }


def write_retrieval_diagnostics(payload: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(payload), indent=2, default=str) + "\n", encoding="utf-8")
    return output
