"""Structured A–J retrieval comparison. Does not overwrite baseline JSON."""

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

from src.config import EVALUATION, RERANKER, RETRIEVAL, UPGRADE, UpgradeConfig
from src.data.loader import get_passage_collection
from src.data.preprocessing import Passage
from src.evaluation.dataset import EvaluationQuery, load_or_create_fixed_queries
from src.evaluation.failure_analysis import categorize_retrieval_failure, summarize_failures
from src.evaluation.metrics import (
    candidate_hit_at_k,
    candidate_recall_at_k,
    compute_retrieval_metrics,
    recall_at_k,
    rerank_recovery_at_k,
)
from src.evaluation.run_evaluation import build_evaluation_retrievers
from src.retrieval.dense import DenseRetriever, Embedder
from src.retrieval.pipeline import StagedRetriever
from src.retrieval.reranker import CrossEncoderReranker, PairScorer

EXPERIMENT_SPECS = (
    ("A", "Existing Hybrid@10"),
    ("B", "Existing Hybrid@50"),
    ("C", "Hybrid@50 → MedCPT@10"),
    ("D", "Query normalization + Hybrid"),
    ("E", "Query expansion + Hybrid"),
    ("F", "Query decomposition + Hybrid"),
    ("G", "Biomedical embeddings + Hybrid"),
    ("H", "Adaptive retrieval"),
    ("I", "Candidate filtering + MedCPT"),
    ("J", "Full upgraded pipeline"),
)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (bytes, str, bool)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def huggingface_model_cached(model_name: str) -> bool:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    slug = "models--" + model_name.replace("/", "--")
    return (cache_root / slug).exists()


def _hit_ids(hits: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(getattr(item, "passage_id") for item in hits)


def _upgrade(
    *,
    pipeline: bool = True,
    terminology: bool = False,
    expansion: bool = False,
    decomposition: bool = False,
    adaptive: bool = False,
    filtering: bool = False,
    biomedical: bool = False,
    pool: int = 50,
    final_top_k: int = 10,
) -> UpgradeConfig:
    return UpgradeConfig(
        pipeline_enabled=pipeline,
        terminology_enabled=terminology,
        expansion_enabled=expansion,
        llm_expansion_enabled=False,
        decomposition_enabled=decomposition,
        adaptive_routing_enabled=adaptive,
        filtering_enabled=filtering,
        biomedical_embeddings_enabled=biomedical,
        biomedical_embedding_model=UPGRADE.biomedical_embedding_model,
        embedding_cache_dir=UPGRADE.embedding_cache_dir,
        filter_drop_duplicates=True,
        filter_min_token_overlap=0.0,
        filter_drop_empty=True,
        expansion_max_variants=4,
        decomposition_min_aspects=2,
        candidate_pool_size=pool,
        final_top_k=final_top_k,
    )


def evaluate_named_retrievers(
    retrievers: Mapping[str, Any],
    queries: Sequence[EvaluationQuery],
    *,
    top_k: int = 10,
    pool_k: int = 50,
    reranked_names: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    reranked_names = reranked_names or set()
    rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    for name, retriever in retrievers.items():
        print(f"  {name}: starting {len(queries)} queries", flush=True)
        for index, query in enumerate(queries, start=1):
            started = time.perf_counter()
            hits = list(retriever.search(query.question, top_k=top_k))
            latency_ms = (time.perf_counter() - started) * 1000.0
            ids = _hit_ids(hits)
            metrics = compute_retrieval_metrics(ids, query.relevant_passage_ids)
            trace = dict(getattr(retriever, "last_trace", {}) or {})
            fusion_ids = [
                item.get("passage_id") if isinstance(item, dict) else item
                for item in (trace.get("fusion_results") or [])
            ]
            if not fusion_ids:
                fusion_ids = list(ids)
            filter_payload = trace.get("filtering_results") or {}
            after_ids = filter_payload.get("kept_passage_ids")
            if after_ids is None:
                after_count = filter_payload.get("after_count")
                after_ids = fusion_ids[: after_count] if after_count is not None else fusion_ids
            removed_ids = filter_payload.get("removed_passage_ids") or []
            timings = dict(trace.get("stage_timings") or {})
            classification = trace.get("query_classification") or {}
            strategy = trace.get("retrieval_strategy") or {}
            decomposition = (trace.get("understanding") or {}).get("decomposition") or {}
            failure = categorize_retrieval_failure(
                question=query.question,
                relevant_ids=query.relevant_passage_ids,
                pool_ids=fusion_ids,
                filtered_ids=after_ids,
                final_ids=ids,
                query_type=classification.get("query_type"),
                decomposition_applied=bool(decomposition.get("applied")),
                reranked=name in reranked_names or bool(trace.get("reranker_enabled")),
                pool_k=pool_k,
                final_k=top_k,
            )
            row = {
                "query_id": _jsonable(query.question_id),
                "question": query.question,
                "configuration": name,
                "retrieved_passage_ids": [_jsonable(item) for item in ids],
                "latency_ms": latency_ms,
                "recall_at_5": metrics["recall_at_5"],
                "recall_at_10": metrics["recall_at_10"],
                "recall_at_50": recall_at_k(fusion_ids or ids, query.relevant_passage_ids, 50),
                "mrr_at_10": metrics["mrr_at_10"],
                "ndcg_at_10": metrics["ndcg_at_10"],
                "candidate_recall_at_50": candidate_recall_at_k(
                    fusion_ids or ids, query.relevant_passage_ids, pool_k
                ),
                "candidate_hit_at_50": candidate_hit_at_k(
                    fusion_ids or ids, query.relevant_passage_ids, pool_k
                ),
                "rerank_recovery_at_10": rerank_recovery_at_k(
                    fusion_ids or ids,
                    ids,
                    query.relevant_passage_ids,
                    pool_k=pool_k,
                    final_k=top_k,
                )
                if name in reranked_names or trace.get("reranker_enabled")
                else None,
                "candidate_count_before_filter": filter_payload.get("before_count", len(fusion_ids)),
                "candidate_count_after_filter": filter_payload.get("after_count", len(after_ids)),
                "filter_removed_ids": [_jsonable(item) for item in removed_ids],
                "query_type": classification.get("query_type"),
                "strategy": strategy.get("strategy"),
                "search_queries": list(trace.get("search_queries") or [query.question]),
                "stage_timings": timings,
                "failure": failure,
                "trace": {
                    "original_query": trace.get("original_query"),
                    "normalized_query": trace.get("normalized_query"),
                    "expansion_variants": list(trace.get("expansion_variants") or []),
                    "subqueries": list(trace.get("subqueries") or []),
                },
            }
            rows.append(row)
            failure_rows.append(
                {
                    "query_id": _jsonable(query.question_id),
                    "question": query.question,
                    "configuration": name,
                    **failure,
                }
            )
            if index == len(queries) or index % 10 == 0:
                print(f"  {name}: {index}/{len(queries)} queries", flush=True)
    summaries: list[dict[str, Any]] = []
    names = list(dict.fromkeys(row["configuration"] for row in rows))
    for name in names:
        group = [row for row in rows if row["configuration"] == name]
        recoveries = [row["rerank_recovery_at_10"] for row in group if row["rerank_recovery_at_10"] is not None]
        latencies = [row["latency_ms"] for row in group]
        filter_ms = [float((row["stage_timings"] or {}).get("filter_ms") or 0.0) for row in group]
        rerank_ms = [
            float((row["stage_timings"] or {}).get("rerank_ms") or 0.0)
            for row in group
            if (row["stage_timings"] or {}).get("rerank_ms") is not None
        ]
        bm25_ms = [float((row["stage_timings"] or {}).get("bm25_ms") or 0.0) for row in group]
        dense_ms = [float((row["stage_timings"] or {}).get("dense_ms") or 0.0) for row in group]
        failures = [row["failure"] for row in group]
        summaries.append(
            {
                "name": name,
                "n_queries": len(group),
                "recall_at_5": _mean([row["recall_at_5"] for row in group]),
                "recall_at_10": _mean([row["recall_at_10"] for row in group]),
                "recall_at_50": _mean([row["recall_at_50"] for row in group]),
                "mrr_at_10": _mean([row["mrr_at_10"] for row in group]),
                "ndcg_at_10": _mean([row["ndcg_at_10"] for row in group]),
                "candidate_recall_at_50": _mean([row["candidate_recall_at_50"] for row in group]),
                "candidate_hit_at_50": _mean([1.0 if row["candidate_hit_at_50"] else 0.0 for row in group]),
                "rerank_recovery_at_10": _mean(recoveries) if recoveries else None,
                "avg_latency_ms": _mean(latencies),
                "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
                "avg_bm25_ms": _mean(bm25_ms),
                "avg_dense_ms": _mean(dense_ms),
                "avg_filter_ms": _mean(filter_ms),
                "avg_rerank_ms": _mean(rerank_ms) if rerank_ms else None,
                "avg_candidates_before_filter": _mean(
                    [float(row["candidate_count_before_filter"] or 0) for row in group]
                ),
                "avg_candidates_after_filter": _mean(
                    [float(row["candidate_count_after_filter"] or 0) for row in group]
                ),
                "failures": summarize_failures(failures),
            }
        )
    return rows, summaries, summarize_failures(failure_rows)


def build_upgrade_retrievers(
    passages: Sequence[Passage],
    *,
    embedder: Embedder | None = None,
    biomedical_embedder: Embedder | None = None,
    scorer: PairScorer | None = None,
    include_medcpt: bool = False,
    include_biomedical: bool = False,
) -> tuple[dict[str, Any], dict[str, float], list[str]]:
    skipped: list[str] = []
    timings: dict[str, float] = {}
    started = time.perf_counter()
    built, index_timings = build_evaluation_retrievers(passages, top_k=50, embedder=embedder)
    timings.update(index_timings)
    timings["shared_index_seconds"] = time.perf_counter() - started
    bm25 = built["BM25"]
    dense = built["Dense"]
    hybrid = built["Hybrid RRF"]
    texts = {passage.passage_id: passage.text for passage in passages}

    def staged(config: UpgradeConfig, *, rerank: bool = False, dense_retriever=None) -> StagedRetriever:
        return StagedRetriever(
            bm25_retriever=bm25,
            dense_retriever=dense_retriever or dense,
            texts=texts,
            passages=passages,
            config=config,
            scorer=scorer,
            rerank_enabled=rerank,
        )

    retrievers: dict[str, Any] = {
        "A Existing Hybrid@10": _TopK(hybrid, 10),
        "B Existing Hybrid@50": _TopK(hybrid, 50),
        "D Query normalization + Hybrid": staged(_upgrade(terminology=True)),
        "E Query expansion + Hybrid": staged(_upgrade(expansion=True)),
        "F Query decomposition + Hybrid": staged(_upgrade(decomposition=True)),
        "H Adaptive retrieval": staged(_upgrade(terminology=True, expansion=True, decomposition=True, adaptive=True)),
    }
    if include_medcpt:
        medcpt = CrossEncoderReranker(scorer=scorer) if scorer is not None else None
        retrievers["C Hybrid@50 → MedCPT@10"] = staged(
            _upgrade(pool=50, final_top_k=10),
            rerank=True,
        )
        if medcpt is not None:
            retrievers["C Hybrid@50 → MedCPT@10"]._reranker = medcpt
        retrievers["I Candidate filtering + MedCPT"] = staged(
            _upgrade(filtering=True, pool=50, final_top_k=10),
            rerank=True,
        )
        retrievers["J Full upgraded pipeline"] = staged(
            _upgrade(
                terminology=True,
                expansion=True,
                decomposition=True,
                adaptive=True,
                filtering=True,
                biomedical=include_biomedical,
                pool=50,
                final_top_k=10,
            ),
            rerank=True,
            dense_retriever=None,
        )
    else:
        skipped.extend(["C Hybrid@50 → MedCPT@10", "I Candidate filtering + MedCPT", "J Full upgraded pipeline"])

    if include_biomedical:
        started = time.perf_counter()
        bio_dense = DenseRetriever(
            passages,
            embedder=biomedical_embedder,
            model_name=UPGRADE.biomedical_embedding_model,
            top_k=50,
            cache_dir=UPGRADE.embedding_cache_dir,
        )
        timings["biomedical_index_seconds"] = time.perf_counter() - started
        retrievers["G Biomedical embeddings + Hybrid"] = StagedRetriever(
            bm25_retriever=bm25,
            dense_retriever=bio_dense,
            texts=texts,
            passages=passages,
            config=_upgrade(biomedical=True),
            rerank_enabled=False,
        )
        if include_medcpt:
            retrievers["J Full upgraded pipeline"] = staged(
                _upgrade(
                    terminology=True,
                    expansion=True,
                    decomposition=True,
                    adaptive=True,
                    filtering=True,
                    biomedical=True,
                ),
                rerank=True,
                dense_retriever=bio_dense,
            )
    else:
        skipped.append("G Biomedical embeddings + Hybrid")
    return retrievers, timings, skipped


class _TopK:
    def __init__(self, retriever: Any, top_k: int) -> None:
        self._retriever = retriever
        self.top_k = top_k
        self.last_trace: dict[str, Any] = {}

    @property
    def _dense(self) -> Any:
        return getattr(self._retriever, "_dense", None)

    def search(self, query: str, top_k: int | None = None) -> list[Any]:
        hits = list(self._retriever.search(query, top_k=self.top_k))
        self.last_trace = dict(getattr(self._retriever, "last_trace", {}) or {})
        fusion_ids = [getattr(item, "passage_id") for item in hits]
        self.last_trace.setdefault("fusion_results", [{"passage_id": pid} for pid in fusion_ids])
        self.last_trace.setdefault("original_query", query)
        self.last_trace.setdefault("search_queries", [query])
        return hits


def run_upgrade_experiments(
    *,
    passages: Sequence[Passage] | None = None,
    queries: Sequence[EvaluationQuery] | None = None,
    retrievers: Mapping[str, Any] | None = None,
    embedder: Embedder | None = None,
    biomedical_embedder: Embedder | None = None,
    scorer: PairScorer | None = None,
    include_medcpt: bool | None = None,
    include_biomedical: bool | None = None,
    top_k: int = 10,
    details_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    details_path = Path(details_path or EVALUATION.upgrade_details_path)
    summary_path = Path(summary_path or EVALUATION.upgrade_summary_path)
    if queries is None:
        queries = load_or_create_fixed_queries()
    skipped: list[str] = []
    timings: dict[str, float] = {}
    if retrievers is None:
        if passages is None:
            print("Loading the full BioASQ passage corpus...", flush=True)
            passages = get_passage_collection()
        if include_medcpt is None:
            include_medcpt = scorer is not None or huggingface_model_cached(RERANKER.model_name)
        if include_biomedical is None:
            include_biomedical = biomedical_embedder is not None or huggingface_model_cached(
                UPGRADE.biomedical_embedding_model
            )
        retrievers, timings, skipped = build_upgrade_retrievers(
            passages,
            embedder=embedder,
            biomedical_embedder=biomedical_embedder,
            scorer=scorer,
            include_medcpt=bool(include_medcpt),
            include_biomedical=bool(include_biomedical),
        )
    reranked = {
        name
        for name in retrievers
        if "MedCPT" in name or name.startswith("J ")
    }
    print(f"Evaluating {len(retrievers)} upgrade configurations on {len(queries)} queries...", flush=True)
    rows, summaries, failure_summary = evaluate_named_retrievers(
        retrievers,
        queries,
        top_k=top_k,
        reranked_names=reranked,
    )
    payload = {
        "n_passages": len(passages) if passages is not None else 0,
        "n_queries": len(queries),
        "queries_path": EVALUATION.queries_path,
        "dense_baseline_model": RETRIEVAL.dense_model_name,
        "biomedical_embedding_model": UPGRADE.biomedical_embedding_model,
        "index_build_seconds": timings,
        "skipped_configurations": skipped,
        "notes": [
            "Disabled stages are omitted rather than fabricated.",
            "LLM expansion is not used in this experiment.",
            "Existing baseline JSON files are not overwritten.",
        ],
        "configurations": summaries,
        "failure_summary": failure_summary,
        "rows": rows,
    }
    details_path.parent.mkdir(parents=True, exist_ok=True)
    details_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary_payload = {key: value for key, value in payload.items() if key != "rows"}
    summary_payload["query_ids"] = [_jsonable(query.question_id) for query in queries]
    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"Upgrade details: {details_path}", flush=True)
    print(f"Upgrade summary: {summary_path}", flush=True)
    if skipped:
        print("Skipped: " + ", ".join(skipped), flush=True)
    return {
        "details_path": details_path,
        "summary_path": summary_path,
        "summaries": summaries,
        "rows": rows,
        "skipped": skipped,
        "failure_summary": failure_summary,
    }


def main() -> None:
    run_upgrade_experiments()


if __name__ == "__main__":
    main()
