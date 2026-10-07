"""Offline retrieval evaluation over a fixed 100-query BioASQ set.

Builds BM25 and dense indexes once, then scores BM25, Dense, and Hybrid RRF
on the same queries and the full passage corpus. No LLM calls.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import EVALUATION, GENERATION, RERANKER, RETRIEVAL, RerankerConfig
from src.data.loader import get_passage_collection
from src.data.preprocessing import Passage
from src.evaluation.dataset import EvaluationQuery, load_or_create_fixed_queries
from src.evaluation.metrics import compute_retrieval_metrics
from src.generation.query_expansion import ExpandingHybridRetriever, QueryExpander
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever, Embedder
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import PairScorer, wrap_retriever


class Searchable(Protocol):
    def search(self, query: str, top_k: int | None = None) -> Sequence[Any]: ...


@dataclass(frozen=True)
class QueryEvaluationRow:
    query_id: Any
    question: str
    configuration: str
    retrieved_passage_ids: tuple[Any, ...]
    retrieved_scores: tuple[float, ...]
    latency_ms: float
    recall_at_5: float
    recall_at_10: float
    mrr_at_10: float
    ndcg_at_10: float
    retrieval_cost_usd: float
    expansions: tuple[str, ...] = ()
    llm_calls: int = 0


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (bytes, str, bool)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def evaluate_retriever(
    name: str,
    retriever: Searchable,
    queries: Sequence[EvaluationQuery],
    *,
    top_k: int = 10,
) -> list[QueryEvaluationRow]:
    """Score one retriever on every query. Index construction is not included."""

    rows: list[QueryEvaluationRow] = []
    for index, query in enumerate(queries, start=1):
        started = time.perf_counter()
        hits = list(retriever.search(query.question, top_k=top_k))
        latency_ms = (time.perf_counter() - started) * 1000.0
        retrieved_ids = tuple(hit.passage_id for hit in hits)
        retrieved_scores = tuple(float(hit.score) for hit in hits)
        metrics = compute_retrieval_metrics(retrieved_ids, query.relevant_passage_ids)
        expansion = getattr(retriever, "last_expansion", None)
        expansions = expansion.expansions if expansion is not None else ()
        llm_calls = int(getattr(retriever, "last_llm_calls", 0) or 0)
        cost = float(getattr(retriever, "last_cost_usd", 0.0) or 0.0)
        rows.append(
            QueryEvaluationRow(
                query_id=query.question_id,
                question=query.question,
                configuration=name,
                retrieved_passage_ids=retrieved_ids,
                retrieved_scores=retrieved_scores,
                latency_ms=latency_ms,
                recall_at_5=metrics["recall_at_5"],
                recall_at_10=metrics["recall_at_10"],
                mrr_at_10=metrics["mrr_at_10"],
                ndcg_at_10=metrics["ndcg_at_10"],
                retrieval_cost_usd=cost,
                expansions=tuple(expansions),
                llm_calls=llm_calls,
            )
        )
        if index == len(queries) or index % 10 == 0:
            print(f"  {name}: {index}/{len(queries)} queries", flush=True)
    return rows


def summarize_rows(rows: Sequence[QueryEvaluationRow]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation result")
    names = {row.configuration for row in rows}
    if len(names) != 1:
        raise ValueError("summarize_rows expects a single configuration")
    latencies = [row.latency_ms for row in rows]
    total_cost = sum(row.retrieval_cost_usd for row in rows)
    return {
        "name": rows[0].configuration,
        "n_queries": len(rows),
        "recall_at_5": sum(row.recall_at_5 for row in rows) / len(rows),
        "recall_at_10": sum(row.recall_at_10 for row in rows) / len(rows),
        "mrr_at_10": sum(row.mrr_at_10 for row in rows) / len(rows),
        "ndcg_at_10": sum(row.ndcg_at_10 for row in rows) / len(rows),
        "avg_latency_ms": sum(latencies) / len(latencies),
        "median_latency_ms": statistics.median(latencies),
        "retrieval_cost_usd": total_cost,
        "llm_calls": sum(row.llm_calls for row in rows),
    }


def format_summary_table(summaries: Sequence[Mapping[str, Any]]) -> str:
    header = (
        "| Configuration | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | "
        "Avg latency (ms) | Median latency (ms) |"
    )
    separator = "|---|---:|---:|---:|---:|---:|---:|"
    lines = [header, separator]
    for item in summaries:
        lines.append(
            "| {name} | {recall_at_5:.4f} | {recall_at_10:.4f} | {mrr_at_10:.4f} | "
            "{ndcg_at_10:.4f} | {avg_latency_ms:.2f} | {median_latency_ms:.2f} |".format(**item)
        )
    return "\n".join(lines)


def _row_to_json(row: QueryEvaluationRow) -> dict[str, Any]:
    return {
        "query_id": _jsonable(row.query_id),
        "question": row.question,
        "configuration": row.configuration,
        "retrieved_passage_ids": [_jsonable(item) for item in row.retrieved_passage_ids],
        "retrieved_scores": [float(score) for score in row.retrieved_scores],
        "latency_ms": float(row.latency_ms),
        "recall_at_5": float(row.recall_at_5),
        "recall_at_10": float(row.recall_at_10),
        "mrr_at_10": float(row.mrr_at_10),
        "ndcg_at_10": float(row.ndcg_at_10),
        "retrieval_cost_usd": float(row.retrieval_cost_usd),
        "expansions": list(row.expansions),
        "llm_calls": int(row.llm_calls),
    }


def save_evaluation_outputs(
    *,
    rows: Sequence[QueryEvaluationRow],
    summaries: Sequence[Mapping[str, Any]],
    details_path: str | Path,
    summary_path: str | Path,
    n_passages: int,
    top_k: int,
    queries_path: str | Path,
    index_build_seconds: Mapping[str, float] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    details_destination = Path(details_path)
    summary_destination = Path(summary_path)
    details_destination.parent.mkdir(parents=True, exist_ok=True)
    summary_destination.parent.mkdir(parents=True, exist_ok=True)

    query_ids = list(dict.fromkeys(_jsonable(row.query_id) for row in rows))
    total_cost = sum(row.retrieval_cost_usd for row in rows)
    details_payload = {
        "n_passages": n_passages,
        "n_queries": len(query_ids),
        "query_ids": query_ids,
        "top_k": top_k,
        "queries_path": str(queries_path),
        "retrieval_cost_usd": total_cost,
        "index_build_seconds": dict(index_build_seconds or {}),
        "rows": [_row_to_json(row) for row in rows],
    }
    summary_payload = {
        "n_passages": n_passages,
        "n_queries": len(query_ids),
        "query_ids": query_ids,
        "top_k": top_k,
        "queries_path": str(queries_path),
        "retrieval_cost_usd": total_cost,
        "index_build_seconds": dict(index_build_seconds or {}),
        "configurations": [dict(item) for item in summaries],
        "table": format_summary_table(summaries),
    }
    if extra:
        details_payload.update(dict(extra))
        summary_payload.update(dict(extra))
    details_destination.write_text(json.dumps(details_payload, indent=2) + "\n", encoding="utf-8")
    summary_destination.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    return details_destination, summary_destination


def evaluate_configurations(
    retrievers: Mapping[str, Searchable],
    queries: Sequence[EvaluationQuery],
    *,
    top_k: int = 10,
) -> tuple[list[QueryEvaluationRow], list[dict[str, Any]]]:
    """Evaluate named retrievers on the same query set."""

    all_rows: list[QueryEvaluationRow] = []
    summaries: list[dict[str, Any]] = []
    for name, retriever in retrievers.items():
        rows = evaluate_retriever(name, retriever, queries, top_k=top_k)
        all_rows.extend(rows)
        summaries.append(summarize_rows(rows))
    return all_rows, summaries


def build_evaluation_retrievers(
    passages: Sequence[Passage],
    *,
    top_k: int = 10,
    embedder: Embedder | None = None,
) -> tuple[dict[str, Searchable], dict[str, float]]:
    """Build BM25 and dense indexes once, then wrap them for hybrid RRF.

    Index-building time is recorded separately from per-query search latency.
    Hybrid reuses the BM25 and dense indexes; it does not re-encode the corpus.
    """

    timings: dict[str, float] = {}
    started = time.perf_counter()
    bm25 = BM25Retriever(passages, top_k=top_k)
    timings["BM25_index_build_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    dense_kwargs: dict[str, Any] = {"top_k": top_k}
    if embedder is not None:
        dense_kwargs["embedder"] = embedder
    dense = DenseRetriever(passages, **dense_kwargs)
    timings["Dense_index_build_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    hybrid = HybridRetriever(
        passages,
        bm25_retriever=bm25,
        dense_retriever=dense,
        top_k=top_k,
    )
    timings["Hybrid_RRF_wrap_seconds"] = time.perf_counter() - started
    return {"BM25": bm25, "Dense": dense, "Hybrid RRF": hybrid}, timings


def run_offline_evaluation(
    *,
    passages: Sequence[Passage] | None = None,
    queries: Sequence[EvaluationQuery] | None = None,
    retrievers: Mapping[str, Searchable] | None = None,
    top_k: int | None = None,
    queries_path: str | Path | None = None,
    details_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    index_build_seconds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Run BM25/Dense/Hybrid evaluation and write result files."""

    top_k = EVALUATION.top_k if top_k is None else top_k
    queries_path = Path(queries_path or EVALUATION.queries_path)
    details_path = Path(details_path or EVALUATION.details_path)
    summary_path = Path(summary_path or EVALUATION.summary_path)

    if queries is None:
        queries = load_or_create_fixed_queries(path=queries_path)
    if passages is None and retrievers is None:
        print("Loading the full BioASQ passage corpus...", flush=True)
        passages = get_passage_collection()
    if retrievers is None:
        if passages is None:
            raise ValueError("passages are required to build evaluation retrievers")
        print(
            f"Building BM25 and dense indexes once over {len(passages)} passages...",
            flush=True,
        )
        retrievers, built_timings = build_evaluation_retrievers(passages, top_k=top_k)
        index_build_seconds = dict(built_timings)
        print(
            "Index build seconds: "
            + ", ".join(f"{name}={value:.2f}" for name, value in built_timings.items()),
            flush=True,
        )

    n_passages = len(passages) if passages is not None else 0
    print(f"Evaluating {len(retrievers)} configurations on {len(queries)} queries...", flush=True)
    rows, summaries = evaluate_configurations(retrievers, queries, top_k=top_k)
    details_destination, summary_destination = save_evaluation_outputs(
        rows=rows,
        summaries=summaries,
        details_path=details_path,
        summary_path=summary_path,
        n_passages=n_passages,
        top_k=top_k,
        queries_path=queries_path,
        index_build_seconds=index_build_seconds,
    )
    table = format_summary_table(summaries)
    print(table, flush=True)
    print(f"Detailed results: {details_destination}", flush=True)
    print(f"Summary results: {summary_destination}", flush=True)
    return {
        "n_passages": n_passages,
        "n_queries": len(queries),
        "rows": rows,
        "summaries": summaries,
        "table": table,
        "details_path": details_destination,
        "summary_path": summary_destination,
        "queries_path": queries_path,
        "index_build_seconds": dict(index_build_seconds or {}),
        "retrieval_cost_usd": 0.0,
        "dense_model_name": RETRIEVAL.dense_model_name,
    }


def query_expansion_config_payload() -> dict[str, Any]:
    return {
        "enabled": True,
        "n_expansions": GENERATION.query_expansion_count,
        "model_name": GENERATION.query_expansion_model_name,
        "estimated_cost_per_call_usd": GENERATION.query_expansion_estimated_cost_per_call_usd,
        "original_query_retained": True,
    }


def write_comparison_summary(
    *,
    qe_summaries: Sequence[Mapping[str, Any]],
    comparison_path: str | Path,
    existing_summary_path: str | Path | None = None,
) -> Path:
    """Combine existing BM25/Dense/Hybrid summaries with Hybrid+QE without rewriting them."""

    configurations: list[dict[str, Any]] = []
    source = Path(existing_summary_path or EVALUATION.summary_path)
    if source.exists():
        payload = json.loads(source.read_text(encoding="utf-8"))
        configurations.extend(dict(item) for item in payload.get("configurations", []))
    existing_names = {item.get("name") for item in configurations}
    for item in qe_summaries:
        if item.get("name") not in existing_names:
            configurations.append(dict(item))
    destination = Path(comparison_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "queries_path": EVALUATION.queries_path,
                "query_expansion": query_expansion_config_payload(),
                "configurations": configurations,
                "table": format_summary_table(configurations),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return destination


def run_hybrid_qe_evaluation(
    *,
    passages: Sequence[Passage] | None = None,
    queries: Sequence[EvaluationQuery] | None = None,
    retrievers: Mapping[str, Searchable] | None = None,
    expander: QueryExpander | None = None,
    top_k: int | None = None,
    queries_path: str | Path | None = None,
    details_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    comparison_path: str | Path | None = None,
    existing_summary_path: str | Path | None = None,
    index_build_seconds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate Hybrid + Query Expansion on the existing fixed 100-query set.

    Does not rewrite BM25/Dense/Hybrid result files.
    """

    top_k = EVALUATION.top_k if top_k is None else top_k
    queries_path = Path(queries_path or EVALUATION.queries_path)
    details_path = Path(details_path or EVALUATION.hybrid_qe_details_path)
    summary_path = Path(summary_path or EVALUATION.hybrid_qe_summary_path)
    comparison_path = Path(comparison_path or EVALUATION.comparison_path)

    if queries is None:
        queries = load_or_create_fixed_queries(path=queries_path)
    if retrievers is None:
        if passages is None:
            print("Loading the full BioASQ passage corpus...", flush=True)
            passages = get_passage_collection()
        print(
            f"Building BM25 and dense indexes once over {len(passages)} passages...",
            flush=True,
        )
        built, timings = build_evaluation_retrievers(passages, top_k=top_k)
        index_build_seconds = dict(timings)
        hybrid = built["Hybrid RRF"]
        qe_retriever = ExpandingHybridRetriever(
            hybrid,
            expander or QueryExpander(),
        )
        retrievers = {"Hybrid + Query Expansion": qe_retriever}

    n_passages = len(passages) if passages is not None else 0
    print(
        f"Evaluating Hybrid + Query Expansion on {len(queries)} queries "
        f"from {queries_path}...",
        flush=True,
    )
    rows, summaries = evaluate_configurations(retrievers, queries, top_k=top_k)
    extra = {
        "query_expansion": query_expansion_config_payload(),
        "llm_calls": sum(row.llm_calls for row in rows),
    }
    details_destination, summary_destination = save_evaluation_outputs(
        rows=rows,
        summaries=summaries,
        details_path=details_path,
        summary_path=summary_path,
        n_passages=n_passages,
        top_k=top_k,
        queries_path=queries_path,
        index_build_seconds=index_build_seconds,
        extra=extra,
    )
    comparison_destination = write_comparison_summary(
        qe_summaries=summaries,
        comparison_path=comparison_path,
        existing_summary_path=existing_summary_path or EVALUATION.summary_path,
    )
    table = format_summary_table(summaries)
    print(table, flush=True)
    print(f"Detailed Hybrid+QE results: {details_destination}", flush=True)
    print(f"Summary Hybrid+QE results: {summary_destination}", flush=True)
    print(f"Comparison table: {comparison_destination}", flush=True)
    return {
        "n_passages": n_passages,
        "n_queries": len(queries),
        "rows": rows,
        "summaries": summaries,
        "table": table,
        "details_path": details_destination,
        "summary_path": summary_destination,
        "comparison_path": comparison_destination,
        "queries_path": queries_path,
        "index_build_seconds": dict(index_build_seconds or {}),
        "retrieval_cost_usd": sum(row.retrieval_cost_usd for row in rows),
        "query_expansion": extra["query_expansion"],
    }


def reranker_config_payload(config: RerankerConfig) -> dict[str, Any]:
    return {
        "enabled": config.enabled,
        "model_name": config.model_name,
        "candidate_pool_size": config.candidate_pool_size,
        "final_top_k": config.final_top_k,
        "device": config.device,
        "batch_size": config.batch_size,
        "max_length": config.max_length,
        "openai_used": False,
    }


def run_rerank_evaluation(
    *,
    passages: Sequence[Passage] | None = None,
    queries: Sequence[EvaluationQuery] | None = None,
    retrievers: Mapping[str, Searchable] | None = None,
    scorer: PairScorer | None = None,
    config: RerankerConfig | None = None,
    top_k: int | None = None,
    queries_path: str | Path | None = None,
    details_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    index_build_seconds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate Hybrid RRF + MedCPT rerank on the existing fixed 100-query set.

    Does not rewrite BM25/Dense/Hybrid or Hybrid+QE result files.
    """

    top_k = EVALUATION.top_k if top_k is None else top_k
    queries_path = Path(queries_path or EVALUATION.queries_path)
    details_path = Path(details_path or EVALUATION.rerank_details_path)
    summary_path = Path(summary_path or EVALUATION.rerank_summary_path)
    rerank_config = config or RerankerConfig(
        enabled=True,
        candidate_pool_size=RERANKER.candidate_pool_size,
        final_top_k=top_k,
        model_name=RERANKER.model_name,
        device=RERANKER.device,
        batch_size=RERANKER.batch_size,
        max_length=RERANKER.max_length,
    )

    if queries is None:
        queries = load_or_create_fixed_queries(path=queries_path)
    if retrievers is None:
        if passages is None:
            print("Loading the full BioASQ passage corpus...", flush=True)
            passages = get_passage_collection()
        print(
            f"Building BM25 and dense indexes once over {len(passages)} passages...",
            flush=True,
        )
        built, timings = build_evaluation_retrievers(passages, top_k=top_k)
        index_build_seconds = dict(timings)
        hybrid = built["Hybrid RRF"]
        retrievers = {
            "Hybrid RRF + MedCPT Rerank": wrap_retriever(
                hybrid,
                config=rerank_config,
                scorer=scorer,
            )
        }

    n_passages = len(passages) if passages is not None else 0
    print(
        f"Evaluating Hybrid RRF + MedCPT rerank on {len(queries)} queries "
        f"from {queries_path}...",
        flush=True,
    )
    rows, summaries = evaluate_configurations(retrievers, queries, top_k=top_k)
    extra = {"reranker": reranker_config_payload(rerank_config)}
    details_destination, summary_destination = save_evaluation_outputs(
        rows=rows,
        summaries=summaries,
        details_path=details_path,
        summary_path=summary_path,
        n_passages=n_passages,
        top_k=top_k,
        queries_path=queries_path,
        index_build_seconds=index_build_seconds,
        extra=extra,
    )
    table = format_summary_table(summaries)
    print(table, flush=True)
    print(f"Detailed rerank results: {details_destination}", flush=True)
    print(f"Summary rerank results: {summary_destination}", flush=True)
    return {
        "n_passages": n_passages,
        "n_queries": len(queries),
        "rows": rows,
        "summaries": summaries,
        "table": table,
        "details_path": details_destination,
        "summary_path": summary_destination,
        "queries_path": queries_path,
        "index_build_seconds": dict(index_build_seconds or {}),
        "retrieval_cost_usd": 0.0,
        "reranker": extra["reranker"],
        "dense_model_name": RETRIEVAL.dense_model_name,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Offline retrieval evaluation")
    parser.add_argument(
        "--hybrid-qe",
        action="store_true",
        help="Evaluate Hybrid + Query Expansion on the existing 100-query set",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Evaluate Hybrid RRF + MedCPT cross-encoder rerank (writes separate files)",
    )
    parser.add_argument(
        "--rerank-comparison",
        action="store_true",
        help=(
            "Three-way Hybrid@10 vs Hybrid@50 vs MedCPT@10 comparison "
            "(does not overwrite baseline JSON)"
        ),
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Run next-generation retrieval comparison A–J (separate JSON files)",
    )
    parser.add_argument(
        "--scientific",
        action="store_true",
        help="Run embedding/pool/failure/expansion/decomposition studies (separate JSON)",
    )
    args = parser.parse_args()
    ran = False
    if args.hybrid_qe:
        run_hybrid_qe_evaluation()
        ran = True
    if args.rerank:
        run_rerank_evaluation()
        ran = True
    if args.rerank_comparison:
        from src.evaluation.rerank_comparison import run_rerank_comparison_experiment

        run_rerank_comparison_experiment()
        ran = True
    if args.upgrade:
        from src.evaluation.upgrade_experiments import run_upgrade_experiments

        run_upgrade_experiments()
        ran = True
    if getattr(args, "scientific", False):
        from src.evaluation.scientific_study import run_scientific_suite

        run_scientific_suite()
        ran = True
    if not ran:
        run_offline_evaluation()


if __name__ == "__main__":
    main()
