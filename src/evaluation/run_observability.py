"""Run data-quality, index-health, and 100-query retrieval diagnostics.

Does not overwrite historical experiment JSON. Does not change production models.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import EVALUATION, RERANKER, RETRIEVAL
from src.data.loader import get_passage_collection
from src.data.quality import inspect_documents, write_data_quality_report
from src.evaluation.dataset import load_or_create_fixed_queries
from src.evaluation.gold_quality import evaluate_gold_quality, write_gold_quality_report
from src.evaluation.index_health import inspect_index_health, write_index_health_report
from src.evaluation.pipeline_diagnostic import categorize_question, trace_query
from src.evaluation.retrieval_diagnostics import run_retrieval_diagnostics, write_retrieval_diagnostics
from src.evaluation.run_evaluation import build_evaluation_retrievers
from src.retrieval.reranker import CrossEncoderReranker, get_shared_scorer


def _write_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def run_observability(
    *,
    skip_rerank: bool = False,
    trace_ids: Sequence[int] = (2, 5),
) -> dict[str, Path]:
    print("Loading corpus and fixed queries...", flush=True)
    passages = get_passage_collection()
    queries = load_or_create_fixed_queries()
    quality = inspect_documents(passages)
    data_path = write_data_quality_report(quality["report"], ROOT / EVALUATION.data_quality_path)
    print(
        f"Data quality: {quality['report']['total_documents']} docs, "
        f"{quality['report']['nan_text']} nan_text, "
        f"{quality['report']['valid_documents']} valid. Wrote {data_path}",
        flush=True,
    )
    gold = evaluate_gold_quality(queries, passages)
    gold_path = write_gold_quality_report(gold, ROOT / EVALUATION.gold_quality_path)
    print(
        f"Gold quality: {gold['invalid_gold_ids']}/{gold['total_gold_ids']} invalid, "
        f"{gold['queries_where_all_gold_ids_are_invalid']} queries with all-invalid gold. Wrote {gold_path}",
        flush=True,
    )

    print("Building production BM25 + MiniLM indexes...", flush=True)
    built, timings = build_evaluation_retrievers(passages, top_k=10)
    health = inspect_index_health(passages, bm25=built["BM25"], dense=built["Dense"])
    health["index_build_seconds"] = timings
    health_path = write_index_health_report(health, ROOT / EVALUATION.index_health_path)
    print(f"Index health aligned={health['aligned']}. Wrote {health_path}", flush=True)

    reranker = None
    if not skip_rerank:
        reranker = CrossEncoderReranker(scorer=get_shared_scorer())
        reranker.preload()
        print("Running 100-query BM25 / MiniLM / RRF / MedCPT diagnostics...", flush=True)
    else:
        print("Skipping MedCPT; still collecting BM25/Dense/RRF diagnostics...", flush=True)
    diagnostics = run_retrieval_diagnostics(
        queries,
        passages=passages,
        bm25=built["BM25"],
        dense=built["Dense"],
        hybrid=built["Hybrid RRF"],
        reranker=reranker,
        pool_k=50,
        final_k=10,
    )
    diag_path = write_retrieval_diagnostics(diagnostics, ROOT / EVALUATION.retrieval_diagnostics_path)
    print(f"Wrote {diag_path}", flush=True)

    trace_dir = ROOT / EVALUATION.query_trace_dir
    trace_dir.mkdir(parents=True, exist_ok=True)
    by_id = {query.question_id: query for query in queries}
    written = {}
    for query_id in trace_ids:
        query = by_id[int(query_id)]
        print(f"Tracing query_id={query_id}...", flush=True)
        trace = trace_query(
            query,
            passages=passages,
            bm25=built["BM25"],
            dense=built["Dense"],
            hybrid=built["Hybrid RRF"],
            reranker=reranker or CrossEncoderReranker(scorer=get_shared_scorer()),
            category=categorize_question(query.question),
        )
        path = _write_json(trace, trace_dir / f"query_{query_id}.json")
        written[str(query_id)] = path
        print(f"  first loss={trace['compact_trace']['first_loss_label']} wrote {path}", flush=True)

    return {
        "data_quality": data_path,
        "gold_quality": gold_path,
        "index_health": health_path,
        "retrieval_diagnostics": diag_path,
        **{f"trace_{key}": value for key, value in written.items()},
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write RAG observability JSON reports")
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--trace-ids", default="2,5")
    args = parser.parse_args(list(argv) if argv is not None else None)
    ids = [int(part) for part in args.trace_ids.split(",") if part.strip()]
    run_observability(skip_rerank=args.skip_rerank, trace_ids=ids)


if __name__ == "__main__":
    main()
