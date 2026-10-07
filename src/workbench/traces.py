"""Trace list from stored diagnostics and live playground results."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.config import EVALUATION
from src.evaluation.live_metrics import load_live_evaluations

ROOT = Path(__file__).resolve().parents[2]

TRACE_STAGES = (
    "Query",
    "Query understanding",
    "BM25",
    "Dense",
    "RRF",
    "Candidate pool",
    "MedCPT",
    "Evidence",
    "Generation",
    "Citation validation",
    "Evaluation",
)


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def stored_traces(*, root: Path | None = None) -> list[dict[str, Any]]:
    base = root or ROOT
    rows: list[dict[str, Any]] = []
    traces_dir = base / EVALUATION.query_trace_dir
    if traces_dir.exists():
        for path in sorted(traces_dir.glob("query_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            gold_traces = list(payload.get("gold_traces") or [])
            category = None
            if gold_traces:
                category = gold_traces[0].get("stage_code") or gold_traces[0].get("failure_point")
            rows.append(
                {
                    "id": path.stem,
                    "source": "offline_diagnostic",
                    "path": str(path.relative_to(base)),
                    "query_id": payload.get("query_id"),
                    "question": payload.get("original_query") or payload.get("question"),
                    "experiment": "hybrid-medcpt",
                    "model": payload.get("reranker_model") or payload.get("dense_model"),
                    "failure_category": category,
                    "latency_ms": (payload.get("stage_timings") or {}).get("total_ms")
                    or payload.get("retrieval_ms"),
                    "quality_score": None,
                    "retrieval_outcome": gold_traces[0].get("first_loss_label") if gold_traces else None,
                    "payload": payload,
                }
            )
    for index, record in enumerate(load_live_evaluations(base / EVALUATION.live_evaluation_path)):
        rows.append(
            {
                "id": f"live-{index}",
                "source": "online_sample",
                "path": EVALUATION.live_evaluation_path,
                "query_id": None,
                "question": record.get("question"),
                "experiment": "playground",
                "model": "heuristic_live_evaluator",
                "failure_category": None,
                "latency_ms": record.get("latency_ms"),
                "quality_score": (record.get("metrics") or {}).get("faithfulness"),
                "retrieval_outcome": "hit" if (record.get("metrics") or {}).get("context_precision") else None,
                "payload": record,
            }
        )
    return rows


def filter_traces(
    rows: list[Mapping[str, Any]],
    *,
    query: str | None = None,
    experiment: str | None = None,
    model: str | None = None,
    failure_category: str | None = None,
    min_latency_ms: float | None = None,
    max_latency_ms: float | None = None,
    source: str | None = None,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    needle = (query or "").strip().lower()
    for row in rows:
        if needle and needle not in str(row.get("question") or "").lower() and needle not in str(row.get("query_id") or ""):
            continue
        if experiment and row.get("experiment") != experiment:
            continue
        if model and row.get("model") != model:
            continue
        if failure_category and row.get("failure_category") != failure_category:
            continue
        if source and row.get("source") != source:
            continue
        latency = row.get("latency_ms")
        if min_latency_ms is not None and (latency is None or float(latency) < min_latency_ms):
            continue
        if max_latency_ms is not None and (latency is None or float(latency) > max_latency_ms):
            continue
        selected.append(dict(row))
    return selected


def trace_timeline(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Stage timeline. Missing input/output/latency is left as Not measured rather than invented."""

    timings = dict(payload.get("stage_timings") or payload.get("latency") or {})
    gold = list(payload.get("gold_traces") or [])
    stages = [
        {
            "stage": "Query",
            "input": payload.get("original_query") or payload.get("question"),
            "output": payload.get("question") or payload.get("original_query"),
            "latency_ms": timings.get("query_ms"),
            "model": None,
            "status": "ok",
            "metadata": {"query_id": payload.get("query_id"), "category": payload.get("category")},
        },
        {
            "stage": "Query understanding",
            "input": payload.get("question") or payload.get("original_query"),
            "output": payload.get("category") or payload.get("query_type"),
            "latency_ms": timings.get("classify_ms"),
            "model": None,
            "status": "ok" if payload.get("category") else "not recorded",
            "metadata": {"inspect_k": payload.get("inspect_k")},
        },
        {
            "stage": "BM25",
            "input": payload.get("question"),
            "output": payload.get("bm25_top10") or payload.get("bm25_results"),
            "latency_ms": timings.get("bm25_ms"),
            "model": "BM25",
            "status": "ok",
            "metadata": {"top_preview": (payload.get("bm25_top10") or [])[:3]},
        },
        {
            "stage": "Dense",
            "input": payload.get("question"),
            "output": payload.get("dense_top10") or payload.get("dense_results"),
            "latency_ms": timings.get("dense_ms"),
            "model": payload.get("dense_model"),
            "status": "ok",
            "metadata": {"top_preview": (payload.get("dense_top10") or [])[:3]},
        },
        {
            "stage": "RRF",
            "input": "BM25 + Dense rankings",
            "output": payload.get("rrf_top50") or payload.get("rrf_results"),
            "latency_ms": timings.get("rrf_ms"),
            "model": None,
            "status": "ok",
            "metadata": {"rrf_k": payload.get("rrf_k")},
        },
        {
            "stage": "Candidate pool",
            "input": payload.get("pool_k"),
            "output": payload.get("candidate_pool") or payload.get("rrf_top50"),
            "latency_ms": timings.get("pool_ms"),
            "model": None,
            "status": "ok",
            "metadata": {"pool_k": payload.get("pool_k")},
        },
        {
            "stage": "MedCPT",
            "input": payload.get("pool_k"),
            "output": payload.get("medcpt_top10") or payload.get("final_top10"),
            "latency_ms": timings.get("rerank_ms") or timings.get("medcpt_ms"),
            "model": payload.get("reranker_model"),
            "status": "ok" if payload.get("reranker_model") else "not recorded",
            "metadata": {"final_k": payload.get("final_k")},
        },
        {
            "stage": "Evidence",
            "input": payload.get("final_k"),
            "output": payload.get("final_top10") or payload.get("evidence"),
            "latency_ms": None,
            "model": None,
            "status": "ok",
            "metadata": {"gold_in_top10": [item.get("gold_id") for item in gold if item.get("in_top10")]},
        },
        {
            "stage": "Generation",
            "input": payload.get("question"),
            "output": payload.get("generated_answer") or payload.get("answer"),
            "latency_ms": timings.get("generation_ms"),
            "model": payload.get("generation_model"),
            "status": "ok" if payload.get("generated_answer") or payload.get("answer") else "not on this diagnostic trace",
            "metadata": {},
        },
        {
            "stage": "Citation validation",
            "input": payload.get("generated_answer") or payload.get("answer"),
            "output": payload.get("citation_validation"),
            "latency_ms": timings.get("citation_ms"),
            "model": None,
            "status": "not on this diagnostic trace" if payload.get("citation_validation") is None else "ok",
            "metadata": {},
        },
        {
            "stage": "Evaluation",
            "input": payload.get("question"),
            "output": payload.get("metrics") or gold,
            "latency_ms": timings.get("evaluation_ms") or timings.get("total_ms"),
            "model": payload.get("evaluator") or "offline diagnostic",
            "status": "ok",
            "metadata": {
                "failure_labels": [item.get("stage_code") or item.get("failure_point") for item in gold],
            },
        },
    ]
    return stages
