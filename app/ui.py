"""Local Streamlit UI for the biomedical RAG pipeline.

Presentation only: retrieval, fusion, generation, and citation checks stay
in `src/`. This module displays GroundedQA results and existing evaluation
artifacts. It does not rerun offline evaluation.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import EVALUATION, GENERATION, RERANKER, RETRIEVAL, UI, UPGRADE, RerankerConfig
from src.data.loader import get_passage_collection
from src.evaluation.answer_evaluation import evaluate_live_answer_quality
from src.evaluation.claim_coverage import claim_evidence_coverage
from src.evaluation.experiment_catalog import (
    experiment_lab_rows,
    experiment_table_rows,
    load_experiment_catalog,
)
from src.evaluation.live_metrics import append_live_evaluation, evaluate_live_query, load_live_evaluations
from src.evaluation.rerank_comparison import (
    comparison_matrix_rows,
    gain_rows,
)
from src.generation.query_expansion import QueryExpander
from src.generation.rag import GroundedQA, GroundedQAResult
from src.inspection.config_view import runtime_configuration
from src.inspection.decisions import architecture_decisions
from src.inspection.model_registry import model_registry
from src.inspection.prompt_inspector import (
    non_llm_prompt_notes,
    prompt_inspector_tabs,
    runtime_generation_variables,
)
from src.inspection.prompts import llm_prompt_catalog
from src.inspection.runtime_state import (
    RANKING_COLUMN_HELP,
    RECOGNIZED_QUERY_TYPES,
    build_query_understanding,
    build_runtime_state,
    infer_medcpt_executed,
    passage_id_pmid,
)
from src.inspection.status import experimental_feature_status
from src.validation.citations import CITATION_PATTERN, CitationValidation
from src.workbench.production import (
    PLAYGROUND_DEFAULT_RERANK_ENABLED,
    resolve_rerank_enabled,
)
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.embedding_models import resolve_embedding_model
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.pipeline import StagedRetriever
from src.retrieval.reranker import wrap_retriever

_SECRET_ENV_NAMES = ("OPENAI_API_KEY", "LLM_API_KEY", "HF_TOKEN")
SUCCESS_REVIEW_IDS = (0, 10, 19)
FAILURE_REVIEW_IDS = (5, 9, 15)
PIPELINE_STEPS = (
    "Question",
    "Query Expansion",
    "BM25 + Dense",
    "RRF Fusion",
    "Evidence",
    "Grounded LLM",
    "Citation Validation",
    "LLM Judge",
)
RETRIEVAL_CONFIG_ORDER = (
    "BM25",
    "Dense",
    "Hybrid RRF",
    "Hybrid + Query Expansion",
)
METRIC_GLOSSARY = (
    ("Recall@5", "Whether relevant evidence appears in the top 5."),
    ("Recall@10", "Whether relevant evidence appears in the top 10."),
    ("MRR@10", "How highly the first relevant passage is ranked."),
    ("nDCG@10", "How good the overall top-10 ranking is, giving more weight to higher-ranked relevant passages."),
    ("Correctness", "Whether the generated answer agrees with the reference answer."),
    ("Groundedness", "Whether the answer is supported by the retrieved evidence."),
    ("Context relevance", "Whether the retrieved context is relevant to the question."),
    ("Candidate Recall@50", "Fraction of golden relevant IDs that appear in Hybrid RRF top-50."),
    ("Candidate Hit@50", "Whether at least one golden relevant ID appears in Hybrid RRF top-50."),
    ("Rerank recovery@10", "When a relevant ID is in the top-50 pool, whether MedCPT puts a relevant ID in the top-10."),
)


def redact_secrets(text: str) -> str:
    """Remove credential values from a message shown in the UI."""

    redacted = text
    for name in _SECRET_ENV_NAMES:
        value = os.environ.get(name)
        if value:
            redacted = redacted.replace(value, "[redacted]")
    return re.sub(r"Bearer\s+\S+", "Bearer [redacted]", redacted)


def public_error_message(exc: BaseException, *, stage: str) -> str:
    """User-facing error text without secrets."""

    prefixes = {
        "init": "Dataset or index initialization failed.",
        "retrieval": "Retrieval failed.",
        "llm": "The language-model call failed.",
    }
    prefix = prefixes.get(stage, "The request failed.")
    detail = redact_secrets(str(exc)).strip()
    if not detail:
        return prefix
    return f"{prefix} {detail}"


def classify_ask_error(exc: BaseException) -> str:
    message = str(exc).lower()
    if any(token in message for token in ("language model", "llm request", "api_key", "llm_api_key")):
        return "llm"
    return "retrieval"


def citation_status_messages(
    citations: CitationValidation,
    *,
    insufficient_evidence: bool,
) -> list[tuple[str, str]]:
    """Return (level, message) pairs for citation validation. Never hide invalid IDs."""

    if insufficient_evidence:
        return []
    messages: list[tuple[str, str]] = []
    if citations.invalid_ids:
        ids = ", ".join(str(item) for item in citations.invalid_ids)
        messages.append(
            ("error", f"Invalid citation IDs (not in the supplied evidence): {ids}")
        )
    if citations.missing_citations:
        messages.append(("warning", "The answer contains no [PASSAGE_ID: ...] citations."))
    elif citations.is_valid:
        ids = ", ".join(str(item) for item in citations.valid_ids)
        messages.append(("success", f"Citations match the supplied evidence: {ids}"))
    return messages


def request_overview(
    result: GroundedQAResult,
    *,
    expand: bool,
    latency_ms: float,
    n_searches: int,
    expansion_calls: int,
    llm_calls: int,
) -> dict[str, str]:
    """Single Explorer overview. No offline Recall/MRR/nDCG and no duplicate live metrics."""

    overview = {
        "Total latency": _format_ms(latency_ms) or "0.0 ms",
        "Retrieval strategy": (
            "Hybrid + Query Expansion" if expand else "Hybrid RRF"
        ),
        "BM25 searches": str(n_searches),
        "Dense searches": str(n_searches),
        "RRF fusion count": "1",
        "Evidence returned": str(len(result.evidence)),
        "LLM calls": str(llm_calls),
        "Valid/invalid citations": (
            f"{'Valid' if result.citations.is_valid else 'Invalid'} "
            f"({len(result.citations.valid_ids)} valid, "
            f"{len(result.citations.invalid_ids)} invalid)"
        ),
    }
    if expansion_calls:
        cost = expansion_calls * GENERATION.query_expansion_estimated_cost_per_call_usd
        overview["QE cost"] = f"${cost:.4f}"
    return overview


def pipeline_caption(*, reranker_enabled: bool | None = None) -> str:
    enabled = RERANKER.enabled if reranker_enabled is None else reranker_enabled
    steps = list(PIPELINE_STEPS)
    if enabled:
        fusion_index = steps.index("RRF Fusion")
        steps.insert(fusion_index + 1, "MedCPT Rerank")
    return " → ".join(steps)


def format_judge_score(value: Any) -> str:
    return f"{float(value):.2f}/5"


def format_heuristic_score(value: Any) -> str:
    """Live heuristic scores are 0–1. Do not present them as /5."""

    if value is None:
        return "—"
    return f"{float(value):.2f}"


NAV_GROUPS = {
    "Overview": ("Overview",),
    "Playground": ("Playground",),
    "Evaluations": ("Datasets", "Benchmark Runs", "Metrics", "Failure Analysis", "Regression"),
    "Experiments": ("Experiment List", "Compare", "Per-query Diff"),
    "Traces": ("Trace Explorer", "Trace Detail"),
    "Data": ("Data Quality", "Gold Quality"),
    "System": ("Models", "Prompts", "Architecture", "Observability"),
}


def _format_ms(value: Any) -> str | None:
    if value is None:
        return None
    return f"{float(value):,.1f} ms"


def format_latency_display(value: Any) -> str:
    """Milliseconds plus human-readable seconds for the dashboard."""

    if value is None:
        return "—"
    ms = float(value)
    return f"{ms:,.1f} ms ({ms / 1000.0:.2f} s)"


def _trace_stage(title: str, fields: Mapping[str, Any], latency_ms: Any = None) -> dict[str, Any]:
    payload = {"title": title, "fields": dict(fields)}
    formatted = _format_ms(latency_ms)
    if formatted is not None:
        payload["fields"]["Latency"] = formatted
    return payload


def build_request_trace(
    result: GroundedQAResult,
    *,
    expand: bool,
    total_latency_ms: float,
    live_quality: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Live Explorer pipeline trace. Omits offline Recall/MRR/nDCG."""

    trace = dict(result.trace or {})
    if "expansion_llm_called" in trace:
        expansion_calls = 1 if trace["expansion_llm_called"] else 0
    else:
        expansion_calls = 1 if result.query_expansion_used else 0
    generation_calls = 1 if result.llm_called else 0
    llm_calls = expansion_calls + generation_calls
    n_queries = trace.get("n_search_queries")
    if n_queries is None:
        n_queries = 1 + len(result.expansions) if result.query_expansion_used else 1
    candidate_k = trace.get("candidate_k")
    if candidate_k is None:
        candidate_k = GENERATION.evidence_top_k
    rrf_k = trace.get("rrf_k")
    if rrf_k is None:
        rrf_k = RETRIEVAL.rrf_k
    dense_model = trace.get("dense_model_name") or RETRIEVAL.dense_model_name
    quality = dict(live_quality or {})
    judge_fields: dict[str, Any] = {
        "Model": quality.get("model_name") or EVALUATION.judge_model_name,
        "Calls": quality.get("llm_calls", 0),
    }
    if not quality.get("evaluated"):
        judge_fields["Status"] = quality.get("status") or "Not evaluated"

    expansion_fields: dict[str, Any] = {
        "Model": GENERATION.query_expansion_model_name,
        "Expansions generated": len(result.expansions),
    }
    if not expand or not result.query_expansion_used:
        expansion_fields["Status"] = "Skipped"

    dense_fields: dict[str, Any] = {
        "Embedding model": dense_model,
    }
    if trace.get("embedding_dim") is not None:
        dense_fields["Embedding dimension"] = trace["embedding_dim"]

    hybrid_fields: dict[str, Any] = {
        "Fusion method": "Reciprocal Rank Fusion",
        "RRF k": rrf_k,
    }
    if trace.get("n_rankings") is not None:
        hybrid_fields["Number of input rankings"] = trace["n_rankings"]

    rerank_enabled = bool(trace.get("reranker_enabled"))
    rerank_fields: dict[str, Any] = {
        "Model": trace.get("reranker_model_name") or RERANKER.model_name,
        "Device": trace.get("reranker_device") or RERANKER.device,
    }
    if trace.get("reranker_candidate_pool_size") is not None:
        rerank_fields["Candidate pool"] = trace["reranker_candidate_pool_size"]
    if trace.get("rerank_candidate_count") is not None:
        rerank_fields["Candidates scored"] = trace["rerank_candidate_count"]
    if not rerank_enabled:
        rerank_fields["Status"] = "Skipped"

    citation_fields: dict[str, Any] = {}
    if result.citations.invalid_ids:
        citation_fields["Invalid citation IDs"] = ", ".join(
            str(item) for item in result.citations.invalid_ids
        )
    if result.citations.missing_citations:
        citation_fields["Missing citations"] = "Yes"

    stages = [
        _trace_stage(
            "1. Query Expansion",
            expansion_fields,
            trace.get("query_expansion_ms"),
        ),
        _trace_stage(
            "2. BM25 Retrieval",
            {"Top-k": candidate_k},
            trace.get("bm25_ms"),
        ),
        _trace_stage("3. Dense Retrieval", dense_fields, trace.get("dense_ms")),
        _trace_stage("4. Hybrid / RRF Fusion", hybrid_fields, trace.get("rrf_ms")),
    ]
    if rerank_enabled:
        stages.append(
            _trace_stage(
                "5. MedCPT Cross-Encoder",
                rerank_fields,
                trace.get("rerank_ms"),
            )
        )
        generation_number = 6
        citation_number = 7
        judge_number = 8
    else:
        generation_number = 5
        citation_number = 6
        judge_number = 7
    stages.extend(
        [
        _trace_stage(
            f"{generation_number}. Answer Generation",
            {"Model": GENERATION.model_name},
            trace.get("generation_ms"),
        ),
        _trace_stage(
            f"{citation_number}. Citation Validation",
            citation_fields,
            trace.get("citation_ms"),
        ),
        _trace_stage(
            f"{judge_number}. LLM Judge",
            judge_fields,
            quality.get("latency_ms"),
        ),
        ]
    )
    return {
        "summary": request_overview(
            result,
            expand=expand,
            latency_ms=total_latency_ms,
            n_searches=int(n_queries),
            expansion_calls=expansion_calls,
            llm_calls=llm_calls,
        ),
        "stages": stages,
        "judge_note": (
            "LIVE ANSWER QUALITY is for this query only. "
            "OFFLINE LLM ANSWER EVALUATION (including correctness) is on the Evaluation tab."
        ),
    }


def load_json_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def retrieval_table_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in payload.get("configurations", []):
        row = {
            "Configuration": item.get("name", ""),
            "Recall@5": item.get("recall_at_5"),
            "Recall@10": item.get("recall_at_10"),
            "MRR@10": item.get("mrr_at_10"),
            "nDCG@10": item.get("ndcg_at_10"),
            "Avg latency (ms)": item.get("avg_latency_ms"),
        }
        if "retrieval_cost_usd" in item:
            row["Cost (USD)"] = item["retrieval_cost_usd"]
        rows.append(row)
    return rows


def _display_metric_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Round floats for the Streamlit table without changing saved JSON values."""

    displayed = []
    for row in rows:
        formatted: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, float):
                formatted[key] = round(value, 4 if "latency" not in key.lower() else 2)
            else:
                formatted[key] = value
        displayed.append(formatted)
    return displayed


def selected_configuration_notes(configurations: Sequence[Mapping[str, Any]]) -> list[str]:
    """Concise measured conclusion for Hybrid + Query Expansion."""

    if not configurations:
        return ["No retrieval comparison results were found."]
    by_metric = {
        "Recall@5": max(configurations, key=lambda item: item.get("recall_at_5") or 0),
        "Recall@10": max(configurations, key=lambda item: item.get("recall_at_10") or 0),
        "nDCG@10": max(configurations, key=lambda item: item.get("ndcg_at_10") or 0),
    }
    notes = [
        "Selected configuration: Hybrid + Query Expansion.",
        (
            f"Highest Recall@5 ({by_metric['Recall@5']['recall_at_5']:.4f}), "
            f"Recall@10 ({by_metric['Recall@10']['recall_at_10']:.4f}), "
            f"and nDCG@10 ({by_metric['nDCG@10']['ndcg_at_10']:.4f})."
        ),
    ]
    hybrid = next((item for item in configurations if item.get("name") == "Hybrid RRF"), None)
    qe = next(
        (item for item in configurations if item.get("name") == "Hybrid + Query Expansion"),
        None,
    )
    if hybrid and qe:
        notes.append(
            f"Hybrid RRF has slightly higher MRR@10 "
            f"({hybrid['mrr_at_10']:.4f} vs {qe['mrr_at_10']:.4f})."
        )
        notes.append(
            "Latency trade-off: "
            f"Hybrid ≈ {hybrid['avg_latency_ms']:.0f} ms vs "
            f"Hybrid + Query Expansion ≈ {qe['avg_latency_ms']:,.0f} ms."
        )
    return notes


def rerank_comparison_table_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return comparison_matrix_rows(payload)


def rerank_gain_table_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return gain_rows(payload)


def rerank_experiment_notes(payload: Mapping[str, Any]) -> list[str]:
    findings = list(payload.get("findings") or [])
    if findings:
        return findings
    return ["No MedCPT rerank comparison results were found."]


UPGRADE_PIPELINE_STAGES = (
    ("Original Query", "original"),
    ("Query Classification", "classification"),
    ("Terminology Normalization", "terminology"),
    ("Query Expansion / Decomposition", "variants"),
    ("BM25 + Dense", "retrieval"),
    ("Fusion", "fusion"),
    ("Candidate Filtering", "filter"),
    ("MedCPT", "rerank"),
    ("Final Evidence", "final"),
)


def _positive_count(*values: Any) -> int:
    for value in values:
        if isinstance(value, Mapping) and value:
            return len(value)
        if isinstance(value, (list, tuple)) and value:
            return len(value)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        if isinstance(value, float) and value > 0:
            return int(value)
    return 0


def pipeline_stage_cards(
    trace: Mapping[str, Any] | None,
    *,
    evidence_count: int | None = None,
) -> list[dict[str, Any]]:
    """Explorer/pipeline view of one query trace. Output counts come from actual stage outputs."""

    trace = dict(trace or {})
    timings = dict(trace.get("stage_timings") or {})
    classification = trace.get("query_classification") or {}
    strategy = trace.get("retrieval_strategy") or {}
    filtering = trace.get("filtering_results") or {}
    expansions = list(trace.get("expansion_variants") or [])
    subqueries = list(trace.get("subqueries") or [])
    fusion = list(trace.get("fusion_results") or [])
    final = list(trace.get("final_evidence") or [])
    query_type = classification.get("query_type") or strategy.get("query_type")
    if query_type in RECOGNIZED_QUERY_TYPES:
        classification_decision = query_type
    else:
        classification_decision = (
            "Unclassified — classifier did not produce a recognized category"
        )
    original = trace.get("original_query")
    normalized = trace.get("normalized_query")
    terminology_on = bool((trace.get("understanding") or {}).get("terminology", {}).get("mappings")) or (
        normalized not in {None, original}
    )
    if normalized and original and str(normalized).strip() == str(original).strip():
        terminology_decision = "Normalization applied: no change"
    elif normalized:
        terminology_decision = normalized
    else:
        terminology_decision = "Normalization applied: no change"
    bm25_n = _positive_count(
        trace.get("bm25_results") or [],
        trace.get("bm25_ranks") or {},
        trace.get("rerank_candidate_count"),
        trace.get("candidate_k"),
        evidence_count,
        len(final),
    )
    dense_n = _positive_count(
        trace.get("dense_results") or [],
        trace.get("dense_ranks") or {},
        trace.get("rerank_candidate_count"),
        trace.get("candidate_k"),
        evidence_count,
        len(final),
    )
    fusion_n = _positive_count(
        fusion,
        trace.get("rrf_ranks") or {},
        trace.get("rerank_candidate_count"),
        trace.get("candidate_k"),
        evidence_count,
        len(final),
    )
    medcpt_on = infer_medcpt_executed(trace)
    rerank_out = _positive_count(
        trace.get("reranker_results") or [],
        trace.get("rerank_returned"),
        len(final) if medcpt_on else 0,
        evidence_count if medcpt_on else 0,
    )
    final_n = _positive_count(final, evidence_count)
    bm25_dense_out = len(trace.get("bm25_results") or []) + len(trace.get("dense_results") or [])
    if bm25_dense_out == 0:
        bm25_dense_out = bm25_n + dense_n if bm25_n or dense_n else 0
    return [
        {
            "stage": "Original Query",
            "enabled": True,
            "output_count": 1,
            "latency_ms": None,
            "decision": original or "",
        },
        {
            "stage": "Query Classification",
            "enabled": True,
            "output_count": 1,
            "latency_ms": None,
            "decision": classification_decision,
        },
        {
            "stage": "Terminology Normalization",
            "enabled": True if (terminology_on or original) else terminology_on,
            "output_count": 1 if (normalized or original) else 0,
            "latency_ms": None,
            "decision": terminology_decision,
        },
        {
            "stage": "Query Expansion / Decomposition",
            "enabled": bool(expansions or subqueries),
            "output_count": len(expansions) + len(subqueries),
            "latency_ms": timings.get("query_expansion_ms") or trace.get("query_expansion_ms"),
            "decision": f"{len(expansions)} expansions, {len(subqueries)} subqueries",
        },
        {
            "stage": "BM25 + Dense",
            "enabled": True,
            "output_count": bm25_dense_out,
            "latency_ms": (timings.get("bm25_ms") or 0) + (timings.get("dense_ms") or 0) or None,
            "decision": f"BM25 {bm25_n} · Dense {dense_n}",
        },
        {
            "stage": "Fusion",
            "enabled": True,
            "output_count": fusion_n,
            "latency_ms": timings.get("rrf_ms") or trace.get("rrf_ms"),
            "decision": strategy.get("strategy") or "fixed_hybrid",
        },
        {
            "stage": "Candidate Filtering",
            "enabled": bool(filtering) and filtering.get("before_count") is not None,
            "output_count": filtering.get("after_count", fusion_n),
            "latency_ms": timings.get("filter_ms") or filtering.get("latency_ms"),
            "decision": f"{filtering.get('before_count', fusion_n)} → {filtering.get('after_count', fusion_n)}",
        },
        {
            "stage": "MedCPT",
            "enabled": medcpt_on,
            "output_count": rerank_out,
            "latency_ms": timings.get("rerank_ms") or trace.get("rerank_ms"),
            "decision": "enabled/active" if medcpt_on else "disabled/not executed",
        },
        {
            "stage": "Final Evidence",
            "enabled": True,
            "output_count": final_n,
            "latency_ms": timings.get("total_ms") or trace.get("retrieval_ms"),
            "decision": f"{final_n} passages",
        },
    ]


def evidence_inspector_rows(evidence: Sequence[Any], trace: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    trace = dict(trace or {})
    fusion = {
        str(item.get("passage_id")): item
        for item in (trace.get("fusion_results") or [])
        if isinstance(item, dict)
    }
    decisions = {
        str(item.get("passage_id")): item.get("reason")
        for item in ((trace.get("filtering_results") or {}).get("decisions") or [])
        if isinstance(item, dict)
    }
    rows = []
    for rank, item in enumerate(evidence, start=1):
        if isinstance(item, Mapping):
            pid = item.get("passage_id")
            score = item.get("score")
            pmid = item.get("pmid", pid)
            bm25_rank = item.get("bm25_rank")
            dense_rank = item.get("dense_rank")
            rrf_score = item.get("rrf_score", item.get("score"))
            rrf_rank = item.get("rrf_rank")
            reranker_score = item.get("reranker_score")
            stage = item.get("retrieval_stage", "hybrid")
            filter_decision = item.get("filter_decision")
        else:
            pid = getattr(item, "passage_id", None)
            score = getattr(item, "score", None)
            pmid = getattr(item, "pmid", pid)
            bm25_rank = getattr(item, "bm25_rank", None)
            dense_rank = getattr(item, "dense_rank", None)
            rrf_score = getattr(item, "rrf_score", score)
            rrf_rank = getattr(item, "rrf_rank", None)
            reranker_score = getattr(item, "reranker_score", None)
            stage = getattr(item, "retrieval_stage", "hybrid")
            filter_decision = getattr(item, "filter_decision", None)
        fused = fusion.get(str(pid), {})
        final_rank = getattr(item, "final_rank", rank) if not isinstance(item, Mapping) else item.get("final_rank", rank)
        passage_display = passage_id_pmid(item) if not isinstance(item, Mapping) else (
            str(pid) if str(pid) == str(pmid) else f"{pid} / {pmid}"
        )
        rows.append(
            {
                "final_rank": final_rank,
                "passage_id": pid,
                "pmid": pmid,
                "passage_id_pmid": passage_display,
                "source": "rag-mini-bioasq",
                "retrieval_stage": stage,
                "bm25_rank": bm25_rank if bm25_rank is not None else fused.get("bm25_rank"),
                "dense_rank": dense_rank if dense_rank is not None else fused.get("dense_rank"),
                "rrf_score": rrf_score if rrf_score is not None else fused.get("rrf_score"),
                "rrf_rank": rrf_rank if rrf_rank is not None else fused.get("rrf_rank"),
                "filter_decision": filter_decision or decisions.get(str(pid), "kept"),
                "medcpt_score": reranker_score,
                "final_score": score,
            }
        )
    return rows


def evidence_table_display_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Main evidence table: no duplicate Passage ID / PMID columns."""

    displayed = []
    for row in rows:
        displayed.append(
            {
                "Final rank": row.get("final_rank"),
                "Passage ID / PMID": row.get("passage_id_pmid") or row.get("passage_id"),
                "BM25 rank": row.get("bm25_rank"),
                "Dense rank": row.get("dense_rank"),
                "RRF rank": row.get("rrf_rank"),
                "RRF score": row.get("rrf_score"),
                "MedCPT score": row.get("medcpt_score"),
                "Final score": row.get("final_score"),
            }
        )
    return displayed


def query_analysis_from_trace(trace: Mapping[str, Any] | None, *, original: str | None = None) -> dict[str, Any]:
    trace = dict(trace or {})
    understanding = dict(trace.get("understanding") or {})
    terminology = understanding.get("terminology") or {}
    classification = trace.get("query_classification") or understanding.get("query_class") or {}
    understanding = build_query_understanding(original or "", trace)
    query_type = classification.get("query_type") or understanding["intent_code"] or "unclassified"
    return {
        "original_query": understanding["original_query"] or original or "",
        "normalized_query": understanding["normalized_query"] or original or "",
        "query_type": query_type,
        "query_type_label": understanding["intent"],
        "query_type_reason": understanding["intent_reason"],
        "classifier_mechanism": understanding["classifier_mechanism"],
        "normalization_status": understanding["normalization_status"],
        "entities": understanding["entities"],
        "expanded_queries": list(trace.get("expansion_variants") or []),
        "subqueries": list(trace.get("subqueries") or []),
        "biomedical_terms": list(terminology.get("detected_terms") or understanding["medical_terms"]),
        "strategy": (trace.get("retrieval_strategy") or {}).get("strategy"),
        "search_queries": list(trace.get("search_queries") or []),
    }


def upgrade_experiment_table_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in payload.get("configurations") or []:
        rows.append(
            {
                "Configuration": item.get("name"),
                "Recall@5": item.get("recall_at_5"),
                "Recall@10": item.get("recall_at_10"),
                "Recall@50": item.get("recall_at_50"),
                "MRR@10": item.get("mrr_at_10"),
                "nDCG@10": item.get("ndcg_at_10"),
                "Candidate Recall@50": item.get("candidate_recall_at_50"),
                "Avg latency (ms)": item.get("avg_latency_ms"),
            }
        )
    return rows


def existing_result_files(root: Path | None = None) -> dict[str, Path]:
    base = root or ROOT
    candidates = {
        "Baseline Hybrid": base / EVALUATION.summary_path,
        "Hybrid + Query Expansion": base / EVALUATION.hybrid_qe_summary_path,
        "MedCPT comparison": base / EVALUATION.rerank_comparison_summary_path,
        "Upgrade comparison": base / EVALUATION.upgrade_summary_path,
        "Embedding comparison": base / EVALUATION.embedding_comparison_summary_path,
        "Candidate pool": base / EVALUATION.candidate_pool_summary_path,
        "MedCPT pool size": base / EVALUATION.medcpt_pool_summary_path,
        "Missing evidence": base / EVALUATION.missing_evidence_summary_path,
        "Objective failure": base / EVALUATION.objective_failure_summary_path,
        "Recovery matrix": base / EVALUATION.recovery_matrix_summary_path,
        "Expansion study": base / EVALUATION.expansion_study_summary_path,
        "Decomposition study": base / EVALUATION.decomposition_study_summary_path,
        "Sparse study": base / EVALUATION.sparse_study_summary_path,
    }
    return {name: path for name, path in candidates.items() if path.exists()}


def candidate_pool_table_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in payload.get("curve") or []:
        rows.append(
            {
                "K": item.get("k"),
                "Recall@K": item.get("recall_at_k"),
                "MRR@10": item.get("mrr_at_10"),
                "nDCG@10": item.get("ndcg_at_10"),
                "Avg latency (ms)": item.get("avg_latency_ms"),
            }
        )
    return rows


def medcpt_pool_table_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in payload.get("configurations") or []:
        rows.append(
            {
                "Pool": item.get("candidate_pool"),
                "Recall@10": item.get("recall_at_10"),
                "MRR@10": item.get("mrr_at_10"),
                "nDCG@10": item.get("ndcg_at_10"),
                "Candidate recall": item.get("candidate_recall"),
                "Est. MedCPT (ms)": item.get("estimated_medcpt_ms"),
                "Est. total (ms)": item.get("estimated_total_ms"),
            }
        )
    return rows


def missing_evidence_table_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in payload.get("table") or []:
        rows.append(
            {
                "Failure category": item.get("failure_category"),
                "Count": item.get("count"),
                "Percentage": item.get("percentage"),
                "Potential remedy": item.get("potential_remedy"),
            }
        )
    return rows


def missing_evidence_passage_table_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in payload.get("passage_table") or []:
        rows.append(
            {
                "Failure category": item.get("failure_category"),
                "Count": item.get("count"),
                "Percentage": item.get("percentage"),
                "Potential remedy": item.get("potential_remedy"),
            }
        )
    return rows


def objective_failure_table_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    counts = payload.get("objective_failure_counts") or {}
    percents = payload.get("objective_failure_percent_of_misses") or {}
    order = (
        "BOTH_RETRIEVERS_MISS",
        "DENSE_ONLY_MISS",
        "BM25_ONLY_MISS",
        "FUSION_CUTOFF",
        "NOT_IN_CORPUS",
    )
    rows = []
    for name in order:
        if name not in counts:
            continue
        rows.append(
            {
                "Objective category": name,
                "Count": counts[name],
                "Share of Hybrid@50 misses": percents.get(name),
            }
        )
    for name, count in counts.items():
        if name in order:
            continue
        rows.append(
            {
                "Objective category": name,
                "Count": count,
                "Share of Hybrid@50 misses": percents.get(name),
            }
        )
    return rows


def recovery_matrix_table_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    counts = payload.get("counts") or payload.get("recovery_matrix_counts") or {}
    n = payload.get("n_hybrid50_failures") or sum(counts.values()) or 1
    rows = []
    for name, count in counts.items():
        rows.append(
            {
                "Earliest recovery mechanism @100": name,
                "Count": count,
                "Share of Hybrid@50 misses": count / n if n else None,
            }
        )
    return rows


def sparse_study_metric_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in list(payload.get("full_retrieval_metrics") or []) + list(payload.get("fusion") or []):
        rows.append(
            {
                "Configuration": item.get("name"),
                "Recall@5": item.get("recall_at_5"),
                "Recall@10": item.get("recall_at_10"),
                "Recall@50": item.get("recall_at_50"),
                "Recall@100": item.get("recall_at_100"),
                "MRR@10": item.get("mrr_at_10"),
                "nDCG@10": item.get("ndcg_at_10"),
                "Avg latency (ms)": item.get("avg_latency_ms"),
            }
        )
    return rows


def sparse_fusion_comparison_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    wanted = {"A BM25 + MiniLM", "B BM25 + MiniLM + Sparse"}
    rows = []
    for item in payload.get("fusion") or []:
        if item.get("name") not in wanted:
            continue
        dual = next(
            (row for row in (payload.get("fusion_dual_miss") or []) if row.get("name") == item.get("name")),
            {},
        )
        rows.append(
            {
                "Configuration": item.get("name"),
                "Status": "BASELINE" if str(item.get("name", "")).startswith("A ") else "EXPERIMENTAL",
                "Recall@10": item.get("recall_at_10"),
                "Recall@50": item.get("recall_at_50"),
                "MRR@10": item.get("mrr_at_10"),
                "nDCG@10": item.get("ndcg_at_10"),
                "Avg latency (ms)": item.get("avg_latency_ms"),
                "Dual-miss@50": dual.get("dual_miss_hit_at_50"),
                "Dual-miss@100": dual.get("dual_miss_hit_at_100"),
            }
        )
    return rows


def sparse_recovery_matrix_preview_rows(payload: Mapping[str, Any], *, limit: int = 20) -> list[dict[str, Any]]:
    rows = []
    for item in (payload.get("recovery_matrix_rows") or [])[:limit]:
        rows.append(
            {
                "Gold ID": item.get("gold_passage_id"),
                "BM25": item.get("BM25"),
                "MiniLM": item.get("MiniLM"),
                "PubMedBERT": item.get("PubMedBERT"),
                "Sparse": item.get("Sparse"),
                "Fused": item.get("Fused_B") or item.get("Fused"),
            }
        )
    return rows


def sparse_inspector_rows(payload: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    payload = dict(payload or {})
    sparse = payload.get("sparse") or {}
    latency = payload.get("latency") or {}
    skipped = bool(payload.get("skipped"))
    return [
        {"Field": "Model", "Value": sparse.get("model") or UPGRADE.sparse_model_name},
        {"Field": "Retrieval type", "Value": "first-stage learned sparse (SPLADE)"},
        {"Field": "Nature", "Value": "sparse term-weight / not dense / not BM25 lexical"},
        {"Field": "Index", "Value": UPGRADE.sparse_cache_dir},
        {"Field": "Top-K (experiment)", "Value": "10 / 50 / 100 / 200"},
        {"Field": "Index build (s)", "Value": latency.get("sparse_index_seconds")},
        {"Field": "Index bytes", "Value": latency.get("sparse_index_nbytes")},
        {"Field": "Avg search (ms)", "Value": latency.get("sparse_avg_search_ms")},
        {"Field": "Runtime flag", "Value": "ENABLED" if UPGRADE.sparse_enabled else "DISABLED"},
        {"Field": "Status", "Value": "EXPERIMENTAL" if not skipped else "EXPERIMENTAL (not loaded)"},
    ]


def embedding_comparison_table_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in payload.get("configurations") or []:
        rows.append(
            {
                "Configuration": item.get("name"),
                "Recall@5": item.get("recall_at_5"),
                "Recall@10": item.get("recall_at_10"),
                "Recall@50": item.get("recall_at_50"),
                "MRR@10": item.get("mrr_at_10"),
                "nDCG@10": item.get("ndcg_at_10"),
                "Dimension": item.get("embedding_dimension"),
                "Avg latency (ms)": item.get("avg_latency_ms"),
            }
        )
    return rows


def feature_status_table_rows() -> list[dict[str, Any]]:
    rows = []
    for item in experimental_feature_status():
        rows.append(
            {
                "Feature": item["name"],
                "Lane": item["lane"],
                "Runtime": item["runtime"],
                "Default": item["default"],
                "Benchmark": item["benchmark"],
            }
        )
    return rows


def failure_analysis_table_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in payload.get("rows") or []:
        failure = item.get("failure") or {}
        if failure.get("category") in {None, "success"}:
            continue
        rows.append(
            {
                "query_id": item.get("query_id"),
                "question": item.get("question"),
                "configuration": item.get("configuration"),
                "category": failure.get("category"),
                "query_type": failure.get("query_type"),
                "recall_at_10": item.get("recall_at_10"),
            }
        )
    return rows


def architecture_document_text() -> str:
    path = ROOT / "architecture.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "architecture.md was not found."


def research_document_sections() -> list[tuple[str, str]]:
    sections = []
    for name in ("README.md", "architecture_essentials.md", "PRD.md"):
        path = ROOT / name
        if path.exists():
            sections.append((name, path.read_text(encoding="utf-8")))
    return sections


def _short(text: str, limit: int = 280) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"


def _manual_example_note(row: Mapping[str, Any]) -> str:
    question = str(row.get("question", ""))
    reference = str(row.get("reference_answer", ""))
    generated = str(row.get("generated_answer", ""))
    evidence = " ".join(str(item) for item in row.get("retrieved_evidence", []))
    citations = row.get("citation_validation") or {}
    parts: list[str] = []
    if "insufficient evidence" in generated.lower():
        parts.append("The system returned an insufficient-evidence response.")
    if "metformin" in question.lower() and "metformin" not in evidence.lower():
        parts.append(
            "Retrieved passages discuss levothyroxine absorption but do not mention metformin."
        )
    if re.search(r"DVL-3", reference, re.I) and not re.search(r"DVL-3", generated, re.I):
        parts.append(
            "Reference lists DVL-1, DVL-2, and DVL-3; the generated answer includes DVL-1 and DVL-2 but not DVL-3."
        )
    if citations.get("is_valid"):
        parts.append("Citation validation is valid.")
    elif citations.get("missing_citations"):
        parts.append("Citation validation reports missing citations.")
    elif citations.get("invalid_ids"):
        parts.append("Citation validation reports invalid citation IDs.")
    return " ".join(parts)


def load_manual_review_examples(
    details: Mapping[str, Any],
    query_ids: Sequence[int],
) -> list[dict[str, Any]]:
    wanted = {int(item) for item in query_ids}
    examples = []
    for row in details.get("rows", []):
        query_id = row.get("query_id")
        if query_id not in wanted:
            continue
        examples.append(
            {
                "query_id": query_id,
                "question": row.get("question", ""),
                "reference_answer": _short(str(row.get("reference_answer", ""))),
                "generated_answer": _short(str(row.get("generated_answer", ""))),
                "citation_validation": (
                    "Valid" if (row.get("citation_validation") or {}).get("is_valid") else "Invalid"
                ),
                "note": _manual_example_note(row),
            }
        )
    examples.sort(key=lambda item: list(query_ids).index(item["query_id"]))
    return examples


def quality_latency_points(configurations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    points = []
    for item in configurations:
        if item.get("name") is None:
            continue
        if item.get("recall_at_10") is None or item.get("avg_latency_ms") is None:
            continue
        points.append(
            {
                "Configuration": item["name"],
                "Recall@10": item["recall_at_10"],
                "Avg latency (ms)": item["avg_latency_ms"],
            }
        )
    return points


def index_retrieval_details(
    *payloads: Mapping[str, Any],
) -> dict[int, dict[str, dict[str, Any]]]:
    indexed: dict[int, dict[str, dict[str, Any]]] = {}
    for payload in payloads:
        for row in payload.get("rows", []):
            query_id = row.get("query_id")
            configuration = row.get("configuration")
            if query_id is None or not configuration:
                continue
            indexed.setdefault(int(query_id), {})[str(configuration)] = dict(row)
    return indexed


def format_query_option(query_id: Any, question: str) -> str:
    text = " ".join(str(question).split())
    if len(text) > 90:
        text = text[:89] + "…"
    return f"{query_id}: {text}"


def query_id_from_choice(label: str) -> int:
    return int(str(label).split(":", 1)[0])


def retrieval_query_choices(indexed: Mapping[int, Mapping[str, Mapping[str, Any]]]) -> list[str]:
    labels = []
    for query_id in sorted(indexed):
        question = ""
        for row in indexed[query_id].values():
            question = str(row.get("question") or "")
            if question:
                break
        labels.append(format_query_option(query_id, question))
    return labels


def per_query_retrieval_rows(
    configs: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in RETRIEVAL_CONFIG_ORDER:
        source = configs.get(name)
        if not source:
            continue
        row: dict[str, Any] = {"Configuration": name}
        for label, key in (
            ("Recall@5", "recall_at_5"),
            ("Recall@10", "recall_at_10"),
            ("MRR@10", "mrr_at_10"),
            ("nDCG@10", "ndcg_at_10"),
            ("Latency (ms)", "latency_ms"),
        ):
            if key in source and source[key] is not None:
                row[label] = source[key]
        ids = source.get("retrieved_passage_ids")
        if ids:
            row["Retrieved passage IDs"] = ", ".join(str(item) for item in ids)
        rows.append(row)
    return rows


def answer_query_choices(details: Mapping[str, Any]) -> list[str]:
    labels = []
    for row in details.get("rows", []):
        if row.get("query_id") is None:
            continue
        labels.append(format_query_option(row["query_id"], str(row.get("question") or "")))
    return labels


def answer_row_by_id(details: Mapping[str, Any], query_id: int) -> dict[str, Any] | None:
    for row in details.get("rows", []):
        if row.get("query_id") == query_id:
            return dict(row)
    return None


def answer_inspection_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for label, key in (
        ("Query ID", "query_id"),
        ("Question", "question"),
        ("Reference answer", "reference_answer"),
        ("Generated answer", "generated_answer"),
        ("Correctness", "correctness"),
        ("Groundedness", "groundedness"),
        ("Context relevance", "context_relevance"),
        ("Judge reason", "judge_reason"),
    ):
        if key in row and row[key] is not None:
            fields[label] = row[key]
    citations = row.get("citation_validation") or {}
    if "is_valid" in citations:
        fields["Citation validation"] = "Valid" if citations.get("is_valid") else "Invalid"
    return fields


def answer_review_summary_counts(
    details: Mapping[str, Any],
    *,
    success_ids: Sequence[int] = SUCCESS_REVIEW_IDS,
    failure_ids: Sequence[int] = FAILURE_REVIEW_IDS,
) -> dict[str, int]:
    rows = list(details.get("rows", []))
    present = {int(row["query_id"]) for row in rows if row.get("query_id") is not None}
    citation_valid = 0
    citation_problems = 0
    insufficient = 0
    for row in rows:
        citations = row.get("citation_validation") or {}
        if citations.get("is_valid") is True:
            citation_valid += 1
        if (
            citations.get("is_valid") is False
            or citations.get("missing_citations")
            or citations.get("invalid_ids")
        ):
            citation_problems += 1
        generated = str(row.get("generated_answer") or "").lower()
        if row.get("insufficient_evidence") is True or "insufficient evidence" in generated:
            insufficient += 1
    return {
        "Successful manual-review examples": sum(1 for item in success_ids if int(item) in present),
        "Limitation / failure examples": sum(1 for item in failure_ids if int(item) in present),
        "Insufficient-evidence cases": insufficient,
        "Citation-valid cases": citation_valid,
        "Missing / invalid citation cases": citation_problems,
    }


def build_pipeline(*, rerank_enabled: bool | None = None) -> GroundedQA:
    """Load the passage corpus and wire the existing RAG components."""

    split = os.environ.get("BIOMED_UI_CORPUS_SPLIT", UI.corpus_split)
    passages = get_passage_collection(split=split)
    if not passages:
        raise RuntimeError("The passage collection is empty.")
    model_name = resolve_embedding_model()
    cache_dir = UPGRADE.embedding_cache_dir if UPGRADE.biomedical_embeddings_enabled else None
    bm25 = BM25Retriever(passages)
    dense = DenseRetriever(passages, model_name=model_name, cache_dir=cache_dir)
    hybrid = HybridRetriever(passages, bm25_retriever=bm25, dense_retriever=dense)
    enabled = RERANKER.enabled if rerank_enabled is None else rerank_enabled
    rerank_config = RerankerConfig(
        enabled=enabled,
        candidate_pool_size=RERANKER.candidate_pool_size,
        final_top_k=RERANKER.final_top_k,
        model_name=RERANKER.model_name,
        device=RERANKER.device,
        batch_size=RERANKER.batch_size,
        max_length=RERANKER.max_length,
    )
    if UPGRADE.pipeline_enabled:
        retriever = StagedRetriever(
            bm25_retriever=bm25,
            dense_retriever=dense,
            passages=passages,
            config=UPGRADE,
            rerank_enabled=enabled,
        )
        if enabled:
            retriever.preload()
    else:
        retriever = wrap_retriever(hybrid, config=rerank_config)
        if enabled:
            retriever.preload()
    return GroundedQA(retriever, expander=QueryExpander())


@st.cache_resource
def load_pipeline(rerank_enabled: bool | None = None, lane: str = "library") -> GroundedQA:
    """Load a retrieval stack for a named lane.

    Default lane is library (not production): MedCPT follows RERANKER.enabled.
    ``lane='production'`` / ``load_production_pipeline()`` always resolve MedCPT on.
    Playground passes an explicit checkbox override; its default is MedCPT on.
    """

    enabled = resolve_rerank_enabled(override=rerank_enabled, lane=lane)
    return build_pipeline(rerank_enabled=enabled)


@st.cache_resource
def load_production_pipeline() -> GroundedQA:
    """Explicit production entry point. MedCPT on, pool 50, RRF k=60."""

    return load_pipeline(lane="production")


@st.cache_data
def load_evaluation_payload(path: str) -> dict[str, Any]:
    return load_json_result(Path(path))


def _latency_history_ms() -> list[float]:
    rows = load_live_evaluations(ROOT / EVALUATION.live_evaluation_path)
    values = []
    for row in rows:
        value = row.get("latency_ms")
        if value is not None:
            values.append(float(value))
    return values


def _store_runtime_state(
    result: GroundedQAResult,
    *,
    expand: bool,
    rerank: bool | None,
    latency_ms: float,
    live_quality: Mapping[str, Any] | None,
) -> dict[str, Any]:
    state = build_runtime_state(
        result,
        expand_requested=expand,
        rerank_requested=rerank,
        total_latency_ms=latency_ms,
        live_quality=live_quality,
        latency_history_ms=_latency_history_ms(),
    )
    st.session_state["last_runtime_state"] = state
    return state


def _current_runtime_state() -> dict[str, Any] | None:
    result = st.session_state.get("last_qa_result")
    if result is None:
        return st.session_state.get("last_runtime_state")
    return _store_runtime_state(
        result,
        expand=bool(st.session_state.get("last_expand")),
        rerank=st.session_state.get("last_rerank"),
        latency_ms=float(st.session_state.get("last_latency_ms") or 0.0),
        live_quality=st.session_state.get("last_live_quality"),
    )


def _seconds_label(ms: Any) -> str:
    if ms is None:
        return "—"
    return f"{float(ms) / 1000.0:.2f} s"


def main() -> None:
    st.set_page_config(page_title=UI.page_title, layout="wide")
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1240px; }
        h1 { font-weight: 600; letter-spacing: -0.03em; font-size: 1.7rem; color: #0f172a; }
        h2 { font-weight: 600; letter-spacing: -0.02em; margin-top: 1.8rem; color: #0f172a; }
        h3 { font-weight: 600; letter-spacing: -0.01em; margin-top: 1.3rem; color: #1e293b; }
        .answer-body { font-size: 1.12rem; line-height: 1.7; }
        .cite-chip { display:inline-block; margin-right:0.35rem; font-variant-numeric: tabular-nums; }
        div[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }
        div[data-testid="stMetric"] { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 0.35rem 0.6rem; }
        section.main { background: #ffffff; }
        .stApp { background: #ffffff; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("**RAG Evaluation Workbench**")
        st.caption("Dataset → Experiment → Trace → Evaluation")
        group = st.radio("Workspace", list(NAV_GROUPS), label_visibility="collapsed", key="nav_workspace")
        pages = list(NAV_GROUPS[group])
        if len(pages) == 1:
            page = pages[0]
        else:
            page = st.radio(group, pages, key=f"nav_{group}")
        st.divider()
        st.markdown("**Project**")
        st.caption("BioASQ Mini · 100-query benchmark")
        st.markdown("**Production path**")
        st.caption(
            "BM25 + MiniLM + RRF k=60 → Top 50 → MedCPT Cross-Encoder → Top 10 → "
            f"{GENERATION.model_name}"
        )
        st.caption("Offline evaluation is repeatable. Playground is online/sample traffic.")
        if "last_expand" not in st.session_state:
            st.session_state["last_expand"] = False
        if "last_rerank" not in st.session_state:
            st.session_state["last_rerank"] = True

    from app import workbench_pages as wp

    routers = {
        ("Overview", "Overview"): wp.render_overview,
        ("Playground", "Playground"): _render_playground,
        ("Evaluations", "Datasets"): wp.render_datasets,
        ("Evaluations", "Benchmark Runs"): wp.render_benchmark_runs,
        ("Evaluations", "Metrics"): wp.render_metrics,
        ("Evaluations", "Failure Analysis"): wp.render_failure_analysis,
        ("Evaluations", "Regression"): wp.render_regression,
        ("Experiments", "Experiment List"): wp.render_experiment_list,
        ("Experiments", "Compare"): wp.render_compare,
        ("Experiments", "Per-query Diff"): wp.render_per_query_diff,
        ("Traces", "Trace Explorer"): wp.render_trace_explorer,
        ("Traces", "Trace Detail"): wp.render_trace_detail,
        ("Data", "Data Quality"): wp.render_data_quality,
        ("Data", "Gold Quality"): wp.render_gold_quality,
        ("System", "Models"): wp.render_models,
        ("System", "Prompts"): wp.render_prompts,
        ("System", "Architecture"): wp.render_architecture,
        ("System", "Observability"): wp.render_observability,
    }
    routers[(group, page)]()


def _render_explorer() -> None:
    _render_playground()


def _render_playground() -> None:
    st.header("Playground")
    st.caption("Ask a biomedical question. Ranking and answering are research tools, not medical advice.")
    with st.form("question_form"):
        question = st.text_area(
            "Question",
            height=90,
            placeholder="What is cerebral hemiatrophy?",
        )
        controls = st.columns(2)
        with controls[0]:
            deep_rerank = st.checkbox(
                "Deep reranking (MedCPT)",
                value=PLAYGROUND_DEFAULT_RERANK_ENABLED,
                help="Playground override. Default on matches declared production (Hybrid@50 → MedCPT@10). Uncheck to run Hybrid only for this request. Library default RERANKER.enabled remains False.",
            )
        with controls[1]:
            use_query_expansion = st.checkbox(
                "Query expansion",
                value=False,
                help="Off in production. Original query is kept; alternates are extra search strings only.",
            )
        submitted = st.form_submit_button("Ask", type="primary")

    if not submitted:
        if st.session_state.get("last_qa_result") is not None:
            _render_result(
                st.session_state["last_qa_result"],
                expand=bool(st.session_state.get("last_expand")),
                latency_ms=float(st.session_state.get("last_latency_ms") or 0.0),
                live_quality=st.session_state.get("last_live_quality"),
            )
        return
    if not question.strip():
        st.warning("Enter a biomedical question to search.")
        return

    try:
        with st.spinner("Loading retrieval indexes..."):
            pipeline = load_pipeline(rerank_enabled=bool(deep_rerank), lane="playground")
    except Exception as exc:
        st.error(public_error_message(exc, stage="init"))
        return

    try:
        with st.spinner("Retrieving evidence and generating an answer..."):
            started = time.perf_counter()
            result = pipeline.ask(question.strip(), expand=use_query_expansion)
            latency_ms = (time.perf_counter() - started) * 1000.0
    except Exception as exc:
        st.error(public_error_message(exc, stage=classify_ask_error(exc)))
        return

    live_record = evaluate_live_query(
        question=result.question,
        answer=result.answer,
        evidence=result.evidence,
        citations=result.citations,
        insufficient_evidence=result.insufficient_evidence,
        latency_ms=latency_ms,
        stage_timings={
            "bm25_ms": (result.trace or {}).get("bm25_ms"),
            "dense_ms": (result.trace or {}).get("dense_ms"),
            "rrf_ms": (result.trace or {}).get("rrf_ms"),
            "rerank_ms": (result.trace or {}).get("rerank_ms"),
            "generation_ms": (result.trace or {}).get("generation_ms"),
            "total_ms": latency_ms,
        },
        estimated_cost_usd=None,
    )
    try:
        append_live_evaluation(live_record, ROOT / EVALUATION.live_evaluation_path)
    except OSError:
        pass
    live_quality = {
        "evaluated": True,
        "status": "Heuristic (non-blocking)",
        "evaluator_type": "heuristic",
        "groundedness": (live_record["metrics"] or {}).get("faithfulness"),
        "context_relevance": (live_record["metrics"] or {}).get("context_precision"),
        "reason": "Live evaluation uses deterministic/heuristic checks and does not call an LLM.",
        "parsed": True,
        "model_name": "heuristic_live_evaluator",
        "latency_ms": 0.0,
        "llm_calls": 0,
        "live_record": live_record,
    }
    st.session_state["last_qa_result"] = result
    st.session_state["last_expand"] = use_query_expansion
    st.session_state["last_latency_ms"] = latency_ms
    st.session_state["last_rerank"] = deep_rerank
    st.session_state["last_live_quality"] = live_quality
    _store_runtime_state(
        result,
        expand=use_query_expansion,
        rerank=deep_rerank,
        latency_ms=latency_ms,
        live_quality=live_quality,
    )
    _render_result(
        result,
        expand=use_query_expansion,
        latency_ms=latency_ms,
        live_quality=live_quality,
    )


def _run_live_answer_quality(result: GroundedQAResult) -> dict[str, Any]:
    try:
        with st.spinner("Scoring live answer quality..."):
            return evaluate_live_answer_quality(
                question=result.question,
                generated_answer=result.answer,
                evidence=result.evidence,
                insufficient_evidence=result.insufficient_evidence,
                generation_llm_called=result.llm_called,
            )
    except Exception as exc:
        return {
            "evaluated": False,
            "status": "Not evaluated",
            "groundedness": None,
            "context_relevance": None,
            "reason": public_error_message(exc, stage="llm"),
            "parsed": False,
            "model_name": EVALUATION.judge_model_name,
            "latency_ms": None,
            "llm_calls": 0,
            "evaluator_type": "llm_judge",
        }


def _render_answer_html(answer: str, evidence: Sequence[Any]) -> str:
    ranks = {str(getattr(item, "passage_id", "")): index for index, item in enumerate(evidence, start=1)}

    def _replace(match: re.Match[str]) -> str:
        pid = match.group(1).strip()
        label = ranks.get(pid, pid)
        return f'<a class="cite-chip" href="#passage-{pid}">[{label}]</a>'

    return CITATION_PATTERN.sub(_replace, answer or "")


def _require_last_result() -> GroundedQAResult | None:
    result = st.session_state.get("last_qa_result")
    if result is None:
        st.info("Ask a question in Playground to populate this view.")
        return None
    return result


def _render_answer_page() -> None:
    result = _require_last_result()
    if result is None:
        return
    _render_question_and_answer(result, _current_runtime_state() or {})


def _render_evidence_page() -> None:
    result = _require_last_result()
    if result is None:
        return
    _render_evidence_section(result, _current_runtime_state() or {})


def _render_question_and_answer(result: GroundedQAResult, state: Mapping[str, Any]) -> None:
    st.markdown("### Question")
    st.write(result.question)

    st.markdown("### Grounded answer")
    if result.insufficient_evidence:
        st.warning(result.answer)
    else:
        html = _render_answer_html(result.answer, result.evidence)
        st.markdown(f'<div class="answer-body">{html}</div>', unsafe_allow_html=True)
    cited = list(result.citations.valid_ids)
    if cited:
        chips = " ".join(f"[{index}] `{pid}`" for index, pid in enumerate(cited, start=1))
        st.markdown(chips)
    st.write(f"**Citation validation:** {state.get('citation_status') or ('Valid' if result.citations.is_valid else 'Invalid')}")
    for level, message in citation_status_messages(
        result.citations,
        insufficient_evidence=result.insufficient_evidence,
    ):
        if level == "error":
            st.error(message)
        elif level == "warning":
            st.warning(message)
        else:
            st.success(message)


def _render_evidence_section(result: GroundedQAResult, state: Mapping[str, Any]) -> None:
    st.markdown("### Evidence supporting answer")
    rows = evidence_inspector_rows(result.evidence, state.get("trace") or result.trace)
    display = evidence_table_display_rows(rows)
    if display:
        help_bits = "  \n".join(f"**{name}:** {text}" for name, text in RANKING_COLUMN_HELP.items())
        if state.get("final_score_equals_medcpt"):
            help_bits += "  \n**Final score:** Final score = MedCPT score."
        st.caption(help_bits)
        st.dataframe(display, hide_index=True, use_container_width=True)
    if not result.evidence:
        st.info("No evidence passages were retrieved.")
        return
    focus = st.session_state.get("focus_passage")
    for rank, item in enumerate(result.evidence, start=1):
        pid = passage_id_pmid(item)
        expanded = focus is not None and str(getattr(item, "passage_id", "")) == str(focus)
        st.markdown(f'<div id="passage-{item.passage_id}"></div>', unsafe_allow_html=True)
        with st.expander(f"{rank}. {pid}  ·  score {float(item.score):.4f}", expanded=expanded or rank == 1):
            st.write(item.text)
            row = rows[rank - 1] if rank - 1 < len(rows) else {}
            with st.expander("Details", expanded=False):
                st.write(f"**source:** rag-mini-bioasq")
                st.write(f"**retrieval_stage:** `{row.get('retrieval_stage')}`")
                st.write(f"**filter_decision:** `{row.get('filter_decision')}`")
            with st.expander("Why this passage ranked here", expanded=False):
                st.write(f"**Passage** `{pid}`")
                st.write(f"**Final rank** #{row.get('final_rank') or rank}")
                st.markdown("**Retrieval journey**")
                st.write(f"BM25 rank: #{row.get('bm25_rank') if row.get('bm25_rank') is not None else '—'}")
                st.write(f"Dense rank: #{row.get('dense_rank') if row.get('dense_rank') is not None else '—'}")
                st.write(f"RRF rank: #{row.get('rrf_rank') if row.get('rrf_rank') is not None else '—'}")
                st.write(f"RRF score: {row.get('rrf_score')}")
                st.write(f"MedCPT score: {row.get('medcpt_score')}")
                final_note = "Final score = MedCPT score" if state.get("final_score_equals_medcpt") else "Final score"
                st.write(f"{final_note}: {row.get('final_score')}")


def _render_query_understanding(result: GroundedQAResult, state: Mapping[str, Any]) -> None:
    understanding = dict(state.get("understanding") or build_query_understanding(result.question, result.trace, expansions=result.expansions))
    st.markdown("### Query Understanding")
    st.write(f"**Original:** {understanding.get('original_query') or result.question}")
    st.write(f"**Intent:** {understanding.get('intent')}")
    if understanding.get("intent_code") == "unclassified" or not understanding.get("intent_recognized", True):
        st.caption(f"Reason: {understanding.get('intent_reason')}")
    st.caption(str(understanding.get("classifier_mechanism") or ""))
    entities = understanding.get("entities") or []
    st.write("**Entities:** " + (", ".join(str(item) for item in entities) if entities else "—"))
    terms = understanding.get("medical_terms") or []
    st.write("**Medical terms:** " + (", ".join(str(item) for item in terms) if terms else "—"))
    st.write(f"**Normalized:** {understanding.get('normalized_query')}")
    st.caption(str(understanding.get("normalization_status")))
    rewrites = list(result.expansions or understanding.get("rewrites") or [])
    if rewrites:
        st.write("**Rewrites:**")
        for index, item in enumerate(rewrites, start=1):
            st.write(f"{index}. {item}")
    else:
        status = state.get("query_expansion_status") or "OFF"
        st.caption(f"Rewrites: none (Query Expansion {status})")
    if understanding.get("subqueries"):
        st.write("**Subqueries:**")
        for item in understanding["subqueries"]:
            st.write(f"- {item}")


def _render_retrieval_summary(state: Mapping[str, Any]) -> None:
    st.markdown("### Retrieval summary")
    by_key = dict(state.get("stage_by_key") or {})
    order = ("query", "understanding", "bm25", "dense", "rrf", "medcpt", "evidence", "generation", "citations", "evaluation")
    lines = []
    for key in order:
        stage = by_key.get(key) or {}
        if not stage:
            continue
        latency = _format_ms(stage.get("latency_ms")) or ""
        count = stage.get("output_count")
        detail = stage.get("detail") or ""
        lines.append(
            f"**{stage.get('label')}** · {stage.get('status')} · "
            f"{detail or (str(count) + ' outputs')} · {latency}".rstrip(" ·")
        )
    st.markdown("  \n↓  \n".join(lines))
    for key in ("bm25", "dense", "rrf", "medcpt", "evidence", "generation", "citations"):
        stage = by_key.get(key) or {}
        if stage.get("output_count") == 0 and key in {"bm25", "dense", "rrf"} and state.get("evidence_count"):
            st.error(f"Inconsistent telemetry: {stage.get('label')} reports 0 outputs.")


def _render_latency_waterfall(state: Mapping[str, Any]) -> None:
    latency = dict(state.get("latency") or {})
    st.markdown("### Latency")
    st.write(f"**Total latency:** {_seconds_label(latency.get('total_ms'))}")
    series = {
        "Query Expansion": (latency.get("query_expansion_ms") or 0) / 1000.0,
        "BM25": (latency.get("bm25_ms") or 0) / 1000.0,
        "Dense": (latency.get("dense_ms") or 0) / 1000.0,
        "RRF": (latency.get("rrf_ms") or 0) / 1000.0,
        "MedCPT": (latency.get("medcpt_ms") or 0) / 1000.0,
        "Answer Generation": (latency.get("generation_ms") or 0) / 1000.0,
        "Citation Validation": (latency.get("citation_ms") or 0) / 1000.0,
    }
    if not state.get("query_expansion_executed"):
        series.pop("Query Expansion", None)
    if not state.get("medcpt_executed"):
        series.pop("MedCPT", None)
    st.bar_chart(series)
    for name, seconds in series.items():
        st.write(f"{name:<22} {seconds:.2f} s")
    pct = st.columns(3)
    pct[0].metric("P50", _seconds_label(latency.get("p50_ms")))
    pct[1].metric("P95", _seconds_label(latency.get("p95_ms")))
    pct[2].metric("P99", _seconds_label(latency.get("p99_ms")))
    if not latency.get("n"):
        st.caption("Percentiles appear after live-query history is stored.")
    st.write(f"BM25 searches: {state.get('bm25_searches')}")
    st.write(f"Dense searches: {state.get('dense_searches')}")
    st.write(f"RRF operations: {state.get('rrf_operations')}")
    st.write(f"Candidate pool: {state.get('candidate_pool')}")
    st.write(f"Candidates reranked: {state.get('candidates_reranked')}")
    st.write(f"Evidence returned: {state.get('evidence_count')}")
    st.write(f"LLM calls: {state.get('llm_calls')}")
    cost = dict(state.get("cost") or {})
    qe = cost.get("query_expansion_usd")
    st.write(f"Query expansion cost: ${qe:.4f}" if qe is not None else "Query expansion cost: —")
    st.write("Generation cost: Not measured")
    total = cost.get("total_estimated_usd")
    st.write(f"Total estimated cost: ${total:.4f}" if total is not None else "Total estimated cost: Not measured")
    st.caption(str(cost.get("generation_note") or ""))


def _render_claim_coverage(result: GroundedQAResult) -> None:
    coverage = claim_evidence_coverage(result.answer, result.evidence, result.citations)
    st.markdown("### Claim → Evidence Coverage")
    st.caption("Evaluation metric only. It does not change generation.")
    st.write(coverage["summary"])
    if coverage.get("claim_support_rate") is not None:
        st.write(f"Claim support rate: {coverage['claim_support_rate']:.1%}")
    for row in coverage.get("claims") or []:
        mark = "✓" if row["supported"] else "✗"
        pids = ", ".join(f"PMID {pid}" for pid in row["passage_ids"]) or "No supporting evidence"
        st.write(f"Claim {row['claim_index']} {mark} → {pids}")
        st.caption(row["claim"])


def _render_result(
    result: GroundedQAResult,
    *,
    expand: bool,
    latency_ms: float,
    live_quality: Mapping[str, Any] | None = None,
) -> None:
    state = _store_runtime_state(
        result,
        expand=expand,
        rerank=bool(st.session_state.get("last_rerank")),
        latency_ms=latency_ms,
        live_quality=live_quality,
    )
    _render_question_and_answer(result, state)
    _render_evidence_section(result, state)
    _render_retrieval_summary(state)
    _render_query_understanding(result, state)
    _render_live_answer_quality(live_quality or {}, result=result, state=state)
    _render_claim_coverage(result)
    _render_deep_evaluation_section(live_quality or {}, state)
    with st.expander("Full trace / advanced diagnostics", expanded=False):
        _render_latency_waterfall(state)
        payload = build_request_trace(
            result,
            expand=expand,
            total_latency_ms=latency_ms,
            live_quality=live_quality,
        )
        _render_request_trace(payload)
        if result.prompt:
            st.markdown("**Answer generation prompt (this request)**")
            st.code(result.prompt)
        else:
            st.caption("No generation prompt (insufficient evidence or LLM not called).")


def _render_live_answer_quality(
    quality: Mapping[str, Any],
    *,
    result: GroundedQAResult | None = None,
    state: Mapping[str, Any] | None = None,
) -> None:
    st.markdown("### Answer Quality")
    st.caption("Live evaluation uses deterministic/heuristic checks and does not call an LLM.")
    evaluator = (state or {}).get("evaluator_label") or "Heuristic evaluator"
    st.write(f"**Evaluator:** {evaluator}")
    st.write(f"**Evaluator latency:** {_format_ms(quality.get('latency_ms')) or '0.0 ms'}")
    judge_calls = (state or {}).get("llm_judge_calls")
    if judge_calls is None:
        judge_calls = quality.get("llm_calls", 0)
    st.write(f"**LLM judge calls:** {judge_calls}")
    if not quality.get("evaluated"):
        st.info(quality.get("status") or "Not evaluated")
        return
    metrics = dict((quality.get("live_record") or {}).get("metrics") or {})
    citations = getattr(result, "citations", None) if result is not None else None
    checks = [
        ("Citation validity", "Valid" if (state or {}).get("citations_valid") else "Invalid"),
        (
            "Citation coverage",
            format_heuristic_score(metrics.get("citation_correctness")),
        ),
        ("Evidence count", str((state or {}).get("evidence_count") if state else "")),
        ("Heuristic groundedness", format_heuristic_score(quality.get("groundedness") or metrics.get("faithfulness"))),
        ("Context relevance", format_heuristic_score(quality.get("context_relevance") or metrics.get("context_precision"))),
        ("Unsupported citation detection", "none" if citations is None or not citations.invalid_ids else ", ".join(str(item) for item in citations.invalid_ids)),
        ("Retrieval hit/miss", "hit" if (state or {}).get("evidence_count") else "miss"),
        ("Answer generated successfully", "yes" if result is not None and result.answer else "no"),
    ]
    for label, value in checks:
        st.write(f"**{label}:** {value}")
    st.caption("Scores above are 0–1 heuristic values, not a 1–5 LLM-judge scale.")


def _render_deep_evaluation_section(quality: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    st.markdown("### Deep Evaluation")
    st.caption("Offline / optional LLM judge. Not part of the Ask critical path.")
    judge_calls = int(state.get("llm_judge_calls") or quality.get("llm_calls") or 0)
    if judge_calls == 0:
        st.info("LLM judge did not run for this live query.")
        st.write("**LLM Judge:** not executed")
        st.write("**Judge calls:** 0")
        return
    st.write(f"**LLM Judge:** `{quality.get('model_name') or EVALUATION.judge_model_name}`")
    st.write(f"**Judge latency:** {_format_ms(quality.get('latency_ms')) or '—'}")
    st.write(f"**Judge calls:** {judge_calls}")
    if quality.get("correctness") is not None:
        st.write(f"**Answer correctness:** {format_judge_score(quality['correctness'])}")
    if quality.get("groundedness") is not None and quality.get("evaluator_type") == "llm_judge":
        st.write(f"**Faithfulness:** {format_judge_score(quality['groundedness'])}")


def _render_request_summary(summary: Mapping[str, Any]) -> None:
    st.markdown("### Request Summary")
    items = [
        ("Total latency", summary.get("Total latency")),
        ("Retrieval strategy", summary.get("Retrieval strategy")),
        ("BM25 searches", summary.get("BM25 searches")),
        ("Dense searches", summary.get("Dense searches")),
        ("RRF fusion count", summary.get("RRF fusion count")),
        ("Evidence returned", summary.get("Evidence returned")),
        ("LLM calls", summary.get("LLM calls")),
        ("Valid/invalid citations", summary.get("Valid/invalid citations")),
    ]
    if summary.get("QE cost"):
        items.append(("QE cost", summary["QE cost"]))
    for start in range(0, len(items), 4):
        cols = st.columns(4)
        for column, (label, value) in zip(cols, items[start : start + 4]):
            column.metric(label, value)


def _render_request_trace(payload: Mapping[str, Any]) -> None:
    st.markdown("### Request Trace")
    st.caption(str(payload.get("judge_note", "")))
    for stage in payload.get("stages") or []:
        fields = stage.get("fields") or {}
        latency = fields.get("Latency", "")
        title = str(stage.get("title", "Stage"))
        if latency:
            title = f"{title}  ·  {latency}"
        with st.expander(title, expanded=False):
            if not fields:
                st.caption("Stage completed. No extra fields beyond the measured latency.")
                continue
            for label, value in fields.items():
                st.write(f"**{label}:** {value}")


def _render_evaluation() -> None:
    st.header("OFFLINE RETRIEVAL EVALUATION")
    st.write(f"**{EVALUATION.n_queries} fixed BioASQ queries**")
    st.caption(
        "Same 100 questions → BM25 vs Dense vs Hybrid vs Hybrid+QE → retrieval metrics. "
        "These scores are loaded from saved results and are not recomputed on app start."
    )

    comparison_path = ROOT / EVALUATION.comparison_path
    comparison: dict[str, Any] = {}
    if comparison_path.exists():
        comparison = load_evaluation_payload(str(comparison_path))
        rows = retrieval_table_rows(comparison)
        st.table(_display_metric_rows(rows))
        points = quality_latency_points(comparison.get("configurations", []))
        if points:
            quality_col, latency_col = st.columns(2)
            with quality_col:
                st.caption("Quality (Recall@10)")
                st.bar_chart({item["Configuration"]: item["Recall@10"] for item in points})
            with latency_col:
                st.caption("Avg latency (ms)")
                st.bar_chart({item["Configuration"]: item["Avg latency (ms)"] for item in points})
        st.header("EXPERIMENT CONCLUSION")
        for note in selected_configuration_notes(comparison.get("configurations", [])):
            st.write(f"- {note}")
        st.caption(
            "This is a measured trade-off on the fixed 100-query set, not a universal winner."
        )
    else:
        st.warning(f"Retrieval comparison file not found: `{comparison_path}`")

    _render_rerank_comparison()
    _render_retrieval_query_inspector()

    st.header("LLM ANSWER EVALUATION")
    st.caption("20 questions → generated answers → LLM judge → correctness/groundedness/context relevance")
    details_path = ROOT / EVALUATION.answer_details_path
    details: dict[str, Any] = {}
    summary_path = ROOT / EVALUATION.answer_summary_path
    if summary_path.exists():
        summary = load_evaluation_payload(str(summary_path))
        n_queries = summary.get("n_queries", EVALUATION.n_answer_queries)
        judge_model = summary.get("judge_model_name") or EVALUATION.judge_model_name
        st.write(f"**{n_queries} queries**")
        st.write(f"**LLM Judge model:** `{judge_model}`")
        score_cols = st.columns(3)
        score_cols[0].metric("Correctness", format_judge_score(summary["average_correctness"]))
        score_cols[1].metric("Groundedness", format_judge_score(summary["average_groundedness"]))
        score_cols[2].metric(
            "Context relevance",
            format_judge_score(summary["average_context_relevance"]),
        )
    else:
        st.warning(f"Answer evaluation summary not found: `{summary_path}`")
        st.write(f"**LLM Judge model:** `{EVALUATION.judge_model_name}`")

    if details_path.exists():
        details = load_evaluation_payload(str(details_path))
        counts = answer_review_summary_counts(details)
        st.subheader("Answer Review Summary")
        count_items = list(counts.items())
        count_cols = st.columns(len(count_items))
        for column, (label, value) in zip(count_cols, count_items):
            column.metric(label, value)
        _render_answer_query_inspector(details)
        st.header("Manual Review Examples")
        st.subheader("Successful")
        for example in load_manual_review_examples(details, SUCCESS_REVIEW_IDS):
            _render_review_example(example)
        st.subheader("Failure / limitations")
        for example in load_manual_review_examples(details, FAILURE_REVIEW_IDS):
            _render_review_example(example)
    else:
        st.warning(f"Answer evaluation details not found: `{details_path}`")

    st.header("What the metrics mean")
    for name, meaning in METRIC_GLOSSARY:
        st.write(f"**{name}:** {meaning}")


def _render_rerank_comparison() -> None:
    path = ROOT / EVALUATION.rerank_comparison_summary_path
    st.header("MEDCPT RERANK EXPERIMENT")
    st.caption(
        "Baseline Hybrid RRF Top-10 vs Hybrid Top-50 vs Hybrid Top-50 → MedCPT Top-10. "
        "Scores are loaded from saved results and do not overwrite the baseline JSON."
    )
    if not path.exists():
        st.warning(
            f"Rerank comparison file not found: `{path}`. "
            "Run `python -m src.evaluation.run_evaluation --rerank-comparison`."
        )
        return
    payload = load_evaluation_payload(str(path))
    st.table(_display_metric_rows(rerank_comparison_table_rows(payload)))

    st.subheader("Rerank gain")
    st.caption("Experiment C minus Experiment A. Negative values mean MedCPT scored worse.")
    st.table(_display_metric_rows(rerank_gain_table_rows(payload)))

    ceiling = payload.get("candidate_ceiling") or {}
    recovery = payload.get("rerank_recovery") or {}
    latency = payload.get("latency") or {}
    st.subheader("Candidate ceiling")
    ceiling_cols = st.columns(3)
    ceiling_cols[0].metric("Candidate Recall@50", _fmt_optional(ceiling.get("candidate_recall_at_50")))
    ceiling_cols[1].metric("Reranked Recall@10", _fmt_optional(ceiling.get("reranked_recall_at_10")))
    ceiling_cols[2].metric("Ceiling gap", _fmt_optional(ceiling.get("gap")))

    st.subheader("Rerank recovery")
    recovery_cols = st.columns(2)
    recovery_cols[0].metric(
        "Recovery@10",
        _fmt_optional(recovery.get("rerank_recovery_at_10")),
    )
    recovery_cols[1].metric(
        "Queries with relevant in pool",
        str(recovery.get("n_queries_with_relevant_in_pool", 0)),
    )

    st.subheader("Latency overhead")
    latency_cols = st.columns(4)
    latency_cols[0].metric("Baseline (ms)", _fmt_optional(latency.get("baseline_avg_ms"), digits=1))
    latency_cols[1].metric("Hybrid@50 (ms)", _fmt_optional(latency.get("hybrid_avg_ms"), digits=1))
    latency_cols[2].metric("MedCPT (ms)", _fmt_optional(latency.get("rerank_avg_ms"), digits=1))
    latency_cols[3].metric("Reranked total (ms)", _fmt_optional(latency.get("total_reranked_avg_ms"), digits=1))
    if latency.get("overhead_vs_baseline_ms") is not None:
        st.caption(
            "Overhead versus baseline: "
            f"{float(latency['overhead_vs_baseline_ms']):,.1f} ms "
            "(Hybrid@50 + MedCPT − Hybrid@10)."
        )

    st.subheader("Measured findings")
    for note in rerank_experiment_notes(payload):
        st.write(f"- {note}")
    st.caption("This experiment reports the measured result, including regressions.")

    inspection = payload.get("query_inspection") or []
    if inspection:
        st.subheader("Per-query inspection")
        st.table(_display_metric_rows(inspection))


def _fmt_optional(value: Any, *, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _render_retrieval_query_inspector() -> None:
    details_path = ROOT / EVALUATION.details_path
    hybrid_qe_path = ROOT / EVALUATION.hybrid_qe_details_path
    payloads = []
    if details_path.exists():
        payloads.append(load_evaluation_payload(str(details_path)))
    if hybrid_qe_path.exists():
        payloads.append(load_evaluation_payload(str(hybrid_qe_path)))
    if not payloads:
        return
    indexed = index_retrieval_details(*payloads)
    choices = retrieval_query_choices(indexed)
    if not choices:
        return
    st.subheader("Inspect a query")
    st.caption("Per-query retrieval scores from the saved 100-query benchmark files.")
    selected = st.selectbox("Inspect a query", choices, key="retrieval_query_inspect")
    query_id = query_id_from_choice(str(selected))
    configs = indexed.get(query_id, {})
    question = ""
    for row in configs.values():
        question = str(row.get("question") or "")
        if question:
            break
    st.write(f"**Query ID:** {query_id}")
    st.write(f"**Question:** {question}")
    rows = per_query_retrieval_rows(configs)
    if rows:
        st.table(_display_metric_rows(rows))
    for name in RETRIEVAL_CONFIG_ORDER:
        row = configs.get(name)
        if not row:
            continue
        ids = row.get("retrieved_passage_ids") or []
        expansions = row.get("expansions") or []
        with st.expander(f"{name} retrieved passages", expanded=False):
            if ids:
                st.write(", ".join(str(item) for item in ids))
            if expansions:
                st.write("**Query expansions:**")
                for index, expansion in enumerate(expansions, start=1):
                    st.write(f"{index}. {expansion}")


def _render_answer_query_inspector(details: Mapping[str, Any]) -> None:
    choices = answer_query_choices(details)
    if not choices:
        return
    st.subheader("Inspect an answer")
    selected = st.selectbox("Inspect an answer", choices, key="answer_query_inspect")
    query_id = query_id_from_choice(str(selected))
    row = answer_row_by_id(details, query_id)
    if row is None:
        return
    fields = answer_inspection_fields(row)
    st.write(f"**Query ID:** {fields.get('Query ID', query_id)}")
    st.write(f"**Question:** {fields.get('Question', '')}")
    if "Reference answer" in fields:
        st.write(f"**Reference answer:** {fields['Reference answer']}")
    if "Generated answer" in fields:
        st.write(f"**Generated answer:** {fields['Generated answer']}")
    score_cols = st.columns(3)
    if "Correctness" in fields:
        score_cols[0].metric("Correctness", fields["Correctness"])
    if "Groundedness" in fields:
        score_cols[1].metric("Groundedness", fields["Groundedness"])
    if "Context relevance" in fields:
        score_cols[2].metric("Context relevance", fields["Context relevance"])
    if "Citation validation" in fields:
        st.write(f"**Citation validation:** {fields['Citation validation']}")
    if "Judge reason" in fields:
        st.write(f"**Judge reason:** {fields['Judge reason']}")


def _render_review_example(example: Mapping[str, Any]) -> None:
    with st.expander(f"Query {example['query_id']}: {example['question']}", expanded=False):
        st.write(f"**Reference:** {example['reference_answer']}")
        st.write(f"**Generated:** {example['generated_answer']}")
        st.write(f"**Citations:** {example['citation_validation']}")
        if example.get("note"):
            st.write(f"**From the saved result:** {example['note']}")


def _render_live_pipeline_panels(result: GroundedQAResult) -> None:
    state = _current_runtime_state() or build_runtime_state(
        result,
        expand_requested=bool(st.session_state.get("last_expand")),
        rerank_requested=st.session_state.get("last_rerank"),
        total_latency_ms=float(st.session_state.get("last_latency_ms") or 0.0),
        live_quality=st.session_state.get("last_live_quality"),
    )
    _render_retrieval_summary(state)
    _render_query_understanding(result, state)
    _render_evidence_section(result, state)


def _render_pipeline_workspace() -> None:
    st.header("Retrieval Pipeline")
    st.caption("Live stage trace from the last Playground query.")
    result = st.session_state.get("last_qa_result")
    if result is None:
        st.info("Ask a question in Playground to populate the pipeline.")
        return
    _render_live_pipeline_panels(result)


def _render_upgrade_experiments() -> None:
    st.header("Retrieval Experiments")
    st.caption("Only experiments with saved result files are shown. Missing files are not invented.")
    available = existing_result_files()
    if not available:
        st.warning("No evaluation result files were found.")
        return
    if "Baseline Hybrid" in available:
        payload = load_evaluation_payload(str(available["Baseline Hybrid"]))
        st.subheader("Baseline Hybrid")
        st.table(_display_metric_rows(retrieval_table_rows(payload)))
    if "Hybrid + Query Expansion" in available:
        payload = load_evaluation_payload(str(available["Hybrid + Query Expansion"]))
        st.subheader("Query Expansion")
        st.table(_display_metric_rows(retrieval_table_rows(payload)))
    if "MedCPT comparison" in available:
        st.subheader("MedCPT")
        _render_rerank_comparison()
    upgrade_path = ROOT / EVALUATION.upgrade_summary_path
    if upgrade_path.exists():
        payload = load_evaluation_payload(str(upgrade_path))
        st.subheader("Upgrade comparison (A–J)")
        rows = upgrade_experiment_table_rows(payload)
        if rows:
            st.table(_display_metric_rows(rows))
        skipped = payload.get("skipped_configurations") or []
        if skipped:
            st.caption("Skipped (not run): " + ", ".join(skipped))
    else:
        st.info(
            "Upgrade comparison JSON is not present yet. "
            "Run `python -m src.evaluation.run_evaluation --upgrade`."
        )

    embedding_path = ROOT / EVALUATION.embedding_comparison_summary_path
    if embedding_path.exists():
        st.subheader("Embedding model comparison")
        payload = load_evaluation_payload(str(embedding_path))
        st.table(_display_metric_rows(embedding_comparison_table_rows(payload)))
        if payload.get("biomedical") and not payload["biomedical"].get("available"):
            st.caption("Biomedical encoder was not available; MiniLM-only numbers are shown.")
    pool_path = ROOT / EVALUATION.candidate_pool_summary_path
    if pool_path.exists():
        st.subheader("Candidate pool curve")
        payload = load_evaluation_payload(str(pool_path))
        st.table(_display_metric_rows(candidate_pool_table_rows(payload)))
    medcpt_pool_path = ROOT / EVALUATION.medcpt_pool_summary_path
    if medcpt_pool_path.exists():
        st.subheader("MedCPT pool size")
        payload = load_evaluation_payload(str(medcpt_pool_path))
        st.caption(str(payload.get("scoring_note") or ""))
        st.table(_display_metric_rows(medcpt_pool_table_rows(payload)))
    expansion_path = ROOT / EVALUATION.expansion_study_summary_path
    if expansion_path.exists():
        st.subheader("Query expansion study")
        payload = load_evaluation_payload(str(expansion_path))
        st.write(
            f"Original Recall@10={payload.get('original_recall_at_10')} · "
            f"Equal-weight={payload.get('equal_recall_at_10')} · "
            f"Priority={payload.get('priority_recall_at_10')}"
        )
        st.write(
            f"New relevant from expansion: {payload.get('n_new_relevant')} · "
            f"Displaced relevant: {payload.get('n_displaced_relevant')}"
        )
    decomp_path = ROOT / EVALUATION.decomposition_study_summary_path
    if decomp_path.exists():
        st.subheader("Query decomposition study")
        payload = load_evaluation_payload(str(decomp_path))
        st.write(f"Decomposed queries: {payload.get('n_decomposed')} / {payload.get('n_queries')}")
        st.json({key: value for key, value in payload.items() if key not in {"notes"}})
    sparse_path = ROOT / EVALUATION.sparse_study_summary_path
    if sparse_path.exists():
        payload = load_evaluation_payload(str(sparse_path))
        st.subheader("Learned sparse (EXPERIMENTAL)")
        st.caption("Production remains BM25 + MiniLM. Sparse is not ENABLED.")
        if payload.get("skipped"):
            st.info(payload.get("reason") or "Sparse model was not loaded.")
        else:
            rec = payload.get("dual_miss_recovery") or {}
            st.write(
                f"Sparse Recovery@100: {rec.get('hit_at_100')} / {rec.get('denominator')} "
                f"(rate {rec.get('sparse_recovery_at_100')}). "
                f"Sparse-only among dual misses: {payload.get('sparse_only_recovery_among_dual_misses')}."
            )
            fusion_rows = sparse_fusion_comparison_rows(payload)
            if fusion_rows:
                st.write("Current baseline BM25 + MiniLM versus BM25 + MiniLM + Sparse")
                st.table(_display_metric_rows(fusion_rows))
            metric_rows = sparse_study_metric_rows(payload)
            if metric_rows:
                st.table(_display_metric_rows(metric_rows))
            if payload.get("medcpt") and not payload["medcpt"].get("skipped"):
                st.write("MedCPT on fused B (measured)")
                st.json(payload["medcpt"])
            elif payload.get("medcpt"):
                st.caption("MedCPT on sparse fusion was not run: " + str(payload["medcpt"].get("reason") or ""))


def _render_failure_analysis() -> None:
    st.header("Retrieval Failure Analysis")
    st.caption(
        "Objective categories are BM25/Dense/RRF ranks at K=100. "
        "Diagnostic signals and hypotheses are separate and are not treated as proven causes."
    )
    objective_path = ROOT / EVALUATION.objective_failure_summary_path
    if objective_path.exists():
        payload = load_evaluation_payload(str(objective_path))
        st.subheader("Objective retrieval failure")
        st.caption(
            f"{payload.get('n_hybrid50_failures')} gold IDs absent from Hybrid@"
            f"{payload.get('hybrid_k')} out of {payload.get('n_gold_ids')} gold IDs "
            f"({payload.get('n_queries')} queries). Rank None means absent from top "
            f"{payload.get('probe_k')}."
        )
        st.table(_display_metric_rows(objective_failure_table_rows(payload)))
        signals = payload.get("diagnostic_signal_summary") or {}
        st.subheader("Diagnostic signals")
        st.caption("Descriptors of missed gold passages. Not failure labels.")
        if signals:
            st.json(signals)
        bio = payload.get("biomedical_dense_recovery") or {}
        st.subheader("Biomedical dense recovery (observational)")
        if bio.get("skipped"):
            st.info(bio.get("reason") or "Biomedical encoder not evaluated.")
        else:
            st.write(bio.get("denominator_definition") or "")
            st.write(
                f"MiniLM Dense@100 misses among Hybrid@50 misses: {bio.get('minilm_dense_misses')}. "
                f"Newly retrieved by PubMedBERT@100: {bio.get('newly_recovered_by_pubmedbert')}. "
                f"Rate: {bio.get('biomedical_dense_recovery_rate')}."
            )
        st.subheader("Lexical / expansion / decomposition recovery")
        st.json(
            {
                "lexical": payload.get("lexical_recovery"),
                "expansion": payload.get("expansion_recovery"),
                "decomposition": payload.get("decomposition_recovery"),
            }
        )
        recovery_path = ROOT / EVALUATION.recovery_matrix_summary_path
        if recovery_path.exists():
            recovery = load_evaluation_payload(str(recovery_path))
            st.subheader("Recovery matrix (Hybrid@50 misses, hit@100)")
            st.table(_display_metric_rows(recovery_matrix_table_rows(recovery)))
        sparse_details_path = ROOT / EVALUATION.sparse_study_details_path
        sparse_summary_path = ROOT / EVALUATION.sparse_study_summary_path
        if sparse_summary_path.exists():
            sparse_summary = load_evaluation_payload(str(sparse_summary_path))
            st.subheader("Sparse recovery matrix (Hybrid@50 misses, EXPERIMENTAL)")
            if sparse_summary.get("skipped"):
                st.info(sparse_summary.get("reason") or "Sparse study skipped.")
            else:
                st.table(_display_metric_rows(recovery_matrix_table_rows(sparse_summary)))
                st.caption(
                    f"Sparse-only recovery among the dual misses: "
                    f"{sparse_summary.get('sparse_only_recovery_among_dual_misses')}"
                )
                if sparse_details_path.exists():
                    sparse_details = load_evaluation_payload(str(sparse_details_path))
                    preview = sparse_recovery_matrix_preview_rows(sparse_details)
                    if preview:
                        st.caption("First 20 Hybrid@50 miss rows. Full table is in sparse_study_details.json.")
                        st.table(preview)
        details_path = ROOT / EVALUATION.objective_failure_details_path
        if details_path.exists():
            details = load_evaluation_payload(str(details_path))
            miss_rows = details.get("failure_rows") or []
            if miss_rows:
                choices = [
                    f"{item['query_id']}:{item['gold_passage_id']} {item['question'][:70]}"
                    for item in miss_rows
                ]
                selected = st.selectbox("Inspect a Hybrid@50 miss", choices, key="objective_fail_inspect")
                item = miss_rows[choices.index(selected)]
                st.write(f"**Objective category:** `{item.get('objective_category')}`")
                st.write(
                    f"BM25@100 rank={item.get('bm25_rank_at_100')} · "
                    f"MiniLM Dense@100 rank={item.get('dense_rank_at_100')} · "
                    f"RRF@100 rank={item.get('rrf_rank_at_100')} · "
                    f"PubMedBERT Dense@100 rank={item.get('pubmedbert_dense_rank_at_100')}"
                )
                st.write("**Diagnostic signals**")
                st.json(item.get("diagnostic_signals") or {})
                st.write("**Hypotheses (not proven causes)**")
                hyps = item.get("hypotheses") or []
                if hyps:
                    st.table(hyps)
                else:
                    st.caption("No hypotheses recorded for this ID.")
                st.write(f"**Earliest recovery @100:** `{item.get('earliest_recovery')}`")

    missing_path = ROOT / EVALUATION.missing_evidence_summary_path
    if missing_path.exists():
        with st.expander("Previous heuristic labels (not causal evidence)", expanded=False):
            st.caption(
                "These labels mixed rank facts with query-type heuristics "
                "(for example lexical mismatch from token overlap). "
                "Do not treat them as demonstrated causes."
            )
            payload = load_evaluation_payload(str(missing_path))
            st.table(_display_metric_rows(missing_evidence_table_rows(payload)))
            passage_rows = missing_evidence_passage_table_rows(payload)
            if passage_rows:
                st.table(_display_metric_rows(passage_rows))

    path = ROOT / EVALUATION.upgrade_details_path
    if not objective_path.exists() and not missing_path.exists() and not path.exists():
        st.info("No failure-analysis JSON yet. Run `python -m src.evaluation.objective_failure`.")
        return
    if not path.exists():
        return
    payload = load_evaluation_payload(str(path))
    summary = payload.get("failure_summary") or {}
    counts = summary.get("counts") or {}
    if counts:
        with st.expander("Upgrade-run heuristic counts (not causal)", expanded=False):
            st.table([{"category": key, "count": value} for key, value in counts.items()])
            rows = failure_analysis_table_rows(payload)
            if rows:
                st.table(_display_metric_rows(rows))


def _render_architecture() -> None:
    st.header("System Architecture")
    st.markdown(
        """
Question → classification → terminology → expansion/decomposition →
BM25 + Dense → RRF fusion → candidate filtering → MedCPT → final evidence →
grounded LLM → citation validation.
        """
    )
    st.subheader("Architecture Decisions")
    for item in architecture_decisions():
        with st.expander(item["component"], expanded=False):
            st.write(f"**Decision:** {item['decision']}")
            st.write(f"**Why chosen:** {item['why_chosen']}")
            st.write(f"**Alternative considered:** {item['alternative_considered']}")
            st.write(f"**Measured result:** {item['measured_result']}")
            st.write(f"**Tradeoff:** {item['tradeoff']}")
    st.markdown(architecture_document_text())


def _render_model_prompt_inspector() -> None:
    st.header("Model & Prompt Inspector")
    st.caption("Read-only. Templates are the ones the code actually sends. Deterministic stages have no LLM prompt.")
    st.subheader("Feature status")
    st.table(feature_status_table_rows())

    st.subheader("Dense Retrieval")
    st.write("**Model:** `sentence-transformers/all-MiniLM-L6-v2`")
    st.write("**Dimension:** `384`")
    st.write("**Role:** Dense retrieval baseline")
    st.write("**Current/default:** production default (biomedical encoder off unless flagged)")
    st.write(
        "**Why selected:** Lightweight local baseline (~80MB, 384-d) that runs on CPU without an API key. "
        "It is a comparison baseline, not a claimed biomedical SOTA encoder."
    )
    st.write(
        "**Alternatives evaluated:** `pritamdeka/S-PubMedBert-MS-MARCO`, then "
        "`NeuML/pubmedbert-base-embeddings` / `sentence-transformers/allenai-specter` if needed."
    )
    embedding_path = ROOT / EVALUATION.embedding_comparison_summary_path
    if embedding_path.exists():
        payload = load_evaluation_payload(str(embedding_path))
        st.table(_display_metric_rows(embedding_comparison_table_rows(payload)))
    else:
        st.info("Embedding comparison JSON not present (NOT EVALUATED).")

    st.subheader("MedCPT")
    st.write(f"**Model:** `{RERANKER.model_name}`")
    st.write("**Source:** Hugging Face / local PyTorch")
    st.write("**Task:** Cross-encoder rerank after Hybrid RRF")
    st.write(f"**Candidate pool:** `{RERANKER.candidate_pool_size}`")
    st.write(f"**Final top-K:** `{RERANKER.final_top_k}`")
    st.write(f"**Device:** `{RERANKER.device}`")
    st.write("**Config default:** disabled (`RERANKER.enabled=False`). Playground can enable MedCPT for the current request.")
    state = st.session_state.get("last_runtime_state") or {}
    if state:
        st.write(
            "**Current request:** "
            + ("enabled/active" if state.get("medcpt_executed") else "disabled/not executed")
        )
    st.write("**Measured:** Recall@10=0.5082, MRR@10=0.8587, nDCG@10=0.6706, rerank ≈ 3131 ms on CPU")

    st.subheader("Learned sparse retrieval (EXPERIMENTAL)")
    st.caption("Not production. Flag default OFF. Do not treat as ENABLED.")
    sparse_path = ROOT / EVALUATION.sparse_study_summary_path
    payload = load_evaluation_payload(str(sparse_path)) if sparse_path.exists() else {}
    st.table(sparse_inspector_rows(payload))
    if not sparse_path.exists():
        st.info("Sparse study JSON not present (NOT EVALUATED).")
    elif payload.get("skipped"):
        st.info(payload.get("reason") or "Sparse model was not loaded.")

    st.subheader("Runtime configuration")
    st.json(runtime_configuration())

    st.subheader("Prompts")
    for item in llm_prompt_catalog():
        title = item["name"]
        with st.expander(title, expanded=False):
            st.write(f"**Source:** `{item['source_file']}`")
            st.write(f"**Model:** `{item['model']}`")
            st.write(f"**Deterministic:** {item['deterministic']}")
            if item["deterministic"] or not item["template"]:
                st.info("Deterministic — no LLM prompt")
            else:
                st.write(f"**Temperature:** {item['temperature']}")
                st.write(f"**Max tokens:** {item['max_tokens']}")
                st.write("**Input variables:** " + ", ".join(item.get("input_variables") or []))
                st.code(item["template"])
            if item.get("notes"):
                st.caption(item["notes"])


def _render_research() -> None:
    st.header("Research / Documentation")
    for name, text in research_document_sections():
        with st.expander(name, expanded=name == "README.md"):
            st.markdown(text)


def _load_optional_json(path: str) -> dict[str, Any]:
    file_path = ROOT / path
    if not file_path.exists():
        return {}
    return load_json_result(file_path)


def _render_live_retrieval_trace() -> None:
    st.header("Retrieval Trace")
    result = _require_last_result()
    if result is None:
        return
    state = _current_runtime_state() or {}
    _render_retrieval_summary(state)
    _render_latency_waterfall(state)
    payload = build_request_trace(
        result,
        expand=bool(st.session_state.get("last_expand")),
        total_latency_ms=float(st.session_state.get("last_latency_ms") or 0.0),
        live_quality=st.session_state.get("last_live_quality"),
    )
    with st.expander("Stage fields", expanded=False):
        _render_request_trace(payload)


def _render_metrics_page() -> None:
    st.header("Retrieval diagnostics")
    st.caption("RAW benchmark numbers are never replaced by the valid-gold diagnostic view.")
    diag = _load_optional_json(EVALUATION.retrieval_diagnostics_path)
    catalog = load_experiment_catalog()
    baseline = next((item for item in catalog if item.get("id") == "baseline"), {})
    baseline_metrics = dict(baseline.get("metrics") or {})
    raw = dict((diag or {}).get("raw_original_benchmark") or {})
    valid = dict((diag or {}).get("valid_gold_diagnostic_view") or {})
    summary = dict((diag or {}).get("summary") or {})

    st.subheader("RAW benchmark")
    st.write(raw.get("note") or "Includes every labelled gold ID.")
    raw_cols = st.columns(4)
    raw_cols[0].metric("Recall@10", _fmt_optional(raw.get("recall_at_10") or summary.get("raw_recall_at_10")))
    raw_cols[1].metric("Candidate Recall@50", _fmt_optional(raw.get("candidate_recall_at_50") or summary.get("candidate_recall_at_50")))
    raw_cols[2].metric("MRR@10", _fmt_optional(baseline_metrics.get("mrr_at_10")))
    raw_cols[3].metric("nDCG@10", _fmt_optional(baseline_metrics.get("ndcg_at_10")))
    st.write(f"Pool hit rate: `{baseline_metrics.get('pool_hit_rate')}`")
    st.write(f"Rerank recovery@10: `{baseline_metrics.get('rerank_recovery_at_10') or summary.get('rerank_recovery_at_10')}`")

    st.subheader("VALID-GOLD diagnostic view")
    st.write(valid.get("note") or "Does not replace the raw benchmark.")
    vg = st.columns(3)
    vg[0].metric("Valid-gold Recall@10", _fmt_optional(valid.get("recall_at_10") or summary.get("valid_gold_recall_at_10")))
    vg[1].metric("Valid-gold Recall@50", _fmt_optional(valid.get("candidate_recall_at_50") or summary.get("valid_gold_recall_at_50")))
    vg[2].metric("Invalid gold IDs", valid.get("invalid_gold_ids") or summary.get("invalid_gold_ids"))

    st.subheader("Recall by cutoff")
    st.table(
        [
            {"Cutoff": "Recall@1", "MedCPT": summary.get("rerank_recall_at_1")},
            {"Cutoff": "Recall@3", "MedCPT": summary.get("rerank_recall_at_3")},
            {"Cutoff": "Recall@5", "MedCPT": summary.get("rerank_recall_at_5"), "BM25": summary.get("bm25_recall_at_5"), "Dense": summary.get("dense_recall_at_5")},
            {"Cutoff": "Recall@10", "MedCPT": summary.get("rerank_recall_at_10"), "BM25": summary.get("bm25_recall_at_10"), "Dense": summary.get("dense_recall_at_10")},
            {"Cutoff": "Recall@50", "RRF": summary.get("rrf_recall_at_50"), "BM25": summary.get("bm25_recall_at_50"), "Dense": summary.get("dense_recall_at_50")},
        ]
    )
    st.write(f"Candidate Recall@50: `{summary.get('candidate_recall_at_50')}`")
    st.write(f"Rerank Recall@10: `{summary.get('rerank_recall_at_10')}`")
    st.write(f"Rerank recovery: `{summary.get('rerank_recovery_at_10')}`")
    st.write(f"Pool hit rate: `{baseline_metrics.get('pool_hit_rate')}`")

    signals = dict(summary.get("signal_recovery") or {})
    counts = dict(signals.get("counts") or {})
    st.subheader("BM25 / Dense hit split")
    st.write(f"BM25-only hits: {counts.get('BM25 only', signals.get('BM25_only'))}")
    st.write(f"Dense-only hits: {counts.get('Dense only', signals.get('Dense_only'))}")
    st.write(f"Both: {counts.get('Both BM25 and Dense', signals.get('Both'))}")
    st.write(f"Neither: {counts.get('Neither BM25 nor Dense', signals.get('Neither'))}")

    rrf = dict(summary.get("rrf_loss_counts") or {})
    st.subheader("RRF diagnostics")
    st.write(f"BM25-found but dropped by RRF: {rrf.get('found by BM25 but pushed out by RRF')}")
    st.write(f"Dense-found but dropped by RRF: {rrf.get('found by Dense but pushed out by RRF')}")
    st.write(f"Both found and retained: {rrf.get('found by both and retained')}")

    st.subheader("Reranker diagnostics")
    st.write(f"Relevant candidates not in final Top-10: {summary.get('gold_present_in_top50_but_lost_by_reranking')}")
    st.write(f"Reranker recovery@10: {summary.get('rerank_recovery_at_10')}")
    if not diag:
        st.warning("Diagnostic JSON is not present. Run observability diagnostics to populate this page.")


def _render_query_inspector_page() -> None:
    st.header("Query Inspector")
    _render_retrieval_query_inspector()
    details_path = ROOT / EVALUATION.answer_details_path
    if details_path.exists():
        details = load_evaluation_payload(str(details_path))
        _render_answer_query_inspector(details)


def _render_experiment_compare_page() -> None:
    st.header("Compare")
    st.caption("One variable changed. Stored experiment files are not rewritten.")
    rows = experiment_lab_rows()
    if not rows:
        st.info("No experiment catalog entries.")
        return
    names = [row["Experiment"] for row in rows]
    left, right = st.columns(2)
    with left:
        a = st.selectbox("Baseline", names, index=0)
    with right:
        b = st.selectbox("Candidate", names, index=min(1, len(names) - 1))
    by_name = {row["Experiment"]: row for row in rows}
    first, second = by_name[a], by_name[b]
    st.write(f"**Changed:** {second.get('Changed variable')}")
    st.table(
        [
            {
                "Side": "Baseline",
                "Run": first["Experiment"],
                "Recall@10": first.get("Recall@10"),
                "Recall@50": first.get("Recall@50"),
                "MRR": first.get("MRR"),
                "nDCG": first.get("nDCG"),
                "Latency": first.get("Latency"),
            },
            {
                "Side": "Candidate",
                "Run": second["Experiment"],
                "Recall@10": second.get("Recall@10"),
                "Recall@50": second.get("Recall@50"),
                "MRR": second.get("MRR"),
                "nDCG": second.get("nDCG"),
                "Latency": second.get("Latency"),
            },
        ]
    )
    st.write(f"**Result:** {second.get('Result')}")
    st.write(f"**Conclusion:** {second.get('Conclusion')}")


def _render_model_registry_page() -> None:
    st.header("Models")
    state = _current_runtime_state() or {}
    registry = model_registry(
        medcpt_active=bool(state.get("medcpt_executed")),
        expansion_active=bool(state.get("query_expansion_executed")),
    )
    st.subheader("Production")
    for item in registry["production"]:
        activity = "active" if item["active"] else "not active this request"
        st.markdown(f"**{item['name']}** · {item['role']} · {item['lane']} · {activity}")
        st.caption(item["notes"])
    st.subheader("Experimental")
    st.caption("Experimental models are not production dependencies.")
    for item in registry["experimental"]:
        activity = "active" if item["active"] else "Not active"
        st.markdown(f"**{item['name']}** · {item['role']} · Experimental · {activity}")
        st.caption(item["notes"])


def _render_prompt_inspector_page() -> None:
    st.header("Prompt Inspector")
    tabs = prompt_inspector_tabs()
    names = [item["tab"] for item in tabs]
    selected = st.radio("Prompt", names, horizontal=True)
    record = next(item for item in tabs if item["tab"] == selected)
    st.write(f"**Stage:** {record['stage']}")
    st.write(f"**Model:** `{record['model']}`")
    st.write("**System instructions:** none (single user message)")
    st.markdown("**User template**")
    st.code(record["user_template"])
    st.markdown("**Variables**")
    variables = dict(record.get("variables") or {})
    result = st.session_state.get("last_qa_result")
    if selected == "Answer Generation" and result is not None:
        variables = runtime_generation_variables(result)
        variables["query"] = result.question
    for key, value in variables.items():
        st.write(f"**{key}:** `{value}`")
    if selected == "Answer Generation" and result is not None and result.prompt:
        with st.expander("Bound prompt from last request", expanded=False):
            st.code(result.prompt)
    if record.get("notes"):
        st.caption(record["notes"])
    st.divider()
    for note in non_llm_prompt_notes():
        st.write(f"**{note['component']}:** {note['note']}")


def _render_gold_quality_page() -> None:
    st.header("Gold Quality")
    gold = _load_optional_json(EVALUATION.gold_quality_path)
    if not gold:
        st.warning("Gold-quality JSON is not present.")
        return
    g1, g2, g3 = st.columns(3)
    g1.metric("Total gold IDs", gold.get("total_gold_ids"))
    g2.metric("Valid gold IDs", gold.get("valid_gold_ids"))
    g3.metric("Invalid gold IDs", gold.get("invalid_gold_ids"))
    st.write(
        f"Queries with ≥1 invalid gold: **{gold.get('queries_with_at_least_one_invalid_gold')}**. "
        f"Queries where all gold IDs are invalid: **{gold.get('queries_where_all_gold_ids_are_invalid')}**."
    )
    st.caption(gold.get("note"))
    st.json({key: value for key, value in gold.items() if key != "per_query"})


def _render_evaluation_dashboard() -> None:
    _render_evaluation()


def _render_experiments_dashboard() -> None:
    st.header("Experiment laboratory")
    st.caption("One variable changed. Historical files are never overwritten from this view.")
    rows = experiment_lab_rows()
    display = [
        {
            "Experiment": row["Experiment"],
            "Date": row["Date"],
            "Changed variable": row["Changed variable"],
            "Baseline": row["Baseline"],
            "Candidate": row["Candidate"],
            "Recall@5": row["Recall@5"],
            "Recall@10": row["Recall@10"],
            "Recall@50": row["Recall@50"],
            "MRR": row["MRR"],
            "nDCG": row["nDCG"],
            "Latency": row["Latency"],
            "Conclusion": row["Conclusion"],
        }
        for row in rows
    ]
    st.dataframe(display, hide_index=True, use_container_width=True)
    for row in rows:
        with st.expander(row["Experiment"], expanded=row.get("Status") == "production"):
            st.write(f"**Date:** {row['Date']}")
            st.write(f"**Baseline:** {row['Baseline']}")
            st.write(f"**Experiment:** {row['Candidate']}")
            st.write(f"**Changed:** {row['Changed variable']}")
            st.write(f"**Result:** {row['Result']}")
            st.write(f"**Conclusion:** {row['Conclusion']}")
            st.write(f"**Status:** {row['Status']}")
            st.caption(f"Result file: `{row['result_file']}`")
    with st.expander("Additional upgrade studies", expanded=False):
        _render_upgrade_experiments()


def _render_data_quality_dashboard() -> None:
    st.header("Data Quality")
    quality = _load_optional_json(EVALUATION.data_quality_path)
    if not quality:
        st.warning("Run `python -m src.evaluation.run_observability` to generate data-quality JSON.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total documents", quality.get("total_documents"))
    c2.metric("Valid documents", quality.get("valid_documents"))
    c3.metric("NaN text", quality.get("nan_text"))
    c4.metric("Rejected (classified, not deleted)", quality.get("rejected_documents"))
    st.info(
        f"Literal 'nan' passages: {quality.get('nan_text')}. "
        "They remain in the production index and are reported here; they are not silently deleted."
    )
    st.json(
        {
            "empty_text": quality.get("empty_text"),
            "duplicate_ids": quality.get("duplicate_ids"),
            "duplicate_texts": quality.get("duplicate_texts"),
            "missing_ids": quality.get("missing_ids"),
            "preprocessing": quality.get("preprocessing"),
            "statistics": quality.get("statistics"),
        }
    )


def _render_architecture_dashboard() -> None:
    st.header("Architecture")
    st.markdown(
        """
```mermaid
flowchart TD
  question[Query] --> understanding[Query Understanding]
  understanding --> bm25[BM25]
  understanding --> dense[Dense]
  bm25 --> rrf[RRF Fusion]
  dense --> rrf
  rrf --> top50[Top-50 Candidates]
  top50 --> medcpt[MedCPT Reranker]
  medcpt --> evidence[Final Evidence]
  evidence --> answer[Grounded Answer]
  answer --> cite[Citation Validation]
  cite --> evaluation[Evaluation]
```
        """
    )
    st.caption("Production retrieval is BM25 + MiniLM + RRF k=60 → Top 50 → MedCPT → Top 10.")
    _render_architecture()
    _render_research()


def _render_observability_dashboard() -> None:
    st.header("Observability")
    state = _current_runtime_state()
    if state:
        _render_latency_waterfall(state)
        if state.get("inconsistencies"):
            st.error("Pipeline state inconsistencies: " + "; ".join(state["inconsistencies"]))
        else:
            st.caption("Current-request stage counts and MedCPT/QE status share one runtime state object.")
    else:
        st.info("Ask a question to populate current-request telemetry.")
    health = _load_optional_json(EVALUATION.index_health_path)
    if health:
        st.subheader("Index health")
        st.write(f"Aligned: **{health.get('aligned')}** · embeddings `{health.get('embedding_count')}` × `{health.get('embedding_dimension')}`")
    traces_dir = ROOT / EVALUATION.query_trace_dir
    if traces_dir.exists():
        files = sorted(traces_dir.glob("query_*.json"))
        st.subheader("Stored query traces")
        for path in files:
            with st.expander(path.name):
                st.json(json.loads(path.read_text(encoding="utf-8")))
    _render_pipeline_workspace()


if __name__ == "__main__":
    main()

