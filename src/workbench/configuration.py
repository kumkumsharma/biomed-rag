"""Experiment configuration, provenance, and lineage. No inferred timestamps."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.workbench.dataset import DATASET_ID, DATASET_VERSION
from src.workbench.experiments import experiment_by_id
from src.workbench.production import PRODUCTION

NOT_RECORDED = "Not recorded"

RERANKER_NAMES = {
    "medcpt-ce": PRODUCTION.reranker_model_name,
    "biomedbert-reranker": "NeuML/biomedbert-base-reranker",
}
DENSE_NAMES = {
    "minilm": PRODUCTION.dense_model_name,
    "pubmedbert": "S-PubMedBert-MS-MARCO",
    "medcpt-query": "ncbi/MedCPT-Query-Encoder",
}

CONFIG_FIELDS = (
    ("retriever", "Retriever", False),
    ("dense_model_id", "Dense model", False),
    ("fusion", "Fusion", False),
    ("rrf_k", "RRF k", False),
    ("candidate_pool", "Candidate pool", False),
    ("reranker_id", "Reranker", False),
    ("query_expansion", "Query expansion", False),
    ("generation_model_id", "Generation model", False),
    ("prompt_version", "Prompt version", True),
    ("dataset_id", "Dataset", False),
    ("dataset_version", "Dataset version", False),
)

PROVENANCE_FIELDS = (
    ("experiment_id", "Experiment ID"),
    ("dataset", "Dataset"),
    ("dataset_version", "Dataset version"),
    ("query_count", "Query count"),
    ("source_artifact", "Source artifact"),
    ("retriever", "Retriever"),
    ("dense_model", "Dense model"),
    ("fusion", "Fusion"),
    ("rrf_k", "RRF k"),
    ("candidate_pool", "Candidate pool"),
    ("reranker", "Reranker"),
    ("query_expansion", "Query expansion"),
    ("generation_model", "Generation model"),
    ("prompt_version", "Prompt version"),
    ("code_version", "Code version"),
    ("model_revision", "Model revision"),
    ("timestamp", "Timestamp"),
)


def _display_reranker(reranker_id: Any) -> Any:
    if reranker_id is None:
        return None
    return RERANKER_NAMES.get(str(reranker_id), reranker_id)


def _display_dense(dense_id: Any) -> Any:
    if dense_id is None:
        return None
    return DENSE_NAMES.get(str(dense_id), dense_id)


def experiment_configuration(experiment_id: str) -> dict[str, Any]:
    """Known configuration for a stored experiment. Missing fields stay Not recorded."""

    specs: dict[str, dict[str, Any]] = {
        "bm25": {
            "retriever": "BM25",
            "dense_model_id": None,
            "fusion": None,
            "rrf_k": None,
            "candidate_pool": 10,
            "reranker_id": None,
            "query_expansion": False,
            "generation_model_id": None,
            "prompt_ids": [],
            "prompt_version": None,
            "model_ids": ["bm25"],
        },
        "dense-minilm": {
            "retriever": "Dense",
            "dense_model_id": "minilm",
            "fusion": None,
            "rrf_k": None,
            "candidate_pool": 10,
            "reranker_id": None,
            "query_expansion": False,
            "generation_model_id": None,
            "prompt_ids": [],
            "prompt_version": None,
            "model_ids": ["minilm"],
        },
        "hybrid": {
            "retriever": "Hybrid",
            "dense_model_id": "minilm",
            "fusion": "rrf",
            "rrf_k": PRODUCTION.rrf_k,
            "candidate_pool": 10,
            "reranker_id": None,
            "query_expansion": False,
            "generation_model_id": None,
            "prompt_ids": [],
            "prompt_version": None,
            "model_ids": ["bm25", "minilm"],
        },
        "hybrid-qe": {
            "retriever": "Hybrid",
            "dense_model_id": "minilm",
            "fusion": "rrf",
            "rrf_k": PRODUCTION.rrf_k,
            "candidate_pool": 10,
            "reranker_id": None,
            "query_expansion": True,
            "generation_model_id": "gpt-4o-mini",
            "prompt_ids": ["query_expansion"],
            "prompt_version": None,
            "model_ids": ["bm25", "minilm", "gpt-4o-mini"],
        },
        "hybrid-medcpt": {
            "retriever": "Hybrid",
            "dense_model_id": "minilm",
            "fusion": "rrf",
            "rrf_k": PRODUCTION.rrf_k,
            "candidate_pool": 50,
            "reranker_id": "medcpt-ce",
            "query_expansion": False,
            "generation_model_id": None,
            "prompt_ids": [],
            "prompt_version": None,
            "model_ids": ["bm25", "minilm", "medcpt-ce"],
        },
        "hybrid-pool100": {
            "retriever": "Hybrid",
            "dense_model_id": "minilm",
            "fusion": "rrf",
            "rrf_k": PRODUCTION.rrf_k,
            "candidate_pool": 100,
            "reranker_id": "medcpt-ce",
            "query_expansion": False,
            "generation_model_id": None,
            "prompt_ids": [],
            "prompt_version": None,
            "model_ids": ["bm25", "minilm", "medcpt-ce"],
        },
        "hybrid-medcpt-biencoder": {
            "retriever": "Hybrid",
            "dense_model_id": "medcpt-query",
            "fusion": "rrf",
            "rrf_k": PRODUCTION.rrf_k,
            "candidate_pool": 50,
            "reranker_id": "medcpt-ce",
            "query_expansion": False,
            "generation_model_id": None,
            "prompt_ids": [],
            "prompt_version": None,
            "model_ids": ["bm25", "medcpt-query", "medcpt-article", "medcpt-ce"],
        },
        "hybrid-qe-medcpt": {
            "retriever": "Hybrid",
            "dense_model_id": "minilm",
            "fusion": "rrf",
            "rrf_k": PRODUCTION.rrf_k,
            "candidate_pool": 50,
            "reranker_id": "medcpt-ce",
            "query_expansion": True,
            "generation_model_id": "gpt-4o-mini",
            "prompt_ids": ["query_expansion"],
            "prompt_version": None,
            "model_ids": ["bm25", "minilm", "medcpt-ce", "gpt-4o-mini"],
        },
        "hybrid-pubmedbert": {
            "retriever": "Hybrid",
            "dense_model_id": "pubmedbert",
            "fusion": "rrf",
            "rrf_k": PRODUCTION.rrf_k,
            "candidate_pool": 10,
            "reranker_id": None,
            "query_expansion": False,
            "generation_model_id": None,
            "prompt_ids": [],
            "prompt_version": None,
            "model_ids": ["bm25", "pubmedbert"],
        },
        "hybrid-biomedbert-reranker": {
            "retriever": "Hybrid",
            "dense_model_id": "minilm",
            "fusion": "rrf",
            "rrf_k": PRODUCTION.rrf_k,
            "candidate_pool": 50,
            "reranker_id": "biomedbert-reranker",
            "query_expansion": False,
            "generation_model_id": None,
            "prompt_ids": [],
            "prompt_version": None,
            "model_ids": ["bm25", "minilm", "biomedbert-reranker"],
        },
        "hybrid-splade-medcpt": {
            "retriever": "Hybrid+SPLADE",
            "dense_model_id": "minilm",
            "fusion": "rrf+splade",
            "rrf_k": PRODUCTION.rrf_k,
            "candidate_pool": 50,
            "reranker_id": "medcpt-ce",
            "query_expansion": False,
            "generation_model_id": None,
            "prompt_ids": [],
            "prompt_version": None,
            "model_ids": ["bm25", "minilm", "splade", "medcpt-ce"],
        },
    }
    spec = dict(specs.get(experiment_id) or {})
    spec["experiment_id"] = experiment_id
    spec["dataset_id"] = DATASET_ID
    spec["dataset_version"] = DATASET_VERSION
    spec.setdefault("prompt_version", None)
    spec["reranker_name"] = _display_reranker(spec.get("reranker_id"))
    spec["dense_model_name"] = _display_dense(spec.get("dense_model_id"))
    spec["model_revision"] = None
    return spec


def _missing(value: Any) -> bool:
    return value is None or value == [] or value == ""


def field_status(a: Any, b: Any, *, unrecorded_if_both_missing: bool = False) -> str:
    if unrecorded_if_both_missing and _missing(a) and _missing(b):
        return "NOT RECORDED"
    if a == b:
        return "UNCHANGED"
    return "CHANGED"


def configuration_diff(baseline_id: str, candidate_id: str) -> list[dict[str, Any]]:
    left = experiment_configuration(baseline_id)
    right = experiment_configuration(candidate_id)
    rows = []
    for key, label, unrecorded in CONFIG_FIELDS:
        a, b = left.get(key), right.get(key)
        if key == "reranker_id":
            a_display = _display_reranker(a)
            b_display = _display_reranker(b)
        elif key == "dense_model_id":
            a_display = _display_dense(a)
            b_display = _display_dense(b)
        else:
            a_display, b_display = a, b
        status = field_status(a, b, unrecorded_if_both_missing=unrecorded)
        rows.append(
            {
                "component": label,
                "field": key,
                "baseline": a if not (unrecorded and _missing(a)) else None,
                "candidate": b if not (unrecorded and _missing(b)) else None,
                "baseline_display": a_display,
                "candidate_display": b_display,
                "changed": status == "CHANGED",
                "status": status,
            }
        )
    return rows


MATERIAL_FIELDS = (
    "retriever",
    "dense_model_id",
    "fusion",
    "rrf_k",
    "candidate_pool",
    "reranker_id",
    "query_expansion",
    "generation_model_id",
)


def configuration_change_summary(baseline_id: str, candidate_id: str) -> dict[str, Any]:
    """Count material config changes. Does not invent isolation that the experiment lacks."""

    rows = configuration_diff(baseline_id, candidate_id)
    changed = [item for item in rows if item["status"] == "CHANGED" and item["field"] in MATERIAL_FIELDS]
    fields = {item["field"] for item in changed}
    n_changed = len(changed)
    isolation_note = None
    if {"candidate_pool", "reranker_id"} <= fields:
        isolation_note = (
            "The observed metric difference reflects both candidate-pool size and reranking. "
            "This stored experiment does not isolate the reranker effect."
        )
    warning = None
    if n_changed > 1:
        warning = "Multiple variables changed. This experiment does not isolate a single intervention."
    noun = "change" if n_changed == 1 else "changes"
    return {
        "n_changed": n_changed,
        "changed_components": [item["component"] for item in changed],
        "changed_fields": sorted(fields),
        "headline": f"{n_changed} configuration {noun} detected",
        "isolation_note": isolation_note,
        "warning": warning,
        "isolates_single_intervention": n_changed == 1,
    }


def provenance(experiment_id: str, *, root: Any = None) -> dict[str, Any]:
    item = experiment_by_id(experiment_id, root=root)
    config = experiment_configuration(experiment_id)
    if item is None:
        return {"error": "Unknown experiment", "experiment_id": experiment_id}
    return {
        "experiment_id": experiment_id,
        "dataset": item.get("dataset"),
        "dataset_id": item.get("dataset_id"),
        "dataset_version": item.get("dataset_version"),
        "query_count": item.get("n_queries"),
        "source_artifact": item.get("source_file"),
        "model_ids": config.get("model_ids") or [],
        "retriever": config.get("retriever"),
        "dense_model": config.get("dense_model_name") or config.get("dense_model_id"),
        "fusion": config.get("fusion"),
        "rrf_k": config.get("rrf_k"),
        "candidate_pool": config.get("candidate_pool"),
        "reranker": config.get("reranker_name") if config.get("reranker_id") else config.get("reranker_id"),
        "query_expansion": config.get("query_expansion"),
        "generation_model": config.get("generation_model_id"),
        "retrieval_configuration": {
            "retriever": config.get("retriever"),
            "dense_model_id": config.get("dense_model_id"),
            "fusion": config.get("fusion"),
            "rrf_k": config.get("rrf_k"),
        },
        "prompt_ids": config.get("prompt_ids") or [],
        "prompt_version": item.get("prompt_versions"),
        "timestamp": item.get("timestamp"),
        "code_version": item.get("git_version"),
        "model_revision": None,
    }


def _fmt_value(value: Any) -> str:
    if value is None or value == [] or value == {}:
        return NOT_RECORDED
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def display_provenance(record: Mapping[str, Any]) -> dict[str, str]:
    shown: dict[str, str] = {}
    for key, label in PROVENANCE_FIELDS:
        shown[label] = _fmt_value(record.get(key))
    return shown


def provenance_completeness(record: Mapping[str, Any]) -> dict[str, Any]:
    """Count recorded vs missing provenance fields. Not a quality score."""

    missing = []
    recorded = 0
    for key, label in PROVENANCE_FIELDS:
        value = record.get(key)
        if value is None or value == [] or value == {}:
            missing.append(label)
        else:
            recorded += 1
    total = len(PROVENANCE_FIELDS)
    return {
        "recorded": recorded,
        "total": total,
        "missing": missing,
        "summary": f"{recorded} / {total} fields recorded",
    }


def run_lineage(experiment_id: str, *, root: Any = None) -> list[dict[str, Any]]:
    record = provenance(experiment_id, root=root)
    config = experiment_configuration(experiment_id)
    item = experiment_by_id(experiment_id, root=root) or {}
    return [
        {"stage": "Dataset", "value": record.get("dataset") or record.get("dataset_id") or NOT_RECORDED},
        {"stage": "Dataset version", "value": record.get("dataset_version") or NOT_RECORDED},
        {"stage": "Code version", "value": record.get("code_version") or NOT_RECORDED},
        {"stage": "Models", "value": ", ".join(config.get("model_ids") or []) or NOT_RECORDED},
        {"stage": "Prompts", "value": ", ".join(config.get("prompt_ids") or []) or NOT_RECORDED},
        {
            "stage": "Retrieval configuration",
            "value": (
                f"{config.get('retriever')} / dense={config.get('dense_model_id') or 'none'} / "
                f"fusion={config.get('fusion') or 'none'} / rrf_k={config.get('rrf_k') or 'none'} / "
                f"pool={config.get('candidate_pool')} / reranker={config.get('reranker_id') or 'none'}"
            ),
        },
        {
            "stage": "Evaluation",
            "value": (
                f"{item.get('n_queries')} queries · {item.get('source_file')}"
                if item
                else NOT_RECORDED
            ),
        },
        {
            "stage": "Results",
            "value": (
                f"Recall@10={item.get('recall_at_10')} · MRR@10={item.get('mrr_at_10')}"
                if item
                else NOT_RECORDED
            ),
        },
    ]
