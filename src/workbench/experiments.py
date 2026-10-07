"""Load stored experiment results into a leaderboard. Never invents missing metrics."""

from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.config import EVALUATION
from src.workbench.dataset import DATASET_ID, DATASET_VERSION, evaluation_dataset

ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _cfg(payload: Mapping[str, Any], name: str) -> dict[str, Any]:
    for item in payload.get("configurations") or []:
        if item.get("name") == name:
            return dict(item)
    return {}


def _percentile(values: Sequence[float], p: float) -> float | None:
    numbers = sorted(float(v) for v in values if v is not None)
    if not numbers:
        return None
    index = min(len(numbers) - 1, max(0, int(round((p / 100.0) * (len(numbers) - 1)))))
    return numbers[index]


def _latencies_from_details(path: Path, *, configuration: str | None = None) -> list[float]:
    payload = _load(path)
    values: list[float] = []
    for row in payload.get("rows") or payload.get("results") or []:
        if configuration and row.get("configuration") not in {None, configuration}:
            continue
        latency = row.get("latency_ms")
        if latency is None and isinstance(row.get("latency"), Mapping):
            latency = row["latency"].get("total_reranked_ms") or row["latency"].get("baseline_ms")
        if latency is not None:
            values.append(float(latency))
    return values


def _row(
    *,
    experiment_id: str,
    name: str,
    source: str,
    metrics: Mapping[str, Any],
    n_queries: int | None,
    production: bool,
    promotion: str,
    changed: str,
    baseline_id: str | None,
    cost_per_query: float | None,
    cost_note: str | None,
    p50: float | None = None,
    p95: float | None = None,
    p99: float | None = None,
    recall_at_1: float | None = None,
    recall_at_3: float | None = None,
    rerank_recovery: float | None = None,
    details_path: str | None = None,
) -> dict[str, Any]:
    dataset = evaluation_dataset()
    mean_latency = metrics.get("avg_latency_ms")
    if mean_latency is None:
        mean_latency = metrics.get("latency_ms")
    candidate_recall = metrics.get("candidate_recall_at_50")
    if candidate_recall is None:
        candidate_recall = metrics.get("recall_at_50")
    rec1 = recall_at_1 if recall_at_1 is not None else metrics.get("recall_at_1")
    rec3 = recall_at_3 if recall_at_3 is not None else metrics.get("recall_at_3")
    return {
        "id": experiment_id,
        "name": name,
        "dataset_id": DATASET_ID,
        "dataset_version": DATASET_VERSION,
        "dataset": dataset["name"],
        "n_queries": n_queries,
        "source_file": source,
        "details_path": details_path,
        "production": production,
        "production_label": "Yes" if production else "No",
        "promotion": promotion,
        "changed_variable": changed,
        "baseline_id": baseline_id,
        "candidate_recall_at_50": candidate_recall,
        "recall_at_1": rec1,
        "recall_at_3": rec3,
        "recall_at_5": metrics.get("recall_at_5"),
        "recall_at_10": metrics.get("recall_at_10"),
        "mrr_at_10": metrics.get("mrr_at_10"),
        "ndcg_at_10": metrics.get("ndcg_at_10"),
        "rerank_recovery": rerank_recovery,
        "mean_latency_ms": mean_latency,
        "p50_latency_ms": p50 if p50 is not None else metrics.get("median_latency_ms"),
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "cost_per_query_usd": cost_per_query,
        "cost_note": cost_note,
        "metrics_source": "stored JSON",
        "git_version": None,
        "timestamp": None,
        "prompt_versions": None,
    }


def load_leaderboard(*, root: Path | None = None) -> list[dict[str, Any]]:
    """One row per stored experiment that actually exists. Missing files are omitted."""

    base = root or ROOT
    rows: list[dict[str, Any]] = []

    comparison = _load(base / EVALUATION.comparison_path)
    details = base / EVALUATION.details_path
    for spec in (
        ("bm25", "BM25", "BM25", False, "retired-first-stage", "Lexical only", None),
        ("dense-minilm", "Dense MiniLM", "Dense", False, "retired-first-stage", "Dense MiniLM only", None),
        ("hybrid", "Hybrid", "Hybrid RRF", False, "candidate", "BM25 + MiniLM + RRF k=60 Top-10", "bm25"),
        ("hybrid-qe", "Hybrid + Query Expansion", "Hybrid + Query Expansion", False, "rejected", "gpt-4o-mini QE before Hybrid@10", "hybrid"),
    ):
        exp_id, name, cfg_name, prod, promo, changed, baseline = spec
        cfg = _cfg(comparison, cfg_name)
        if not cfg:
            continue
        lats = _latencies_from_details(details, configuration=cfg_name)
        if exp_id == "hybrid-qe":
            lats = _latencies_from_details(base / EVALUATION.hybrid_qe_details_path)
        cost = cfg.get("retrieval_cost_usd")
        nq = cfg.get("n_queries") or comparison.get("n_queries")
        per_q = None
        note = None
        if cost is not None and nq:
            per_q = float(cost) / float(nq) if float(nq) else None
        elif cost is None:
            note = "Not measured"
        rows.append(
            _row(
                experiment_id=exp_id,
                name=name,
                source=EVALUATION.comparison_path,
                metrics=cfg,
                n_queries=nq,
                production=prod,
                promotion=promo,
                changed=changed,
                baseline_id=baseline,
                cost_per_query=per_q,
                cost_note=note if per_q is None else None,
                p50=_percentile(lats, 50) or cfg.get("median_latency_ms"),
                p95=_percentile(lats, 95),
                p99=_percentile(lats, 99),
                details_path=str(
                    EVALUATION.hybrid_qe_details_path if exp_id == "hybrid-qe" else EVALUATION.details_path
                ),
            )
        )

    rerank = _load(base / EVALUATION.rerank_comparison_summary_path)
    medcpt = _cfg(rerank, "+ MedCPT")
    if medcpt:
        diag = _load(base / EVALUATION.retrieval_diagnostics_path)
        summary = dict((diag or {}).get("summary") or {})
        recov = (rerank.get("rerank_recovery") or {}).get("rerank_recovery_at_10")
        lats = []
        details_payload = _load(base / EVALUATION.rerank_comparison_details_path)
        for item in details_payload.get("rows") or []:
            latency = (item.get("latency") or {}).get("total_reranked_ms")
            if latency is not None:
                lats.append(float(latency))
        rows.append(
            _row(
                experiment_id="hybrid-medcpt",
                name="Hybrid + MedCPT",
                source=EVALUATION.rerank_comparison_summary_path,
                metrics=medcpt,
                n_queries=rerank.get("n_queries") or medcpt.get("n_queries"),
                production=True,
                promotion="production",
                changed="MedCPT Cross-Encoder after Hybrid RRF Top-50",
                baseline_id="hybrid",
                cost_per_query=None,
                cost_note="Not measured (local cross-encoder; no token usage recorded)",
                p50=_percentile(lats, 50),
                p95=_percentile(lats, 95),
                p99=_percentile(lats, 99),
                recall_at_1=summary.get("rerank_recall_at_1"),
                recall_at_3=summary.get("rerank_recall_at_3"),
                rerank_recovery=recov,
                details_path=EVALUATION.rerank_comparison_details_path,
            )
        )

    pool = _load(base / "experiments/results/retrieval_evaluation_rerank_comparison_pool100_summary.json")
    pool_cfg = _cfg(pool, "+ MedCPT")
    if pool_cfg:
        rows.append(
            _row(
                experiment_id="hybrid-pool100",
                name="Hybrid + candidate pool 100",
                source="experiments/results/retrieval_evaluation_rerank_comparison_pool100_summary.json",
                metrics=pool_cfg,
                n_queries=pool.get("n_queries"),
                production=False,
                promotion="rejected",
                changed="Candidate pool 50 → 100, then MedCPT Top-10",
                baseline_id="hybrid-medcpt",
                cost_per_query=None,
                cost_note="Not measured",
                p50=pool_cfg.get("median_latency_ms"),
                rerank_recovery=(pool.get("rerank_recovery") or {}).get("rerank_recovery_at_10"),
                details_path="experiments/results/retrieval_evaluation_rerank_comparison_pool100_details.json",
            )
        )

    bio = _load(base / "experiments/results/retrieval_evaluation_rerank_comparison_medcpt_biencoder_summary.json")
    bio_cfg = _cfg(bio, "+ MedCPT")
    if bio_cfg:
        rows.append(
            _row(
                experiment_id="hybrid-medcpt-biencoder",
                name="Hybrid + MedCPT bi-encoder",
                source="experiments/results/retrieval_evaluation_rerank_comparison_medcpt_biencoder_summary.json",
                metrics=bio_cfg,
                n_queries=bio.get("n_queries"),
                production=False,
                promotion="rejected",
                changed="Dense encoder MiniLM → MedCPT Query+Article, MedCPT CE retained",
                baseline_id="hybrid-medcpt",
                cost_per_query=None,
                cost_note="Not measured",
                p50=bio_cfg.get("median_latency_ms"),
                rerank_recovery=(bio.get("rerank_recovery") or {}).get("rerank_recovery_at_10"),
                details_path="experiments/results/retrieval_evaluation_rerank_comparison_medcpt_biencoder_details.json",
            )
        )

    qe_med = _load(base / "experiments/results/retrieval_evaluation_rerank_comparison_qe_summary.json")
    qe_cfg = _cfg(qe_med, "+ MedCPT")
    # Already have Hybrid+QE without MedCPT. This file is QE then MedCPT — include as distinct if present.
    if qe_cfg:
        rows.append(
            _row(
                experiment_id="hybrid-qe-medcpt",
                name="Hybrid + Query Expansion + MedCPT",
                source="experiments/results/retrieval_evaluation_rerank_comparison_qe_summary.json",
                metrics=qe_cfg,
                n_queries=qe_med.get("n_queries"),
                production=False,
                promotion="rejected",
                changed="QE then Hybrid@50 → MedCPT Top-10",
                baseline_id="hybrid-medcpt",
                cost_per_query=None,
                cost_note="Not measured",
                p50=qe_cfg.get("median_latency_ms"),
                rerank_recovery=(qe_med.get("rerank_recovery") or {}).get("rerank_recovery_at_10"),
                details_path="experiments/results/retrieval_evaluation_rerank_comparison_qe_details.json",
            )
        )

    pubmed = _load(base / EVALUATION.embedding_comparison_summary_path)
    pubmed_cfg = _cfg(pubmed, "B Biomedical embedding")
    if pubmed_cfg:
        rows.append(
            _row(
                experiment_id="hybrid-pubmedbert",
                name="Hybrid + PubMedBERT",
                source=EVALUATION.embedding_comparison_summary_path,
                metrics=pubmed_cfg,
                n_queries=pubmed.get("n_queries") or pubmed_cfg.get("n_queries"),
                production=False,
                promotion="rejected",
                changed="Dense encoder MiniLM → S-PubMedBert-MS-MARCO (Hybrid prefixes)",
                baseline_id="hybrid",
                cost_per_query=None,
                cost_note="Not measured",
                p50=pubmed_cfg.get("median_latency_ms"),
                details_path=EVALUATION.embedding_comparison_details_path,
            )
        )

    biomed = _load(base / "experiments/results/retrieval_evaluation_rerank_comparison_biomedbert_summary.json")
    biomed_cfg = _cfg(biomed, "+ MedCPT")
    if biomed_cfg:
        rows.append(
            _row(
                experiment_id="hybrid-biomedbert-reranker",
                name="Hybrid + BiomedBERT reranker",
                source="experiments/results/retrieval_evaluation_rerank_comparison_biomedbert_summary.json",
                metrics=biomed_cfg,
                n_queries=biomed.get("n_queries"),
                production=False,
                promotion="rejected",
                changed="Reranker ncbi/MedCPT-Cross-Encoder → NeuML/biomedbert-base-reranker",
                baseline_id="hybrid-medcpt",
                cost_per_query=None,
                cost_note="Not measured",
                p50=biomed_cfg.get("median_latency_ms"),
                rerank_recovery=(biomed.get("rerank_recovery") or {}).get("rerank_recovery_at_10"),
                details_path="experiments/results/retrieval_evaluation_rerank_comparison_biomedbert_details.json",
            )
        )

    sparse = _load(base / EVALUATION.sparse_study_summary_path)
    medcpt_sparse = dict(sparse.get("medcpt") or {})
    if sparse and not sparse.get("skipped") and medcpt_sparse and not medcpt_sparse.get("skipped"):
        metrics = {
            "recall_at_10": medcpt_sparse.get("recall_at_10"),
            "mrr_at_10": medcpt_sparse.get("mrr_at_10"),
            "ndcg_at_10": medcpt_sparse.get("ndcg_at_10"),
            "candidate_recall_at_50": medcpt_sparse.get("candidate_recall_50"),
            "avg_latency_ms": medcpt_sparse.get("avg_latency_ms"),
        }
        rows.append(
            _row(
                experiment_id="hybrid-splade-medcpt",
                name="Hybrid + SPLADE + MedCPT",
                source=EVALUATION.sparse_study_summary_path,
                metrics=metrics,
                n_queries=sparse.get("n_queries"),
                production=False,
                promotion="rejected",
                changed="Added SPLADE first-stage fused with BM25+MiniLM, then MedCPT@10",
                baseline_id="hybrid-medcpt",
                cost_per_query=None,
                cost_note="Not measured",
                rerank_recovery=None,
                details_path=EVALUATION.sparse_study_details_path,
            )
        )

    return rows


def experiment_by_id(experiment_id: str, *, root: Path | None = None) -> dict[str, Any] | None:
    for row in load_leaderboard(root=root):
        if row["id"] == experiment_id:
            return row
    return None


def leaderboard_query_counts(rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, int | None]:
    return {str(item["id"]): item.get("n_queries") for item in (rows or load_leaderboard())}


def compare_experiments(baseline_id: str, candidate_id: str, *, root: Path | None = None) -> dict[str, Any]:
    """Neutral delta description. Does not declare a winner."""

    left = experiment_by_id(baseline_id, root=root)
    right = experiment_by_id(candidate_id, root=root)
    if left is None or right is None:
        return {"error": "One or both experiments are missing from stored results."}

    def _delta(key: str) -> dict[str, Any] | None:
        a, b = left.get(key), right.get(key)
        if a is None or b is None:
            return None
        change = float(b) - float(a)
        return {"baseline": a, "candidate": b, "delta": change}

    metrics = {}
    statements = []
    for key, label, scale in (
        ("candidate_recall_at_50", "Candidate Recall@50", "pp"),
        ("recall_at_1", "Recall@1", "pp"),
        ("recall_at_3", "Recall@3", "pp"),
        ("recall_at_5", "Recall@5", "pp"),
        ("recall_at_10", "Recall@10", "pp"),
        ("mrr_at_10", "MRR@10", "abs"),
        ("ndcg_at_10", "nDCG@10", "abs"),
        ("rerank_recovery", "Rerank recovery", "abs"),
        ("mean_latency_ms", "Mean latency", "ms"),
        ("p50_latency_ms", "P50 latency", "ms"),
        ("p95_latency_ms", "P95 latency", "ms"),
        ("cost_per_query_usd", "Cost/query", "usd"),
    ):
        payload = _delta(key)
        if not payload:
            continue
        metrics[key] = payload
        delta = payload["delta"]
        if scale == "pp":
            if delta > 0:
                statements.append(f"{label} increased by {abs(delta) * 100:.2f} percentage points")
            elif delta < 0:
                statements.append(f"{label} decreased by {abs(delta) * 100:.2f} percentage points")
            else:
                statements.append(f"{label} unchanged")
        elif scale == "ms":
            statements.append(
                f"{label} {'increased' if delta > 0 else 'decreased' if delta < 0 else 'unchanged'}"
                + (f" by {abs(delta):.1f} ms" if delta else "")
            )
        elif scale == "usd":
            statements.append(f"{label} changed by {delta:+.6f} USD")
        else:
            statements.append(f"{label} {'increased' if delta > 0 else 'decreased' if delta < 0 else 'unchanged'} by {abs(delta):.4f}")
    return {
        "baseline": left,
        "candidate": right,
        "metrics": metrics,
        "statements": statements,
        "same_dataset": left.get("dataset_id") == right.get("dataset_id"),
        "same_query_count": left.get("n_queries") == right.get("n_queries"),
        "dataset_id": left.get("dataset_id"),
        "dataset_version": left.get("dataset_version"),
        "query_count_baseline": left.get("n_queries"),
        "query_count_candidate": right.get("n_queries"),
    }


def _query_map_from_config_details(path: Path, configuration: str) -> dict[int, dict[str, Any]]:
    payload = _load(path)
    out: dict[int, dict[str, Any]] = {}
    for row in payload.get("rows") or payload.get("results") or []:
        if row.get("configuration") != configuration:
            continue
        qid = int(row["query_id"])
        out[qid] = {
            "query_id": qid,
            "question": row.get("question"),
            "recall_at_10": row.get("recall_at_10"),
            "recall_at_5": row.get("recall_at_5"),
            "mrr_at_10": row.get("mrr_at_10"),
            "ndcg_at_10": row.get("ndcg_at_10"),
            "latency_ms": row.get("latency_ms"),
            "retrieved_passage_ids": row.get("retrieved_passage_ids") or [],
            "final_ids": row.get("retrieved_passage_ids") or [],
            "candidate_ids": None,
            "gold_ids": row.get("relevant_passage_ids") or row.get("gold_passage_ids") or [],
        }
    return out


def _query_map_from_rerank_details(path: Path, side: str) -> dict[int, dict[str, Any]]:
    payload = _load(path)
    key = {"baseline": "baseline", "hybrid": "hybrid_50", "reranked": "reranked"}.get(side, side)
    out: dict[int, dict[str, Any]] = {}
    for row in payload.get("rows") or []:
        block = row.get(key) or {}
        if not block:
            continue
        qid = int(row["query_id"])
        latency = None
        lat = row.get("latency") or {}
        if key == "reranked":
            latency = lat.get("total_reranked_ms")
        elif key == "hybrid_50":
            latency = lat.get("hybrid_ms")
        else:
            latency = lat.get("baseline_ms")
        candidate_ids = (row.get("hybrid_50") or {}).get("retrieved_passage_ids")
        final_ids = block.get("retrieved_passage_ids") or []
        out[qid] = {
            "query_id": qid,
            "question": row.get("question"),
            "recall_at_10": block.get("recall_at_10"),
            "recall_at_5": block.get("recall_at_5"),
            "mrr_at_10": block.get("mrr_at_10"),
            "ndcg_at_10": block.get("ndcg_at_10"),
            "latency_ms": latency,
            "retrieved_passage_ids": final_ids,
            "final_ids": final_ids,
            "candidate_ids": candidate_ids,
            "gold_ids": row.get("relevant_passage_ids") or [],
        }
    return out


QUERY_LOADERS = {
    "bm25": ("config", EVALUATION.details_path, "BM25"),
    "dense-minilm": ("config", EVALUATION.details_path, "Dense"),
    "hybrid": ("config", EVALUATION.details_path, "Hybrid RRF"),
    "hybrid-qe": ("config", EVALUATION.hybrid_qe_details_path, "Hybrid + Query Expansion"),
    "hybrid-medcpt": ("rerank", EVALUATION.rerank_comparison_details_path, "reranked"),
    "hybrid-pool100": ("rerank", "experiments/results/retrieval_evaluation_rerank_comparison_pool100_details.json", "reranked"),
    "hybrid-medcpt-biencoder": ("rerank", "experiments/results/retrieval_evaluation_rerank_comparison_medcpt_biencoder_details.json", "reranked"),
    "hybrid-qe-medcpt": ("rerank", "experiments/results/retrieval_evaluation_rerank_comparison_qe_details.json", "reranked"),
    "hybrid-biomedbert-reranker": ("rerank", "experiments/results/retrieval_evaluation_rerank_comparison_biomedbert_details.json", "reranked"),
    "hybrid-pubmedbert": ("config", EVALUATION.embedding_comparison_details_path, "B Biomedical embedding"),
}


TOP_K = 10
AB_IMPROVED = "IMPROVED"
AB_REGRESSED = "REGRESSED"
AB_UNCHANGED = "UNCHANGED"
AB_INSUFFICIENT = "INSUFFICIENT DATA"
AB_MIXED = "MIXED"
FIXED = "FIXED"


def _signal_dir(a: Any, b: Any, *, lower_better: bool = False) -> int | None:
    if a is None or b is None:
        return None
    try:
        left, right = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if abs(left - right) < 1e-12:
        return 0
    if lower_better:
        return 1 if right < left else -1
    return 1 if right > left else -1


def explain_classification(
    *,
    a_rank: int | None = None,
    b_rank: int | None = None,
    a_recall: Any = None,
    b_recall: Any = None,
    a_ndcg: Any = None,
    b_ndcg: Any = None,
    a_candidate_hit: bool | None = None,
    b_candidate_hit: bool | None = None,
) -> dict[str, Any]:
    """Human-readable reason. Mixed signals are flagged instead of being forced."""

    dirs = [
        _signal_dir(a_rank, b_rank, lower_better=True),
        _signal_dir(a_recall, b_recall),
        _signal_dir(a_ndcg, b_ndcg),
        _signal_dir(
            None if a_candidate_hit is None else int(bool(a_candidate_hit)),
            None if b_candidate_hit is None else int(bool(b_candidate_hit)),
        ),
    ]
    nonzero = [item for item in dirs if item not in (None, 0)]
    mixed = 1 in nonzero and -1 in nonzero
    if mixed:
        extra = ""
        if a_rank is not None and b_rank is not None and a_rank != b_rank:
            extra = f". Best gold rank moved {a_rank} → {b_rank}"
        return {"reason": f"Mixed / inspect metrics{extra}", "ambiguous": True}
    if a_rank is not None and b_rank is not None:
        if a_rank != b_rank:
            return {"reason": f"Best gold rank moved {a_rank} → {b_rank}", "ambiguous": False}
        return {"reason": "Same best gold rank and same retrieval outcome", "ambiguous": False}
    if a_recall is not None and b_recall is not None:
        if abs(float(a_recall) - float(b_recall)) < 1e-12:
            return {"reason": "Same retrieval outcome; best gold rank not recorded", "ambiguous": False}
        return {"reason": f"Recall@10 moved {a_recall} → {b_recall}", "ambiguous": False}
    return {"reason": "Insufficient data", "ambiguous": False}


def _fmt_metric(value: Any, *, kind: str) -> str:
    if value is None:
        return "Not available"
    if kind == "rank":
        return str(int(value))
    if kind == "pct":
        return f"{float(value):.2f}"
    return f"{float(value):.4f}"


def metric_disagreement_diagnostic(
    *,
    a_rank: int | None = None,
    b_rank: int | None = None,
    a_recall: Any = None,
    b_recall: Any = None,
    a_rr: float | None = None,
    b_rr: float | None = None,
    a_candidate_hit: bool | None = None,
    b_candidate_hit: bool | None = None,
    a_ndcg: Any = None,
    b_ndcg: Any = None,
) -> dict[str, Any]:
    """Diagnostic flag only. Does not change IMPROVED / REGRESSED / UNCHANGED labels."""

    rank_dir = _signal_dir(a_rank, b_rank, lower_better=True)
    rec_dir = _signal_dir(a_recall, b_recall)
    rr_dir = _signal_dir(a_rr, b_rr)
    ndcg_dir = _signal_dir(a_ndcg, b_ndcg)
    cand_dir = _signal_dir(
        None if a_candidate_hit is None else int(bool(a_candidate_hit)),
        None if b_candidate_hit is None else int(bool(b_candidate_hit)),
    )
    comparable = sum(
        1
        for item in (rank_dir, rec_dir, rr_dir, ndcg_dir, cand_dir)
        if item is not None
    )
    if comparable < 2:
        return {
            "metric_disagreement": False,
            "metric_disagreement_reason": "Not available",
            "metric_disagreement_cases": [],
        }

    cases: list[str] = []
    if rank_dir == 1 and rec_dir == -1:
        cases.append(
            "Best gold rank improved "
            f"{_fmt_metric(a_rank, kind='rank')} → {_fmt_metric(b_rank, kind='rank')}, "
            "but Recall@10 decreased "
            f"{_fmt_metric(a_recall, kind='pct')} → {_fmt_metric(b_recall, kind='pct')}."
        )
    if rec_dir == 1 and rank_dir == -1:
        cases.append(
            "Recall@10 improved "
            f"{_fmt_metric(a_recall, kind='pct')} → {_fmt_metric(b_recall, kind='pct')}, "
            "but best-gold rank decreased "
            f"{_fmt_metric(a_rank, kind='rank')} → {_fmt_metric(b_rank, kind='rank')}."
        )
    if cand_dir == 1 and rank_dir == -1:
        cases.append(
            "Candidate retrieval improved, but final ranking decreased "
            f"{_fmt_metric(a_rank, kind='rank')} → {_fmt_metric(b_rank, kind='rank')}."
        )
    if cand_dir == 0 and rank_dir not in (None, 0):
        cases.append(
            "Candidate retrieval unchanged, but final ranking changed "
            f"{_fmt_metric(a_rank, kind='rank')} → {_fmt_metric(b_rank, kind='rank')}."
        )
    if rank_dir == 1 and ndcg_dir == -1:
        cases.append(
            "Best gold rank improved "
            f"{_fmt_metric(a_rank, kind='rank')} → {_fmt_metric(b_rank, kind='rank')}, "
            "but nDCG contribution decreased "
            f"{_fmt_metric(a_ndcg, kind='pct')} → {_fmt_metric(b_ndcg, kind='pct')}."
        )
    if rec_dir == 1 and ndcg_dir == -1 and rank_dir != 1:
        cases.append(
            "Recall@10 improved "
            f"{_fmt_metric(a_recall, kind='pct')} → {_fmt_metric(b_recall, kind='pct')}, "
            "but nDCG contribution decreased "
            f"{_fmt_metric(a_ndcg, kind='pct')} → {_fmt_metric(b_ndcg, kind='pct')}."
        )
    if rec_dir == -1 and rr_dir == 1 and rank_dir != 1:
        cases.append(
            "Reciprocal rank improved "
            f"{_fmt_metric(a_rr, kind='ratio')} → {_fmt_metric(b_rr, kind='ratio')}, "
            "but Recall@10 decreased "
            f"{_fmt_metric(a_recall, kind='pct')} → {_fmt_metric(b_recall, kind='pct')}."
        )

    if not cases:
        return {
            "metric_disagreement": False,
            "metric_disagreement_reason": None,
            "metric_disagreement_cases": [],
        }
    if len(cases) > 1:
        reason = "Multiple metrics disagree. " + " ".join(cases)
    else:
        reason = cases[0]
    return {
        "metric_disagreement": True,
        "metric_disagreement_reason": reason,
        "metric_disagreement_cases": cases,
    }

FAILURE_FILTER_BUCKETS = {
    "Data quality": {"DATA_QUALITY", "GOLD_LABEL"},
    "Retrieval": {"RETRIEVAL", "FUSION"},
    "Reranking": {"RERANKING"},
    "Generation": {"GENERATION"},
    "Citation": {"CITATION"},
}


def _gold_set(gold: Sequence[Any]) -> set[int]:
    out: set[int] = set()
    for item in gold:
        try:
            out.add(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _first_gold_rank(retrieved: Sequence[Any], gold: Sequence[Any]) -> int | None:
    gold_set = _gold_set(gold)
    if not retrieved:
        return None
    for index, pid in enumerate(retrieved, start=1):
        try:
            if int(pid) in gold_set:
                return index
        except (TypeError, ValueError):
            continue
    return None


def _candidate_hit(ids: Any, gold: Sequence[Any]) -> bool | None:
    if ids is None:
        return None
    gold_set = _gold_set(gold)
    if not gold_set:
        return None
    try:
        return bool(gold_set & {int(pid) for pid in ids})
    except (TypeError, ValueError):
        return None


def reciprocal_rank(rank: int | None) -> float | None:
    if rank is None:
        return None
    return 1.0 / float(rank)


def rank_movement_phrase(a_rank: int | None, b_rank: int | None) -> str:
    if a_rank is None or b_rank is None:
        return "Insufficient data"
    delta = int(a_rank) - int(b_rank)
    if delta > 0:
        return f"+{delta} positions toward top"
    if delta < 0:
        return f"{abs(delta)} positions toward bottom"
    return "unchanged"


def in_topk(rank: int | None, recall: Any = None, *, k: int = TOP_K) -> bool | None:
    if rank is not None:
        return int(rank) <= k
    if recall is not None:
        return float(recall) > 0
    return None


def classify_query_outcome(
    *,
    a_rank: int | None = None,
    b_rank: int | None = None,
    a_recall: Any = None,
    b_recall: Any = None,
    a_rr: float | None = None,
    b_rr: float | None = None,
    a_ndcg: Any = None,
    b_ndcg: Any = None,
    a_candidate_hit: bool | None = None,
    b_candidate_hit: bool | None = None,
) -> dict[str, Any]:
    """Label a stored A/B query change. Missing values stay insufficient, never invented."""

    diagnoses: list[str] = []
    improved = False
    regressed = False
    comparable = False

    if a_rank is not None and b_rank is not None:
        comparable = True
        if b_rank < a_rank:
            improved = True
            diagnoses.append("improved best gold rank")
            if a_rank > TOP_K and b_rank <= TOP_K:
                diagnoses.append("moved from outside Top-K to inside Top-K")
            elif a_rank <= TOP_K and b_rank <= TOP_K:
                diagnoses.append("moved higher within Top-K")
        elif b_rank > a_rank:
            regressed = True
            diagnoses.append("lower rank")
            if a_rank <= TOP_K and b_rank > TOP_K:
                diagnoses.append("moved from inside Top-K to outside Top-K")

    if a_recall is not None and b_recall is not None:
        comparable = True
        a_r, b_r = float(a_recall), float(b_recall)
        if b_r > a_r:
            improved = True
            diagnoses.append("newly retrieved gold" if a_r == 0 and b_r > 0 else "increased Recall@10")
        elif b_r < a_r:
            regressed = True
            diagnoses.append("lost gold" if b_r == 0 and a_r > 0 else "decreased Recall@10")

    if a_rr is not None and b_rr is not None:
        comparable = True
        if b_rr > a_rr:
            improved = True
            diagnoses.append("improved reciprocal rank")
        elif b_rr < a_rr:
            regressed = True
            diagnoses.append("lower reciprocal rank")

    if a_ndcg is not None and b_ndcg is not None:
        comparable = True
        if float(b_ndcg) > float(a_ndcg):
            improved = True
            diagnoses.append("improved nDCG contribution")
        elif float(b_ndcg) < float(a_ndcg):
            regressed = True
            diagnoses.append("lower nDCG contribution")

    if a_candidate_hit is not None and b_candidate_hit is not None:
        comparable = True
        if a_candidate_hit and not b_candidate_hit:
            regressed = True
            diagnoses.append("candidate disappearance")
        elif (not a_candidate_hit) and b_candidate_hit:
            improved = True
            diagnoses.append("newly retrieved gold in candidate pool")
        a_in = in_topk(a_rank, a_recall)
        b_in = in_topk(b_rank, b_recall)
        if a_candidate_hit and b_candidate_hit and a_in is True and b_in is False:
            regressed = True
            diagnoses.append("reranker demotion")

    if not comparable:
        return {
            "outcome": AB_INSUFFICIENT,
            "ab_class": AB_INSUFFICIENT,
            "diagnoses": ["Insufficient data"],
        }

    a_in = in_topk(a_rank, a_recall)
    b_in = in_topk(b_rank, b_recall)
    outcome = AB_UNCHANGED
    if a_in is False and b_in is True:
        outcome = FIXED
    elif improved and not regressed:
        outcome = AB_IMPROVED
    elif regressed and not improved:
        outcome = AB_REGRESSED
    elif improved and regressed:
        if a_rank is not None and b_rank is not None:
            if b_rank < a_rank:
                outcome = FIXED if a_in is False and b_in is True else AB_IMPROVED
            elif b_rank > a_rank:
                outcome = AB_REGRESSED
            else:
                outcome = AB_UNCHANGED
        elif a_recall is not None and b_recall is not None:
            if float(b_recall) > float(a_recall):
                outcome = FIXED if a_in is False and b_in is True else AB_IMPROVED
            elif float(b_recall) < float(a_recall):
                outcome = AB_REGRESSED
            else:
                outcome = AB_UNCHANGED
        else:
            outcome = AB_UNCHANGED

    ab_class = AB_IMPROVED if outcome == FIXED else outcome
    return {"outcome": outcome, "ab_class": ab_class, "diagnoses": diagnoses or ["unchanged"]}


def failure_filter_bucket(category: str | None) -> str | None:
    if not category:
        return None
    for bucket, names in FAILURE_FILTER_BUCKETS.items():
        if category in names:
            return bucket
    return None


def load_per_query_results(experiment_id: str, *, root: Path | None = None) -> dict[int, dict[str, Any]]:
    base = root or ROOT
    spec = QUERY_LOADERS.get(experiment_id)
    if not spec:
        return {}
    kind, rel, key = spec
    path = base / rel
    if kind == "config":
        return _query_map_from_config_details(path, key)
    return _query_map_from_rerank_details(path, key)


def _stage_changes(a: Mapping[str, Any], b: Mapping[str, Any], gold: Sequence[Any]) -> list[str]:
    notes: list[str] = []
    a_pool, b_pool = _candidate_hit(a.get("candidate_ids"), gold), _candidate_hit(b.get("candidate_ids"), gold)
    a_final, b_final = _candidate_hit(a.get("final_ids") or a.get("retrieved_passage_ids"), gold), _candidate_hit(
        b.get("final_ids") or b.get("retrieved_passage_ids"), gold
    )
    if a_pool is not None and b_pool is not None and a_pool != b_pool:
        notes.append(
            "Candidate pool: gold present in A, absent in B" if a_pool and not b_pool else "Candidate pool: gold absent in A, present in B"
        )
    if a_final is not None and b_final is not None and a_final != b_final:
        notes.append(
            "Final Top-K: gold present in A, absent in B" if a_final and not b_final else "Final Top-K: gold absent in A, present in B"
        )
    if a.get("candidate_ids") is None and b.get("candidate_ids") is None:
        notes.append("Candidate-pool IDs not recorded for either experiment")
    elif a.get("candidate_ids") is None:
        notes.append("Candidate-pool IDs not recorded for experiment A")
    elif b.get("candidate_ids") is None:
        notes.append("Candidate-pool IDs not recorded for experiment B")
    return notes


def per_query_diff(baseline_id: str, candidate_id: str, *, root: Path | None = None) -> list[dict[str, Any]]:
    left = load_per_query_results(baseline_id, root=root)
    right = load_per_query_results(candidate_id, root=root)
    from src.workbench.failures import query_failure_rows

    failure_by_q = {int(item["query_id"]): item.get("category") for item in query_failure_rows(root=root)}
    ids = sorted(set(left) | set(right))
    rows = []
    for qid in ids:
        a = left.get(qid) or {}
        b = right.get(qid) or {}
        rec_a = a.get("recall_at_10")
        rec_b = b.get("recall_at_10")
        lat_a = a.get("latency_ms")
        lat_b = b.get("latency_ms")
        rec_delta = None if rec_a is None or rec_b is None else float(rec_b) - float(rec_a)
        gold = b.get("gold_ids") or a.get("gold_ids") or []
        rank_a = _first_gold_rank(a.get("final_ids") or a.get("retrieved_passage_ids") or [], gold)
        rank_b = _first_gold_rank(b.get("final_ids") or b.get("retrieved_passage_ids") or [], gold)
        cand_rank_a = _first_gold_rank(a.get("candidate_ids") or [], gold) if a.get("candidate_ids") is not None else None
        cand_rank_b = _first_gold_rank(b.get("candidate_ids") or [], gold) if b.get("candidate_ids") is not None else None
        rr_a = reciprocal_rank(rank_a)
        rr_b = reciprocal_rank(rank_b)
        cand_hit_a = _candidate_hit(a.get("candidate_ids"), gold)
        cand_hit_b = _candidate_hit(b.get("candidate_ids"), gold)
        labelled = classify_query_outcome(
            a_rank=rank_a,
            b_rank=rank_b,
            a_recall=rec_a,
            b_recall=rec_b,
            a_rr=rr_a,
            b_rr=rr_b,
            a_ndcg=a.get("ndcg_at_10"),
            b_ndcg=b.get("ndcg_at_10"),
            a_candidate_hit=cand_hit_a,
            b_candidate_hit=cand_hit_b,
        )
        explained = explain_classification(
            a_rank=rank_a,
            b_rank=rank_b,
            a_recall=rec_a,
            b_recall=rec_b,
            a_ndcg=a.get("ndcg_at_10"),
            b_ndcg=b.get("ndcg_at_10"),
            a_candidate_hit=cand_hit_a,
            b_candidate_hit=cand_hit_b,
        )
        disagreement = metric_disagreement_diagnostic(
            a_rank=rank_a,
            b_rank=rank_b,
            a_recall=rec_a,
            b_recall=rec_b,
            a_rr=rr_a,
            b_rr=rr_b,
            a_candidate_hit=cand_hit_a,
            b_candidate_hit=cand_hit_b,
            a_ndcg=a.get("ndcg_at_10"),
            b_ndcg=b.get("ndcg_at_10"),
        )
        ab_class = labelled["ab_class"]
        if not a or not b:
            ab_class = AB_INSUFFICIENT
            labelled = {
                "outcome": AB_INSUFFICIENT,
                "ab_class": AB_INSUFFICIENT,
                "diagnoses": ["Insufficient data"],
            }
            explained = {"reason": "Insufficient data", "ambiguous": False}
            disagreement = {
                "metric_disagreement": False,
                "metric_disagreement_reason": "Not available",
                "metric_disagreement_cases": [],
            }
        rank_delta = None if rank_a is None or rank_b is None else rank_b - rank_a
        toward_top = None if rank_a is None or rank_b is None else rank_a - rank_b
        rows.append(
            {
                "query_id": qid,
                "question": b.get("question") or a.get("question"),
                "a_result": rec_a,
                "b_result": rec_b,
                "a_recall_at_10": rec_a,
                "b_recall_at_10": rec_b,
                "recall_change": rec_delta,
                "a_rank": rank_a,
                "b_rank": rank_b,
                "rank_change": rank_delta,
                "rank_movement": toward_top,
                "rank_movement_phrase": rank_movement_phrase(rank_a, rank_b),
                "a_rr": rr_a,
                "b_rr": rr_b,
                "a_ndcg": a.get("ndcg_at_10"),
                "b_ndcg": b.get("ndcg_at_10"),
                "a_candidate_hit": cand_hit_a,
                "b_candidate_hit": cand_hit_b,
                "a_candidate_rank": cand_rank_a,
                "b_candidate_rank": cand_rank_b,
                "candidate_rank_movement_phrase": rank_movement_phrase(cand_rank_a, cand_rank_b),
                "a_latency_ms": lat_a,
                "b_latency_ms": lat_b,
                "latency_change_ms": None if lat_a is None or lat_b is None else float(lat_b) - float(lat_a),
                "a_ids": a.get("retrieved_passage_ids") or [],
                "b_ids": b.get("retrieved_passage_ids") or [],
                "a_candidate_ids": a.get("candidate_ids"),
                "b_candidate_ids": b.get("candidate_ids"),
                "a_final_ids": a.get("final_ids") or a.get("retrieved_passage_ids") or [],
                "b_final_ids": b.get("final_ids") or b.get("retrieved_passage_ids") or [],
                "gold_ids": gold,
                "stage_changes": _stage_changes(a, b, gold),
                "failure_category": failure_by_q.get(qid),
                "failure_bucket": failure_filter_bucket(failure_by_q.get(qid)),
                "classification": ab_class,
                "outcome": labelled["outcome"],
                "diagnoses": labelled["diagnoses"],
                "reason": explained["reason"],
                "ambiguous": explained["ambiguous"],
                "metric_disagreement": disagreement["metric_disagreement"],
                "metric_disagreement_reason": disagreement["metric_disagreement_reason"],
                "helped": ab_class == AB_IMPROVED,
                "hurt": ab_class == AB_REGRESSED,
                "unchanged": ab_class == AB_UNCHANGED,
                "insufficient": ab_class == AB_INSUFFICIENT,
            }
        )
    return rows


def query_diff_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "total": len(rows),
        "improved": sum(1 for item in rows if item.get("classification") == AB_IMPROVED),
        "regressed": sum(1 for item in rows if item.get("classification") == AB_REGRESSED),
        "unchanged": sum(1 for item in rows if item.get("classification") == AB_UNCHANGED),
        "insufficient": sum(1 for item in rows if item.get("classification") == AB_INSUFFICIENT),
        "ambiguous": sum(1 for item in rows if item.get("ambiguous")),
        "metric_disagreement": sum(1 for item in rows if item.get("metric_disagreement")),
        "explainable_rank_movement": sum(
            1
            for item in rows
            if item.get("a_rank") is not None
            and item.get("b_rank") is not None
            and item.get("a_rank") != item.get("b_rank")
        ),
    }


def promotion_packet(experiment_id: str, *, root: Path | None = None) -> dict[str, Any]:
    item = experiment_by_id(experiment_id, root=root)
    if item is None:
        return {"error": "Unknown experiment"}
    return {
        "experiment": item["name"],
        "stage": item["promotion"],
        "production": item["production"],
        "dataset": item["dataset"],
        "n_queries": item["n_queries"],
        "recall_at_10": item["recall_at_10"],
        "mean_latency_ms": item["mean_latency_ms"],
        "cost_per_query_usd": item["cost_per_query_usd"],
        "note": "Promotion is manual. One metric is not sufficient to promote.",
        "path": "Experiment → Candidate → Validated → Production",
        "auto_promote": False,
    }


def leaderboard_table_rows(rows: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Flatten stored experiments for the UI table. Missing metrics stay None."""

    table = []
    for item in rows or load_leaderboard():
        table.append(
            {
                "Experiment": item["name"],
                "Dataset": item["dataset"],
                "Queries": item["n_queries"],
                "Production": item["production_label"],
                "Recall@10": item.get("recall_at_10"),
                "Candidate Recall@50": item.get("candidate_recall_at_50"),
                "Recall@1": item.get("recall_at_1"),
                "Recall@3": item.get("recall_at_3"),
                "Recall@5": item.get("recall_at_5"),
                "MRR@10": item.get("mrr_at_10"),
                "nDCG@10": item.get("ndcg_at_10"),
                "Rerank Recovery": item.get("rerank_recovery"),
                "Mean latency": item.get("mean_latency_ms"),
                "P50 latency": item.get("p50_latency_ms"),
                "P95 latency": item.get("p95_latency_ms"),
                "Cost/query": item.get("cost_per_query_usd"),
                "id": item["id"],
            }
        )
    return table


def dataset_consistency_report(rows: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    payload = evaluation_dataset()
    items = list(rows or load_leaderboard())
    mismatches = [
        {
            "id": item.get("id"),
            "dataset_id": item.get("dataset_id"),
            "n_queries": item.get("n_queries"),
        }
        for item in items
        if item.get("dataset_id") != payload["id"] or item.get("n_queries") != payload["n_queries"]
    ]
    return {
        "dataset_id": payload["id"],
        "expected_n_queries": payload["n_queries"],
        "n_experiments": len(items),
        "consistent": not mismatches,
        "mismatches": mismatches,
    }
