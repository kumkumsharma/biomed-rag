"""Regression dataset. Separate from the 100-query benchmark. No invented outcomes."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from src.workbench.experiments import (
    classify_query_outcome,
    load_per_query_results,
    reciprocal_rank,
    _candidate_hit,
    _first_gold_rank,
)
from src.workbench.failures import query_failure_rows

ROOT = Path(__file__).resolve().parents[2]
REGRESSION_RELPATH = "experiments/results/workbench/regression_set.json"


def regression_path(*, root: Path | None = None) -> Path:
    return (root or ROOT) / REGRESSION_RELPATH


def load_regression_set(*, root: Path | None = None) -> list[dict[str, Any]]:
    path = regression_path(root=root)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return list(payload)
    return list(payload.get("items") or [])


def save_regression_set(items: list[dict[str, Any]], *, root: Path | None = None) -> Path:
    path = regression_path(root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"items": items}, indent=2) + "\n", encoding="utf-8")
    return path


def query_in_regression_set(query_id: int, *, root: Path | None = None) -> dict[str, Any] | None:
    for existing in load_regression_set(root=root):
        if int(existing.get("query_id")) == int(query_id):
            return existing
    return None


def add_to_regression_set(
    *,
    query_id: int,
    question: str,
    gold_ids: list[Any],
    failure_category: str | None,
    source_experiment: str,
    diagnosis: str | None = None,
    date_added: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Append a benchmark failure. Does not modify evaluation_100.json."""

    existing = query_in_regression_set(query_id, root=root)
    if existing is not None:
        return {**existing, "already_present": True}
    record = {
        "query_id": int(query_id),
        "question": question,
        "gold_ids": list(gold_ids),
        "failure_category": failure_category,
        "source_experiment": source_experiment,
        "diagnosis": diagnosis,
        "date_added": date_added or date.today().isoformat(),
    }
    items = load_regression_set(root=root)
    items.append(record)
    save_regression_set(items, root=root)
    return {**record, "already_present": False}


def add_failure_query(
    *,
    query_id: int,
    source_experiment: str = "hybrid-medcpt",
    root: Path | None = None,
    results_root: Path | None = None,
) -> dict[str, Any]:
    existing = query_in_regression_set(query_id, root=root)
    if existing is not None:
        return {**existing, "already_present": True}
    eval_root = results_root or ROOT
    failures = {int(item["query_id"]): item for item in query_failure_rows(root=eval_root)}
    row = failures.get(int(query_id))
    if row is None:
        per_query = load_per_query_results(source_experiment, root=eval_root).get(int(query_id))
        if per_query is None:
            raise KeyError(f"Query {query_id} is not a stored failure for {source_experiment}")
        row = {
            "query_id": query_id,
            "question": per_query.get("question"),
            "category": None,
            "gold_ids": per_query.get("gold_ids") or [],
        }
        if (per_query.get("recall_at_10") or 0) > 0:
            raise ValueError("Query is not a stored Top-10 miss; not added.")
    return add_to_regression_set(
        query_id=int(query_id),
        question=str(row.get("question") or ""),
        gold_ids=list(row.get("gold_ids") or []),
        failure_category=row.get("category") or row.get("failure_category"),
        source_experiment=source_experiment,
        diagnosis=None,
        root=root,
    )


def _row_signals(row: dict[str, Any], gold: list[Any]) -> dict[str, Any]:
    final_ids = row.get("final_ids") or row.get("retrieved_passage_ids") or []
    rank = _first_gold_rank(final_ids, gold)
    return {
        "recall_at_10": row.get("recall_at_10"),
        "rank": rank,
        "rr": reciprocal_rank(rank),
        "ndcg": row.get("ndcg_at_10"),
        "candidate_hit": _candidate_hit(row.get("candidate_ids"), gold),
    }


def run_regression(
    experiment_id: str,
    *,
    root: Path | None = None,
    results_root: Path | None = None,
) -> dict[str, Any]:
    """Score the regression set using stored per-query results. Does not re-retrieve.

    ``root`` is the regression-set location. Experiment artifacts default to the
    project tree so tests can isolate the set without copying historical JSON.
    """

    items = load_regression_set(root=root)
    eval_root = results_root or ROOT
    candidate = load_per_query_results(experiment_id, root=eval_root)
    buckets: dict[str, list[dict[str, Any]]] = {
        "FIXED": [],
        "IMPROVED": [],
        "REGRESSED": [],
        "UNCHANGED": [],
        "INSUFFICIENT DATA": [],
        "passed": [],
        "failed": [],
    }
    for item in items:
        qid = int(item["query_id"])
        source = load_per_query_results(
            str(item.get("source_experiment") or "hybrid-medcpt"),
            root=eval_root,
        )
        source_row = source.get(qid) or {}
        cand_row = candidate.get(qid)
        gold = list(item.get("gold_ids") or source_row.get("gold_ids") or (cand_row or {}).get("gold_ids") or [])
        if cand_row is None or cand_row.get("recall_at_10") is None:
            entry = {
                **item,
                "outcome": "INSUFFICIENT DATA",
                "diagnoses": ["Per-query result not stored for this experiment"],
                "before": None,
                "after": None,
            }
            buckets["INSUFFICIENT DATA"].append(entry)
            continue
        before = _row_signals(source_row, gold)
        after = _row_signals(cand_row, gold)
        labelled = classify_query_outcome(
            a_rank=before["rank"],
            b_rank=after["rank"],
            a_recall=before["recall_at_10"],
            b_recall=after["recall_at_10"],
            a_rr=before["rr"],
            b_rr=after["rr"],
            a_ndcg=before["ndcg"],
            b_ndcg=after["ndcg"],
            a_candidate_hit=before["candidate_hit"],
            b_candidate_hit=after["candidate_hit"],
        )
        if source_row.get("recall_at_10") is None and before["rank"] is None:
            labelled = {
                "outcome": "INSUFFICIENT DATA",
                "ab_class": "INSUFFICIENT DATA",
                "diagnoses": ["Insufficient data"],
            }
        entry = {
            **item,
            "source_recall_at_10": before["recall_at_10"],
            "candidate_recall_at_10": after["recall_at_10"],
            "before": before,
            "after": after,
            "outcome": labelled["outcome"],
            "diagnoses": labelled["diagnoses"],
        }
        buckets[labelled["outcome"]].append(entry)
        rec = after["recall_at_10"]
        if rec is not None and float(rec) > 0:
            buckets["passed"].append(entry)
        else:
            buckets["failed"].append(entry)
    fixed = buckets["FIXED"]
    improved = buckets["IMPROVED"]
    return {
        "experiment_id": experiment_id,
        "n_items": len(items),
        "passed": buckets["passed"],
        "failed": buckets["failed"],
        "fixed_failures": fixed,
        "improved": improved,
        "regressed": buckets["REGRESSED"],
        "unchanged": buckets["UNCHANGED"],
        "new_regressions": buckets["REGRESSED"],
        "not_recorded": buckets["INSUFFICIENT DATA"],
        "insufficient": buckets["INSUFFICIENT DATA"],
        "counts": {
            "size": len(items),
            "passed": len(buckets["passed"]),
            "failed": len(buckets["failed"]),
            "fixed_failures": len(fixed),
            "improved": len(improved) + len(fixed),
            "regressed": len(buckets["REGRESSED"]),
            "unchanged": len(buckets["UNCHANGED"]),
            "insufficient": len(buckets["INSUFFICIENT DATA"]),
            "not_recorded": len(buckets["INSUFFICIENT DATA"]),
            "new_regressions": len(buckets["REGRESSED"]),
        },
        "note": (
            "Outcomes use stored rank, Recall@10, reciprocal rank, nDCG, and candidate-hit "
            "signals. Recall@10 unchanged is UNCHANGED, not a pass of the set. No retrieval is re-run."
        ),
    }
