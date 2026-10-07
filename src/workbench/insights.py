"""Factual experiment observations. Never ranks configurations."""

from __future__ import annotations

from typing import Any

from src.workbench.experiments import compare_experiments

_BANNED = ("better", "best", "wins", "winner", "recommended", "superior")


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "Not recorded"
    return f"{float(value) * 100:.2f}%"


def _fmt_ratio(value: Any) -> str:
    if value is None:
        return "Not recorded"
    return f"{float(value):.4f}"


def _fmt_ms(value: Any) -> str:
    if value is None:
        return "Not recorded"
    return f"{float(value):.1f} ms"


def _direction(delta: float, *, kind: str) -> str:
    if delta > 0:
        if kind == "pp":
            return f"Increased by {abs(delta) * 100:.2f} percentage points."
        if kind == "ms":
            return f"Increased by {abs(delta):.1f} ms."
        return f"Increased by {abs(delta):.4f}."
    if delta < 0:
        if kind == "pp":
            return f"Decreased by {abs(delta) * 100:.2f} percentage points."
        if kind == "ms":
            return f"Decreased by {abs(delta):.1f} ms."
        return f"Decreased by {abs(delta):.4f}."
    return "No observed change."


def _metric_block(
    comparison: dict[str, Any],
    key: str,
    label: str,
    *,
    kind: str,
    fmt,
) -> dict[str, Any]:
    left = comparison["baseline"].get(key)
    right = comparison["candidate"].get(key)
    payload = (comparison.get("metrics") or {}).get(key)
    if left is None or right is None:
        return {
            "label": label,
            "baseline": fmt(left),
            "candidate": fmt(right),
            "delta": None,
            "statement": "Insufficient data.",
            "insufficient": True,
        }
    delta = payload["delta"] if payload else float(right) - float(left)
    return {
        "label": label,
        "baseline": fmt(left),
        "candidate": fmt(right),
        "delta": delta,
        "statement": _direction(delta, kind=kind),
        "insufficient": False,
    }


def candidate_vs_rerank_observation(comparison: dict[str, Any]) -> str | None:
    cand = (comparison.get("metrics") or {}).get("candidate_recall_at_50")
    rec = (comparison.get("metrics") or {}).get("recall_at_10")
    if not cand or not rec:
        return None
    cand_delta = cand["delta"]
    rec_delta = rec["delta"]
    if abs(cand_delta) < 1e-12 and rec_delta > 0:
        return "The candidate recall was unchanged while final Recall@10 increased."
    if abs(cand_delta) < 1e-12 and rec_delta < 0:
        return "The candidate recall was unchanged while final Recall@10 decreased."
    if abs(cand_delta) < 1e-12 and abs(rec_delta) < 1e-12:
        return "Candidate recall and final Recall@10 were both unchanged."
    return None


def observed_tradeoff(comparison: dict[str, Any]) -> str | None:
    rec = (comparison.get("metrics") or {}).get("recall_at_10")
    lat = (comparison.get("metrics") or {}).get("mean_latency_ms")
    if not rec or not lat:
        return None
    if rec["delta"] > 0 and lat["delta"] > 0:
        return "Observed trade-off: final retrieval metrics increased while measured latency also increased."
    if rec["delta"] < 0 and lat["delta"] < 0:
        return "Observed trade-off: final retrieval metrics decreased while measured latency also decreased."
    if rec["delta"] > 0 and lat["delta"] < 0:
        return "Final retrieval metrics increased while measured latency decreased."
    if rec["delta"] < 0 and lat["delta"] > 0:
        return "Final retrieval metrics decreased while measured latency increased."
    return None


def experiment_insights(baseline_id: str, candidate_id: str, *, root: Any = None) -> dict[str, Any]:
    comparison = compare_experiments(baseline_id, candidate_id, root=root)
    if comparison.get("error"):
        return {"error": comparison["error"], "observations": []}
    observations = [
        _metric_block(comparison, "candidate_recall_at_50", "Candidate Recall@50", kind="pp", fmt=_fmt_pct),
        _metric_block(comparison, "recall_at_10", "Recall@10", kind="pp", fmt=_fmt_pct),
        _metric_block(comparison, "mrr_at_10", "MRR@10", kind="abs", fmt=_fmt_ratio),
        _metric_block(comparison, "ndcg_at_10", "nDCG@10", kind="abs", fmt=_fmt_ratio),
        _metric_block(comparison, "mean_latency_ms", "Latency", kind="ms", fmt=_fmt_ms),
    ]
    statements = []
    for item in observations:
        statements.append(f"{item['label']}: {item['baseline']} → {item['candidate']}. {item['statement']}")
    stage = candidate_vs_rerank_observation(comparison)
    tradeoff = observed_tradeoff(comparison)
    extra = [text for text in (stage, tradeoff) if text]
    joined = " ".join(statements + extra).lower()
    for word in _BANNED:
        if word in joined:
            extra = [text for text in extra if word not in text.lower()]
    return {
        "baseline_id": baseline_id,
        "candidate_id": candidate_id,
        "observations": observations,
        "statements": statements,
        "candidate_vs_rerank": stage,
        "tradeoff": tradeoff,
        "notes": extra,
    }
