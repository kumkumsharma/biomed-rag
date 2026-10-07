"""Canonical evaluation dataset. The 100-query BioASQ set is first-class."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import EVALUATION

ROOT = Path(__file__).resolve().parents[2]

DATASET_ID = "bioasq-mini-100"
DATASET_VERSION = "1.0"


def evaluation_dataset(*, root: Path | None = None) -> dict[str, Any]:
    """Metadata for the fixed 100-query benchmark. Does not reload the corpus."""

    base = root or ROOT
    queries_path = base / EVALUATION.queries_path
    gold = {}
    gold_path = base / EVALUATION.gold_quality_path
    if gold_path.exists():
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
    n_from_file = None
    created = None
    if queries_path.exists():
        payload = json.loads(queries_path.read_text(encoding="utf-8"))
        n_from_file = payload.get("n_queries")
        created = payload.get("created") or payload.get("created_at")
    return {
        "id": DATASET_ID,
        "name": "BioASQ Mini 100-query benchmark",
        "version": DATASET_VERSION,
        "n_queries": n_from_file or EVALUATION.n_queries,
        "n_gold_ids": gold.get("total_gold_ids") or 1142,
        "valid_gold_ids": gold.get("valid_gold_ids"),
        "invalid_gold_ids": gold.get("invalid_gold_ids"),
        "domain": "biomedical / BioASQ (rag-mini-bioasq)",
        "creation_date": created or "fixed evaluation set (evaluation_100.json)",
        "queries_path": EVALUATION.queries_path,
        "protocol": (
            "Same 100 questions, same gold passage IDs, same metrics "
            "(Recall@k, MRR@10, nDCG@10, latency). "
            "Every retrieval/reranking experiment must use this dataset unless another is selected."
        ),
        "reusable": True,
    }
