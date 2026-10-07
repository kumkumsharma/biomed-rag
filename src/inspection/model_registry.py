"""Production vs experimental model registry for the System → Models page."""

from __future__ import annotations

from typing import Any

from src.config import EVALUATION, GENERATION, RERANKER, RETRIEVAL, UPGRADE


def model_registry(*, medcpt_active: bool | None = None, expansion_active: bool | None = None) -> dict[str, Any]:
    """Catalog of models. Experimental entries are never implied as production."""

    medcpt_on = bool(medcpt_active) if medcpt_active is not None else False
    expansion_on = bool(expansion_active) if expansion_active is not None else False
    production = [
        {
            "name": "BM25",
            "id": "bm25",
            "role": "lexical retrieval",
            "lane": "Production",
            "active": True,
            "neural": False,
            "notes": "No neural model. First-stage sparse retrieval.",
        },
        {
            "name": RETRIEVAL.dense_model_name,
            "id": "minilm",
            "role": "dense first-stage retrieval",
            "lane": "Production",
            "active": True,
            "neural": True,
            "notes": "384-d sentence embeddings. Production dense encoder.",
        },
        {
            "name": RERANKER.model_name,
            "id": "medcpt-ce",
            "role": "cross-encoder reranking",
            "lane": "Production",
            "active": medcpt_on,
            "neural": True,
            "notes": (
                "ncbi/MedCPT-Cross-Encoder. Current request: "
                + ("active." if medcpt_on else "not executed on the current request.")
            ),
        },
        {
            "name": GENERATION.model_name,
            "id": "gpt-4o-mini",
            "role": "query expansion · grounded answer generation · LLM judge (offline only)",
            "lane": "Production",
            "active": True,
            "neural": True,
            "notes": (
                "Query expansion is request-scoped "
                f"({'ON' if expansion_on else 'OFF'} this request). "
                f"LLM judge model `{EVALUATION.judge_model_name}` is not on the Ask critical path."
            ),
        },
    ]
    experimental = [
        {
            "name": "naver/splade-v3-distilbert",
            "id": "splade",
            "role": "learned sparse retrieval",
            "lane": "Experimental",
            "active": bool(UPGRADE.sparse_enabled),
            "neural": True,
            "notes": "Not a production dependency. Dual-miss recovery experiment only.",
        },
        {
            "name": "PubMedBERT (S-PubMedBert-MS-MARCO)",
            "id": "pubmedbert",
            "role": "dense retrieval experiment",
            "lane": "Experimental",
            "active": bool(UPGRADE.biomedical_embeddings_enabled),
            "neural": True,
            "notes": "Measured vs MiniLM; not swapped into production.",
        },
        {
            "name": "MedCPT Query Encoder",
            "id": "medcpt-query",
            "role": "dense retrieval experiment",
            "lane": "Experimental",
            "active": False,
            "neural": True,
            "notes": "Bi-encoder experiment. Not active in production.",
        },
        {
            "name": "MedCPT Article Encoder",
            "id": "medcpt-article",
            "role": "dense retrieval experiment",
            "lane": "Experimental",
            "active": False,
            "neural": True,
            "notes": "Bi-encoder experiment. Not active in production.",
        },
        {
            "name": "NeuML/biomedbert-base-reranker",
            "id": "biomedbert-reranker",
            "role": "reranker experiment",
            "lane": "Experimental",
            "active": False,
            "neural": True,
            "notes": "Worse Top-10 than MedCPT on the same pool. Not promoted.",
        },
    ]
    return {
        "production": production,
        "experimental": experimental,
        "active_production": [item for item in production if item["active"]],
        "inactive_experimental": [item for item in experimental if not item["active"]],
    }
