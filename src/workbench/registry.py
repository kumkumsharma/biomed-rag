"""Evaluator, prompt, and model registries for the workbench."""

from __future__ import annotations

from typing import Any

from src.config import EVALUATION
from src.inspection.model_registry import model_registry as base_model_registry
from src.inspection.prompt_inspector import prompt_inspector_tabs
from src.inspection.prompts import llm_prompt_catalog


def evaluator_registry() -> list[dict[str, Any]]:
    return [
        {
            "name": "Heuristic evaluator",
            "id": "heuristic_live_evaluator",
            "version": "1",
            "type": "Heuristic evaluator",
            "description": "Deterministic overlap and citation checks for a live query. Does not call an LLM.",
            "input": "question, answer, evidence, citations",
            "output": "faithfulness, answer_relevancy, context_precision, citation_correctness",
            "score_range": "0–1",
            "model": None,
            "prompt": None,
            "critical_path": False,
        },
        {
            "name": "Citation validator",
            "id": "code_citation_validator",
            "version": "1",
            "type": "Code evaluator",
            "description": "Parses [PASSAGE_ID: id] tokens and compares them to supplied evidence IDs.",
            "input": "answer text, evidence IDs",
            "output": "valid / invalid / missing",
            "score_range": "boolean + ID lists",
            "model": None,
            "prompt": None,
            "critical_path": True,
        },
        {
            "name": "LLM-as-judge (offline)",
            "id": "llm_judge_offline",
            "version": "1",
            "type": "LLM-as-judge",
            "description": "Offline judge for the 20-query answer sample. Not on the Ask critical path.",
            "input": "question, reference answer, generated answer, evidence",
            "output": "correctness, groundedness, context_relevance (integers)",
            "score_range": "1–5",
            "model": EVALUATION.judge_model_name,
            "prompt": "LLM Judge (offline)",
            "critical_path": False,
            "configurable": True,
        },
        {
            "name": "Manual review",
            "id": "human_manual_review",
            "version": "1",
            "type": "Human evaluator",
            "description": "Stored successful / failure examples from the 20-query answer sample.",
            "input": "generated answer, reference, evidence",
            "output": "review notes",
            "score_range": "qualitative",
            "model": None,
            "prompt": None,
            "critical_path": False,
        },
    ]


def prompt_registry() -> list[dict[str, Any]]:
    rows = []
    for item in prompt_inspector_tabs():
        rows.append(
            {
                "id": item["tab"].lower().replace(" ", "_"),
                "name": item["tab"],
                "version": "1",
                "stage": item["stage"],
                "model": item["model"],
                "system_prompt": item.get("system_instructions"),
                "user_template": item.get("user_template"),
                "variables": item.get("variables") or {},
                "created_date": "implementation (source templates)",
                "exists": True,
            }
        )
    return rows


def workbench_model_registry(*, production_medcpt: bool = True) -> dict[str, Any]:
    base = base_model_registry(medcpt_active=production_medcpt, expansion_active=False)
    experiments = {
        "bm25": ["bm25", "hybrid", "hybrid-medcpt", "hybrid-qe"],
        "minilm": ["dense-minilm", "hybrid", "hybrid-medcpt"],
        "medcpt-ce": ["hybrid-medcpt", "hybrid-pool100", "hybrid-qe-medcpt", "hybrid-splade-medcpt"],
        "gpt-4o-mini": ["hybrid-qe", "hybrid-qe-medcpt"],
        "splade": ["hybrid-splade-medcpt"],
        "pubmedbert": ["hybrid-pubmedbert"],
        "medcpt-query": ["hybrid-medcpt-biencoder"],
        "medcpt-article": ["hybrid-medcpt-biencoder"],
        "biomedbert-reranker": ["hybrid-biomedbert-reranker"],
    }
    production = []
    for item in base["production"]:
        entry = dict(item)
        entry["status"] = "Production"
        entry["experiments_used_in"] = experiments.get(item["id"], [])
        entry["dimensions"] = 384 if item["id"] == "minilm" else None
        production.append(entry)
    experimental = []
    for item in base["experimental"]:
        entry = dict(item)
        entry["status"] = "Experimental"
        entry["experiments_used_in"] = experiments.get(item["id"], [])
        experimental.append(entry)
    return {
        "production": production,
        "experimental": experimental,
        "retired": [],
    }


def llm_judge_spec() -> dict[str, Any]:
    catalog = {item["component"]: item for item in llm_prompt_catalog()}
    judge = catalog.get("llm_judge") or {}
    return {
        "enabled_on_ask_path": False,
        "model": EVALUATION.judge_model_name,
        "temperature": EVALUATION.judge_temperature,
        "max_tokens": EVALUATION.judge_max_tokens,
        "dataset": "20-query answer sample (not the 100-query retrieval set unless configured)",
        "metrics": [
            "Answer correctness",
            "Faithfulness",
            "Answer relevance",
            "Citation correctness (code validator, not the judge)",
            "Completeness",
            "Claim support",
        ],
        "prompt_version": "1",
        "template": judge.get("template"),
        "note": "Completeness and claim support are heuristic unless a judge run stores them. Do not imply they were scored by the LLM if they were not.",
    }
