"""Prompt inspector records for prompts that exist in the implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.config import EVALUATION, GENERATION
from src.evaluation.answer_evaluation import JUDGE_PROMPT
from src.generation.answer_generator import GROUNDED_PROMPT_TEMPLATE, INSUFFICIENT_EVIDENCE_ANSWER
from src.generation.query_expansion import EXPANSION_PROMPT
from src.inspection.prompts import llm_prompt_catalog
from src.validation.citations import format_citation


def prompt_inspector_tabs(
    *,
    expansion_available: bool = True,
    judge_available: bool = True,
) -> list[dict[str, Any]]:
    """Tabs that correspond to real LLM prompts. No invented templates."""

    catalog = {item["component"]: item for item in llm_prompt_catalog()}
    tabs: list[dict[str, Any]] = []
    expansion = catalog.get("llm_query_expansion")
    if expansion_available and expansion and expansion.get("template"):
        tabs.append(
            {
                "tab": "Query Expansion",
                "stage": "Query Expansion",
                "model": expansion.get("model") or GENERATION.query_expansion_model_name,
                "system_instructions": None,
                "user_template": expansion.get("template") or EXPANSION_PROMPT,
                "variables": {
                    "count": GENERATION.query_expansion_count,
                    "query": "<runtime question>",
                },
                "notes": expansion.get("notes"),
                "exists": True,
            }
        )
    generation = catalog.get("answer_generation")
    if generation and generation.get("template"):
        tabs.append(
            {
                "tab": "Answer Generation",
                "stage": "Answer Generation",
                "model": generation.get("model") or GENERATION.model_name,
                "system_instructions": None,
                "user_template": generation.get("template") or GROUNDED_PROMPT_TEMPLATE,
                "variables": {
                    "query": "<runtime question>",
                    "evidence": "<retrieved passages with [PASSAGE_ID: id]>",
                    "citation_format": format_citation("<id>"),
                    "insufficient_evidence_rule": INSUFFICIENT_EVIDENCE_ANSWER,
                },
                "notes": (
                    "Single user message; there is no separate system prompt. "
                    + str(generation.get("notes") or "")
                ),
                "exists": True,
            }
        )
    judge = catalog.get("llm_judge")
    if judge_available and judge and judge.get("template"):
        tabs.append(
            {
                "tab": "LLM Judge",
                "stage": "Offline LLM Judge",
                "model": judge.get("model") or EVALUATION.judge_model_name,
                "system_instructions": None,
                "user_template": judge.get("template") or JUDGE_PROMPT,
                "variables": {
                    "question": "<benchmark question>",
                    "reference_answer": "<gold answer>",
                    "generated_answer": "<model answer>",
                    "evidence": "<evidence passages>",
                },
                "notes": (
                    "Not on the Ask critical path. Live playground evaluation is heuristic. "
                    + str(judge.get("notes") or "")
                ),
                "exists": True,
            }
        )
    return tabs


def non_llm_prompt_notes() -> list[dict[str, str]]:
    return [
        {
            "component": "Query Classification",
            "note": "Not LLM-prompt driven. Heuristic classifier in `src/query/classify.py`.",
        },
        {
            "component": "Terminology Normalization",
            "note": "Not LLM-prompt driven. Local term map in `src/query/terminology.py`.",
        },
        {
            "component": "Live Answer Quality",
            "note": "Deterministic/heuristic checks. Does not call an LLM judge.",
        },
    ]


def runtime_generation_variables(result: Any) -> dict[str, Any]:
    evidence = list(getattr(result, "evidence", ()) or [])
    return {
        "query": getattr(result, "question", ""),
        "evidence": [
            {"passage_id": getattr(item, "passage_id", None), "chars": len(str(getattr(item, "text", "") or ""))}
            for item in evidence
        ],
        "citation_format": format_citation("<id>"),
        "insufficient_evidence_rule": INSUFFICIENT_EVIDENCE_ANSWER,
        "prompt_present": bool(getattr(result, "prompt", "")),
    }


def bind_prompt_variables(template: str | None, variables: Mapping[str, Any]) -> str | None:
    if not template:
        return None
    rendered = template
    for key, value in variables.items():
        rendered = rendered.replace("{" + str(key) + "}", str(value))
    return rendered
