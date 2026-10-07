"""Catalog of actual LLM prompts used by the application.

Templates are imported from the modules that send them. Deterministic
components are marked as having no LLM prompt.
"""

from __future__ import annotations

from typing import Any

from src.config import EVALUATION, GENERATION, UPGRADE
from src.evaluation.answer_evaluation import JUDGE_PROMPT, LIVE_JUDGE_PROMPT
from src.generation.answer_generator import GROUNDED_PROMPT_TEMPLATE
from src.generation.query_expansion import EXPANSION_PROMPT


def llm_prompt_catalog() -> list[dict[str, Any]]:
    """Return inspectable prompt records. No invented templates."""

    return [
        {
            "name": "Answer Generation",
            "component": "answer_generation",
            "source_file": "src/generation/answer_generator.py",
            "model": GENERATION.model_name,
            "temperature": GENERATION.temperature,
            "max_tokens": GENERATION.max_tokens,
            "deterministic": False,
            "enabled": True,
            "input_variables": ["question", "evidence"],
            "template": GROUNDED_PROMPT_TEMPLATE,
            "notes": "Single user message. Evidence passages are appended with [PASSAGE_ID: id].",
        },
        {
            "name": "LLM Query Expansion",
            "component": "llm_query_expansion",
            "source_file": "src/generation/query_expansion.py",
            "model": GENERATION.query_expansion_model_name,
            "temperature": GENERATION.temperature,
            "max_tokens": 200,
            "deterministic": False,
            "enabled": bool(GENERATION.query_expansion_enabled),
            "input_variables": ["count", "query"],
            "template": EXPANSION_PROMPT,
            "notes": "Original query is kept. LLM proposes alternate search strings as JSON.",
        },
        {
            "name": "Deterministic Query Expansion",
            "component": "deterministic_query_expansion",
            "source_file": "src/query/expansion.py",
            "model": None,
            "temperature": None,
            "max_tokens": None,
            "deterministic": True,
            "enabled": UPGRADE.expansion_enabled,
            "input_variables": [],
            "template": None,
            "notes": "Local synonym substitution. No LLM prompt.",
        },
        {
            "name": "Query Decomposition",
            "component": "query_decomposition",
            "source_file": "src/query/decomposition.py",
            "model": None,
            "temperature": None,
            "max_tokens": None,
            "deterministic": True,
            "enabled": UPGRADE.decomposition_enabled,
            "input_variables": [],
            "template": None,
            "notes": "Heuristic aspect split for multi-part questions. No LLM prompt.",
        },
        {
            "name": "LLM Judge (offline)",
            "component": "llm_judge",
            "source_file": "src/evaluation/answer_evaluation.py",
            "model": EVALUATION.judge_model_name,
            "temperature": EVALUATION.judge_temperature,
            "max_tokens": EVALUATION.judge_max_tokens,
            "deterministic": False,
            "enabled": True,
            "input_variables": ["question", "reference_answer", "generated_answer", "evidence"],
            "template": JUDGE_PROMPT,
            "notes": "Scores correctness, groundedness, and context relevance as integers 1–5.",
        },
        {
            "name": "LLM Judge (live Explorer)",
            "component": "live_llm_judge",
            "source_file": "src/evaluation/answer_evaluation.py",
            "model": EVALUATION.judge_model_name,
            "temperature": EVALUATION.judge_temperature,
            "max_tokens": EVALUATION.judge_max_tokens,
            "deterministic": False,
            "enabled": True,
            "input_variables": ["question", "generated_answer", "evidence"],
            "template": LIVE_JUDGE_PROMPT,
            "notes": "No correctness score: live Explorer has no reference answer.",
        },
    ]


def prompt_by_name(name: str) -> dict[str, Any] | None:
    for item in llm_prompt_catalog():
        if item["name"] == name or item["component"] == name:
            return item
    return None
