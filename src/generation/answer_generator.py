"""Grounded answer generation from selected evidence only.

The LLM never sees the full corpus. It receives the original question and
the selected evidence passages. API keys are read from the environment.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from src.config import GENERATION
from src.validation.citations import format_citation

INSUFFICIENT_EVIDENCE_ANSWER = (
    "Insufficient evidence in the provided documents to answer this question."
)


class LLMClient(Protocol):
    """Minimal text-completion interface for grounded generation."""

    def complete(self, prompt: str) -> str: ...


class OpenAICompatibleClient:
    """Chat Completions client. Credentials come from the environment."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        self.model_name = model_name or GENERATION.model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
        self.base_url = (
            base_url
            or os.environ.get("LLM_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        ).rstrip("/")
        self.temperature = GENERATION.temperature if temperature is None else temperature
        self.max_tokens = GENERATION.max_tokens if max_tokens is None else max_tokens

    def complete(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError(
                "Set OPENAI_API_KEY or LLM_API_KEY to call the language model."
            )

        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        try:
            return str(body["choices"][0]["message"]["content"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("LLM response did not contain message content.") from exc


GROUNDED_PROMPT_TEMPLATE = """You are a biomedical question-answering assistant.

Answer ONLY using the supplied evidence.
Do not use outside knowledge.
Do not invent facts.
Do not invent passage IDs.
Cite supporting passage IDs using the exact format [PASSAGE_ID: <id>].
If the evidence is insufficient, explicitly say that there is insufficient evidence in the provided documents.

Question:
{question}

Evidence:
{evidence}"""


def build_grounded_prompt(question: str, evidence: Sequence[Any]) -> str:
    """Build a prompt that restricts the model to the supplied evidence."""

    blocks = []
    for item in evidence:
        passage_id = getattr(item, "passage_id")
        text = getattr(item, "text")
        blocks.extend(["", format_citation(passage_id), str(text)])
    evidence_text = "\n".join(blocks).strip()
    return GROUNDED_PROMPT_TEMPLATE.replace("{question}", question.strip()).replace(
        "{evidence}", evidence_text
    )


@dataclass(frozen=True)
class GeneratedAnswer:
    question: str
    answer: str
    prompt: str
    llm_called: bool
    insufficient_evidence: bool


class AnswerGenerator:
    """Generate a grounded answer from selected evidence passages."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    def generate(self, question: str, evidence: Sequence[Any]) -> GeneratedAnswer:
        if not isinstance(question, str):
            raise TypeError("question must be a string")

        evidence = list(evidence)
        if len(evidence) < GENERATION.min_evidence_count:
            return GeneratedAnswer(
                question=question,
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                prompt="",
                llm_called=False,
                insufficient_evidence=True,
            )

        prompt = build_grounded_prompt(question, evidence)
        answer = self._client().complete(prompt)
        return GeneratedAnswer(
            question=question,
            answer=answer,
            prompt=prompt,
            llm_called=True,
            insufficient_evidence=False,
        )

    def _client(self) -> LLMClient:
        if self._llm is None:
            self._llm = OpenAICompatibleClient()
        return self._llm
