from dataclasses import dataclass

from src.generation.answer_generator import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    AnswerGenerator,
    build_grounded_prompt,
)
from src.validation.citations import format_citation


@dataclass(frozen=True)
class FakePassage:
    passage_id: object
    text: str
    score: float = 1.0


class FakeLLM:
    def __init__(self, response: str = "Papilin is secreted. [PASSAGE_ID: 1001]"):
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def _evidence():
    return [
        FakePassage(1001, "Papilin is a secreted extracellular matrix protein."),
        FakePassage(42, "Insulin regulates blood glucose."),
    ]


def test_prompt_contains_question_evidence_ids_and_grounding_rules():
    question = "Is papilin secreted?"
    prompt = build_grounded_prompt(question, _evidence())

    assert question in prompt
    assert "Papilin is a secreted extracellular matrix protein." in prompt
    assert "Insulin regulates blood glucose." in prompt
    assert format_citation(1001) in prompt
    assert format_citation(42) in prompt
    assert "Answer ONLY using the supplied evidence." in prompt
    assert "Do not use outside knowledge." in prompt
    assert "Do not invent facts." in prompt
    assert "Do not invent passage IDs." in prompt
    assert "insufficient evidence in the provided documents" in prompt


def test_generator_returns_llm_output_and_passes_the_grounded_prompt():
    llm = FakeLLM("Yes, papilin is secreted. [PASSAGE_ID: 1001]")
    result = AnswerGenerator(llm=llm).generate("Is papilin secreted?", _evidence())

    assert result.llm_called is True
    assert result.insufficient_evidence is False
    assert result.answer == "Yes, papilin is secreted. [PASSAGE_ID: 1001]"
    assert result.question == "Is papilin secreted?"
    assert llm.prompts == [result.prompt]
    assert "Is papilin secreted?" in result.prompt
    assert format_citation(1001) in result.prompt


def test_no_evidence_does_not_call_the_llm():
    llm = FakeLLM("should not be used")
    result = AnswerGenerator(llm=llm).generate("Is papilin secreted?", [])

    assert llm.prompts == []
    assert result.llm_called is False
    assert result.insufficient_evidence is True
    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.prompt == ""
