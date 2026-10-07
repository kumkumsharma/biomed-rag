from dataclasses import dataclass

from src.config import EVALUATION
from src.data.loader import GoldenQA
from src.data.preprocessing import prepare_corpus
from src.evaluation.answer_evaluation import (
    AnswerEvaluationQuery,
    LLMJudge,
    attach_reference_answers,
    build_live_judge_prompt,
    evaluate_live_answer_quality,
    evaluate_one_answer,
    load_or_create_answer_sample,
    parse_judge_response,
    parse_live_judge_response,
    run_answer_evaluation,
    select_answer_sample,
    select_manual_review_queries,
    summarize_answer_rows,
)
from src.evaluation.dataset import EvaluationQuery
from src.generation.answer_generator import AnswerGenerator
from src.generation.rag import GroundedQA
from src.retrieval.hybrid import HybridRetriever
from src.validation.citations import validate_citations


@dataclass(frozen=True)
class FakeHit:
    passage_id: object
    score: float
    text: str


class FakeRetriever:
    def __init__(self, ranking: list[FakeHit]):
        self.ranking = ranking

    def search(self, query: str, top_k: int | None = None) -> list[FakeHit]:
        if top_k is None:
            return list(self.ranking)
        return list(self.ranking[:top_k])


class FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def _retrieval_queries(n: int = 25) -> list[EvaluationQuery]:
    return [
        EvaluationQuery(i, f"question {i}", (i * 10,))
        for i in range(n)
    ]


def _golden(n: int = 25) -> list[GoldenQA]:
    return [
        GoldenQA(i, f"question {i}", f"reference {i}", (i * 10,))
        for i in range(n)
    ]


def _hybrid(hits: list[FakeHit]) -> HybridRetriever:
    passages = prepare_corpus(
        [{"id": hit.passage_id, "passage": hit.text} for hit in hits]
        or [{"id": 0, "passage": "empty"}]
    )
    retriever = FakeRetriever(hits)
    return HybridRetriever(passages, bm25_retriever=retriever, dense_retriever=retriever)


def test_select_answer_sample_is_the_first_20_of_the_fixed_set():
    queries = _retrieval_queries(25)
    first = select_answer_sample(queries, n=20)
    second = select_answer_sample(queries, n=20)
    assert [query.question_id for query in first] == list(range(20))
    assert first == second


def test_attach_reference_answers_preserves_ids_and_golden_answers():
    queries = attach_reference_answers(select_answer_sample(_retrieval_queries(3), n=2), _golden(3))
    assert queries[0].question_id == 0
    assert queries[0].reference_answer == "reference 0"
    assert queries[0].relevant_passage_ids == (0,)


def test_load_or_create_answer_sample_reuses_saved_ids(tmp_path):
    path = tmp_path / "answer_evaluation_20.json"
    created = load_or_create_answer_sample(
        n=3,
        path=path,
        retrieval_queries=_retrieval_queries(10),
        golden=_golden(10),
    )
    reused = load_or_create_answer_sample(
        n=3,
        path=path,
        retrieval_queries=_retrieval_queries(10),
        golden=[],
    )
    assert [query.question_id for query in created] == [0, 1, 2]
    assert reused == created


def test_judge_prompt_contains_question_reference_generated_and_evidence():
    llm = FakeLLM(
        '{"correctness": 4, "groundedness": 5, "context_relevance": 3, "reason": "ok"}'
    )
    evidence = [FakeHit(1001, 1.0, "Papilin is secreted.")]
    verdict = LLMJudge(llm=llm).evaluate(
        question="Is papilin secreted?",
        reference_answer="Yes.",
        generated_answer="Papilin is secreted. [PASSAGE_ID: 1001]",
        evidence=evidence,
    )
    prompt = llm.prompts[0]
    assert "Is papilin secreted?" in prompt
    assert "Yes." in prompt
    assert "Papilin is secreted. [PASSAGE_ID: 1001]" in prompt
    assert "Papilin is secreted." in prompt
    assert "[PASSAGE_ID: 1001]" in prompt
    assert "Do not check" in prompt or "citation" in prompt.lower()
    assert verdict.correctness == 4
    assert verdict.groundedness == 5
    assert verdict.context_relevance == 3
    assert verdict.parsed is True


def test_parse_judge_response_reads_json_and_fenced_json():
    plain = parse_judge_response(
        '{"correctness": 2, "groundedness": 3, "context_relevance": 4, "reason": "partial"}'
    )
    fenced = parse_judge_response(
        "```json\n"
        '{"correctness": 2, "groundedness": 3, "context_relevance": 4, "reason": "partial"}\n'
        "```"
    )
    assert plain.parsed is True
    assert fenced.parsed is True
    assert plain.correctness == fenced.correctness == 2


def test_invalid_judge_output_is_handled_without_raising():
    verdict = parse_judge_response("the answer looks good")
    assert verdict.parsed is False
    assert verdict.correctness is None
    assert verdict.groundedness is None
    assert verdict.context_relevance is None
    assert "not valid JSON" in verdict.reason


def test_out_of_range_judge_scores_are_not_accepted():
    verdict = parse_judge_response(
        '{"correctness": 9, "groundedness": 0, "context_relevance": 3, "reason": "bad"}'
    )
    assert verdict.parsed is False
    assert verdict.correctness is None
    assert verdict.groundedness is None
    assert verdict.context_relevance == 3


def test_citation_validation_stays_deterministic_and_ignores_the_judge():
    hits = [FakeHit(1001, 1.0, "Papilin is a secreted protein.")]
    qa = GroundedQA(
        _hybrid(hits),
        generator=AnswerGenerator(llm=FakeLLM("Invented claim. [PASSAGE_ID: 999]")),
    )
    judge = LLMJudge(
        llm=FakeLLM(
            '{"correctness": 5, "groundedness": 5, "context_relevance": 5, "reason": "ignore me"}'
        )
    )
    query = AnswerEvaluationQuery(1, "Is papilin secreted?", "Yes.", (1001,))
    row = evaluate_one_answer(query, qa, judge, top_k=1)

    independent = validate_citations(row.generated_answer, [1001])
    assert row.citation_is_valid is False
    assert independent.is_valid is False
    assert row.citation_invalid_ids == ("999",)
    assert row.correctness == 5


def test_answer_evaluation_smoke_does_not_call_a_real_llm(tmp_path):
    hits = [FakeHit(1001, 1.0, "Papilin is a secreted extracellular matrix protein.")]
    qa = GroundedQA(
        _hybrid(hits),
        generator=AnswerGenerator(
            llm=FakeLLM("Papilin is secreted. [PASSAGE_ID: 1001]")
        ),
    )
    judge_llm = FakeLLM(
        '{"correctness": 4, "groundedness": 4, "context_relevance": 5, "reason": "supported"}'
    )
    queries = [
        AnswerEvaluationQuery(1, "Is papilin secreted?", "Yes, papilin is secreted.", (1001,)),
        AnswerEvaluationQuery(2, "Is papilin secreted?", "Yes.", (1001,)),
    ]
    result = run_answer_evaluation(
        qa=qa,
        judge=LLMJudge(llm=judge_llm),
        queries=queries,
        passages=prepare_corpus([{"id": 1001, "passage": hits[0].text}]),
        top_k=1,
        queries_path=tmp_path / "answer_evaluation_20.json",
        details_path=tmp_path / "details.json",
        summary_path=tmp_path / "summary.json",
        manual_review_path=tmp_path / "manual.md",
        manual_review_json_path=tmp_path / "manual.json",
    )
    assert len(result["rows"]) == 2
    assert result["summary"]["n_queries"] == 2
    assert result["summary"]["average_correctness"] == 4
    assert result["summary"]["citation_validity_rate"] == 1.0
    assert result["details_path"].exists()
    assert "supported answer" in result["manual_review_path"].read_text(encoding="utf-8")
    assert judge_llm.prompts
    assert all("Is papilin secreted?" in prompt for prompt in judge_llm.prompts)


def test_summarize_skips_unparsed_judge_scores():
    hits = [FakeHit(1001, 1.0, "Papilin is secreted.")]
    qa = GroundedQA(
        _hybrid(hits),
        generator=AnswerGenerator(llm=FakeLLM("Papilin is secreted. [PASSAGE_ID: 1001]")),
    )

    class FlakyJudge:
        def __init__(self):
            self.calls = 0

        def evaluate(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return parse_judge_response("not json")
            return parse_judge_response(
                '{"correctness": 4, "groundedness": 4, "context_relevance": 4, "reason": "ok"}'
            )

    judge = FlakyJudge()
    rows = [
        evaluate_one_answer(
            AnswerEvaluationQuery(i, "Is papilin secreted?", "Yes.", (1001,)),
            qa,
            judge,
        )
        for i in (0, 1)
    ]
    summary = summarize_answer_rows(rows)
    assert summary["n_judge_parsed"] == 1
    assert summary["average_correctness"] == 4
    assert summary["citation_validity_rate"] == 1.0


def test_manual_review_picks_first_middle_and_last():
    queries = [
        AnswerEvaluationQuery(i, f"q{i}", f"a{i}", (i,))
        for i in range(20)
    ]
    selected = select_manual_review_queries(queries, n=3)
    assert [query.question_id for query in selected] == [0, 10, 19]


def test_evaluation_config_uses_a_20_query_judge_sample():
    assert EVALUATION.n_answer_queries == 20
    assert EVALUATION.n_manual_review == 3
    assert EVALUATION.judge_model_name
    assert EVALUATION.judge_temperature == 0.0


def test_live_judge_skips_insufficient_evidence_and_does_not_call_llm():
    llm = FakeLLM('{"groundedness": 5, "context_relevance": 5, "reason": "no"}')
    quality = evaluate_live_answer_quality(
        question="Does metformin interfere thyroxine absorption?",
        generated_answer="Insufficient evidence in the provided documents to answer this question.",
        evidence=(),
        insufficient_evidence=True,
        generation_llm_called=False,
        llm=llm,
    )
    assert quality["status"] == "Not evaluated"
    assert quality["evaluated"] is False
    assert quality["llm_calls"] == 0
    assert quality["latency_ms"] is None
    assert llm.prompts == []
    assert quality["model_name"] == EVALUATION.judge_model_name


def test_live_judge_scores_groundedness_and_context_without_correctness():
    llm = FakeLLM(
        '{"groundedness": 4, "context_relevance": 5, "reason": "supported by evidence"}'
    )
    evidence = [FakeHit(1001, 1.0, "Papilin is secreted.")]
    quality = evaluate_live_answer_quality(
        question="Is papilin secreted?",
        generated_answer="Yes. Papilin is secreted [PASSAGE_ID: 1001].",
        evidence=evidence,
        insufficient_evidence=False,
        generation_llm_called=True,
        llm=llm,
    )
    assert quality["evaluated"] is True
    assert quality["groundedness"] == 4
    assert quality["context_relevance"] == 5
    assert quality["llm_calls"] == 1
    assert quality["latency_ms"] is not None
    assert quality["model_name"] == EVALUATION.judge_model_name
    prompt = llm.prompts[0]
    assert "Is papilin secreted?" in prompt
    assert "Papilin is secreted" in prompt
    assert "Reference answer:" not in prompt
    assert "- correctness:" not in prompt
    parsed = parse_live_judge_response(llm.response)
    assert parsed["parsed"] is True
    assert "correctness" not in parsed
    live_prompt = build_live_judge_prompt(
        question="Is papilin secreted?",
        generated_answer="Yes.",
        evidence=evidence,
    )
    assert "Do not use outside knowledge." in live_prompt
