"""Lightweight final-answer evaluation with a separate LLM judge.

Hybrid RRF retrieves evidence, GroundedQA generates the answer, and
src/validation/citations.py checks citations deterministically. The judge
scores correctness, groundedness, and context relevance. It does not
validate citations.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import EVALUATION, GENERATION, RETRIEVAL
from src.data.loader import GoldenQA, get_passage_collection, load_golden_qa
from src.data.preprocessing import Passage
from src.evaluation.dataset import EvaluationQuery, load_or_create_fixed_queries
from src.evaluation.run_evaluation import build_evaluation_retrievers
from src.generation.answer_generator import AnswerGenerator, LLMClient, OpenAICompatibleClient
from src.generation.rag import GroundedQA
from src.retrieval.hybrid import HybridRetriever
from src.validation.citations import format_citation, validate_citations

ANSWER_SAMPLE_METHOD = "first_20_of_fixed_100_query_evaluation_set"
MANUAL_REVIEW_METHOD = "first_middle_last_of_answer_sample"
OBSERVATION_LABELS = (
    "supported answer",
    "missing evidence",
    "incorrect answer",
    "citation problem",
    "retrieval problem",
)

JUDGE_PROMPT = """You are an evaluation judge for biomedical RAG answers.

Score the generated answer using only the information below. Do not check
whether citation tokens such as [PASSAGE_ID: 123] are valid; citation
validity is determined separately by application code.

Score each dimension as an integer from 1 to 5:
- correctness: how well the generated answer matches the reference/golden answer
- groundedness: how well the generated answer is supported by the retrieved evidence
- context_relevance: how relevant the retrieved evidence is to the question

Return JSON only, with this exact shape:
{"correctness": 1, "groundedness": 1, "context_relevance": 1, "reason": "short explanation"}

Question:
{question}

Reference answer:
{reference_answer}

Generated answer:
{generated_answer}

Retrieved evidence:
{evidence}
"""


@dataclass(frozen=True)
class AnswerEvaluationQuery:
    """One answer-evaluation question with the golden reference answer."""

    question_id: Any
    question: str
    reference_answer: str
    relevant_passage_ids: tuple[Any, ...]


@dataclass(frozen=True)
class JudgeVerdict:
    correctness: int | None
    groundedness: int | None
    context_relevance: int | None
    reason: str
    parsed: bool
    raw_response: str
    prompt: str


@dataclass(frozen=True)
class AnswerEvaluationRow:
    query_id: Any
    question: str
    reference_answer: str
    retrieved_passage_ids: tuple[Any, ...]
    retrieved_evidence: tuple[str, ...]
    generated_answer: str
    citation_is_valid: bool
    citation_cited_ids: tuple[Any, ...]
    citation_valid_ids: tuple[Any, ...]
    citation_invalid_ids: tuple[Any, ...]
    citation_missing: bool
    insufficient_evidence: bool
    correctness: int | None
    groundedness: int | None
    context_relevance: int | None
    judge_reason: str
    judge_parsed: bool
    generation_latency_ms: float
    judge_latency_ms: float


def llm_api_key_configured() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY"))


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (bytes, str, bool)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def _id_key(value: Any) -> str:
    return str(value).strip()


def format_evidence_for_judge(evidence: Sequence[Any]) -> str:
    if not evidence:
        return "(no retrieved evidence)"
    blocks = []
    for item in evidence:
        passage_id = getattr(item, "passage_id")
        text = getattr(item, "text")
        blocks.append(f"{format_citation(passage_id)}\n{text}")
    return "\n\n".join(blocks)


def build_judge_prompt(
    *,
    question: str,
    reference_answer: str,
    generated_answer: str,
    evidence: Sequence[Any],
) -> str:
    return (
        JUDGE_PROMPT.replace("{question}", question.strip())
        .replace("{reference_answer}", reference_answer.strip())
        .replace("{generated_answer}", generated_answer.strip())
        .replace("{evidence}", format_evidence_for_judge(evidence))
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        payload = json.loads(stripped)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None


def _score_1_to_5(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            return None
    if number < 1 or number > 5:
        return None
    return number


def parse_judge_response(text: str, *, prompt: str = "") -> JudgeVerdict:
    """Parse judge JSON. Invalid output yields null scores instead of raising."""

    raw = "" if text is None else str(text)
    payload = _extract_json_object(raw)
    if payload is None:
        return JudgeVerdict(
            correctness=None,
            groundedness=None,
            context_relevance=None,
            reason="Judge output was not valid JSON.",
            parsed=False,
            raw_response=raw,
            prompt=prompt,
        )
    correctness = _score_1_to_5(payload.get("correctness"))
    groundedness = _score_1_to_5(payload.get("groundedness"))
    context_relevance = _score_1_to_5(payload.get("context_relevance"))
    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)
    parsed = (
        correctness is not None
        and groundedness is not None
        and context_relevance is not None
    )
    if not parsed and not reason.strip():
        reason = "Judge JSON was missing or out-of-range scores."
    return JudgeVerdict(
        correctness=correctness,
        groundedness=groundedness,
        context_relevance=context_relevance,
        reason=reason.strip(),
        parsed=parsed,
        raw_response=raw,
        prompt=prompt,
    )


class AnswerJudge(Protocol):
    def evaluate(
        self,
        *,
        question: str,
        reference_answer: str,
        generated_answer: str,
        evidence: Sequence[Any],
    ) -> JudgeVerdict: ...


class LLMJudge:
    """Separate from the production answer-generation LLM."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm

    def evaluate(
        self,
        *,
        question: str,
        reference_answer: str,
        generated_answer: str,
        evidence: Sequence[Any],
    ) -> JudgeVerdict:
        prompt = build_judge_prompt(
            question=question,
            reference_answer=reference_answer,
            generated_answer=generated_answer,
            evidence=evidence,
        )
        raw = self._client().complete(prompt)
        return parse_judge_response(raw, prompt=prompt)

    def _client(self) -> LLMClient:
        if self._llm is None:
            self._llm = OpenAICompatibleClient(
                model_name=EVALUATION.judge_model_name,
                temperature=EVALUATION.judge_temperature,
                max_tokens=EVALUATION.judge_max_tokens,
            )
        return self._llm


LIVE_JUDGE_PROMPT = """You are an evaluation judge for biomedical RAG answers.

Score the generated answer using only the retrieved evidence below.
Do not use outside knowledge.
Do not invent facts.
Do not score correctness; there is no reference answer for this live query.

Score each dimension as an integer from 1 to 5:
- groundedness: how well the generated answer is supported by the retrieved evidence
- context_relevance: how relevant the retrieved evidence is to the question

Return JSON only, with this exact shape:
{"groundedness": 1, "context_relevance": 1, "reason": "short explanation"}

Question:
{question}

Generated answer:
{generated_answer}

Retrieved evidence:
{evidence}
"""


def build_live_judge_prompt(
    *,
    question: str,
    generated_answer: str,
    evidence: Sequence[Any],
) -> str:
    return (
        LIVE_JUDGE_PROMPT.replace("{question}", question.strip())
        .replace("{generated_answer}", generated_answer.strip())
        .replace("{evidence}", format_evidence_for_judge(evidence))
    )


def parse_live_judge_response(text: str) -> dict[str, Any]:
    """Parse live-judge JSON. Does not require or display correctness."""

    payload = _extract_json_object("" if text is None else str(text))
    if payload is None:
        return {
            "groundedness": None,
            "context_relevance": None,
            "reason": "Judge output was not valid JSON.",
            "parsed": False,
        }
    groundedness = _score_1_to_5(payload.get("groundedness"))
    context_relevance = _score_1_to_5(payload.get("context_relevance"))
    reason = payload.get("reason", "")
    if not isinstance(reason, str):
        reason = str(reason)
    return {
        "groundedness": groundedness,
        "context_relevance": context_relevance,
        "reason": reason.strip(),
        "parsed": groundedness is not None and context_relevance is not None,
    }


def should_run_live_judge(
    *,
    insufficient_evidence: bool,
    generation_llm_called: bool,
    evidence: Sequence[Any],
) -> bool:
    return (
        not insufficient_evidence
        and bool(generation_llm_called)
        and bool(evidence)
    )


def evaluate_live_answer_quality(
    *,
    question: str,
    generated_answer: str,
    evidence: Sequence[Any],
    insufficient_evidence: bool,
    generation_llm_called: bool,
    llm: LLMClient | None = None,
) -> dict[str, Any]:
    """Score one Explorer answer for groundedness and context relevance only."""

    model_name = EVALUATION.judge_model_name
    skipped = {
        "evaluated": False,
        "status": "Not evaluated",
        "groundedness": None,
        "context_relevance": None,
        "reason": "",
        "parsed": False,
        "model_name": model_name,
        "latency_ms": None,
        "llm_calls": 0,
    }
    if not should_run_live_judge(
        insufficient_evidence=insufficient_evidence,
        generation_llm_called=generation_llm_called,
        evidence=evidence,
    ):
        return skipped

    prompt = build_live_judge_prompt(
        question=question,
        generated_answer=generated_answer,
        evidence=evidence,
    )
    client = llm or OpenAICompatibleClient(
        model_name=model_name,
        temperature=EVALUATION.judge_temperature,
        max_tokens=EVALUATION.judge_max_tokens,
    )
    started = time.perf_counter()
    raw = client.complete(prompt)
    latency_ms = (time.perf_counter() - started) * 1000.0
    parsed = parse_live_judge_response(raw)
    return {
        "evaluated": True,
        "status": "Evaluated",
        "groundedness": parsed["groundedness"],
        "context_relevance": parsed["context_relevance"],
        "reason": parsed["reason"],
        "parsed": parsed["parsed"],
        "model_name": model_name,
        "latency_ms": latency_ms,
        "llm_calls": 1,
    }


def select_answer_sample(
    queries: Sequence[EvaluationQuery],
    *,
    n: int = 20,
) -> list[EvaluationQuery]:
    """Take the first n queries from the fixed retrieval evaluation set."""

    if n < 1:
        raise ValueError("n must be >= 1")
    if len(queries) < n:
        raise ValueError(f"Need {n} queries from the fixed set, found {len(queries)}")
    return list(queries[:n])


def _manual_review_indexes(length: int, n: int) -> list[int]:
    if length < 1:
        raise ValueError("Cannot select a manual review set from no queries")
    limit = min(n, length)
    if limit == 1:
        return [0]
    if limit == 2:
        return [0, length - 1]
    return [0, length // 2, length - 1]


def select_manual_review_queries(
    queries: Sequence[AnswerEvaluationQuery],
    *,
    n: int = 3,
) -> list[AnswerEvaluationQuery]:
    """Pick a small deterministic subset for human inspection."""

    return [queries[index] for index in _manual_review_indexes(len(queries), n)]


def attach_reference_answers(
    queries: Sequence[EvaluationQuery],
    golden: Sequence[GoldenQA],
) -> list[AnswerEvaluationQuery]:
    by_id = {_id_key(item.question_id): item for item in golden}
    attached: list[AnswerEvaluationQuery] = []
    for query in queries:
        record = by_id.get(_id_key(query.question_id))
        if record is None:
            raise KeyError(f"No golden answer for question_id={query.question_id!r}")
        attached.append(
            AnswerEvaluationQuery(
                question_id=query.question_id,
                question=query.question,
                reference_answer=record.answer,
                relevant_passage_ids=tuple(query.relevant_passage_ids),
            )
        )
    return attached


def answer_sample_to_json(queries: Sequence[AnswerEvaluationQuery]) -> dict[str, Any]:
    return {
        "n_queries": len(queries),
        "source_queries_path": EVALUATION.queries_path,
        "selection_method": ANSWER_SAMPLE_METHOD,
        "query_ids": [query.question_id for query in queries],
        "queries": [
            {
                "question_id": query.question_id,
                "question": query.question,
                "reference_answer": query.reference_answer,
                "relevant_passage_ids": list(query.relevant_passage_ids),
            }
            for query in queries
        ],
    }


def save_answer_sample(queries: Sequence[AnswerEvaluationQuery], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(answer_sample_to_json(queries), indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def load_answer_sample(path: str | Path) -> list[AnswerEvaluationQuery]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    queries = []
    for item in payload["queries"]:
        queries.append(
            AnswerEvaluationQuery(
                question_id=item["question_id"],
                question=item["question"],
                reference_answer=item["reference_answer"],
                relevant_passage_ids=tuple(item["relevant_passage_ids"]),
            )
        )
    return queries


def load_or_create_answer_sample(
    *,
    n: int | None = None,
    path: str | Path | None = None,
    retrieval_queries: Sequence[EvaluationQuery] | None = None,
    golden: Sequence[GoldenQA] | None = None,
) -> list[AnswerEvaluationQuery]:
    """Load the saved 20-query set, or select it from the fixed 100-query set."""

    n = EVALUATION.n_answer_queries if n is None else n
    destination = Path(path or EVALUATION.answer_queries_path)
    if destination.exists():
        queries = load_answer_sample(destination)
        if len(queries) != n:
            raise ValueError(f"{destination} has {len(queries)} queries, expected {n}")
        return queries

    source = (
        list(retrieval_queries)
        if retrieval_queries is not None
        else load_or_create_fixed_queries()
    )
    selected = select_answer_sample(source, n=n)
    records = list(golden) if golden is not None else load_golden_qa()
    queries = attach_reference_answers(selected, records)
    save_answer_sample(queries, destination)
    return queries


def evaluate_one_answer(
    query: AnswerEvaluationQuery,
    qa: GroundedQA,
    judge: AnswerJudge,
    *,
    top_k: int | None = None,
) -> AnswerEvaluationRow:
    """Generate a grounded answer, validate citations, then ask the judge."""

    started = time.perf_counter()
    result = qa.ask(query.question, top_k=top_k)
    generation_latency_ms = (time.perf_counter() - started) * 1000.0

    citations = validate_citations(
        result.answer,
        [item.passage_id for item in result.evidence],
    )

    started = time.perf_counter()
    verdict = judge.evaluate(
        question=query.question,
        reference_answer=query.reference_answer,
        generated_answer=result.answer,
        evidence=result.evidence,
    )
    judge_latency_ms = (time.perf_counter() - started) * 1000.0

    return AnswerEvaluationRow(
        query_id=query.question_id,
        question=query.question,
        reference_answer=query.reference_answer,
        retrieved_passage_ids=tuple(item.passage_id for item in result.evidence),
        retrieved_evidence=tuple(item.text for item in result.evidence),
        generated_answer=result.answer,
        citation_is_valid=citations.is_valid,
        citation_cited_ids=citations.cited_ids,
        citation_valid_ids=citations.valid_ids,
        citation_invalid_ids=citations.invalid_ids,
        citation_missing=citations.missing_citations,
        insufficient_evidence=result.insufficient_evidence,
        correctness=verdict.correctness,
        groundedness=verdict.groundedness,
        context_relevance=verdict.context_relevance,
        judge_reason=verdict.reason,
        judge_parsed=verdict.parsed,
        generation_latency_ms=generation_latency_ms,
        judge_latency_ms=judge_latency_ms,
    )


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def summarize_answer_rows(rows: Sequence[AnswerEvaluationRow]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty answer evaluation")
    correctness = [row.correctness for row in rows if row.correctness is not None]
    groundedness = [row.groundedness for row in rows if row.groundedness is not None]
    relevance = [row.context_relevance for row in rows if row.context_relevance is not None]
    return {
        "n_queries": len(rows),
        "query_ids": [_jsonable(row.query_id) for row in rows],
        "average_correctness": _mean(correctness),
        "average_groundedness": _mean(groundedness),
        "average_context_relevance": _mean(relevance),
        "n_judge_parsed": sum(1 for row in rows if row.judge_parsed),
        "citation_validity_rate": sum(1 for row in rows if row.citation_is_valid) / len(rows),
        "average_generation_latency_ms": _mean([row.generation_latency_ms for row in rows]),
        "average_judge_latency_ms": _mean([row.judge_latency_ms for row in rows]),
        "median_generation_latency_ms": statistics.median(
            [row.generation_latency_ms for row in rows]
        ),
        "median_judge_latency_ms": statistics.median([row.judge_latency_ms for row in rows]),
        "judge_model_name": EVALUATION.judge_model_name,
        "retrieval": "Hybrid RRF",
        "generation_model_name": GENERATION.model_name,
    }


def _row_to_json(row: AnswerEvaluationRow) -> dict[str, Any]:
    return {
        "query_id": _jsonable(row.query_id),
        "question": row.question,
        "reference_answer": row.reference_answer,
        "retrieved_passage_ids": [_jsonable(item) for item in row.retrieved_passage_ids],
        "retrieved_evidence": list(row.retrieved_evidence),
        "generated_answer": row.generated_answer,
        "citation_validation": {
            "is_valid": row.citation_is_valid,
            "cited_ids": [_jsonable(item) for item in row.citation_cited_ids],
            "valid_ids": [_jsonable(item) for item in row.citation_valid_ids],
            "invalid_ids": [_jsonable(item) for item in row.citation_invalid_ids],
            "missing_citations": row.citation_missing,
        },
        "insufficient_evidence": row.insufficient_evidence,
        "correctness": row.correctness,
        "groundedness": row.groundedness,
        "context_relevance": row.context_relevance,
        "judge_reason": row.judge_reason,
        "judge_parsed": row.judge_parsed,
        "generation_latency_ms": float(row.generation_latency_ms),
        "judge_latency_ms": float(row.judge_latency_ms),
    }


def format_manual_review_markdown(
    rows: Sequence[AnswerEvaluationRow],
    *,
    sample_size: int,
) -> str:
    lines = [
        "# Manual review of RAG answers",
        "",
        f"Reviewing {len(rows)} queries from the {sample_size}-query answer-evaluation sample.",
        "Record observations below. Do not convert these notes into an automatic score.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## Query {row.query_id}",
                "",
                f"**Question:** {row.question}",
                "",
                f"**Reference answer:** {row.reference_answer}",
                "",
                f"**Generated answer:** {row.generated_answer}",
                "",
                "**Citation validation:** "
                f"valid={str(row.citation_is_valid).lower()}; "
                f"cited={list(row.citation_cited_ids)}; "
                f"invalid={list(row.citation_invalid_ids)}; "
                f"missing={str(row.citation_missing).lower()}",
                "",
                "**LLM judge:** "
                f"correctness={row.correctness} "
                f"groundedness={row.groundedness} "
                f"context_relevance={row.context_relevance}",
                "",
                f"**Judge reason:** {row.judge_reason}",
                "",
                "**Retrieved evidence:**",
            ]
        )
        if not row.retrieved_passage_ids:
            lines.append("- (none)")
        else:
            for passage_id, text in zip(row.retrieved_passage_ids, row.retrieved_evidence):
                lines.append(f"- {format_citation(passage_id)} {text}")
        lines.extend(["", "**Observations:**"])
        for label in OBSERVATION_LABELS:
            lines.append(f"- [ ] {label}")
        lines.extend(["", "**Notes:**", "", ""])
    return "\n".join(lines).rstrip() + "\n"


def manual_review_to_json(
    rows: Sequence[AnswerEvaluationRow],
    *,
    sample_size: int,
) -> dict[str, Any]:
    return {
        "n_reviewed": len(rows),
        "sample_size": sample_size,
        "selection_method": MANUAL_REVIEW_METHOD,
        "query_ids": [_jsonable(row.query_id) for row in rows],
        "reviews": [
            {
                **_row_to_json(row),
                "observations": {label: None for label in OBSERVATION_LABELS},
                "notes": "",
            }
            for row in rows
        ],
    }


def save_answer_evaluation_outputs(
    *,
    rows: Sequence[AnswerEvaluationRow],
    details_path: str | Path,
    summary_path: str | Path,
    manual_review_path: str | Path,
    manual_review_json_path: str | Path,
    n_passages: int,
    top_k: int,
    queries_path: str | Path,
    index_build_seconds: Mapping[str, float] | None = None,
) -> dict[str, Path]:
    details_destination = Path(details_path)
    summary_destination = Path(summary_path)
    review_destination = Path(manual_review_path)
    review_json_destination = Path(manual_review_json_path)
    for path in (
        details_destination,
        summary_destination,
        review_destination,
        review_json_destination,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    summary = summarize_answer_rows(rows)
    details_payload = {
        "n_passages": n_passages,
        "top_k": top_k,
        "queries_path": str(queries_path),
        "retrieval": "Hybrid RRF",
        "dense_model_name": RETRIEVAL.dense_model_name,
        "generation_model_name": GENERATION.model_name,
        "judge_model_name": EVALUATION.judge_model_name,
        "index_build_seconds": dict(index_build_seconds or {}),
        "rows": [_row_to_json(row) for row in rows],
    }
    summary_payload = {
        **summary,
        "n_passages": n_passages,
        "top_k": top_k,
        "queries_path": str(queries_path),
        "index_build_seconds": dict(index_build_seconds or {}),
    }
    review_detail_rows = [
        rows[index] for index in _manual_review_indexes(len(rows), EVALUATION.n_manual_review)
    ]

    details_destination.write_text(json.dumps(details_payload, indent=2) + "\n", encoding="utf-8")
    summary_destination.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
    review_destination.write_text(
        format_manual_review_markdown(review_detail_rows, sample_size=len(rows)),
        encoding="utf-8",
    )
    review_json_destination.write_text(
        json.dumps(
            manual_review_to_json(review_detail_rows, sample_size=len(rows)),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "details_path": details_destination,
        "summary_path": summary_destination,
        "manual_review_path": review_destination,
        "manual_review_json_path": review_json_destination,
    }


def run_answer_evaluation(
    *,
    qa: GroundedQA | None = None,
    judge: AnswerJudge | None = None,
    queries: Sequence[AnswerEvaluationQuery] | None = None,
    passages: Sequence[Passage] | None = None,
    top_k: int | None = None,
    queries_path: str | Path | None = None,
    details_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    manual_review_path: str | Path | None = None,
    manual_review_json_path: str | Path | None = None,
    index_build_seconds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Run grounded generation + citation checks + LLM judge on the 20-query sample."""

    top_k = GENERATION.evidence_top_k if top_k is None else top_k
    queries_path = Path(queries_path or EVALUATION.answer_queries_path)
    if queries is None:
        queries = load_or_create_answer_sample(path=queries_path)

    timings = dict(index_build_seconds or {})
    if qa is None:
        if passages is None:
            print("Loading the full BioASQ passage corpus...", flush=True)
            passages = get_passage_collection()
        print(
            f"Building Hybrid RRF indexes once over {len(passages)} passages...",
            flush=True,
        )
        retrievers, built = build_evaluation_retrievers(passages, top_k=top_k)
        timings.update(built)
        hybrid = retrievers["Hybrid RRF"]
        if not isinstance(hybrid, HybridRetriever):
            raise TypeError("Answer evaluation requires Hybrid RRF")
        qa = GroundedQA(hybrid, generator=AnswerGenerator())
        print(
            "Index build seconds: "
            + ", ".join(f"{name}={value:.2f}" for name, value in timings.items()),
            flush=True,
        )
    if judge is None:
        judge = LLMJudge()

    print(f"Evaluating {len(queries)} grounded answers with the LLM judge...", flush=True)
    rows = []
    for index, query in enumerate(queries, start=1):
        row = evaluate_one_answer(query, qa, judge, top_k=top_k)
        rows.append(row)
        print(
            f"  {index}/{len(queries)} query_id={query.question_id} "
            f"correctness={row.correctness} citations_valid={row.citation_is_valid}",
            flush=True,
        )

    n_passages = len(passages) if passages is not None else 0
    paths = save_answer_evaluation_outputs(
        rows=rows,
        details_path=details_path or EVALUATION.answer_details_path,
        summary_path=summary_path or EVALUATION.answer_summary_path,
        manual_review_path=manual_review_path or EVALUATION.manual_review_path,
        manual_review_json_path=manual_review_json_path or EVALUATION.manual_review_json_path,
        n_passages=n_passages,
        top_k=top_k,
        queries_path=queries_path,
        index_build_seconds=timings,
    )
    summary = summarize_answer_rows(rows)
    print(
        "Answer evaluation: "
        f"n={summary['n_queries']} "
        f"correctness={summary['average_correctness']} "
        f"groundedness={summary['average_groundedness']} "
        f"context_relevance={summary['average_context_relevance']} "
        f"citation_validity_rate={summary['citation_validity_rate']:.4f}",
        flush=True,
    )
    for label, path in paths.items():
        print(f"{label}: {path}", flush=True)
    return {
        "rows": rows,
        "summary": summary,
        "n_passages": n_passages,
        **paths,
        "queries_path": queries_path,
        "index_build_seconds": timings,
    }


def main() -> None:
    if not llm_api_key_configured():
        print(
            "OPENAI_API_KEY or LLM_API_KEY is not set. "
            "The 20-query answer evaluation was not run."
        )
        raise SystemExit(1)
    run_answer_evaluation()


if __name__ == "__main__":
    main()
