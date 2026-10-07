from src.config import EVALUATION
from src.data.loader import GoldenQA
from src.evaluation.dataset import (
    EvaluationQuery,
    load_evaluation_queries,
    load_or_create_fixed_queries,
    save_evaluation_queries,
    select_fixed_queries,
)


def _qa(question_id, question, relevant_passage_ids):
    return GoldenQA(
        question_id=question_id,
        question=question,
        answer="yes",
        relevant_passage_ids=tuple(relevant_passage_ids),
    )


def test_evaluation_config_requests_exactly_100_queries():
    assert EVALUATION.n_queries == 100
    assert EVALUATION.top_k == 10


def test_select_fixed_queries_is_deterministic_and_numeric_sorted():
    records = [
        _qa("10", "q10", [100]),
        _qa(2, "q2", [20]),
        _qa(1, "q1", [10]),
        _qa("3", "q3", [30]),
        _qa(0, "empty", []),
        _qa(4, "q4", [40]),
    ]
    first = select_fixed_queries(records, n=4)
    second = select_fixed_queries(list(reversed(records)), n=4)

    assert [query.question_id for query in first] == [1, 2, "3", 4]
    assert [query.question_id for query in first] == [query.question_id for query in second]
    assert first[0].relevant_passage_ids == (10,)


def test_select_fixed_queries_skips_empty_relevant_ids_and_keeps_original_ids():
    records = [
        _qa(1, "skip me", []),
        _qa(2, "keep me", [20598273, "42"]),
        _qa(3, "also keep", [7]),
    ]
    selected = select_fixed_queries(records, n=2)
    assert [query.question_id for query in selected] == [2, 3]
    assert selected[0].relevant_passage_ids == (20598273, "42")
    assert selected[0].question == "keep me"


def test_select_fixed_queries_takes_the_first_100_after_sorting():
    records = [_qa(i, f"q{i}", [i * 10] if i != 50 else []) for i in range(120, 0, -1)]
    selected = select_fixed_queries(records, n=100)
    assert len(selected) == 100
    assert selected[0].question_id == 1
    assert 50 not in {query.question_id for query in selected}
    assert selected[-1].question_id == 101


def test_save_and_load_preserve_query_and_relevant_passage_ids(tmp_path):
    queries = [
        EvaluationQuery(question_id=2, question="q2", relevant_passage_ids=(20, 21)),
        EvaluationQuery(question_id="3", question="q3", relevant_passage_ids=("30",)),
    ]
    path = tmp_path / "evaluation_100.json"
    save_evaluation_queries(queries, path)
    loaded = load_evaluation_queries(path)
    assert loaded == queries
    payload = path.read_text(encoding="utf-8")
    assert "query_ids" in payload
    assert "relevant_passage_ids" in payload


def test_load_or_create_reuses_the_saved_query_set(tmp_path):
    path = tmp_path / "evaluation_100.json"
    records = [_qa(i, f"q{i}", [i]) for i in range(1, 6)]
    created = load_or_create_fixed_queries(n=3, path=path, records=records)
    different_records = [_qa(i, f"other{i}", [i + 100]) for i in range(10, 20)]
    reused = load_or_create_fixed_queries(n=3, path=path, records=different_records)
    assert reused == created
    assert [query.question_id for query in reused] == [1, 2, 3]
