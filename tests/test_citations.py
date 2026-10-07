from src.validation.citations import (
    extract_cited_passage_ids,
    format_citation,
    validate_citations,
)


def test_valid_citation_passes():
    result = validate_citations("Papilin is secreted. [PASSAGE_ID: 1001]", [1001, 42])

    assert result.has_citations is True
    assert result.missing_citations is False
    assert result.valid_ids == (1001,)
    assert result.invalid_ids == ()
    assert result.is_valid is True


def test_multiple_valid_citations_pass():
    answer = "Both findings hold. [PASSAGE_ID: 1001] [PASSAGE_ID: 42]"
    result = validate_citations(answer, [1001, 42, 99])

    assert result.cited_ids == ("1001", "42")
    assert result.valid_ids == (1001, 42)
    assert result.invalid_ids == ()
    assert result.is_valid is True


def test_invalid_non_retrieved_passage_id_is_detected():
    result = validate_citations(
        "Claim. [PASSAGE_ID: 1001] [PASSAGE_ID: 99999]",
        [1001, 42],
    )

    assert result.valid_ids == (1001,)
    assert result.invalid_ids == ("99999",)
    assert result.is_valid is False


def test_no_citation_is_handled_explicitly():
    result = validate_citations("This answer cites nothing.", [1001])

    assert result.has_citations is False
    assert result.missing_citations is True
    assert result.cited_ids == ()
    assert result.valid_ids == ()
    assert result.invalid_ids == ()
    assert result.is_valid is False
    assert extract_cited_passage_ids("This answer cites nothing.") == []


def test_duplicate_citations_are_unique_and_still_valid():
    answer = "Repeated support. [PASSAGE_ID: 1001] and again [PASSAGE_ID: 1001]"
    result = validate_citations(answer, [1001])

    assert extract_cited_passage_ids(answer) == ["1001", "1001"]
    assert result.cited_ids == ("1001",)
    assert result.valid_ids == (1001,)
    assert result.invalid_ids == ()
    assert result.is_valid is True


def test_citation_format_matches_validator():
    assert format_citation(20598273) == "[PASSAGE_ID: 20598273]"
    result = validate_citations(format_citation(20598273), [20598273])
    assert result.is_valid is True
