from src.data.preprocessing import Passage
from src.evaluation.pipeline_diagnostic import (
    categorize_question,
    diagnose_loss,
    inspect_gold_passages,
    preview_text,
)


def test_categorize_papilin_as_fact():
    assert categorize_question("Is the protein Papilin secreted?") == (
        "Fact / definition / biological relationship"
    )


def test_categorize_metformin_as_treatment():
    assert categorize_question("Does metformin interfere thyroxine absorption?") == (
        "Treatment / intervention"
    )


def test_categorize_association_before_treatment_wording():
    assert categorize_question(
        "What is the association of estrogen replacement therapy and intracranial meningioma risk?"
    ) == "Risk / association / prognosis"


def test_categorize_hirschsprung_as_mechanism():
    assert categorize_question(
        "Is Hirschsprung disease a mendelian or a multifactorial disorder?"
    ) == "Cause / mechanism / pathophysiology"


def test_inspect_gold_passages_keeps_full_text_and_does_not_chunk():
    passages = [Passage(passage_id=42, text="Papilin is a secreted extracellular matrix protein.")]
    reports = inspect_gold_passages([42], passages)
    assert reports[0]["exists_in_index"] is True
    assert reports[0]["chunked"] is False
    assert reports[0]["truncated"] is False
    assert reports[0]["text_unchanged"] is True
    assert reports[0]["n_chunks"] == 1


def test_inspect_gold_passages_flags_missing_ids():
    reports = inspect_gold_passages([99], [Passage(passage_id=1, text="x")])
    assert reports[0]["exists_in_index"] is False
    assert reports[0]["failure_point"] == "DATA / PARSING"


def test_inspect_gold_passages_flags_literal_nan_text_as_data():
    reports = inspect_gold_passages([7], [Passage(passage_id=7, text="nan")])
    assert reports[0]["exists_in_index"] is True
    assert reports[0]["empty_or_nan_text"] is True
    assert reports[0]["failure_point"] == "DATA / PARSING"


def test_diagnose_nan_text_is_data_not_retrieval():
    assert diagnose_loss(
        gold_exists=True,
        empty_or_nan_text=True,
        in_bm25_pool=False,
        in_dense_pool=False,
        rrf_rank=None,
        medcpt_rank=None,
    ) == "DATA / PARSING"


def test_diagnose_dual_miss_is_first_stage():
    assert diagnose_loss(
        gold_exists=True,
        in_bm25_pool=False,
        in_dense_pool=False,
        rrf_rank=None,
        medcpt_rank=None,
    ) == "FIRST-STAGE RETRIEVAL / FUSION"


def test_diagnose_pool_hit_but_not_top10_is_reranking():
    assert diagnose_loss(
        gold_exists=True,
        in_bm25_pool=True,
        in_dense_pool=True,
        rrf_rank=12,
        medcpt_rank=None,
    ) == "RERANKING"


def test_preview_text_collapses_whitespace():
    assert preview_text("a\n\nb", limit=10) == "a b"
