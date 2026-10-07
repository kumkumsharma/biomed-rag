import json
from pathlib import Path

import numpy as np

from src.data.preprocessing import prepare_corpus
from src.evaluation.dataset import EvaluationQuery
from src.evaluation.objective_failure import (
    BM25_ONLY_MISS,
    BOTH_RETRIEVERS_MISS,
    DENSE_ONLY_MISS,
    FUSION_CUTOFF,
    NOT_IN_CORPUS,
    TOP50_RETRIEVED,
    biomedical_dense_recovery_rate,
    diagnostic_signals,
    earliest_recovery,
    fusion_cutoff_flag,
    hypotheses_from_observation,
    lexical_transformations,
    objective_failure_category,
    observed_facts,
    run_objective_failure_study,
)
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever


class KeywordEmbedder:
    _vocab = ("hirschsprung", "intestine", "insulin", "glucose", "papilin", "secreted", "egfr")

    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([1.0 if token in lowered else 0.0 for token in self._vocab])
        return np.asarray(vectors, dtype=float)


class AltEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "papilin" in lowered or "secreted" in lowered else 0.0,
                    1.0 if "hirschsprung" in lowered else 0.0,
                    1.0 if "unrelated" in lowered else 0.0,
                ]
            )
        return np.asarray(vectors, dtype=float)


def test_objective_category_not_in_corpus():
    assert (
        objective_failure_category(
            in_corpus=False, bm25_rank=None, dense_rank=None, rrf_rank=None
        )
        == NOT_IN_CORPUS
    )


def test_objective_category_top50_is_not_a_failure():
    assert (
        objective_failure_category(
            in_corpus=True, bm25_rank=3, dense_rank=8, rrf_rank=12
        )
        == TOP50_RETRIEVED
    )
    assert (
        objective_failure_category(
            in_corpus=True, bm25_rank=None, dense_rank=1, rrf_rank=50
        )
        == TOP50_RETRIEVED
    )


def test_objective_category_bm25_only_miss():
    assert (
        objective_failure_category(
            in_corpus=True, bm25_rank=None, dense_rank=4, rrf_rank=80
        )
        == BM25_ONLY_MISS
    )


def test_objective_category_dense_only_miss():
    assert (
        objective_failure_category(
            in_corpus=True, bm25_rank=2, dense_rank=None, rrf_rank=70
        )
        == DENSE_ONLY_MISS
    )


def test_objective_category_both_retrievers_miss():
    assert (
        objective_failure_category(
            in_corpus=True, bm25_rank=None, dense_rank=None, rrf_rank=None
        )
        == BOTH_RETRIEVERS_MISS
    )


def test_objective_category_fusion_cutoff_requires_both_first_stages():
    assert (
        objective_failure_category(
            in_corpus=True, bm25_rank=3, dense_rank=9, rrf_rank=61
        )
        == FUSION_CUTOFF
    )
    assert (
        objective_failure_category(
            in_corpus=True, bm25_rank=3, dense_rank=9, rrf_rank=None
        )
        == FUSION_CUTOFF
    )


def test_none_rank_means_absent_from_probe_k_not_full_collection():
    category = objective_failure_category(
        in_corpus=True, bm25_rank=None, dense_rank=None, rrf_rank=None, probe_k=100
    )
    assert category == BOTH_RETRIEVERS_MISS


def test_query_type_does_not_change_objective_category():
    both = objective_failure_category(
        in_corpus=True, bm25_rank=None, dense_rank=None, rrf_rank=None
    )
    assert both == BOTH_RETRIEVERS_MISS
    signals = diagnostic_signals(
        question="What are the risk factors and treatment of Hirschsprung disease?",
        passage_text="Unrelated zinc biology.",
        in_corpus=True,
    )
    assert signals["multi_part_signal"] is True
    assert signals["has_term_map_match"] is True
    hypotheses = hypotheses_from_observation(
        category=both, signals=signals, fusion_cutoff=False
    )
    labels = {item["label"] for item in hypotheses}
    assert "possible_query_formulation_issue" in labels
    assert "possible_terminology_gap" in labels
    assert both == BOTH_RETRIEVERS_MISS


def test_hypotheses_are_not_the_failure_field():
    signals = diagnostic_signals(
        question="Is papilin secreted?",
        passage_text="Completely unrelated cardiovascular outcome trial.",
        in_corpus=True,
    )
    hyps = hypotheses_from_observation(
        category=BOTH_RETRIEVERS_MISS, signals=signals, fusion_cutoff=False
    )
    assert all("label" in item and "evidence" in item and "confidence" in item for item in hyps)
    assert all("caused by" not in item["evidence"].lower() for item in hyps)
    if signals["token_overlap"] < 0.15:
        assert any(item["label"] == "possible_lexical_mismatch" for item in hyps)


def test_fusion_cutoff_flag_is_independent_of_asymmetric_miss_label():
    assert fusion_cutoff_flag(bm25_rank=None, dense_rank=4, rrf_rank=80) is True
    assert fusion_cutoff_flag(bm25_rank=2, dense_rank=None, rrf_rank=70) is True
    assert fusion_cutoff_flag(bm25_rank=2, dense_rank=4, rrf_rank=10) is False
    assert fusion_cutoff_flag(bm25_rank=None, dense_rank=None, rrf_rank=None) is False


def test_observed_facts_do_not_include_heuristic_labels():
    facts = observed_facts(
        query_id=1,
        question="q",
        gold_passage_id=9,
        in_corpus=True,
        bm25_rank=None,
        dense_rank=3,
        rrf_rank=77,
    )
    blob = json.dumps(facts)
    assert "lexical mismatch" not in blob
    assert "terminology mismatch" not in blob
    assert facts["present_in_hybrid_50"] is False
    assert facts["present_in_bm25_100"] is False
    assert facts["present_in_dense_100"] is True
    assert facts["present_in_either_retriever"] is True
    assert facts["present_in_both_retrievers"] is False
    assert facts["rank_movement"]["dense_to_rrf"] == 3 - 77


def test_biomedical_recovery_rate_documents_denominator():
    result = biomedical_dense_recovery_rate(
        minilm_ranks=[None, None, 4, None],
        biomedical_ranks=[1, None, None, 8],
    )
    assert result["minilm_dense_misses"] == 3
    assert result["newly_recovered_by_pubmedbert"] == 2
    assert result["still_missed_by_pubmedbert"] == 1
    assert result["lost_vs_minilm"] == 1
    assert result["biomedical_dense_recovery_rate"] == 2 / 3
    assert "MiniLM Dense rank is None" in result["denominator_definition"]


def test_earliest_recovery_order():
    assert earliest_recovery({"BM25": True, "MiniLM Dense": True}) == "BM25"
    assert earliest_recovery({"BM25": False, "MiniLM Dense": True}) == "MiniLM Dense"
    assert earliest_recovery({"BM25": False, "MiniLM Dense": False}) == "never_recovered"


def test_lexical_transformations_are_deterministic_and_skip_duplicates():
    variants = lexical_transformations("EGFR signaling")
    queries = [item["query"].lower() for item in variants]
    assert len(queries) == len(set(queries))
    assert any("epidermal" in item["query"].lower() for item in variants)


def test_run_objective_study_writes_facts_signals_hypotheses(tmp_path):
    passages = prepare_corpus(
        [
            {"id": 1, "passage": "Hirschsprung disease of the intestine."},
            {"id": 2, "passage": "Insulin regulates blood glucose."},
            {"id": 3, "passage": "Papilin is a secreted protein."},
        ]
    )
    bm25 = BM25Retriever(passages, top_k=3)
    dense = DenseRetriever(passages, embedder=KeywordEmbedder(), model_name="toy", top_k=3)
    hybrid = HybridRetriever(passages, bm25_retriever=bm25, dense_retriever=dense, top_k=3)
    bio = DenseRetriever(passages, embedder=AltEmbedder(), model_name="toy-bio", top_k=3)
    queries = [
        EvaluationQuery(1, "Hirschsprung intestine", (1,)),
        EvaluationQuery(2, "unrelated zinc transporter", (99,)),
        EvaluationQuery(3, "secreted papilin", (3, 2)),
    ]
    result = run_objective_failure_study(
        passages=passages,
        queries=queries,
        bm25=bm25,
        dense=dense,
        hybrid=hybrid,
        biomedical_dense=bio,
        biomedical_model_name="toy-bio",
        allow_biomedical=False,
        hybrid_k=1,
        probe_k=3,
        details_path=tmp_path / "obj_details.json",
        summary_path=tmp_path / "obj_summary.json",
        recovery_details_path=tmp_path / "rec_details.json",
        recovery_summary_path=tmp_path / "rec_summary.json",
    )
    summary = json.loads(Path(tmp_path / "obj_summary.json").read_text(encoding="utf-8"))
    blob = json.dumps(summary)
    assert "lexical mismatch" not in blob
    assert NOT_IN_CORPUS in summary["objective_failure_counts"]
    assert BOTH_RETRIEVERS_MISS in summary["objective_failure_counts"]
    assert summary["objective_failure_counts"][NOT_IN_CORPUS] >= 1
    rows = result["rows"]
    missing_corpus = next(row for row in rows if row["gold_passage_id"] == 99)
    assert missing_corpus["objective_category"] == NOT_IN_CORPUS
    assert missing_corpus["diagnostic_signals"]["in_corpus"] is False
    assert "hypotheses" in missing_corpus
    recovered = json.loads(Path(tmp_path / "rec_details.json").read_text(encoding="utf-8"))
    assert "MiniLM Dense" in recovered["rows"][0]
    assert "Controlled lexical variant" in recovered["rows"][0]
    assert "n_bm25_misses_newly_recovered_by_transform" in summary["lexical_recovery"]
