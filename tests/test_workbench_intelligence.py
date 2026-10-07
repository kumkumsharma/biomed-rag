"""Production lanes, insights, rank movement, regression semantics, provenance."""

from __future__ import annotations

import inspect

from app.ui import load_pipeline, load_production_pipeline
from src.config import RERANKER, RETRIEVAL
from src.workbench.configuration import (
    configuration_change_summary,
    configuration_diff,
    display_provenance,
    provenance,
    provenance_completeness,
    run_lineage,
)
from src.workbench.experiments import (
    classify_query_outcome,
    explain_classification,
    metric_disagreement_diagnostic,
    per_query_diff,
    query_diff_counts,
    rank_movement_phrase,
)
from src.workbench.insights import experiment_insights
from src.workbench.production import (
    PLAYGROUND_DEFAULT_RERANK_ENABLED,
    PRODUCTION,
    resolve_pipeline_config,
    resolve_rerank_enabled,
)
from src.workbench.regression import (
    add_failure_query,
    add_to_regression_set,
    load_regression_set,
    query_in_regression_set,
    run_regression,
    save_regression_set,
)


def test_production_lane_resolves_medcpt_on():
    config = resolve_pipeline_config(lane="production")
    assert config["reranker_enabled"] is True
    assert config["candidate_pool"] == 50
    assert config["rrf_k"] == 60
    assert config["is_production_stack"] is True
    assert resolve_rerank_enabled(lane="production") is True
    assert RERANKER.enabled is False
    assert RETRIEVAL.rrf_k == 60


def test_playground_default_and_override():
    assert PLAYGROUND_DEFAULT_RERANK_ENABLED is True
    assert resolve_rerank_enabled(lane="playground") is True
    assert resolve_pipeline_config(lane="playground")["reranker_enabled"] is True
    assert resolve_rerank_enabled(override=False, lane="playground") is False
    assert resolve_rerank_enabled(override=True, lane="playground") is True
    signature = inspect.signature(load_pipeline)
    assert signature.parameters["rerank_enabled"].default is None
    assert callable(load_production_pipeline)


def test_experiment_override_does_not_silently_use_production():
    assert resolve_rerank_enabled(lane="experiment") is False
    assert resolve_rerank_enabled(override=True, lane="experiment") is True
    assert resolve_rerank_enabled(override=False, lane="experiment") is False
    library = resolve_pipeline_config(lane="library")
    assert library["reranker_enabled"] is False
    assert library["is_production_stack"] is False
    assert library["label"] == "Library (not production)"


def test_configuration_diff_not_recorded_vs_changed():
    rows = {item["field"]: item for item in configuration_diff("hybrid", "hybrid-medcpt")}
    assert rows["reranker_id"]["status"] == "CHANGED"
    assert rows["reranker_id"]["baseline_display"] is None
    assert rows["reranker_id"]["candidate_display"] == "ncbi/MedCPT-Cross-Encoder"
    assert rows["candidate_pool"]["status"] == "CHANGED"
    assert rows["rrf_k"]["status"] == "UNCHANGED"
    assert rows["dataset_id"]["status"] == "UNCHANGED"
    assert rows["dataset_version"]["status"] == "UNCHANGED"
    assert rows["prompt_version"]["status"] == "NOT RECORDED"
    summary = configuration_change_summary("hybrid", "hybrid-medcpt")
    assert summary["n_changed"] == 2
    assert summary["headline"] == "2 configuration changes detected"
    assert "candidate-pool size and reranking" in summary["isolation_note"]
    assert summary["warning"] == "Multiple variables changed. This experiment does not isolate a single intervention."
    assert summary["isolates_single_intervention"] is False


def test_experiment_insights_are_neutral_and_factual():
    insights = experiment_insights("hybrid", "hybrid-medcpt")
    joined = " ".join(insights["statements"] + insights["notes"]).lower()
    assert "better" not in joined
    assert "best" not in joined
    assert "wins" not in joined
    rec = next(item for item in insights["observations"] if item["label"] == "Recall@10")
    assert rec["baseline"] == "42.86%"
    assert rec["candidate"] == "50.82%"
    assert rec["statement"] == "Increased by 7.96 percentage points."
    cand = next(item for item in insights["observations"] if item["label"] == "Candidate Recall@50")
    assert cand["insufficient"] is True
    lat = next(item for item in insights["observations"] if item["label"] == "Latency")
    assert "Increased" in lat["statement"]
    assert insights["tradeoff"]
    assert "increased while measured latency also increased" in insights["tradeoff"]


def test_rank_movement_and_ab_classification():
    assert rank_movement_phrase(17, 4) == "+13 positions toward top"
    assert rank_movement_phrase(3, 12) == "9 positions toward bottom"
    assert rank_movement_phrase(4, 4) == "unchanged"
    assert rank_movement_phrase(None, 4) == "Insufficient data"
    improved = classify_query_outcome(a_rank=17, b_rank=4, a_recall=0, b_recall=1)
    assert improved["ab_class"] == "IMPROVED"
    assert improved["outcome"] == "FIXED"
    assert explain_classification(a_rank=17, b_rank=4, a_recall=0, b_recall=1)["reason"] == "Best gold rank moved 17 → 4"
    regressed = classify_query_outcome(a_rank=3, b_rank=12, a_recall=1, b_recall=0)
    assert regressed["ab_class"] == "REGRESSED"
    assert explain_classification(a_rank=3, b_rank=12, a_recall=1, b_recall=0)["reason"] == "Best gold rank moved 3 → 12"
    unchanged = classify_query_outcome(a_rank=4, b_rank=4, a_recall=1, b_recall=1, a_rr=0.25, b_rr=0.25)
    assert unchanged["ab_class"] == "UNCHANGED"
    assert explain_classification(a_rank=4, b_rank=4, a_recall=1, b_recall=1)["reason"] == (
        "Same best gold rank and same retrieval outcome"
    )
    missing = classify_query_outcome()
    assert missing["ab_class"] == "INSUFFICIENT DATA"
    rank_only = classify_query_outcome(a_rank=4, b_rank=1, a_recall=1, b_recall=1)
    assert rank_only["ab_class"] == "IMPROVED"
    mixed = explain_classification(a_rank=4, b_rank=1, a_recall=0.67, b_recall=0.33)
    assert mixed["ambiguous"] is True
    assert mixed["reason"].startswith("Mixed / inspect metrics")
    disagreement = metric_disagreement_diagnostic(a_rank=4, b_rank=1, a_recall=0.67, b_recall=0.33)
    assert disagreement["metric_disagreement"] is True
    assert disagreement["metric_disagreement_reason"] == (
        "Best gold rank improved 4 → 1, but Recall@10 decreased 0.67 → 0.33."
    )
    missing = metric_disagreement_diagnostic()
    assert missing["metric_disagreement"] is False
    assert missing["metric_disagreement_reason"] == "Not available"
    rec_vs_rank = metric_disagreement_diagnostic(a_rank=3, b_rank=12, a_recall=0.2, b_recall=0.5)
    assert rec_vs_rank["metric_disagreement"] is True
    assert rec_vs_rank["metric_disagreement_reason"].startswith("Recall@10 improved")
    cand_final = metric_disagreement_diagnostic(
        a_rank=1,
        b_rank=4,
        a_candidate_hit=False,
        b_candidate_hit=True,
    )
    assert cand_final["metric_disagreement"] is True
    assert "Candidate retrieval improved" in cand_final["metric_disagreement_reason"]
    cand_unchanged = metric_disagreement_diagnostic(
        a_rank=1,
        b_rank=3,
        a_candidate_hit=True,
        b_candidate_hit=True,
        a_recall=1,
        b_recall=1,
    )
    assert cand_unchanged["metric_disagreement"] is True
    assert "Candidate retrieval unchanged" in cand_unchanged["metric_disagreement_reason"]


def test_per_query_diff_classifies_from_stored_ranks():
    diff = per_query_diff("hybrid", "hybrid-medcpt")
    counts = query_diff_counts(diff)
    assert counts["total"] == 100
    assert counts["improved"] == 59
    assert counts["regressed"] == 26
    assert counts["unchanged"] == 15
    assert counts["insufficient"] == 0
    assert counts["metric_disagreement"] >= 1
    sample = next(item for item in diff if item.get("metric_disagreement"))
    assert sample["classification"] in {"IMPROVED", "REGRESSED", "UNCHANGED"}
    assert sample["metric_disagreement_reason"]
    assert sample["metric_disagreement_reason"] != "Not available"
    assert counts["improved"] + counts["regressed"] + counts["unchanged"] + counts["insufficient"] == 100
    assert all(item.get("reason") for item in diff)
    moved = next(item for item in diff if item["a_rank"] and item["b_rank"] and item["a_rank"] > item["b_rank"] and not item["ambiguous"])
    assert moved["classification"] == "IMPROVED"
    assert "Best gold rank moved" in moved["reason"]
    same = next(item for item in diff if item["classification"] == "UNCHANGED" and not item["ambiguous"])
    assert "Same best gold rank" in same["reason"] or "same retrieval outcome" in same["reason"].lower()
    ambiguous = [item for item in diff if item["ambiguous"]]
    assert counts["ambiguous"] == len(ambiguous)
    assert all(str(item["reason"]).startswith("Mixed / inspect metrics") for item in ambiguous)


def test_regression_scoring_uses_rank_not_only_recall(tmp_path):
    save_regression_set([], root=tmp_path)
    diff = per_query_diff("hybrid", "hybrid-medcpt")
    improved = next(item for item in diff if item["classification"] == "IMPROVED" and item["a_rank"] and item["b_rank"])
    add_to_regression_set(
        query_id=int(improved["query_id"]),
        question=str(improved["question"]),
        gold_ids=list(improved["gold_ids"]),
        failure_category=improved.get("failure_category"),
        source_experiment="hybrid",
        diagnosis="stored rank movement",
        date_added="2026-10-02",
        root=tmp_path,
    )
    result = run_regression("hybrid-medcpt", root=tmp_path)
    assert result["counts"]["size"] == 1
    assert result["counts"]["improved"] >= 1
    entry = (result["fixed_failures"] or result["improved"])[0]
    assert entry["outcome"] in {"FIXED", "IMPROVED"}
    assert entry["before"]["rank"] is not None
    assert entry["after"]["rank"] is not None
    unchanged_run = run_regression("hybrid", root=tmp_path)
    assert unchanged_run["counts"]["unchanged"] + unchanged_run["counts"]["insufficient"] + unchanged_run["counts"]["failed"] >= 1
    assert unchanged_run["counts"]["fixed_failures"] == 0


def test_duplicate_regression_query_is_rejected(tmp_path):
    save_regression_set([], root=tmp_path)
    first = add_to_regression_set(
        query_id=5,
        question="Does metformin interfere thyroxine absorption?",
        gold_ids=[26191653],
        failure_category="GOLD_LABEL",
        source_experiment="hybrid-medcpt",
        diagnosis="stored Top-10 miss",
        date_added="2026-10-02",
        root=tmp_path,
    )
    second = add_to_regression_set(
        query_id=5,
        question="Does metformin interfere thyroxine absorption?",
        gold_ids=[26191653],
        failure_category="GOLD_LABEL",
        source_experiment="hybrid",
        diagnosis="duplicate",
        date_added="2026-10-02",
        root=tmp_path,
    )
    assert first["already_present"] is False
    assert second["already_present"] is True
    assert len(load_regression_set(root=tmp_path)) == 1
    assert query_in_regression_set(5, root=tmp_path) is not None


def test_failure_to_regression_workflow(tmp_path):
    save_regression_set([], root=tmp_path)
    added = add_failure_query(query_id=5, source_experiment="hybrid-medcpt", root=tmp_path)
    assert added["already_present"] is False
    assert added["query_id"] == 5
    assert "metformin" in added["question"].lower()
    again = add_failure_query(query_id=5, source_experiment="hybrid-medcpt", root=tmp_path)
    assert again["already_present"] is True
    items = load_regression_set(root=tmp_path)
    assert len(items) == 1
    assert items[0]["source_experiment"] == "hybrid-medcpt"


def test_provenance_completeness_and_lineage():
    record = provenance("hybrid-medcpt")
    shown = display_provenance(record)
    completeness = provenance_completeness(record)
    assert completeness["total"] == 17
    assert completeness["recorded"] >= 10
    assert completeness["recorded"] < completeness["total"]
    assert " / 17 fields recorded" in completeness["summary"]
    assert shown["Code version"] == "Not recorded"
    assert shown["Model revision"] == "Not recorded"
    assert shown["Timestamp"] == "Not recorded"
    assert shown["Prompt version"] == "Not recorded"
    lineage = run_lineage("hybrid-medcpt")
    assert [item["stage"] for item in lineage] == [
        "Dataset",
        "Dataset version",
        "Code version",
        "Models",
        "Prompts",
        "Retrieval configuration",
        "Evaluation",
        "Results",
    ]
    assert lineage[1]["value"] == "1.0"
    assert lineage[2]["value"] == "Not recorded"


def test_insights_do_not_claim_candidate_change_when_unrecorded():
    insights = experiment_insights("hybrid", "hybrid-medcpt")
    assert insights["candidate_vs_rerank"] is None
    pool = experiment_insights("hybrid-medcpt", "hybrid-pool100")
    if pool["candidate_vs_rerank"]:
        assert "better" not in pool["candidate_vs_rerank"].lower()
