"""Workbench screens. Presentation only; retrieval implementations stay in src/."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import streamlit as st

from src.config import EVALUATION
from src.workbench.configuration import (
    configuration_change_summary,
    configuration_diff,
    display_provenance,
    experiment_configuration,
    provenance,
    provenance_completeness,
    run_lineage,
)
from src.workbench.dataset import evaluation_dataset
from src.workbench.display import display_table, not_measured
from src.workbench.experiments import (
    compare_experiments,
    experiment_by_id,
    leaderboard_table_rows,
    load_leaderboard,
    load_per_query_results,
    per_query_diff,
    promotion_packet,
    query_diff_counts,
)
from src.workbench.failures import failure_analysis, retrieval_funnel
from src.workbench.insights import experiment_insights
from src.workbench.instrumentation import instrumentation_coverage
from src.workbench.overview import overview_payload
from src.workbench.production import production_runtime_report, resolve_pipeline_config
from src.workbench.registry import evaluator_registry, llm_judge_spec, prompt_registry, workbench_model_registry
from src.workbench.regression import add_failure_query, load_regression_set, query_in_regression_set, run_regression
from src.workbench.traces import filter_traces, stored_traces, trace_timeline

ROOT = Path(__file__).resolve().parents[1]


def format_leaderboard_ui(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pct = {
        "Candidate Recall@50",
        "Recall@1",
        "Recall@3",
        "Recall@5",
        "Recall@10",
        "Rerank Recovery",
    }
    formatted = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, value in row.items():
            if key == "id":
                continue
            if value is None:
                item[key] = "Not measured"
            elif key in pct:
                item[key] = not_measured(value, kind="pct")
            elif key in {"Mean latency", "P50 latency", "P95 latency"}:
                item[key] = not_measured(value, kind="ms")
            elif key in {"MRR@10", "nDCG@10"}:
                item[key] = not_measured(value, kind="ratio")
            elif key == "Cost/query":
                item[key] = not_measured(value, kind="usd")
            else:
                item[key] = value
        formatted.append(item)
    return formatted


def _promotion_label(value: Any) -> str:
    return {
        "production": "Production",
        "candidate": "Candidate",
        "validated": "Validated",
        "rejected": "Experimental",
        "retired-first-stage": "Experimental",
    }.get(str(value or ""), str(value or "Experimental"))


def _config_cell(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, list):
        return ", ".join(str(item) for item in value) or "None"
    return str(value)


def _hit_status(hit: Any, rank: Any) -> str:
    if hit is None:
        return "Not recorded"
    if hit:
        return f"Gold present (rank {rank})" if rank is not None else "Gold present"
    return "Gold absent"
    st.session_state["nav_workspace"] = group
    st.session_state[f"nav_{group}"] = page
    for key, value in extra.items():
        st.session_state[key] = value


def _metric(column: Any, label: str, value: Any, *, kind: str = "raw") -> None:
    column.metric(label, not_measured(value, kind=kind))


def render_overview() -> None:
    payload = overview_payload()
    production = payload.get("production") or {}
    dataset = payload["dataset"]
    coverage = instrumentation_coverage()
    st.header("Overview")
    st.caption("How good is production RAG — why it fails — what changed — what to inspect next.")

    st.subheader("Current production system")
    st.write(payload["production_config"])
    st.write(f"**Production:** {production.get('production_label') or 'No'} · **Promotion:** `{_promotion_label(production.get('promotion'))}`")
    cols = st.columns(6)
    _metric(cols[0], "Recall@10", production.get("recall_at_10"), kind="pct")
    _metric(cols[1], "Candidate Recall@50", production.get("candidate_recall_at_50"), kind="pct")
    _metric(cols[2], "MRR@10", production.get("mrr_at_10"), kind="ratio")
    _metric(cols[3], "nDCG@10", production.get("ndcg_at_10"), kind="ratio")
    _metric(cols[4], "Mean latency", production.get("mean_latency_ms"), kind="ms")
    _metric(cols[5], "Cost/query", production.get("cost_per_query_usd"), kind="usd")
    if st.button("Inspect production", key="inspect_production"):
        _inspect("Experiments", "Experiment List")
        st.rerun()

    st.subheader("Evaluation dataset")
    st.write(
        f"**{dataset['name']}** · version `{dataset['version']}` · "
        f"{dataset['n_queries']} queries · {dataset['n_gold_ids']} gold IDs · {dataset['domain']}"
    )

    latest = payload.get("latest_experiment") or {}
    regression = payload.get("latest_regression") or {}
    failure = payload.get("latest_failure") or {}
    next_row = st.columns(3)
    with next_row[0]:
        st.markdown("**Latest stored experiment**")
        st.write(latest.get("name") or "None")
        st.caption(f"Recall@10 {not_measured(latest.get('recall_at_10'), kind='pct')}")
        if latest and st.button("Inspect", key="inspect_latest"):
            _inspect("Experiments", "Compare", compare_a="hybrid-medcpt", compare_b=latest.get("id"))
            st.rerun()
    with next_row[1]:
        st.markdown("**Largest stored regression vs production**")
        if regression:
            st.write(regression.get("name"))
            st.caption(
                f"Recall@10 {not_measured(regression.get('recall_at_10'), kind='pct')} · Production: No"
            )
            if st.button("Inspect", key="inspect_regression"):
                _inspect("Experiments", "Compare", compare_a="hybrid-medcpt", compare_b=regression.get("id"))
                st.rerun()
        else:
            st.write("None recorded")
    with next_row[2]:
        st.markdown("**Latest failure category**")
        st.write(failure.get("category") or "None")
        st.caption(f"{failure.get('count')} {failure.get('unit')}")
        if failure and st.button("Inspect", key="inspect_failure"):
            _inspect("Evaluations", "Failure Analysis")
            st.rerun()

    dq_path = ROOT / EVALUATION.data_quality_path
    dq: dict[str, Any] = {}
    if dq_path.exists():
        import json

        dq = json.loads(dq_path.read_text(encoding="utf-8"))
    st.markdown("**Data quality issue**")
    st.write(
        f"{dq.get('total_documents')} passages · {dq.get('nan_text')} literal \"nan\" · "
        f"{dq.get('duplicate_texts')} duplicate texts · {dq.get('duplicate_ids')} duplicate IDs"
    )
    if st.button("Inspect", key="inspect_data_quality"):
        _inspect("Data", "Data Quality")
        st.rerun()

    st.markdown("**Instrumentation gaps**")
    gaps = [item for item in coverage if item["status"] != "Measured"]
    st.table([{"Signal": item["signal"], "Status": item["status"]} for item in gaps])
    if st.button("Inspect", key="inspect_instrumentation"):
        _inspect("System", "Observability")
        st.rerun()

    st.divider()
    st.markdown("**Next actions**")
    a, b, c = st.columns(3)
    with a:
        with st.expander("Run benchmark"):
            st.write("Offline evaluation is not executed from the UI. Use the same 100-query dataset:")
            st.code("python -m src.evaluation.run_evaluation --rerank-comparison")
            st.caption("Does not overwrite BM25 / Dense / Hybrid historical files.")
    with b:
        if st.button("Compare experiments"):
            _inspect("Experiments", "Compare")
            st.rerun()
    with c:
        if st.button("Inspect failures"):
            _inspect("Evaluations", "Failure Analysis")
            st.rerun()


def render_datasets() -> None:
    dataset = evaluation_dataset()
    st.header("Datasets")
    st.caption("The 100-query BioASQ set is the default dataset for every retrieval experiment.")
    st.write(f"**Name:** {dataset['name']}")
    st.write(f"**ID / version:** `{dataset['id']}` / `{dataset['version']}`")
    st.write(f"**Queries:** {dataset['n_queries']}")
    st.write(f"**Gold IDs:** {dataset['n_gold_ids']} ({dataset.get('valid_gold_ids')} valid, {dataset.get('invalid_gold_ids')} invalid)")
    st.write(f"**Domain:** {dataset['domain']}")
    st.write(f"**Created:** {dataset['creation_date']}")
    st.write(f"**Protocol:** {dataset['protocol']}")
    st.write(f"**Queries file:** `{dataset['queries_path']}`")
    rows = load_leaderboard()
    st.subheader("Experiments on this dataset")
    st.table(
        [
            {
                "Experiment": item["name"],
                "Queries": item["n_queries"],
                "Production": item["production_label"],
            }
            for item in rows
        ]
    )


def render_benchmark_runs() -> None:
    from app.ui import _render_evaluation

    st.caption("Offline, repeatable. Same 100 queries for every stored run below.")
    _render_evaluation()


def render_metrics() -> None:
    from app.ui import METRIC_GLOSSARY, _render_metrics_page

    _render_metrics_page()
    st.subheader("Evaluator registry")
    for item in evaluator_registry():
        st.markdown(f"**{item['name']}** · `{item['type']}` · v{item['version']}")
        st.caption(item["description"])
        st.write(
            f"Input: {item['input']} · Output: {item['output']} · "
            f"Range: {item['score_range']} · Critical path: {'yes' if item.get('critical_path') else 'no'}"
        )
        if item.get("model"):
            st.write(f"Model: `{item['model']}`")
    st.subheader("LLM-as-judge (offline only)")
    spec = llm_judge_spec()
    st.write(f"**On Ask path:** {'yes' if spec['enabled_on_ask_path'] else 'no'}")
    st.write(f"**Judge model:** `{spec['model']}` · temperature `{spec['temperature']}`")
    st.write(f"**Dataset:** {spec['dataset']}")
    st.write("**Metrics:** " + ", ".join(spec["metrics"]))
    st.caption(spec["note"])
    with st.expander("Metric glossary"):
        for name, meaning in METRIC_GLOSSARY:
            st.write(f"**{name}:** {meaning}")


def render_failure_analysis() -> None:
    from app.ui import _render_failure_analysis

    analysis = failure_analysis()
    st.header("Failure Analysis")
    st.caption(analysis["mapping_note"])
    query_fails = analysis.get("query_failures") or []
    counts = Counter(item["category"] for item in query_fails)
    n_failed = len(query_fails)
    st.subheader("Query-level Top-10 misses (Hybrid + MedCPT, 100 queries)")
    st.write(f"**Failed queries:** {n_failed} of {analysis.get('n_queries')}")
    if n_failed:
        st.bar_chart({key: counts.get(key, 0) for key in counts})
    display = []
    for item in analysis["categories"]:
        display.append(
            {
                "Category": item["category"],
                "Count": item["count"],
                "Unit": item["unit"],
                "Note": item.get("note") or "",
            }
        )
    st.subheader("Stored diagnostic buckets")
    st.caption("ID-level counts and 20-query generation/citation counts are not interchangeable.")
    st.table(display)
    selected = st.selectbox("Failure category", [item["category"] for item in analysis["categories"]])
    group = next(item for item in analysis["categories"] if item["category"] == selected)
    examples = [item for item in query_fails if item["category"] == selected][:8] or group.get("examples") or []
    if examples:
        st.table(examples)
        pick = st.selectbox(
            "Open a failure",
            [f"{item.get('query_id')} · {str(item.get('question') or item.get('gold_passage_id'))[:80]}" for item in examples],
        )
        chosen = examples[[f"{item.get('query_id')} · {str(item.get('question') or item.get('gold_passage_id'))[:80]}" for item in examples].index(pick)]
        st.session_state["trace_query_id"] = chosen.get("query_id")
        st.info("Open **Traces / Trace Detail** for the stored diagnostic if one exists for this query.")
        if chosen.get("query_id") is not None and st.button("Add to regression set", key="add_fail_reg"):
            try:
                result = add_failure_query(query_id=int(chosen["query_id"]), source_experiment="hybrid-medcpt")
                if result.get("already_present"):
                    st.info("Already in regression set")
                else:
                    st.success("Added to the regression set. The original 100-query benchmark was not modified.")
            except (KeyError, ValueError) as exc:
                st.warning(str(exc))
        elif chosen.get("query_id") is not None and query_in_regression_set(int(chosen["query_id"])):
            st.info("Already in regression set")
    with st.expander("Objective Hybrid@50 miss table (gold IDs)", expanded=False):
        _render_failure_analysis()


def render_experiment_list() -> None:
    st.header("Experiment list")
    st.caption("Rows come from stored JSON only. Missing metrics stay Not measured. No invented experiments.")
    rows = load_leaderboard()
    table = leaderboard_table_rows(rows)
    st.dataframe(format_leaderboard_ui(table), use_container_width=True, hide_index=True)
    names = {item["name"]: item["id"] for item in rows}
    if not names:
        st.warning("No stored experiments found.")
        return
    default = 0
    if "Hybrid + MedCPT" in names:
        default = list(names).index("Hybrid + MedCPT")
    chosen = st.selectbox("Open experiment", list(names), index=default)
    render_experiment_detail(names[chosen])


def render_experiment_detail(experiment_id: str) -> None:
    item = experiment_by_id(experiment_id)
    if item is None:
        st.warning("Experiment is not in stored results.")
        return
    baseline = experiment_by_id(item["baseline_id"]) if item.get("baseline_id") else None
    st.subheader(item["name"])
    st.write(
        f"**Dataset:** {item['dataset']} · **Queries:** {item['n_queries']} · "
        f"**Production:** {item['production_label']} · **Promotion:** `{_promotion_label(item['promotion'])}`"
    )
    st.write(f"**Changed variable:** {item['changed_variable']}")
    st.write(f"**Source:** `{item['source_file']}`")
    record = provenance(experiment_id)
    completeness = provenance_completeness(record)
    with st.expander("Provenance", expanded=True):
        st.caption(f"Provenance completeness · {completeness['summary']}")
        for label, value in display_provenance(record).items():
            st.write(f"**{label}:** {value}")
    with st.expander("Run lineage"):
        stages = run_lineage(experiment_id)
        st.caption("Dataset ↓ Dataset version ↓ Code version ↓ Models ↓ Prompts ↓ Retrieval configuration ↓ Evaluation ↓ Results")
        st.table([{"Stage": item["stage"], "Value": item["value"]} for item in stages])

    tabs = st.tabs(["Summary", "Configuration", "Metrics", "Latency", "Failure analysis", "Per-query"])
    with tabs[0]:
        if baseline:
            comparison = compare_experiments(baseline["id"], item["id"])
            st.write(f"{baseline['name']} → {item['name']}")
            insights = experiment_insights(baseline["id"], item["id"])
            st.subheader("Experiment insights")
            for observation in insights.get("observations") or []:
                st.write(f"**{observation['label']}**")
                st.write(f"{observation['baseline']} → {observation['candidate']}")
                st.write(observation["statement"])
            for note in insights.get("notes") or []:
                st.write(note)
            for statement in comparison.get("statements") or []:
                st.write(statement)
            rec = (comparison.get("metrics") or {}).get("recall_at_10")
            if rec:
                st.write(
                    f"Recall@10: {not_measured(rec['baseline'], kind='pct')} → "
                    f"{not_measured(rec['candidate'], kind='pct')} · "
                    f"Delta: {not_measured(rec['delta'], kind='pp')}"
                )
        else:
            st.write("No stored baseline comparison for this row.")
        packet = promotion_packet(experiment_id)
        st.markdown("**Promotion packet** (manual — never auto-promoted from one metric)")
        st.write(packet["path"])
        st.json(
            {
                "recall_at_10": packet.get("recall_at_10"),
                "mean_latency_ms": packet.get("mean_latency_ms"),
                "cost_per_query_usd": packet.get("cost_per_query_usd"),
                "n_queries": packet.get("n_queries"),
                "dataset": packet.get("dataset"),
                "production": packet.get("production"),
            }
        )
    with tabs[1]:
        config = experiment_configuration(experiment_id)
        st.write(f"**Model IDs:** {', '.join(config.get('model_ids') or []) or 'Not recorded'}")
        st.write(f"**Prompt IDs:** {', '.join(config.get('prompt_ids') or []) or 'Not recorded'}")
        st.write(f"**Retriever:** {config.get('retriever')} · **Dense:** {config.get('dense_model_id') or 'Not recorded'}")
        st.write(f"**Fusion:** {config.get('fusion') or 'Not recorded'} · **Pool:** {config.get('candidate_pool')}")
        st.write(f"**Reranker ID:** {config.get('reranker_id') or 'Not recorded'} · **QE:** {config.get('query_expansion')}")
        st.caption("Playground can enable MedCPT per request. Library `RERANKER.enabled` remains off unless env-flagged.")
    with tabs[2]:
        st.table(
            [
                {"Metric": "Candidate Recall@50", "Value": not_measured(item.get("candidate_recall_at_50"), kind="pct")},
                {"Metric": "Recall@1", "Value": not_measured(item.get("recall_at_1"), kind="pct")},
                {"Metric": "Recall@3", "Value": not_measured(item.get("recall_at_3"), kind="pct")},
                {"Metric": "Recall@5", "Value": not_measured(item.get("recall_at_5"), kind="pct")},
                {"Metric": "Recall@10", "Value": not_measured(item.get("recall_at_10"), kind="pct")},
                {"Metric": "MRR@10", "Value": not_measured(item.get("mrr_at_10"), kind="ratio")},
                {"Metric": "nDCG@10", "Value": not_measured(item.get("ndcg_at_10"), kind="ratio")},
                {"Metric": "Rerank recovery", "Value": not_measured(item.get("rerank_recovery"), kind="ratio")},
            ]
        )
    with tabs[3]:
        st.table(
            [
                {"Stat": "Mean", "Value": not_measured(item.get("mean_latency_ms"), kind="ms")},
                {"Stat": "P50", "Value": not_measured(item.get("p50_latency_ms"), kind="ms")},
                {"Stat": "P95", "Value": not_measured(item.get("p95_latency_ms"), kind="ms")},
                {"Stat": "P99", "Value": not_measured(item.get("p99_latency_ms"), kind="ms")},
                {"Stat": "Cost/query", "Value": not_measured(item.get("cost_per_query_usd"), kind="usd")},
            ]
        )
        if item.get("cost_note"):
            st.caption(item["cost_note"])
        st.caption("Retrieval / rerank / generation / evaluation split is Not measured unless the stored JSON records it.")
    with tabs[4]:
        st.caption("Use Evaluations → Failure Analysis for the catalogued buckets. This tab is experiment-scoped.")
        queries = load_per_query_results(experiment_id)
        misses = [row for row in queries.values() if (row.get("recall_at_10") or 0) == 0]
        st.write(f"Queries with Recall@10 = 0 in this experiment: **{len(misses)}** of {len(queries)}")
    with tabs[5]:
        queries = load_per_query_results(experiment_id)
        preview = [
            {
                "Query ID": row["query_id"],
                "Question": row.get("question"),
                "Recall@10": row.get("recall_at_10"),
                "Latency ms": row.get("latency_ms"),
            }
            for row in list(queries.values())[:100]
        ]
        if preview:
            st.dataframe(display_table(preview), use_container_width=True, hide_index=True)
        else:
            st.info("No per-query file is attached to this experiment.")


def render_compare() -> None:
    st.header("Experiment comparison")
    st.caption("Select any two stored experiments. Deltas are absolute metric changes. No ranking is declared.")
    rows = load_leaderboard()
    names = [item["name"] for item in rows]
    ids = {item["name"]: item["id"] for item in rows}
    reverse = {item["id"]: item["name"] for item in rows}
    if len(names) < 2:
        st.warning("Need at least two stored experiments.")
        return
    default_a = names.index("Hybrid") if "Hybrid" in names else 0
    default_b = names.index("Hybrid + MedCPT") if "Hybrid + MedCPT" in names else min(1, len(names) - 1)
    preset_a = reverse.get(st.session_state.get("compare_a"))
    preset_b = reverse.get(st.session_state.get("compare_b"))
    if preset_a in names:
        default_a = names.index(preset_a)
    if preset_b in names:
        default_b = names.index(preset_b)
    left = st.selectbox("Experiment A", names, index=default_a, key="compare_name_a")
    right = st.selectbox("Experiment B", names, index=default_b, key="compare_name_b")
    comparison = compare_experiments(ids[left], ids[right])
    if comparison.get("error"):
        st.error(comparison["error"])
        return
    st.write(
        f"Dataset `{comparison.get('dataset_id')}` v{comparison['baseline'].get('dataset_version')} · "
        f"Queries A={comparison.get('query_count_baseline')} B={comparison.get('query_count_candidate')} · "
        f"Same dataset: **{comparison['same_dataset']}** · Same query count: **{comparison['same_query_count']}"
        + "**"
    )
    st.subheader("What changed")
    diff_rows = configuration_diff(ids[left], ids[right])
    change_summary = configuration_change_summary(ids[left], ids[right])
    st.write(f"**{change_summary['headline']}**")
    if change_summary.get("isolation_note"):
        st.warning(change_summary["isolation_note"])
    elif change_summary.get("warning"):
        st.warning(change_summary["warning"])
    st.table(
        [
            {
                "Component": item["component"],
                "A": "Not recorded" if item["status"] == "NOT RECORDED" else _config_cell(item.get("baseline_display")),
                "B": "Not recorded" if item["status"] == "NOT RECORDED" else _config_cell(item.get("candidate_display")),
                "Status": item["status"],
            }
            for item in diff_rows
        ]
    )
    insights = experiment_insights(ids[left], ids[right])
    st.subheader("Experiment insights")
    for observation in insights.get("observations") or []:
        st.write(f"**{observation['label']}:** {observation['baseline']} → {observation['candidate']}. {observation['statement']}")
    for note in insights.get("notes") or []:
        st.write(note)
    st.subheader("Metrics")
    for statement in comparison.get("statements") or []:
        st.write(statement)
    metric_table = []
    for key, label, kind in (
        ("candidate_recall_at_50", "Candidate Recall@50", "pct"),
        ("recall_at_1", "Recall@1", "pct"),
        ("recall_at_3", "Recall@3", "pct"),
        ("recall_at_5", "Recall@5", "pct"),
        ("recall_at_10", "Recall@10", "pct"),
        ("mrr_at_10", "MRR@10", "ratio"),
        ("ndcg_at_10", "nDCG@10", "ratio"),
        ("rerank_recovery", "Rerank recovery", "ratio"),
        ("mean_latency_ms", "Mean latency", "ms"),
        ("p50_latency_ms", "P50 latency", "ms"),
        ("p95_latency_ms", "P95 latency", "ms"),
        ("cost_per_query_usd", "Cost/query", "usd"),
    ):
        block = (comparison.get("metrics") or {}).get(key)
        a_val = comparison["baseline"].get(key)
        b_val = comparison["candidate"].get(key)
        delta = block["delta"] if block else None
        metric_table.append(
            {
                "Metric": label,
                "A": not_measured(a_val, kind=kind),
                "B": not_measured(b_val, kind=kind),
                "Delta": not_measured(delta, kind="pp" if kind == "pct" else kind),
            }
        )
    st.table(metric_table)
    chart = {}
    for key, label in (("recall_at_10", "Recall@10"), ("mrr_at_10", "MRR@10"), ("ndcg_at_10", "nDCG@10")):
        block = (comparison.get("metrics") or {}).get(key)
        if block:
            chart[label] = block
    if chart:
        st.bar_chart(
            {
                left: {label: block["baseline"] for label, block in chart.items()},
                right: {label: block["candidate"] for label, block in chart.items()},
            }
        )
    if st.button("Open per-query diff"):
        _inspect("Experiments", "Per-query Diff", diff_a=left, diff_b=right)
        st.rerun()

    st.subheader("Query outcomes")
    diff = per_query_diff(ids[left], ids[right])
    counts = query_diff_counts(diff)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Improved", counts["improved"])
    c2.metric("Regressed", counts["regressed"])
    c3.metric("Unchanged", counts["unchanged"])
    c4.metric("Insufficient data", counts["insufficient"])
    if counts.get("metric_disagreement"):
        st.caption(
            f"{counts['metric_disagreement']} queries have a metric-disagreement diagnostic. "
            "Primary classification is unchanged."
        )
    fail_view = st.radio(
        "Filter queries",
        ("All", "Improved", "Regressed", "Unchanged", "Metric disagreement"),
        horizontal=True,
        key="compare_fail_filter",
    )
    fail_rows = diff
    if fail_view == "Improved":
        fail_rows = [item for item in diff if item["classification"] == "IMPROVED"]
    elif fail_view == "Regressed":
        fail_rows = [item for item in diff if item["classification"] == "REGRESSED"]
    elif fail_view == "Unchanged":
        fail_rows = [item for item in diff if item["classification"] == "UNCHANGED"]
    elif fail_view == "Metric disagreement":
        fail_rows = [item for item in diff if item.get("metric_disagreement")]
    if fail_rows:
        st.dataframe(
            display_table(
                [
                    {
                        "Query ID": item["query_id"],
                        "Classification": item["classification"],
                        "Reason": item.get("reason"),
                        "Metric disagreement": item.get("metric_disagreement_reason") or "",
                        "Rank A": item.get("a_rank"),
                        "Rank B": item.get("b_rank"),
                        "RR A": item.get("a_rr"),
                        "RR B": item.get("b_rr"),
                        "Recall@10 A": item.get("a_recall_at_10"),
                        "Recall@10 B": item.get("b_recall_at_10"),
                        "Candidate A": item.get("a_candidate_hit"),
                        "Candidate B": item.get("b_candidate_hit"),
                        "nDCG A": item.get("a_ndcg"),
                        "nDCG B": item.get("b_ndcg"),
                    }
                    for item in fail_rows
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        pick = st.selectbox(
            "Inspect query",
            [f"{item['query_id']}: {str(item.get('question') or '')[:80]}" for item in fail_rows],
            key="compare_fail_pick",
        )
        chosen = fail_rows[[f"{item['query_id']}: {str(item.get('question') or '')[:80]}" for item in fail_rows].index(pick)]
        st.write(f"**Classification:** {chosen['classification']}")
        st.write(f"**Reason:** {chosen.get('reason')}")
        if chosen.get("metric_disagreement"):
            st.write("**Metric disagreement**")
            st.write(f"**Reason:** {chosen.get('metric_disagreement_reason')}")
        elif chosen.get("metric_disagreement_reason") == "Not available":
            st.write("**Metric disagreement:** Not available")
        st.write(f"**Question:** {chosen.get('question')}")
        st.write(f"**Gold IDs:** {chosen.get('gold_ids')}")
        st.write(f"**{left} candidate status:** {_hit_status(chosen.get('a_candidate_hit'), chosen.get('a_candidate_rank'))}")
        st.write(f"**{left} best gold rank:** {chosen.get('a_rank') if chosen.get('a_rank') is not None else 'Insufficient data'}")
        st.write(f"**{right} candidate status:** {_hit_status(chosen.get('b_candidate_hit'), chosen.get('b_candidate_rank'))}")
        st.write(f"**{right} best gold rank:** {chosen.get('b_rank') if chosen.get('b_rank') is not None else 'Insufficient data'}")
        st.write(f"**Rank movement:** {chosen.get('rank_movement_phrase')}")
        st.write(f"**Failure category:** {chosen.get('failure_category') or 'Not recorded'}")
        if st.button("Open Trace Detail", key="compare_open_trace"):
            st.session_state["trace_query_id"] = chosen["query_id"]
            _inspect("Traces", "Trace Detail")
            st.rerun()
    else:
        st.caption("No stored queries in this filter.")


def render_per_query_diff() -> None:
    st.header("Per-query experiment diff")
    st.caption("Which queries did the candidate help, hurt, or leave unchanged?")
    rows = load_leaderboard()
    names = [item["name"] for item in rows]
    ids = {item["name"]: item["id"] for item in rows}
    if len(names) < 2:
        st.warning("Need at least two stored experiments.")
        return
    default_a = names.index("Hybrid") if "Hybrid" in names else 0
    default_b = names.index("Hybrid + MedCPT") if "Hybrid + MedCPT" in names else min(1, len(names) - 1)
    left = st.selectbox("Experiment A", names, index=default_a, key="diff_a")
    right = st.selectbox("Experiment B", names, index=default_b, key="diff_b")
    diff = per_query_diff(ids[left], ids[right])
    if not diff:
        st.info("Per-query details are not stored for one or both of these experiments.")
        return
    counts = query_diff_counts(diff)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total queries", counts["total"])
    c2.metric("Improved", counts["improved"])
    c3.metric("Regressed", counts["regressed"])
    c4.metric("Unchanged", counts["unchanged"])
    c5.metric("Insufficient data", counts["insufficient"])
    view = st.radio(
        "Show",
        ("All", "Improved", "Regressed", "Unchanged", "Insufficient data", "Data quality", "Retrieval", "Reranking", "Generation", "Citation"),
        horizontal=True,
    )
    filtered = diff
    if view == "Improved":
        filtered = [item for item in diff if item["classification"] == "IMPROVED"]
    elif view == "Regressed":
        filtered = [item for item in diff if item["classification"] == "REGRESSED"]
    elif view == "Unchanged":
        filtered = [item for item in diff if item["classification"] == "UNCHANGED"]
    elif view == "Insufficient data":
        filtered = [item for item in diff if item["classification"] == "INSUFFICIENT DATA"]
    elif view in {"Data quality", "Retrieval", "Reranking", "Generation", "Citation"}:
        filtered = [item for item in diff if item.get("failure_bucket") == view]
    table = [
        {
            "Query ID": item["query_id"],
            "Question": item["question"],
            "Class": item["classification"],
            "Reason": item.get("reason"),
            "A Recall@10": item["a_result"],
            "B Recall@10": item["b_result"],
            "A rank": item["a_rank"],
            "B rank": item["b_rank"],
            "Rank movement": item.get("rank_movement_phrase"),
            "Failure": item.get("failure_category"),
        }
        for item in filtered
    ]
    st.dataframe(display_table(table), use_container_width=True, hide_index=True)
    labels = [f"{item['query_id']}: {item['question'][:90]}" for item in filtered]
    if not labels:
        return
    pick = st.selectbox("Inspect query", labels)
    item = filtered[labels.index(pick)]
    st.markdown("**Question**")
    st.write(item["question"])
    st.write(f"**Classification:** {item['classification']}")
    st.write(f"**Reason:** {item.get('reason')}")
    st.write(f"**Failure category:** {item.get('failure_category') or 'Not recorded'}")
    st.markdown("**Gold IDs**")
    st.write(item["gold_ids"])
    st.subheader("Rank movement")
    st.write(f"Best gold rank: {item.get('a_rank') if item.get('a_rank') is not None else 'Insufficient data'} → {item.get('b_rank') if item.get('b_rank') is not None else 'Insufficient data'}")
    st.write(f"Rank movement: {item.get('rank_movement_phrase')}")
    st.write(
        f"A reciprocal rank: {item.get('a_rr') if item.get('a_rr') is not None else 'Insufficient data'} · "
        f"B reciprocal rank: {item.get('b_rr') if item.get('b_rr') is not None else 'Insufficient data'}"
    )
    st.write(
        f"Candidate-stage movement: {item.get('candidate_rank_movement_phrase')} "
        f"(A {item.get('a_candidate_rank') if item.get('a_candidate_rank') is not None else 'Insufficient data'} → "
        f"B {item.get('b_candidate_rank') if item.get('b_candidate_rank') is not None else 'Insufficient data'})"
    )
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**A candidate retrieval**")
        st.write(not_measured(item.get("a_candidate_ids") if item.get("a_candidate_ids") is not None else None))
        st.markdown("**A final ranking**")
        st.write(item.get("a_final_ids") or item.get("a_ids"))
    with cols[1]:
        st.markdown("**B candidate retrieval**")
        st.write(not_measured(item.get("b_candidate_ids") if item.get("b_candidate_ids") is not None else None))
        st.markdown("**B final ranking**")
        st.write(item.get("b_final_ids") or item.get("b_ids"))
    st.write(
        f"Recall@10 {not_measured(item.get('a_result'), kind='pct')} → {not_measured(item.get('b_result'), kind='pct')}"
    )
    st.markdown("**Retrieval stage changes**")
    for note in item.get("stage_changes") or []:
        st.write(note)
    st.session_state["trace_query_id"] = item["query_id"]
    if st.button("Open Trace Detail"):
        _inspect("Traces", "Trace Detail")
        st.rerun()
    if st.button("Add to regression set"):
        try:
            result = add_failure_query(query_id=int(item["query_id"]), source_experiment=ids[right])
            if result.get("already_present"):
                st.info("Already in regression set")
            else:
                st.success("Added to the regression set. The 100-query benchmark file was not modified.")
        except (KeyError, ValueError) as exc:
            st.warning(str(exc))
    elif query_in_regression_set(int(item["query_id"])):
        st.info("Already in regression set")
    st.info("Open **Traces / Trace Detail** for BM25 / dense / RRF / reranker ranks when a diagnostic trace exists.")


def render_trace_explorer() -> None:
    st.header("Trace Explorer")
    traces = stored_traces()
    questions = st.text_input("Query contains")
    sources = ["(all)"] + sorted({str(item.get("source")) for item in traces})
    experiments = ["(all)"] + sorted({str(item.get("experiment")) for item in traces if item.get("experiment")})
    categories = ["(all)"] + sorted({str(item.get("failure_category")) for item in traces if item.get("failure_category")})
    c1, c2, c3 = st.columns(3)
    source = c1.selectbox("Source", sources)
    experiment = c2.selectbox("Experiment", experiments)
    category = c3.selectbox("Failure category", categories)
    filtered = filter_traces(
        traces,
        query=questions or None,
        source=None if source == "(all)" else source,
        experiment=None if experiment == "(all)" else experiment,
        failure_category=None if category == "(all)" else category,
    )
    st.write(f"{len(filtered)} traces")
    st.dataframe(
        display_table(
            [
                {
                    "ID": item["id"],
                    "Source": item["source"],
                    "Query ID": item.get("query_id"),
                    "Question": item.get("question"),
                    "Experiment": item.get("experiment"),
                    "Model": item.get("model"),
                    "Failure": item.get("failure_category"),
                    "Latency": item.get("latency_ms"),
                    "Quality": item.get("quality_score"),
                    "Retrieval": item.get("retrieval_outcome"),
                }
                for item in filtered
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )
    if filtered:
        pick = st.selectbox("Open trace", [item["id"] for item in filtered])
        st.session_state["selected_trace_id"] = pick
        st.info("Open **Trace Detail** for the timeline.")


def render_trace_detail() -> None:
    from app.ui import _render_live_retrieval_trace, _require_last_result

    st.header("Trace Detail")
    traces = stored_traces()
    selected = st.session_state.get("selected_trace_id")
    preferred = st.session_state.get("trace_query_id")
    options = [item["id"] for item in traces]
    index = 0
    if selected in options:
        index = options.index(selected)
    elif preferred is not None:
        for i, item in enumerate(traces):
            if item.get("query_id") == preferred:
                index = i
                break
    if not options:
        st.info("No stored traces yet. Ask a playground question to capture an online sample, or run diagnostics.")
        if _require_last_result() is not None:
            _render_live_retrieval_trace()
        return
    pick = st.selectbox("Trace", options, index=index)
    record = next(item for item in traces if item["id"] == pick)
    st.write(f"**Source:** {record['source']} · **Query:** {record.get('question')}")
    for stage in trace_timeline(record.get("payload") or {}):
        with st.expander(f"{stage['stage']} · {stage.get('status')}", expanded=False):
            st.write(f"**Model:** {not_measured(stage.get('model'))}")
            st.write(f"**Latency:** {not_measured(stage.get('latency_ms'), kind='ms')}")
            st.write("**Input**")
            st.write(stage.get("input"))
            st.write("**Output**")
            st.write(stage.get("output") if not isinstance(stage.get("output"), (list, dict)) else "")
            if isinstance(stage.get("output"), (list, dict)):
                st.json(stage.get("output") if not isinstance(stage.get("output"), list) else stage.get("output")[:5])
            st.caption("Metadata")
            st.json(stage.get("metadata") or {})
    live = _require_last_result()
    if live is not None:
        with st.expander("Last playground request", expanded=False):
            _render_live_retrieval_trace()


def render_data_quality() -> None:
    from app.ui import _render_data_quality_dashboard

    funnel = retrieval_funnel()
    st.header("Data Quality")
    st.caption("Corpus issues remain in the production index. They are reported, not silently deleted.")
    _render_data_quality_dashboard()
    st.subheader("Why this changes retrieval metrics")
    st.write(funnel["raw_vs_valid_gold"]["note"])
    st.write(
        f"Raw Recall@10 {not_measured(funnel['raw_vs_valid_gold'].get('raw_recall_at_10'), kind='pct')} · "
        f"Valid-gold diagnostic Recall@10 {not_measured(funnel['raw_vs_valid_gold'].get('valid_gold_recall_at_10'), kind='pct')}"
    )
    st.caption("The valid-gold view does not replace the raw 100-query benchmark.")


def render_gold_quality() -> None:
    from app.ui import _render_gold_quality_page

    _render_gold_quality_page()
    funnel = retrieval_funnel()
    st.subheader("Retrieval funnel")
    st.caption(funnel["distinction"])
    st.markdown("**ID-level**")
    st.table(
        [
            {
                "Stage": item["stage"],
                "Count": not_measured(item.get("count")),
                "Rate": not_measured(item.get("rate"), kind="pct"),
                "Note": item.get("note") or "",
            }
            for item in funnel["id_level"]
        ]
    )
    st.markdown("**Query-level**")
    st.table(
        [
            {
                "Stage": item["stage"],
                "Count": not_measured(item.get("count")),
                "Rate": not_measured(item.get("rate"), kind="pct"),
                "Note": item.get("note") or "",
            }
            for item in funnel["query_level"]
        ]
    )


def render_regression() -> None:
    st.header("Regression set")
    st.caption("Separate from the 100-query benchmark. Outcomes use stored per-query results only.")
    items = load_regression_set()
    st.write(f"**Items:** {len(items)}")
    if items:
        st.dataframe(display_table(items), use_container_width=True, hide_index=True)
    else:
        st.info("Empty. Add a failed query from Failure Analysis or Per-query Diff.")
    rows = load_leaderboard()
    names = [item["name"] for item in rows]
    ids = {item["name"]: item["id"] for item in rows}
    if not names:
        return
    default = names.index("Hybrid + MedCPT") if "Hybrid + MedCPT" in names else 0
    chosen = st.selectbox("Run stored results of", names, index=default)
    if st.button("Score regression set"):
        result = run_regression(ids[chosen])
        counts = result["counts"]
        st.write(f"**Regression set size:** {counts['size']}")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Passed", counts["passed"])
        c2.metric("Failed", counts["failed"])
        c3.metric("Improved", counts["improved"])
        c4.metric("Regressed", counts["regressed"])
        c5.metric("Unchanged", counts["unchanged"])
        c6.metric("Insufficient data", counts["insufficient"])
        st.caption(result["note"])
        for title, rows in (
            ("Passed", result["passed"]),
            ("Failed", result["failed"]),
            ("Improved", result["improved"] + result["fixed_failures"]),
            ("Regressed", result["regressed"]),
            ("Unchanged", result["unchanged"]),
            ("Insufficient data", result["insufficient"]),
        ):
            with st.expander(f"{title} ({len(rows)})", expanded=bool(rows) and title in {"Regressed", "Improved"}):
                if rows:
                    st.dataframe(
                        display_table(
                            [
                                {
                                    "Query ID": item.get("query_id"),
                                    "Outcome": item.get("outcome"),
                                    "Before Rec@10": (item.get("before") or {}).get("recall_at_10"),
                                    "After Rec@10": (item.get("after") or {}).get("recall_at_10"),
                                    "Before rank": (item.get("before") or {}).get("rank"),
                                    "After rank": (item.get("after") or {}).get("rank"),
                                    "Diagnosis": "; ".join(item.get("diagnoses") or []),
                                }
                                for item in rows
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.write("None")


def render_models() -> None:
    st.header("Models")
    registry = workbench_model_registry(production_medcpt=True)
    for lane in ("production", "experimental", "retired"):
        st.subheader(lane.title())
        items = registry.get(lane) or []
        if not items:
            st.caption("None.")
            continue
        for item in items:
            st.markdown(f"**{item['name']}** · {item['role']} · `{item.get('status')}`")
            st.write(
                f"ID `{item['id']}` · dimensions {not_measured(item.get('dimensions'))} · "
                f"experiments: {', '.join(item.get('experiments_used_in') or []) or 'none listed'}"
            )
            st.caption(item.get("notes"))
            st.write(f"**Latency:** Not measured as a registry statistic (see experiment P50/P95).")


def render_prompts() -> None:
    from app.ui import _render_prompt_inspector_page

    st.header("Prompt registry")
    for item in prompt_registry():
        st.markdown(f"**{item['name']}** · `{item.get('id')}` · stage `{item['stage']}` · v{item['version']}")
        st.write(f"Model: `{item['model']}` · created: {item['created_date']}")
        st.caption("Attach prompt versions to experiments by recording the registry name in the experiment notes. Historical JSON does not store prompt versions.")
    _render_prompt_inspector_page()


def render_architecture() -> None:
    from app.ui import _render_architecture_dashboard

    _render_architecture_dashboard()
    st.subheader("Online vs offline")
    left, right = st.columns(2)
    with left:
        st.markdown("**OFFLINE EVALUATION**")
        st.write("Fixed 100-query benchmark · repeatable · experiment comparison · stored JSON")
    with right:
        st.markdown("**ONLINE / PRODUCTION**")
        st.write("Playground / live traffic · latency · errors · cost · sampled quality · LLM judge sampling (off by default)")
    st.subheader("Production configuration lanes")
    report = production_runtime_report()
    production_cfg = resolve_pipeline_config(lane="production")
    playground_cfg = resolve_pipeline_config(lane="playground")
    library_cfg = resolve_pipeline_config(lane="library")
    st.write(f"**Declared production:** {report['declared_production']['path']}")
    st.write(
        f"**Production lane MedCPT:** `{production_cfg['reranker_enabled']}` · "
        f"pool `{production_cfg['candidate_pool']}` · RRF k `{production_cfg['rrf_k']}`"
    )
    st.write(
        f"**Playground default MedCPT:** `{playground_cfg['reranker_enabled']}` · "
        f"**Library MedCPT:** `{library_cfg['reranker_enabled']}` (not production)"
    )
    st.write(f"**Library `RERANKER.enabled`:** `{report['library_default']['RERANKER.enabled']}`")
    st.write(report["streamlit_playground"]["note"])
    st.warning(report["reason"]) if report["can_accidentally_run_without_medcpt"] else st.write(report["reason"])
    st.caption("Library default was not flipped. Ranking code is unchanged. Production is declared separately from experiment overrides.")


def render_observability() -> None:
    from app.ui import _render_observability_dashboard

    _render_observability_dashboard()
    st.subheader("Instrumentation coverage")
    st.table(
        [
            {
                "Signal": item["signal"],
                "Status": item["status"],
                "Detail": "Not measured" if item.get("detail") in (None, {}) else str(item.get("detail")),
            }
            for item in instrumentation_coverage()
        ]
    )
    st.subheader("Cost & latency")
    production = experiment_by_id("hybrid-medcpt") or {}
    st.table(
        [
            {"Measure": "P50", "Value": not_measured(production.get("p50_latency_ms"), kind="ms")},
            {"Measure": "P95", "Value": not_measured(production.get("p95_latency_ms"), kind="ms")},
            {"Measure": "P99", "Value": not_measured(production.get("p99_latency_ms"), kind="ms")},
            {"Measure": "Mean", "Value": not_measured(production.get("mean_latency_ms"), kind="ms")},
            {"Measure": "Retrieval latency", "Value": "Not measured"},
            {"Measure": "Reranking latency", "Value": "Not measured as a split (included in total where recorded)"},
            {"Measure": "Generation latency", "Value": "Not measured on the 100-query retrieval set"},
            {"Measure": "Evaluation latency", "Value": "Not measured on the 100-query retrieval set"},
            {"Measure": "Token usage", "Value": "Not measured"},
            {"Measure": "Cost/query", "Value": not_measured(production.get("cost_per_query_usd"), kind="usd")},
        ]
    )
    st.caption(production.get("cost_note") or "")
