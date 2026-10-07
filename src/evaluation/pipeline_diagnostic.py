"""End-to-end RAG pipeline diagnostic. Does not change production retrieval.

Reads the fixed 100-query set and the saved MedCPT comparison metrics, then
traces selected queries through the production BM25, MiniLM, RRF, and MedCPT
components. Query expansion, SPLADE, and generation are not invoked.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import EVALUATION, RERANKER, RETRIEVAL
from src.data.loader import get_passage_collection
from src.data.preprocessing import Passage
from src.evaluation.dataset import EvaluationQuery, load_or_create_fixed_queries
from src.evaluation.metrics import passage_id_key, rank_of_id, recall_at_k
from src.evaluation.run_evaluation import build_evaluation_retrievers
from src.retrieval.reranker import CrossEncoderReranker, get_shared_scorer

CATEGORIES = (
    "Treatment / intervention",
    "Diagnosis / disease identification",
    "Cause / mechanism / pathophysiology",
    "Risk / association / prognosis",
    "Fact / definition / biological relationship",
    "Other / unclear",
)

PREVIEW_CHARS = 800
DEEP_K_CAP = 2000

_RISK_CUES = (
    "associated with",
    "association of",
    "association between",
    "prognosis",
    "prevalent",
    "more common in",
    "additional risk",
    "linked to disease",
    "risk for",
    "meningioma risk",
)
_TREATMENT_CUES = (
    "treatment",
    "therapy",
    "vaccine",
    "approved",
    "effective",
    "side effect",
    "indications of",
    "interfere",
    "self management",
    "acupuncture",
    "tens machine",
    "palbociclib",
    "inhibitors are used",
    "developed for treatment",
    "potential use in the treatment",
    "denosumab",
    "trastuzumab",
    "orteronel",
    "lacosamide",
    "ivabradine",
    "sglt2",
    "valproic",
    "levoxyl",
    "dupilumab",
    "evolocumab",
    "imetelstat",
    "sonidegib",
    "everolimus",
    "clopidogrel",
    "metformin",
    "gv1001",
    "renal sympathetic denervation",
    "smoking cessation",
    "chronic pain",
    "anti-arrhythmic",
)
_DIAGNOSIS_CUES = (
    "diagnos",
    "symptom",
    "triad",
    "common features of",
    "pathological condition",
    "weaver syndrome similar",
    "biomarker",
    "synonym of",
    "pendred syndrome present",
)
_MECHANISM_CUES = (
    "mode of inheritance",
    "mode of action",
    "mendelian",
    "multifactorial",
    "mutation",
    "phosphorylat",
    "signaling pathway",
    "ubiquitin",
    "genetic basis",
    "pathophysiolog",
    "mechanism",
    "involved in the development",
    "involved in the activation",
    "required for",
    "cell cycle",
    "co-localize",
    "knockdown",
    "depleting protein",
    "role of extracellular",
    "function of the viral",
    "formation of the ryanodine",
    "formation of r-loops",
    "post-translational",
    "maternal-to-zygotic",
    "transcription and splicing",
    "effect of trh",
    "which map kinase",
    "which transcription factor",
    "which fusion protein",
    "which gene",
    "which human genes",
    "which cell types",
    "driving rheumatoid",
)


def categorize_question(question: str) -> str:
    """Assign one presentation category from the question text. No ML."""

    if not isinstance(question, str):
        raise TypeError("question must be a string")
    q = question.lower()
    if any(cue in q for cue in _RISK_CUES):
        return "Risk / association / prognosis"
    if any(cue in q for cue in _TREATMENT_CUES):
        return "Treatment / intervention"
    if any(cue in q for cue in _DIAGNOSIS_CUES):
        return "Diagnosis / disease identification"
    if any(cue in q for cue in _MECHANISM_CUES):
        return "Cause / mechanism / pathophysiology"
    if q.startswith(("what is", "what are", "what kind", "where is", "list ", "name ")):
        return "Fact / definition / biological relationship"
    if any(
        cue in q
        for cue in (
            "secreted",
            "spliced",
            "meaning of",
            "aim of",
            "located",
            "isoforms",
            "yamanaka",
            "chromatography",
            "clathrin",
            "ultraconserved",
            "transcribed",
            "methyl donor",
            "complete genome",
            "jaspar",
            "method fasp",
            "evaluated using",
            "technique is used",
            "application of",
            "being measured",
            "signaling molecules",
            "dishevelled",
            "notch receptor",
            "bromodomain",
            "pannexin",
            "papilin",
            "rankl",
            "ctcf",
        )
    ):
        return "Fact / definition / biological relationship"
    return "Other / unclear"


def preview_text(text: str, limit: int = PREVIEW_CHARS) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "…"


def _id_map(passages: Sequence[Passage]) -> dict[str, Passage]:
    return {passage_id_key(item.passage_id): item for item in passages}


def _hit_rows(hits: Sequence[Any], limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    items = list(hits) if limit is None else list(hits)[:limit]
    for rank, hit in enumerate(items, start=1):
        rows.append(
            {
                "rank": rank,
                "passage_id": hit.passage_id,
                "score": float(getattr(hit, "score", 0.0)),
                "preview": preview_text(getattr(hit, "text", ""), 160),
            }
        )
    return rows


def _lookup_hit(hits: Sequence[Any], gold_id: Any) -> tuple[int | None, float | None, Any]:
    rank = rank_of_id([hit.passage_id for hit in hits], gold_id)
    if rank is None:
        return None, None, None
    hit = hits[rank - 1]
    return rank, float(getattr(hit, "score", 0.0)), hit


def load_medcpt_baseline_rows(
    path: str | Path | None = None,
) -> dict[Any, dict[str, Any]]:
    details_path = Path(path or EVALUATION.rerank_comparison_details_path)
    payload = json.loads(details_path.read_text(encoding="utf-8"))
    return {row["query_id"]: row for row in payload.get("rows") or []}


def categorize_evaluation_queries(
    queries: Sequence[EvaluationQuery],
    baseline_rows: Mapping[Any, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Category counts and mean MedCPT Recall@10. Does not change metrics."""

    rows = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in queries:
        category = categorize_question(query.question)
        baseline = (baseline_rows or {}).get(query.question_id) or {}
        reranked = baseline.get("reranked") or {}
        rec10 = reranked.get("recall_at_10")
        rec50 = baseline.get("candidate_recall_at_50")
        item = {
            "query_id": query.question_id,
            "question": query.question,
            "category": category,
            "n_gold": len(query.relevant_passage_ids),
            "gold_passage_ids": list(query.relevant_passage_ids),
            "recall_at_10": rec10,
            "candidate_recall_at_50": rec50,
            "candidate_hit_at_50": baseline.get("candidate_hit_at_50"),
            "best_rrf_rank": baseline.get("best_rrf_rank"),
            "best_reranked_rank": baseline.get("best_reranked_rank"),
        }
        rows.append(item)
        by_category[category].append(item)

    table = []
    for name in CATEGORIES:
        group = by_category.get(name) or []
        recs = [float(item["recall_at_10"]) for item in group if item["recall_at_10"] is not None]
        misses = [item for item in group if item["recall_at_10"] == 0]
        low = [item for item in group if item["recall_at_10"] is not None and item["recall_at_10"] < 0.3]
        table.append(
            {
                "category": name,
                "n_queries": len(group),
                "recall_at_10": (sum(recs) / len(recs)) if recs else None,
                "n_complete_misses": len(misses),
                "n_recall_below_0_30": len(low),
                "query_ids": [item["query_id"] for item in group],
            }
        )
    populated = [row for row in table if row["n_queries"]]
    most_misses = max(populated, key=lambda row: (row["n_complete_misses"], -row["n_queries"])) if populated else None
    lowest_recall = min(
        (row for row in populated if row["recall_at_10"] is not None),
        key=lambda row: row["recall_at_10"],
        default=None,
    )
    return {
        "n_queries": len(rows),
        "rows": rows,
        "table": table,
        "most_complete_misses_category": None if most_misses is None else most_misses["category"],
        "lowest_mean_recall_category": None if lowest_recall is None else lowest_recall["category"],
        "n_complete_misses": sum(row["n_complete_misses"] for row in table),
        "note": (
            "Category is assigned from question wording only. "
            "A lower mean Recall@10 is a diagnostic observation, not a demonstrated cause."
        ),
    }


def select_representative_queries(category_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Pick one success and one complete miss that are easy to present."""

    rows = list(category_payload.get("rows") or [])
    by_id = {item["query_id"]: item for item in rows}
    success = by_id.get(2)
    failure = by_id.get(5)
    if success is None or failure is None:
        raise KeyError("Expected query_id 2 (success) and 5 (complete miss) in the fixed 100-query set")
    return {
        "query_a": success,
        "query_b": failure,
        "reason_a": "Simple yes/no fact question; MedCPT Recall@10=0.80 with gold at rank 1.",
        "reason_b": (
            "The only complete miss in the 100-query MedCPT run: one gold passage, "
            "Candidate Recall@50=0 and Final Recall@10=0."
        ),
    }


def _is_empty_or_nan_text(text: str | None) -> bool:
    if text is None:
        return True
    stripped = str(text).strip()
    return stripped == "" or stripped.lower() == "nan"


def inspect_gold_passages(
    gold_ids: Sequence[Any],
    passages: Sequence[Passage],
) -> list[dict[str, Any]]:
    indexed = _id_map(passages)
    reports = []
    for gold_id in gold_ids:
        key = passage_id_key(gold_id)
        passage = indexed.get(key)
        if passage is None:
            reports.append(
                {
                    "gold_id": gold_id,
                    "pmid": gold_id,
                    "exists_in_index": False,
                    "failure_point": "DATA / PARSING",
                    "raw_text_preview": None,
                    "raw_text_length": None,
                    "indexed_text_preview": None,
                    "indexed_text_length": None,
                    "text_unchanged": False,
                    "truncated": None,
                    "chunked": False,
                    "n_chunks": 0,
                    "empty_or_nan_text": True,
                }
            )
            continue
        text = passage.text or ""
        empty = _is_empty_or_nan_text(text)
        reports.append(
            {
                "gold_id": gold_id,
                "pmid": passage.passage_id,
                "exists_in_index": True,
                "failure_point": "DATA / PARSING" if empty else None,
                "raw_text_preview": preview_text(text),
                "raw_text_length": len(text),
                "indexed_text_preview": preview_text(text),
                "indexed_text_length": len(text),
                "text_unchanged": True,
                "truncated": False,
                "chunked": False,
                "n_chunks": 1,
                "empty_or_nan_text": empty,
                "note": (
                    "This corpus stores one full passage per ID. prepare_passage keeps "
                    "the original ID and text; there is no extra cleaning or chunk split. "
                    "A source value of 'nan' is preserved as the three-character string 'nan'."
                    if empty
                    else (
                        "This corpus stores one full passage per ID. prepare_passage keeps "
                        "the original ID and text; there is no extra cleaning or chunk split."
                    )
                ),
            }
        )
    return reports


STAGE_CODES = {
    "DATA / PARSING": "DATA_QUALITY",
    "FIRST-STAGE RETRIEVAL / FUSION": "RRF",
    "BM25 RETRIEVAL": "BM25",
    "DENSE RETRIEVAL": "DENSE",
    "RRF / FUSION": "RRF",
    "RERANKING": "RERANKING",
    "NOT LOST": "NOT_LOST",
    "EVALUATION / GOLD-LABEL ISSUE": "EVALUATION",
}


def stage_code(label: str) -> str:
    return STAGE_CODES.get(label, label.replace(" ", "_").replace("/", ""))


def diagnose_loss(
    *,
    gold_exists: bool,
    empty_or_nan_text: bool = False,
    in_bm25_pool: bool,
    in_dense_pool: bool,
    rrf_rank: int | None,
    medcpt_rank: int | None,
    pool_k: int = 50,
    final_k: int = 10,
) -> str:
    """Return the first supported loss label. Parallel BM25+dense misses map to fusion."""

    if not gold_exists or empty_or_nan_text:
        return "DATA / PARSING"
    in_pool = rrf_rank is not None and rrf_rank <= pool_k
    if not in_pool:
        if not in_bm25_pool and not in_dense_pool:
            return "FIRST-STAGE RETRIEVAL / FUSION"
        if in_bm25_pool and not in_dense_pool:
            return "DENSE RETRIEVAL"
        if in_dense_pool and not in_bm25_pool:
            return "BM25 RETRIEVAL"
        return "RRF / FUSION"
    if medcpt_rank is None or medcpt_rank > final_k:
        return "RERANKING"
    return "NOT LOST"


def trace_query(
    query: EvaluationQuery,
    *,
    passages: Sequence[Passage],
    bm25: Any,
    dense: Any,
    hybrid: Any,
    reranker: Any,
    category: str,
    pool_k: int = 50,
    final_k: int = 10,
    deep_k: int | None = None,
) -> dict[str, Any]:
    """Trace every gold ID through production retrieval. Does not call the LLM."""

    inspect_k = deep_k if deep_k is not None else len(passages)
    gold_reports = inspect_gold_passages(query.relevant_passage_ids, passages)
    bm25_hits = list(bm25.search(query.question, top_k=inspect_k))
    dense_hits = list(dense.search(query.question, top_k=inspect_k))
    rrf_hits = list(hybrid.search(query.question, top_k=pool_k))
    trace = dict(getattr(hybrid, "last_trace", {}) or {})
    bm25_pool = list(bm25.search(query.question, top_k=pool_k))
    dense_pool = list(dense.search(query.question, top_k=pool_k))

    reranked_hits: list[Any] = []
    rerank_ms = None
    if rrf_hits:
        import time

        started = time.perf_counter()
        reranked_hits = list(
            reranker.rerank(
                query.question,
                rrf_hits,
                top_k=final_k,
                bm25_ranks=dict(trace.get("bm25_ranks") or {}),
                dense_ranks=dict(trace.get("dense_ranks") or {}),
            )
        )
        rerank_ms = (time.perf_counter() - started) * 1000.0

    gold_traces = []
    for gold in gold_reports:
        gold_id = gold["gold_id"]
        bm25_rank, bm25_score, _ = _lookup_hit(bm25_hits, gold_id)
        dense_rank, dense_score, _ = _lookup_hit(dense_hits, gold_id)
        rrf_rank, rrf_score, _ = _lookup_hit(rrf_hits, gold_id)
        medcpt_rank, medcpt_score, _ = _lookup_hit(reranked_hits, gold_id)
        in_bm25_pool = rank_of_id([hit.passage_id for hit in bm25_pool], gold_id) is not None
        in_dense_pool = rank_of_id([hit.passage_id for hit in dense_pool], gold_id) is not None
        in_top50 = rrf_rank is not None
        in_top10 = medcpt_rank is not None and medcpt_rank <= final_k
        loss = diagnose_loss(
            gold_exists=bool(gold["exists_in_index"]),
            empty_or_nan_text=bool(gold.get("empty_or_nan_text")),
            in_bm25_pool=in_bm25_pool,
            in_dense_pool=in_dense_pool,
            rrf_rank=rrf_rank,
            medcpt_rank=medcpt_rank,
            pool_k=pool_k,
            final_k=final_k,
        )
        gold_traces.append(
            {
                **gold,
                "bm25_rank": bm25_rank,
                "bm25_score": bm25_score,
                "bm25_in_production_top50": in_bm25_pool,
                "dense_rank": dense_rank,
                "dense_score": dense_score,
                "dense_in_production_top50": in_dense_pool,
                "rrf_rank": rrf_rank,
                "rrf_score": rrf_score,
                "in_top50": in_top50,
                "medcpt_rank": medcpt_rank,
                "medcpt_score": medcpt_score,
                "in_top10": in_top10,
                "first_loss_label": loss,
                "stage_code": stage_code(loss),
            }
        )

    any_in_top50 = any(item["in_top50"] for item in gold_traces)
    any_in_top10 = any(item["in_top10"] for item in gold_traces)
    first_loss = next(
        (item["first_loss_label"] for item in gold_traces if item["first_loss_label"] != "NOT LOST"),
        "NOT LOST",
    )
    above_gold = []
    if any(item["in_top50"] and not item["in_top10"] for item in gold_traces):
        above_gold = _hit_rows(reranked_hits, final_k)

    compact = {
        "question": query.question,
        "category": category,
        "raw_gold_ids": [item["gold_id"] for item in gold_traces],
        "preprocessing": "unchanged full passages; no chunking",
        "best_bm25_rank": min((item["bm25_rank"] for item in gold_traces if item["bm25_rank"] is not None), default=None),
        "best_dense_rank": min((item["dense_rank"] for item in gold_traces if item["dense_rank"] is not None), default=None),
        "best_rrf_rank": min((item["rrf_rank"] for item in gold_traces if item["rrf_rank"] is not None), default=None),
        "in_top50": any_in_top50,
        "best_medcpt_rank": min((item["medcpt_rank"] for item in gold_traces if item["medcpt_rank"] is not None), default=None),
        "in_top10": any_in_top10,
        "first_loss_label": first_loss,
        "stage_code": stage_code(first_loss),
        "recall_at_10": recall_at_k([hit.passage_id for hit in reranked_hits], query.relevant_passage_ids, final_k),
        "candidate_recall_at_50": recall_at_k(
            [hit.passage_id for hit in rrf_hits], query.relevant_passage_ids, pool_k
        ),
    }
    return {
        "query_id": query.question_id,
        "question": query.question,
        "category": category,
        "gold_ids": list(query.relevant_passage_ids),
        "inspect_k": inspect_k,
        "pool_k": pool_k,
        "final_k": final_k,
        "dense_model": RETRIEVAL.dense_model_name,
        "rrf_k": RETRIEVAL.rrf_k,
        "reranker_model": RERANKER.model_name,
        "gold_traces": gold_traces,
        "bm25_top10": _hit_rows(bm25_hits, 10),
        "dense_top10": _hit_rows(dense_hits, 10),
        "rrf_top20": _hit_rows(rrf_hits, 20),
        "medcpt_top10": _hit_rows(reranked_hits, 10),
        "documents_ranked_above_missed_gold": above_gold,
        "rerank_ms": rerank_ms,
        "compact_trace": compact,
        "generation_invoked": False,
    }


def format_category_table(payload: Mapping[str, Any]) -> str:
    lines = [
        f"{'Category':<46} {'N':>4} {'Recall@10':>12} {'Rec=0':>7} {'Rec<0.30':>9}",
        "-" * 80,
    ]
    for row in payload["table"]:
        rec = "n/a" if row["recall_at_10"] is None else f"{row['recall_at_10']:.4f}"
        lines.append(
            f"{row['category']:<46} {row['n_queries']:>4} {rec:>12} "
            f"{row['n_complete_misses']:>7} {row['n_recall_below_0_30']:>9}"
        )
    return "\n".join(lines)


def format_compact_trace(trace: Mapping[str, Any]) -> str:
    compact = trace["compact_trace"]
    def _rank(value: Any) -> str:
        return "absent" if value is None else str(value)

    return (
        f"QUESTION: {compact['question']}\n"
        f"   ↓\n"
        f"RAW GOLD EVIDENCE: {compact['raw_gold_ids']}\n"
        f"   ↓\n"
        f"PREPROCESSING / CHUNKING: {compact['preprocessing']}\n"
        f"   ↓\n"
        f"BM25: rank {_rank(compact['best_bm25_rank'])}\n"
        f"   ↓\n"
        f"MiniLM: rank {_rank(compact['best_dense_rank'])}\n"
        f"   ↓\n"
        f"RRF: rank {_rank(compact['best_rrf_rank'])}\n"
        f"   ↓\n"
        f"Top 50? {'YES' if compact['in_top50'] else 'NO'}\n"
        f"   ↓\n"
        f"MedCPT: rank {_rank(compact['best_medcpt_rank'])}\n"
        f"   ↓\n"
        f"Top 10? {'YES' if compact['in_top10'] else 'NO'}"
    )


def _print_stage_table(title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    print(f"\n{title}")
    print(f"{'rank':>4}  {'id':<12} {'score':>12}  preview")
    for row in rows:
        print(f"{row['rank']:>4}  {str(row['passage_id']):<12} {row['score']:>12.6f}  {row['preview']}")


def print_query_trace(label: str, trace: Mapping[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(f"{label}: query_id={trace['query_id']}")
    print("=" * 72)
    print(f"Question: {trace['question']}")
    print(f"Category: {trace['category']}")
    print(f"Gold IDs: {trace['gold_ids']}")
    print(f"Models: BM25Okapi + {trace['dense_model']} + RRF k={trace['rrf_k']} + {trace['reranker_model']}")
    for gold in trace["gold_traces"]:
        print(f"\n--- Gold {gold['gold_id']} ---")
        print(f"Exists in index: {gold['exists_in_index']}")
        print(f"Chunked: {gold['chunked']}  truncated: {gold['truncated']}  text unchanged: {gold['text_unchanged']}")
        print(f"Indexed length: {gold['indexed_text_length']}")
        print(f"Preview: {gold['raw_text_preview']}")
        print(
            f"BM25 rank={gold['bm25_rank']} score={gold['bm25_score']} "
            f"in production top50={gold['bm25_in_production_top50']}"
        )
        print(
            f"MiniLM rank={gold['dense_rank']} score={gold['dense_score']} "
            f"in production top50={gold['dense_in_production_top50']}"
        )
        print(f"RRF rank={gold['rrf_rank']} score={gold['rrf_score']} in Top50={gold['in_top50']}")
        print(f"MedCPT rank={gold['medcpt_rank']} score={gold['medcpt_score']} in Top10={gold['in_top10']}")
        print(f"First loss label: {gold['first_loss_label']}")
    _print_stage_table("BM25 top 10", trace["bm25_top10"])
    _print_stage_table("MiniLM top 10", trace["dense_top10"])
    _print_stage_table("RRF top 20", trace["rrf_top20"])
    if trace["compact_trace"]["in_top50"]:
        _print_stage_table("MedCPT top 10", trace["medcpt_top10"])
    print("\nCOMPACT PIPELINE TRACE")
    print(format_compact_trace(trace))


def run_pipeline_diagnostic(
    *,
    query_ids: Sequence[Any] | None = None,
    passages: Sequence[Passage] | None = None,
    queries: Sequence[EvaluationQuery] | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    queries = list(queries or load_or_create_fixed_queries())
    baseline_rows = load_medcpt_baseline_rows()
    categories = categorize_evaluation_queries(queries, baseline_rows)
    selected = select_representative_queries(categories)
    wanted = list(query_ids) if query_ids is not None else [
        selected["query_a"]["query_id"],
        selected["query_b"]["query_id"],
    ]
    by_id = {query.question_id: query for query in queries}
    cat_by_id = {item["query_id"]: item["category"] for item in categories["rows"]}

    print("PART 1 — QUERY CATEGORIES (wording only; MedCPT Recall@10 from saved baseline)")
    print(format_category_table(categories))
    print(f"\nComplete misses (Recall@10=0): {categories['n_complete_misses']}")
    print(f"Category with the most complete misses: {categories['most_complete_misses_category']}")
    print(f"Category with the lowest mean Recall@10: {categories['lowest_mean_recall_category']}")
    print(categories["note"])
    print("\nPART 2 — SELECTED QUERIES")
    print(f"Query A (success): [{selected['query_a']['query_id']}] {selected['query_a']['question']}")
    print(f"  category={selected['query_a']['category']}")
    print(f"  gold IDs={selected['query_a']['gold_passage_ids']}")
    print(f"  {selected['reason_a']}")
    print(f"Query B (failure): [{selected['query_b']['query_id']}] {selected['query_b']['question']}")
    print(f"  category={selected['query_b']['category']}")
    print(f"  gold IDs={selected['query_b']['gold_passage_ids']}")
    print(f"  {selected['reason_b']}")

    if passages is None:
        print("\nLoading the BioASQ passage corpus for tracing...", flush=True)
        passages = get_passage_collection()
    nan_gold = 0
    gold_total = 0
    queries_with_nan = 0
    indexed = _id_map(passages)
    for query in queries:
        empty_ids = [
            gold_id
            for gold_id in query.relevant_passage_ids
            if gold_id is not None
            and (
                passage_id_key(gold_id) not in indexed
                or _is_empty_or_nan_text(indexed[passage_id_key(gold_id)].text)
            )
        ]
        gold_total += len(query.relevant_passage_ids)
        nan_gold += len(empty_ids)
        if empty_ids:
            queries_with_nan += 1
    print(
        f"Gold-label data check: {nan_gold}/{gold_total} gold IDs are missing or have empty/'nan' text "
        f"({queries_with_nan}/100 queries).",
        flush=True,
    )
    print(f"Building production BM25 + MiniLM + Hybrid indexes over {len(passages)} passages...", flush=True)
    built, timings = build_evaluation_retrievers(passages, top_k=10)
    reranker = CrossEncoderReranker(scorer=get_shared_scorer())
    reranker.preload()

    traces = []
    labels = {selected["query_a"]["query_id"]: "QUERY A", selected["query_b"]["query_id"]: "QUERY B"}
    for query_id in wanted:
        query = by_id[query_id]
        print(f"\nTracing query_id={query_id}...", flush=True)
        trace = trace_query(
            query,
            passages=passages,
            bm25=built["BM25"],
            dense=built["Dense"],
            hybrid=built["Hybrid RRF"],
            reranker=reranker,
            category=cat_by_id[query_id],
        )
        traces.append(trace)
        print_query_trace(labels.get(query_id, f"QUERY {query_id}"), trace)

    payload = {
        "production_unchanged": True,
        "query_expansion": False,
        "gold_label_data_check": {
            "n_gold_ids": gold_total,
            "n_missing_or_nan_gold_ids": nan_gold,
            "n_queries_with_nan_gold": queries_with_nan,
            "n_queries": len(queries),
        },
        "index_build_seconds": timings,
        "categories": {
            "table": categories["table"],
            "most_complete_misses_category": categories["most_complete_misses_category"],
            "lowest_mean_recall_category": categories["lowest_mean_recall_category"],
            "n_complete_misses": categories["n_complete_misses"],
            "note": categories["note"],
            "assignments": [
                {key: item[key] for key in ("query_id", "question", "category", "recall_at_10", "n_gold")}
                for item in categories["rows"]
            ],
        },
        "selected": selected,
        "traces": traces,
    }
    path = Path(output_path or "experiments/results/pipeline_diagnostic.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nWrote {path}", flush=True)
    return payload


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Diagnostic trace of the production RAG pipeline")
    parser.add_argument(
        "--query-ids",
        default="",
        help="Optional comma-separated query IDs. Default: representative success + failure.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    query_ids = None
    if args.query_ids.strip():
        query_ids = [int(part.strip()) for part in args.query_ids.split(",") if part.strip()]
    return run_pipeline_diagnostic(query_ids=query_ids)


if __name__ == "__main__":
    main()
