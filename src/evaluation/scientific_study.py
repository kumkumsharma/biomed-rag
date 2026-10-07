"""Scientific retrieval studies. Does not overwrite baseline JSON or change defaults."""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import EVALUATION, RETRIEVAL, UPGRADE
from src.data.loader import get_passage_collection
from src.data.preprocessing import Passage
from src.evaluation.dataset import EvaluationQuery, load_or_create_fixed_queries
from src.evaluation.metrics import (
    candidate_hit_at_k,
    candidate_recall_at_k,
    compute_retrieval_metrics,
    passage_id_key,
    rank_of_id,
    recall_at_k,
    rerank_recovery_at_k,
)
from src.evaluation.run_evaluation import build_evaluation_retrievers
from src.query.classify import classify_query
from src.query.decomposition import decompose_query
from src.query.expansion import expand_query
from src.query.terminology import detect_terms
from src.retrieval.bm25 import BM25Retriever, tokenize
from src.retrieval.dense import DenseRetriever, Embedder
from src.retrieval.embedding_models import (
    BIOMEDICAL_CANDIDATES,
    MINILM_DIMENSION,
    MINILM_MODEL,
    embedding_memory_bytes,
    huggingface_model_cached,
)
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker, PairScorer, get_shared_scorer

POOL_SIZES = (10, 20, 30, 50, 75, 100)
MEDCPT_POOLS = (20, 30, 50, 75, 100)
PROBE_K = 100


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (bytes, str, bool)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def _hit_ids(hits: Sequence[Any]) -> list[Any]:
    return [getattr(item, "passage_id") for item in hits]


def _metrics(ids: Sequence[Any], relevant: Sequence[Any], *, k50_ids: Sequence[Any] | None = None) -> dict[str, float]:
    payload = compute_retrieval_metrics(ids, relevant)
    pool = list(k50_ids if k50_ids is not None else ids)
    payload["recall_at_50"] = recall_at_k(pool, relevant, 50)
    return payload


def try_resolve_biomedical_model(*, prefer: str | None = None, allow_download: bool = True) -> dict[str, Any]:
    """Pick a cached or downloadable biomedical encoder. Never returns MiniLM as a silent swap."""

    ordered = []
    if prefer:
        ordered.append(prefer)
    for name in BIOMEDICAL_CANDIDATES:
        if name not in ordered:
            ordered.append(name)
    checked = []
    for name in ordered:
        cached = huggingface_model_cached(name)
        checked.append({"model": name, "cached": cached})
        if cached:
            return {"model": name, "available": True, "source": "cache", "checked": checked}
        if not allow_download:
            continue
        try:
            from sentence_transformers import SentenceTransformer

            print(f"Attempting to load biomedical encoder {name}...", flush=True)
            SentenceTransformer(name)
            return {"model": name, "available": True, "source": "download", "checked": checked}
        except Exception as exc:  # noqa: BLE001 — experiment must continue if a candidate fails
            checked[-1]["error"] = str(exc)[:300]
            print(f"  unavailable: {name}: {exc}", flush=True)
    return {"model": None, "available": False, "source": None, "checked": checked}


def run_embedding_comparison(
    *,
    passages: Sequence[Passage] | None = None,
    queries: Sequence[EvaluationQuery] | None = None,
    minilm_embedder: Embedder | None = None,
    biomedical_embedder: Embedder | None = None,
    biomedical_model_name: str | None = None,
    allow_download: bool = True,
    details_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    queries = list(queries or load_or_create_fixed_queries())
    if passages is None:
        print("Loading the full BioASQ passage corpus...", flush=True)
        passages = get_passage_collection()
    n_passages = len(passages)
    started = time.perf_counter()
    bm25 = BM25Retriever(passages, top_k=PROBE_K)
    bm25_s = time.perf_counter() - started

    started = time.perf_counter()
    minilm_kwargs: dict[str, Any] = {"top_k": PROBE_K, "model_name": MINILM_MODEL, "cache_dir": UPGRADE.embedding_cache_dir}
    if minilm_embedder is not None:
        minilm_kwargs["embedder"] = minilm_embedder
    minilm = DenseRetriever(passages, **minilm_kwargs)
    minilm_s = time.perf_counter() - started
    minilm_dim = int(getattr(minilm._embeddings, "shape", [0, MINILM_DIMENSION])[1])
    hybrid_minilm = HybridRetriever(passages, bm25_retriever=bm25, dense_retriever=minilm, top_k=PROBE_K)

    biomedical_info = {"model": biomedical_model_name, "available": biomedical_embedder is not None, "source": "injected"}
    biomedical_hybrid = None
    bio_s = None
    bio_dim = None
    bio_model = biomedical_model_name
    if biomedical_embedder is None:
        biomedical_info = try_resolve_biomedical_model(
            prefer=biomedical_model_name or UPGRADE.biomedical_embedding_model,
            allow_download=allow_download,
        )
        bio_model = biomedical_info.get("model")
        if biomedical_info.get("available") and bio_model:
            started = time.perf_counter()
            biomedical = DenseRetriever(
                passages,
                model_name=bio_model,
                top_k=PROBE_K,
                cache_dir=UPGRADE.embedding_cache_dir,
            )
            bio_s = time.perf_counter() - started
            bio_dim = int(getattr(biomedical._embeddings, "shape", [0, 0])[1])
            biomedical_hybrid = HybridRetriever(
                passages, bm25_retriever=bm25, dense_retriever=biomedical, top_k=PROBE_K
            )
    else:
        started = time.perf_counter()
        biomedical = DenseRetriever(
            passages,
            embedder=biomedical_embedder,
            model_name=bio_model or "injected-biomedical",
            top_k=PROBE_K,
            cache_dir=None,
        )
        bio_s = time.perf_counter() - started
        bio_dim = int(getattr(biomedical._embeddings, "shape", [0, 0])[1])
        biomedical_hybrid = HybridRetriever(
            passages, bm25_retriever=bm25, dense_retriever=biomedical, top_k=PROBE_K
        )

    configurations = {
        "A MiniLM baseline": hybrid_minilm,
    }
    if biomedical_hybrid is not None:
        configurations["B Biomedical embedding"] = biomedical_hybrid

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for name, retriever in configurations.items():
        print(f"Embedding comparison: {name}", flush=True)
        group: list[dict[str, Any]] = []
        for index, query in enumerate(queries, start=1):
            started = time.perf_counter()
            hits = list(retriever.search(query.question, top_k=PROBE_K))
            latency_ms = (time.perf_counter() - started) * 1000.0
            ids = _hit_ids(hits)
            metrics = _metrics(ids[:10], query.relevant_passage_ids, k50_ids=ids)
            row = {
                "query_id": _jsonable(query.question_id),
                "question": query.question,
                "configuration": name,
                "retrieved_passage_ids": [_jsonable(item) for item in ids[:10]],
                "pool_ids": [_jsonable(item) for item in ids[:50]],
                "latency_ms": latency_ms,
                **{key: float(metrics[key]) for key in ("recall_at_5", "recall_at_10", "recall_at_50", "mrr_at_10", "ndcg_at_10")},
            }
            rows.append(row)
            group.append(row)
            if index == len(queries) or index % 10 == 0:
                print(f"  {name}: {index}/{len(queries)}", flush=True)
        dense = getattr(retriever, "_dense", None)
        dim = int(getattr(getattr(dense, "_embeddings", None), "shape", [0, 0])[1] or 0)
        summaries.append(
            {
                "name": name,
                "n_queries": len(group),
                "recall_at_5": _mean([row["recall_at_5"] for row in group]),
                "recall_at_10": _mean([row["recall_at_10"] for row in group]),
                "recall_at_50": _mean([row["recall_at_50"] for row in group]),
                "mrr_at_10": _mean([row["mrr_at_10"] for row in group]),
                "ndcg_at_10": _mean([row["ndcg_at_10"] for row in group]),
                "avg_latency_ms": _mean([row["latency_ms"] for row in group]),
                "median_latency_ms": statistics.median([row["latency_ms"] for row in group]),
                "embedding_dimension": dim,
                "embedding_memory_bytes": embedding_memory_bytes(n_passages, dim) if dim else None,
            }
        )

    better = None
    if len(summaries) == 2:
        a, b = summaries[0], summaries[1]
        better = {
            "recall_at_10": "B" if b["recall_at_10"] > a["recall_at_10"] else ("A" if a["recall_at_10"] > b["recall_at_10"] else "tie"),
            "do_not_claim_without_measurement": True,
        }
    payload = {
        "n_passages": n_passages,
        "n_queries": len(queries),
        "queries_path": EVALUATION.queries_path,
        "minilm_model": MINILM_MODEL,
        "minilm_dimension": minilm_dim,
        "minilm_index_seconds": minilm_s,
        "bm25_index_seconds": bm25_s,
        "biomedical": biomedical_info,
        "biomedical_model": bio_model,
        "biomedical_dimension": bio_dim,
        "biomedical_index_seconds": bio_s,
        "minilm_memory_bytes": embedding_memory_bytes(n_passages, minilm_dim),
        "biomedical_memory_bytes": embedding_memory_bytes(n_passages, bio_dim) if bio_dim else None,
        "same_bm25": True,
        "same_rrf_k": RETRIEVAL.rrf_k,
        "configurations": summaries,
        "winner_by_recall_at_10": better,
        "rows": rows,
        "notes": [
            "MiniLM is not replaced. Biomedical results appear only when a model loaded.",
            "Do not claim improvement unless B metrics exceed A on this file.",
        ],
    }
    details = _write(details_path or EVALUATION.embedding_comparison_details_path, payload)
    summary = {key: value for key, value in payload.items() if key != "rows"}
    summary_path = _write(summary_path or EVALUATION.embedding_comparison_summary_path, summary)
    print(json.dumps(summaries, indent=2), flush=True)
    return {"details_path": details, "summary_path": summary_path, "summaries": summaries, "skipped": biomedical_hybrid is None}


def run_candidate_pool_experiment(
    *,
    passages: Sequence[Passage] | None = None,
    queries: Sequence[EvaluationQuery] | None = None,
    retrievers: Mapping[str, Any] | None = None,
    embedder: Embedder | None = None,
    details_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    queries = list(queries or load_or_create_fixed_queries())
    timings: dict[str, float] = {}
    if retrievers is None:
        if passages is None:
            print("Loading the full BioASQ passage corpus...", flush=True)
            passages = get_passage_collection()
        built, timings = build_evaluation_retrievers(passages, top_k=PROBE_K, embedder=embedder)
        hybrid = built["Hybrid RRF"]
    else:
        hybrid = retrievers["Hybrid RRF"]
        passages = passages or ()
    rows: list[dict[str, Any]] = []
    print(f"Candidate pool curve on {len(queries)} queries...", flush=True)
    for index, query in enumerate(queries, start=1):
        started = time.perf_counter()
        hits = list(hybrid.search(query.question, top_k=PROBE_K))
        latency_ms = (time.perf_counter() - started) * 1000.0
        ids = _hit_ids(hits)
        point = {
            "query_id": _jsonable(query.question_id),
            "question": query.question,
            "latency_ms": latency_ms,
            "candidate_count": len(ids),
        }
        for k in POOL_SIZES:
            point[f"recall_at_{k}"] = recall_at_k(ids, query.relevant_passage_ids, k)
        metrics10 = compute_retrieval_metrics(ids[:10], query.relevant_passage_ids)
        point["mrr_at_10"] = metrics10["mrr_at_10"]
        point["ndcg_at_10"] = metrics10["ndcg_at_10"]
        rows.append(point)
        if index == len(queries) or index % 10 == 0:
            print(f"  pool curve: {index}/{len(queries)}", flush=True)
    curve = []
    for k in POOL_SIZES:
        curve.append(
            {
                "k": k,
                "recall_at_k": _mean([row[f"recall_at_{k}"] for row in rows]),
                "avg_latency_ms": _mean([row["latency_ms"] for row in rows]),
                "candidate_count": k,
                "mrr_at_10": _mean([row["mrr_at_10"] for row in rows]) if k >= 10 else None,
                "ndcg_at_10": _mean([row["ndcg_at_10"] for row in rows]) if k >= 10 else None,
            }
        )
    payload = {
        "n_passages": len(passages) if passages is not None else 0,
        "n_queries": len(queries),
        "pool_sizes": list(POOL_SIZES),
        "index_build_seconds": timings,
        "curve": curve,
        "avg_mrr_at_10": _mean([row["mrr_at_10"] for row in rows]),
        "avg_ndcg_at_10": _mean([row["ndcg_at_10"] for row in rows]),
        "rows": rows,
        "notes": [
            "One Hybrid search at K=100 per query; smaller K are prefixes of that ranking.",
            "50 is not assumed optimal.",
        ],
    }
    details = _write(details_path or EVALUATION.candidate_pool_details_path, payload)
    summary = {key: value for key, value in payload.items() if key != "rows"}
    summary_path = _write(summary_path or EVALUATION.candidate_pool_summary_path, summary)
    print(json.dumps(curve, indent=2), flush=True)
    return {"details_path": details, "summary_path": summary_path, "curve": curve, "rows": rows}


def run_medcpt_pool_experiment(
    *,
    passages: Sequence[Passage] | None = None,
    queries: Sequence[EvaluationQuery] | None = None,
    hybrid: Any | None = None,
    scorer: PairScorer | None = None,
    embedder: Embedder | None = None,
    details_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    queries = list(queries or load_or_create_fixed_queries())
    timings: dict[str, float] = {}
    if hybrid is None:
        if passages is None:
            print("Loading the full BioASQ passage corpus...", flush=True)
            passages = get_passage_collection()
        built, timings = build_evaluation_retrievers(passages, top_k=PROBE_K, embedder=embedder)
        hybrid = built["Hybrid RRF"]
    reranker = CrossEncoderReranker(scorer=scorer or get_shared_scorer())
    rows: list[dict[str, Any]] = []
    print(f"MedCPT pool-size study on {len(queries)} queries...", flush=True)
    for index, query in enumerate(queries, start=1):
        started = time.perf_counter()
        pool = list(hybrid.search(query.question, top_k=PROBE_K))
        hybrid_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        texts = [getattr(item, "text", "") for item in pool]
        scores = list(reranker._scorer.score_pairs(query.question, texts)) if pool else []
        score_ms = (time.perf_counter() - started) * 1000.0
        pool_ids = _hit_ids(pool)
        per_k = {}
        for k in MEDCPT_POOLS:
            subset = list(range(min(k, len(pool))))
            ranked = sorted(subset, key=lambda i: (-float(scores[i]), i))[:10]
            final_ids = [pool_ids[i] for i in ranked]
            metrics = compute_retrieval_metrics(final_ids, query.relevant_passage_ids)
            per_k[str(k)] = {
                **metrics,
                "candidate_recall": candidate_recall_at_k(pool_ids, query.relevant_passage_ids, k),
                "candidate_hit": candidate_hit_at_k(pool_ids, query.relevant_passage_ids, k),
                "rerank_recovery_at_10": rerank_recovery_at_k(
                    pool_ids, final_ids, query.relevant_passage_ids, pool_k=k, final_k=10
                ),
                "final_ids": [_jsonable(item) for item in final_ids],
            }
        rows.append(
            {
                "query_id": _jsonable(query.question_id),
                "question": query.question,
                "hybrid_ms": hybrid_ms,
                "score_ms_for_100": score_ms,
                "by_pool": per_k,
            }
        )
        if index == len(queries) or index % 10 == 0:
            print(f"  medcpt pools: {index}/{len(queries)}", flush=True)
    summaries = []
    for k in MEDCPT_POOLS:
        recoveries = [
            row["by_pool"][str(k)]["rerank_recovery_at_10"]
            for row in rows
            if row["by_pool"][str(k)]["rerank_recovery_at_10"] is not None
        ]
        # Approximate MedCPT latency for pool k as score_ms_for_100 * (k/100)
        summaries.append(
            {
                "candidate_pool": k,
                "recall_at_5": _mean([row["by_pool"][str(k)]["recall_at_5"] for row in rows]),
                "recall_at_10": _mean([row["by_pool"][str(k)]["recall_at_10"] for row in rows]),
                "mrr_at_10": _mean([row["by_pool"][str(k)]["mrr_at_10"] for row in rows]),
                "ndcg_at_10": _mean([row["by_pool"][str(k)]["ndcg_at_10"] for row in rows]),
                "candidate_recall": _mean([row["by_pool"][str(k)]["candidate_recall"] for row in rows]),
                "rerank_recovery_at_10": _mean(recoveries) if recoveries else None,
                "avg_hybrid_ms": _mean([row["hybrid_ms"] for row in rows]),
                "avg_score_ms_for_100": _mean([row["score_ms_for_100"] for row in rows]),
                "estimated_medcpt_ms": _mean([row["score_ms_for_100"] * (k / PROBE_K) for row in rows]),
                "estimated_total_ms": _mean(
                    [row["hybrid_ms"] + row["score_ms_for_100"] * (k / PROBE_K) for row in rows]
                ),
            }
        )
    payload = {
        "n_passages": len(passages) if passages is not None else 0,
        "n_queries": len(queries),
        "index_build_seconds": timings,
        "scoring_note": (
            "Pairs were scored once on Hybrid@100. Pool-k quality uses the first k candidates "
            "reordered by those scores. estimated_medcpt_ms scales the 100-pair scoring time by k/100; "
            "it is not a separate timed run of each pool."
        ),
        "configurations": summaries,
        "rows": rows,
        "default_pool_unchanged": 50,
    }
    details = _write(details_path or EVALUATION.medcpt_pool_details_path, payload)
    summary = {key: value for key, value in payload.items() if key != "rows"}
    summary_path = _write(summary_path or EVALUATION.medcpt_pool_summary_path, summary)
    print(json.dumps(summaries, indent=2), flush=True)
    return {"details_path": details, "summary_path": summary_path, "summaries": summaries}


def _token_overlap(query: str, text: str) -> float:
    q = set(tokenize(query))
    p = set(tokenize(text))
    if not q or not p:
        return 0.0
    return len(q & p) / len(q)


def categorize_missing_evidence(
    *,
    question: str,
    relevant_id: Any,
    bm25_rank: int | None,
    dense_rank: int | None,
    rrf_rank: int | None,
    in_corpus: bool,
    overlap: float,
    probe_k: int = PROBE_K,
    hybrid_k: int = 50,
) -> str:
    if not in_corpus:
        return "dataset/relevance issue"
    query_type = classify_query(question).query_type
    terms = detect_terms(question)
    if rrf_rank is not None and rrf_rank <= hybrid_k:
        return "other"
    if bm25_rank is None and dense_rank is None:
        if terms:
            return "terminology mismatch"
        if query_type == "multi_part":
            return "query formulation issue"
        if overlap < 0.15:
            return "lexical mismatch"
        if query_type == "semantic":
            return "semantic mismatch"
        return "both BM25 and dense failure"
    if bm25_rank is None and dense_rank is not None:
        return "BM25 failure"
    if dense_rank is None and bm25_rank is not None:
        if query_type == "semantic":
            return "semantic mismatch"
        return "dense retrieval failure"
    return "other"


def run_missing_evidence_study(
    *,
    passages: Sequence[Passage] | None = None,
    queries: Sequence[EvaluationQuery] | None = None,
    bm25: Any | None = None,
    dense: Any | None = None,
    hybrid: Any | None = None,
    embedder: Embedder | None = None,
    hybrid_k: int = 50,
    details_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    queries = list(queries or load_or_create_fixed_queries())
    if hybrid is None or bm25 is None or dense is None:
        if passages is None:
            print("Loading the full BioASQ passage corpus...", flush=True)
            passages = get_passage_collection()
        built, _timings = build_evaluation_retrievers(passages, top_k=PROBE_K, embedder=embedder)
        bm25 = built["BM25"]
        dense = built["Dense"]
        hybrid = built["Hybrid RRF"]
    texts = {}
    if passages is not None:
        texts = {passage.passage_id: passage.text for passage in passages}
        texts.update({passage_id_key(passage.passage_id): passage.text for passage in passages})
    rows: list[dict[str, Any]] = []
    print(f"Missing-evidence study (Hybrid@{hybrid_k}) on {len(queries)} queries...", flush=True)
    for index, query in enumerate(queries, start=1):
        bm25_hits = list(bm25.search(query.question, top_k=PROBE_K))
        dense_hits = list(dense.search(query.question, top_k=PROBE_K))
        hybrid_hits = list(hybrid.search(query.question, top_k=PROBE_K))
        bm25_ids = _hit_ids(bm25_hits)
        dense_ids = _hit_ids(dense_hits)
        hybrid_ids = _hit_ids(hybrid_hits)
        missed = []
        for relevant in query.relevant_passage_ids:
            in_corpus = passage_id_key(relevant) in texts or relevant in texts
            text = texts.get(relevant) or texts.get(passage_id_key(relevant)) or ""
            bm25_rank = rank_of_id(bm25_ids, relevant)
            dense_rank = rank_of_id(dense_ids, relevant)
            rrf_rank = rank_of_id(hybrid_ids, relevant)
            if rrf_rank is not None and rrf_rank <= hybrid_k:
                continue
            category = categorize_missing_evidence(
                question=query.question,
                relevant_id=relevant,
                bm25_rank=bm25_rank,
                dense_rank=dense_rank,
                rrf_rank=rrf_rank,
                in_corpus=in_corpus,
                overlap=_token_overlap(query.question, text),
                hybrid_k=hybrid_k,
            )
            missed.append(
                {
                    "relevant_passage_id": _jsonable(relevant),
                    "bm25_retrieved": bm25_rank is not None,
                    "dense_retrieved": dense_rank is not None,
                    "bm25_rank": bm25_rank,
                    "dense_rank": dense_rank,
                    "rrf_rank": rrf_rank,
                    "in_corpus": in_corpus,
                    "token_overlap": _token_overlap(query.question, text),
                    "category": category,
                }
            )
        if missed:
            rows.append(
                {
                    "query_id": _jsonable(query.question_id),
                    "question": query.question,
                    "query_type": classify_query(query.question).query_type,
                    "detected_terms": detect_terms(query.question),
                    "misses": missed,
                    "primary_category": missed[0]["category"],
                }
            )
        if index == len(queries) or index % 10 == 0:
            print(f"  missing evidence: {index}/{len(queries)}", flush=True)
    counts: dict[str, int] = {}
    passage_counts: dict[str, int] = {}
    n_miss_passages = 0
    for row in rows:
        counts[row["primary_category"]] = counts.get(row["primary_category"], 0) + 1
        for miss in row.get("misses") or []:
            n_miss_passages += 1
            category = str(miss.get("category") or "other")
            passage_counts[category] = passage_counts.get(category, 0) + 1
    n_miss_queries = len(rows)
    remedies = {
        "lexical mismatch": "Stronger lexical matching / domain synonyms for BM25",
        "semantic mismatch": "Biomedical dense encoder or larger dense pool",
        "terminology mismatch": "Better BioASQ-specific abbreviation handling",
        "query formulation issue": "Selective decomposition for multi-part questions",
        "BM25 failure": "Query rewriting for lexical gaps; keep dense",
        "dense retrieval failure": "Biomedical embeddings; do not rely on MiniLM alone",
        "both BM25 and dense failure": "First-stage recall (larger K, domain encoder, or different lexical model)",
        "dataset/relevance issue": "Inspect golden IDs / corpus coverage",
        "other": "Inspect fusion cutoffs and rank distributions",
    }
    table = []
    for category, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        table.append(
            {
                "failure_category": category,
                "count": count,
                "percentage": (count / n_miss_queries) if n_miss_queries else 0.0,
                "potential_remedy": remedies.get(category, ""),
            }
        )
    passage_table = []
    for category, count in sorted(passage_counts.items(), key=lambda item: (-item[1], item[0])):
        passage_table.append(
            {
                "failure_category": category,
                "count": count,
                "percentage": (count / n_miss_passages) if n_miss_passages else 0.0,
                "potential_remedy": remedies.get(category, ""),
            }
        )
    payload = {
        "n_queries": len(queries),
        "hybrid_k": hybrid_k,
        "probe_k": PROBE_K,
        "n_queries_missing_from_hybrid": n_miss_queries,
        "n_missed_relevant_ids": n_miss_passages,
        "ceiling_gap_note": (
            "Query table: queries with at least one golden ID absent from Hybrid@50. "
            "Passage table: individual missed relevant IDs (the Recall@50 gap)."
        ),
        "table": table,
        "passage_table": passage_table,
        "rows": rows,
    }
    details = _write(details_path or EVALUATION.missing_evidence_details_path, payload)
    summary = {key: value for key, value in payload.items() if key != "rows"}
    summary_path = _write(summary_path or EVALUATION.missing_evidence_summary_path, summary)
    print(json.dumps(table, indent=2), flush=True)
    return {"details_path": details, "summary_path": summary_path, "table": table, "rows": rows}


def run_expansion_study(
    *,
    passages: Sequence[Passage] | None = None,
    queries: Sequence[EvaluationQuery] | None = None,
    hybrid: HybridRetriever | None = None,
    embedder: Embedder | None = None,
    top_k: int = 50,
    details_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    queries = list(queries or load_or_create_fixed_queries())
    if hybrid is None:
        if passages is None:
            print("Loading the full BioASQ passage corpus...", flush=True)
            passages = get_passage_collection()
        built, _timings = build_evaluation_retrievers(passages, top_k=top_k, embedder=embedder)
        hybrid = built["Hybrid RRF"]
        assert isinstance(hybrid, HybridRetriever)
    rows = []
    print(f"Expansion study on {len(queries)} queries...", flush=True)
    for index, query in enumerate(queries, start=1):
        expansion = expand_query(query.question, enabled=True, max_variants=4)
        orig_hits = list(hybrid.search(query.question, top_k=top_k, expansions=None))
        equal_hits = list(hybrid.search(query.question, top_k=top_k, expansions=expansion.expansions, expansion_mode="equal"))
        priority_hits = list(
            hybrid.search(query.question, top_k=top_k, expansions=expansion.expansions, expansion_mode="priority")
        )
        orig_ids = _hit_ids(orig_hits)
        equal_ids = _hit_ids(equal_hits)
        priority_ids = _hit_ids(priority_hits)
        relevant = {passage_id_key(item) for item in query.relevant_passage_ids}
        orig_rel = {passage_id_key(item) for item in orig_ids if passage_id_key(item) in relevant}
        equal_rel = {passage_id_key(item) for item in equal_ids if passage_id_key(item) in relevant}
        new_relevant = sorted(equal_rel - orig_rel)
        lost_relevant = sorted(orig_rel - equal_rel)
        rows.append(
            {
                "query_id": _jsonable(query.question_id),
                "question": query.question,
                "expansions": list(expansion.expansions),
                "detected_terms": detect_terms(query.question),
                "original_recall_at_10": recall_at_k(orig_ids, query.relevant_passage_ids, 10),
                "equal_recall_at_10": recall_at_k(equal_ids, query.relevant_passage_ids, 10),
                "priority_recall_at_10": recall_at_k(priority_ids, query.relevant_passage_ids, 10),
                "original_recall_at_50": recall_at_k(orig_ids, query.relevant_passage_ids, 50),
                "equal_recall_at_50": recall_at_k(equal_ids, query.relevant_passage_ids, 50),
                "priority_recall_at_50": recall_at_k(priority_ids, query.relevant_passage_ids, 50),
                "new_relevant_ids": new_relevant,
                "lost_relevant_ids": lost_relevant,
                "expansion_retrieved_new_relevant": bool(new_relevant),
                "expansion_displaced_relevant": bool(lost_relevant),
                "introduced_variants": list(expansion.expansions),
            }
        )
        if index == len(queries) or index % 10 == 0:
            print(f"  expansion: {index}/{len(queries)}", flush=True)
    summary = {
        "n_queries": len(rows),
        "n_with_variants": sum(1 for row in rows if row["expansions"]),
        "n_new_relevant": sum(1 for row in rows if row["expansion_retrieved_new_relevant"]),
        "n_displaced_relevant": sum(1 for row in rows if row["expansion_displaced_relevant"]),
        "original_recall_at_10": _mean([row["original_recall_at_10"] for row in rows]),
        "equal_recall_at_10": _mean([row["equal_recall_at_10"] for row in rows]),
        "priority_recall_at_10": _mean([row["priority_recall_at_10"] for row in rows]),
        "original_recall_at_50": _mean([row["original_recall_at_50"] for row in rows]),
        "equal_recall_at_50": _mean([row["equal_recall_at_50"] for row in rows]),
        "priority_recall_at_50": _mean([row["priority_recall_at_50"] for row in rows]),
        "priority_is_default": False,
        "notes": [
            "Equal-weight fusion is the previous expansion behaviour.",
            "Priority fusion keeps original ranking and appends expansion-only IDs.",
            "Priority is not the production default unless this file shows a gain.",
        ],
    }
    payload = {**summary, "rows": rows}
    details = _write(details_path or EVALUATION.expansion_study_details_path, payload)
    summary_path = _write(summary_path or EVALUATION.expansion_study_summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return {"details_path": details, "summary_path": summary_path, "summary": summary, "rows": rows}


def run_decomposition_study(
    *,
    passages: Sequence[Passage] | None = None,
    queries: Sequence[EvaluationQuery] | None = None,
    hybrid: HybridRetriever | None = None,
    embedder: Embedder | None = None,
    top_k: int = 10,
    details_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    queries = list(queries or load_or_create_fixed_queries())
    if hybrid is None:
        if passages is None:
            print("Loading the full BioASQ passage corpus...", flush=True)
            passages = get_passage_collection()
        built, _timings = build_evaluation_retrievers(passages, top_k=max(top_k, 50), embedder=embedder)
        hybrid = built["Hybrid RRF"]
        assert isinstance(hybrid, HybridRetriever)
    rows = []
    print(f"Decomposition study on {len(queries)} queries...", flush=True)
    for index, query in enumerate(queries, start=1):
        classification = classify_query(query.question)
        decomposition = decompose_query(query.question, enabled=True)
        started = time.perf_counter()
        base_hits = list(hybrid.search(query.question, top_k=top_k, expansions=None))
        base_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        split_hits = list(
            hybrid.search(
                query.question,
                top_k=top_k,
                expansions=decomposition.subqueries if decomposition.applied else None,
            )
        )
        split_ms = (time.perf_counter() - started) * 1000.0
        base_ids = _hit_ids(base_hits)
        split_ids = _hit_ids(split_hits)
        rows.append(
            {
                "query_id": _jsonable(query.question_id),
                "question": query.question,
                "query_type": classification.query_type,
                "aspect_count": classification.aspect_count,
                "decomposed": decomposition.applied,
                "n_subqueries": len(decomposition.subqueries),
                "subqueries": list(decomposition.subqueries),
                "base_recall_at_10": recall_at_k(base_ids, query.relevant_passage_ids, 10),
                "decomp_recall_at_10": recall_at_k(split_ids, query.relevant_passage_ids, 10),
                "base_ndcg_at_10": compute_retrieval_metrics(base_ids, query.relevant_passage_ids)["ndcg_at_10"],
                "decomp_ndcg_at_10": compute_retrieval_metrics(split_ids, query.relevant_passage_ids)["ndcg_at_10"],
                "base_latency_ms": base_ms,
                "decomp_latency_ms": split_ms,
            }
        )
        if index == len(queries) or index % 10 == 0:
            print(f"  decomposition: {index}/{len(queries)}", flush=True)
    complex_rows = [row for row in rows if row["decomposed"]]
    simple_rows = [row for row in rows if not row["decomposed"]]

    def _block(group: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
        if not group:
            return {"n_queries": 0}
        return {
            "n_queries": len(group),
            f"{prefix}_recall_at_10": _mean([row["base_recall_at_10"] for row in group]),
            f"{prefix}_decomp_recall_at_10": _mean([row["decomp_recall_at_10"] for row in group]),
            f"{prefix}_ndcg_at_10": _mean([row["base_ndcg_at_10"] for row in group]),
            f"{prefix}_decomp_ndcg_at_10": _mean([row["decomp_ndcg_at_10"] for row in group]),
            f"{prefix}_latency_ms": _mean([row["base_latency_ms"] for row in group]),
            f"{prefix}_decomp_latency_ms": _mean([row["decomp_latency_ms"] for row in group]),
        }

    summary = {
        "n_queries": len(rows),
        "n_decomposed": len(complex_rows),
        "n_not_decomposed": len(simple_rows),
        "selective_activation": True,
        "all_queries": _block(rows, "all"),
        "complex_queries": _block(complex_rows, "complex"),
        "simple_queries": _block(simple_rows, "simple"),
        "notes": [
            "Simple queries are not decomposed by the heuristic.",
            "Decomposition remains default OFF.",
        ],
    }
    payload = {**summary, "rows": rows}
    details = _write(details_path or EVALUATION.decomposition_study_details_path, payload)
    summary_path = _write(summary_path or EVALUATION.decomposition_study_summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return {"details_path": details, "summary_path": summary_path, "summary": summary, "rows": rows}


def run_scientific_suite(*, include_medcpt: bool = True, allow_download: bool = True) -> dict[str, Any]:
    print("Loading corpus and MiniLM hybrid indexes once...", flush=True)
    passages = get_passage_collection()
    queries = load_or_create_fixed_queries()
    built, timings = build_evaluation_retrievers(passages, top_k=PROBE_K)
    hybrid = built["Hybrid RRF"]
    dense = built["Dense"]
    results = {
        "index_build_seconds": timings,
        "embedding": run_embedding_comparison(
            passages=passages,
            queries=queries,
            minilm_embedder=getattr(dense, "_embedder", None),
            allow_download=allow_download,
        ),
        "candidate_pool": run_candidate_pool_experiment(
            passages=passages,
            queries=queries,
            retrievers=built,
        ),
        "missing_evidence": run_missing_evidence_study(
            passages=passages,
            queries=queries,
            bm25=built["BM25"],
            dense=built["Dense"],
            hybrid=hybrid,
        ),
        "expansion": run_expansion_study(passages=passages, queries=queries, hybrid=hybrid),
        "decomposition": run_decomposition_study(passages=passages, queries=queries, hybrid=hybrid),
    }
    if include_medcpt:
        results["medcpt_pool"] = run_medcpt_pool_experiment(
            passages=passages,
            queries=queries,
            hybrid=hybrid,
        )
    else:
        results["medcpt_pool"] = {"skipped": True}
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Scientific retrieval studies")
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--candidate-pool", action="store_true")
    parser.add_argument("--medcpt-pool", action="store_true")
    parser.add_argument("--missing-evidence", action="store_true")
    parser.add_argument("--expansion", action="store_true")
    parser.add_argument("--decomposition", action="store_true")
    parser.add_argument("--objective-failure", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--skip-medcpt", action="store_true")
    args = parser.parse_args()
    allow_download = not args.no_download
    ran = False
    if args.all:
        run_scientific_suite(include_medcpt=not args.skip_medcpt, allow_download=allow_download)
        return
    if args.embedding:
        run_embedding_comparison(allow_download=allow_download)
        ran = True
    if args.candidate_pool:
        run_candidate_pool_experiment()
        ran = True
    if args.medcpt_pool:
        run_medcpt_pool_experiment()
        ran = True
    if args.missing_evidence:
        run_missing_evidence_study()
        ran = True
    if args.expansion:
        run_expansion_study()
        ran = True
    if args.decomposition:
        run_decomposition_study()
        ran = True
    if args.objective_failure:
        from src.evaluation.objective_failure import run_objective_failure_study

        run_objective_failure_study()
        ran = True
    if not ran:
        run_scientific_suite(include_medcpt=not args.skip_medcpt, allow_download=allow_download)


if __name__ == "__main__":
    main()
