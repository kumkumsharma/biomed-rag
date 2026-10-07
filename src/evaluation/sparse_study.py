"""Learned-sparse retrieval experiment. Does not change production defaults.

Uses query text and passage text only. Gold IDs are used after retrieval to
score recovery; they are never passed into the sparse encoder.
"""

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
    compute_retrieval_metrics,
    passage_id_key,
    rank_of_id,
    recall_at_k,
)
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever, Embedder
from src.retrieval.embedding_models import MINILM_MODEL, huggingface_model_cached
from src.retrieval.hybrid import rrf_fuse_retrievers
from src.retrieval.sparse import SparseEmbedder, SparseRetriever
from src.retrieval.sparse_models import DEFAULT_SPARSE_MODEL, SPARSE_CANDIDATES, try_resolve_sparse_model

PROBE_K = 200
DUAL_MISS = "BOTH_RETRIEVERS_MISS"
MATERIAL_DUAL_RECOVERY_FOR_MEDCPT = 10


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


def load_dual_miss_rows(path: str | Path | None = None) -> list[dict[str, Any]]:
    target = Path(path or EVALUATION.objective_failure_details_path)
    if not target.exists():
        return []
    payload = json.loads(target.read_text(encoding="utf-8"))
    rows = payload.get("failure_rows") or payload.get("rows") or []
    return [row for row in rows if row.get("objective_category") == DUAL_MISS]


def dual_miss_recovery_rate(*, recovered: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return recovered / denominator


def _score_retriever(
    name: str,
    retriever: Any,
    queries: Sequence[EvaluationQuery],
    *,
    top_k: int,
) -> dict[str, Any]:
    rows = []
    for query in queries:
        started = time.perf_counter()
        hits = list(retriever.search(query.question, top_k=top_k))
        latency_ms = (time.perf_counter() - started) * 1000.0
        ids = _hit_ids(hits)
        metrics = compute_retrieval_metrics(ids[:10], query.relevant_passage_ids)
        rows.append(
            {
                "query_id": _jsonable(query.question_id),
                "latency_ms": latency_ms,
                "recall_at_50": recall_at_k(ids, query.relevant_passage_ids, 50),
                "recall_at_100": recall_at_k(ids, query.relevant_passage_ids, 100),
                **{key: float(metrics[key]) for key in ("recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10")},
            }
        )
    return {
        "name": name,
        "n_queries": len(rows),
        "recall_at_5": _mean([row["recall_at_5"] for row in rows]),
        "recall_at_10": _mean([row["recall_at_10"] for row in rows]),
        "recall_at_50": _mean([row["recall_at_50"] for row in rows]),
        "recall_at_100": _mean([row["recall_at_100"] for row in rows]),
        "mrr_at_10": _mean([row["mrr_at_10"] for row in rows]),
        "ndcg_at_10": _mean([row["ndcg_at_10"] for row in rows]),
        "avg_latency_ms": _mean([row["latency_ms"] for row in rows]),
        "median_latency_ms": statistics.median([row["latency_ms"] for row in rows]) if rows else 0.0,
        "rows": rows,
    }


def run_sparse_study(
    *,
    passages: Sequence[Passage] | None = None,
    queries: Sequence[EvaluationQuery] | None = None,
    sparse_embedder: SparseEmbedder | None = None,
    minilm_embedder: Embedder | None = None,
    biomedical_embedder: Embedder | None = None,
    allow_download: bool = True,
    include_medcpt: bool = True,
    dual_miss_rows: Sequence[Mapping[str, Any]] | None = None,
    details_path: str | Path | None = None,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    queries = list(queries or load_or_create_fixed_queries())
    if passages is None:
        print("Loading the full BioASQ passage corpus...", flush=True)
        passages = get_passage_collection()
    passages = list(passages)
    texts = {passage.passage_id: passage.text for passage in passages}
    dual_rows = list(dual_miss_rows if dual_miss_rows is not None else load_dual_miss_rows())
    n_dual = len(dual_rows)

    sparse_info = {"model": None, "available": sparse_embedder is not None, "source": "injected"}
    sparse = None
    if sparse_embedder is not None:
        sparse = SparseRetriever(
            passages,
            embedder=sparse_embedder,
            model_name=getattr(sparse_embedder, "model_name", "injected-sparse"),
            top_k=PROBE_K,
            cache_dir=None,
        )
        sparse_info = {"model": sparse.model_name, "available": True, "source": "injected"}
    else:
        sparse_info = try_resolve_sparse_model(
            prefer=UPGRADE.sparse_model_name or DEFAULT_SPARSE_MODEL,
            allow_download=allow_download,
        )
        if sparse_info.get("available") and sparse_info.get("model"):
            print(f"Indexing sparse retriever {sparse_info['model']}...", flush=True)
            sparse = SparseRetriever(
                passages,
                model_name=str(sparse_info["model"]),
                top_k=PROBE_K,
                cache_dir=UPGRADE.sparse_cache_dir,
            )

    leakage_note = (
        "The sparse encoder is given only query strings and passage text. "
        "Gold passage IDs, relevance labels, and reference answers are used after "
        "retrieval for scoring and are never passed into encode()."
    )

    if sparse is None:
        payload = {
            "skipped": True,
            "reason": "No learned sparse model could be loaded.",
            "candidates": list(SPARSE_CANDIDATES),
            "sparse": sparse_info,
            "leakage_note": leakage_note,
            "n_dual_misses": n_dual,
            "production_defaults_unchanged": True,
            "status": "EXPERIMENTAL",
        }
        details = _write(details_path or EVALUATION.sparse_study_details_path, payload)
        summary = _write(summary_path or EVALUATION.sparse_study_summary_path, payload)
        print(json.dumps(payload, indent=2), flush=True)
        return {"skipped": True, "details_path": details, "summary_path": summary, "summary": payload}

    print(f"Sparse dual-miss recovery on {n_dual} gold IDs...", flush=True)
    dual_by_query: dict[str, list[dict[str, Any]]] = {}
    for row in dual_rows:
        dual_by_query.setdefault(passage_id_key(row.get("query_id")), []).append(dict(row))

    recovered_at: dict[int, int] = {10: 0, 50: 0, 100: 0, 200: 0}
    dual_detail = []
    sparse_ranks: list[int] = []
    query_cache: dict[str, list[Any]] = {}
    for query in queries:
        key = passage_id_key(query.question_id)
        needed = dual_by_query.get(key) or []
        if not needed:
            continue
        hits = list(sparse.search(query.question, top_k=PROBE_K))
        ids = _hit_ids(hits)
        query_cache[key] = ids
        for item in needed:
            gold = item.get("gold_passage_id")
            rank = rank_of_id(ids, gold)
            if rank is not None:
                sparse_ranks.append(rank)
            for k in recovered_at:
                if rank is not None and rank <= k:
                    recovered_at[k] += 1
            dual_detail.append(
                {
                    "query_id": _jsonable(query.question_id),
                    "question": query.question,
                    "gold_passage_id": _jsonable(gold),
                    "sparse_rank": rank,
                    "hit_at_10": rank is not None and rank <= 10,
                    "hit_at_50": rank is not None and rank <= 50,
                    "hit_at_100": rank is not None and rank <= 100,
                    "hit_at_200": rank is not None and rank <= 200,
                }
            )

    scored_keys = set(query_cache)
    for key, needed in dual_by_query.items():
        if key in scored_keys:
            continue
        question = str(needed[0].get("question") or "")
        hits = list(sparse.search(question, top_k=PROBE_K))
        ids = _hit_ids(hits)
        query_cache[key] = ids
        for item in needed:
            gold = item.get("gold_passage_id")
            rank = rank_of_id(ids, gold)
            if rank is not None:
                sparse_ranks.append(rank)
            for k in recovered_at:
                if rank is not None and rank <= k:
                    recovered_at[k] += 1
            dual_detail.append(
                {
                    "query_id": _jsonable(item.get("query_id")),
                    "question": question,
                    "gold_passage_id": _jsonable(gold),
                    "sparse_rank": rank,
                    "hit_at_10": rank is not None and rank <= 10,
                    "hit_at_50": rank is not None and rank <= 50,
                    "hit_at_100": rank is not None and rank <= 100,
                    "hit_at_200": rank is not None and rank <= 200,
                }
            )

    recovery_100 = dual_miss_recovery_rate(recovered=recovered_at[100], denominator=n_dual) if n_dual else None
    newly_recovered_ids = [item["gold_passage_id"] for item in dual_detail if item["hit_at_100"]]
    still_missing_ids = [item["gold_passage_id"] for item in dual_detail if not item["hit_at_100"]]
    recovery_by_query: dict[str, dict[str, Any]] = {}
    for item in dual_detail:
        qid = passage_id_key(item["query_id"])
        bucket = recovery_by_query.setdefault(
            qid,
            {
                "query_id": item["query_id"],
                "question": item["question"],
                "n_dual_miss": 0,
                "hit_at_100": 0,
            },
        )
        bucket["n_dual_miss"] += 1
        if item["hit_at_100"]:
            bucket["hit_at_100"] += 1
    rank_histogram = {"1-10": 0, "11-50": 0, "51-100": 0, "101-200": 0, "not_in_200": 0}
    for item in dual_detail:
        rank = item["sparse_rank"]
        if rank is None:
            rank_histogram["not_in_200"] += 1
        elif rank <= 10:
            rank_histogram["1-10"] += 1
        elif rank <= 50:
            rank_histogram["11-50"] += 1
        elif rank <= 100:
            rank_histogram["51-100"] += 1
        else:
            rank_histogram["101-200"] += 1

    print("Building BM25 and MiniLM for full-corpus comparison...", flush=True)
    bm25 = BM25Retriever(passages, top_k=PROBE_K)
    minilm_kwargs: dict[str, Any] = {
        "top_k": PROBE_K,
        "model_name": MINILM_MODEL,
        "cache_dir": UPGRADE.embedding_cache_dir,
    }
    if minilm_embedder is not None:
        minilm_kwargs["embedder"] = minilm_embedder
        minilm_kwargs["cache_dir"] = None
    minilm = DenseRetriever(passages, **minilm_kwargs)

    pubmed = None
    pubmed_name = UPGRADE.biomedical_embedding_model
    if biomedical_embedder is not None:
        pubmed = DenseRetriever(
            passages, embedder=biomedical_embedder, model_name="injected-biomedical", top_k=PROBE_K, cache_dir=None
        )
    elif minilm_embedder is None and huggingface_model_cached(pubmed_name):
        pubmed = DenseRetriever(
            passages, model_name=pubmed_name, top_k=PROBE_K, cache_dir=UPGRADE.embedding_cache_dir
        )

    comparisons = [
        _score_retriever("BM25", bm25, queries, top_k=100),
        _score_retriever("MiniLM Dense", minilm, queries, top_k=100),
        _score_retriever("Learned sparse", sparse, queries, top_k=100),
    ]
    if pubmed is not None:
        comparisons.insert(2, _score_retriever("PubMedBERT Dense", pubmed, queries, top_k=100))

    def _without_rows(block: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in block.items() if key != "rows"}

    print("Fusion experiments...", flush=True)

    def _fused(name: str, retrievers: Sequence[Any]) -> dict[str, Any]:
        class _Fused:
            def search(self, query: str, top_k: int | None = None):
                return rrf_fuse_retrievers(
                    retrievers,
                    query,
                    texts=texts,
                    top_k=top_k or 100,
                    rrf_k=RETRIEVAL.rrf_k,
                    candidate_k=100,
                )

        return _score_retriever(name, _Fused(), queries, top_k=100)

    fusion = [
        _fused("A BM25 + MiniLM", [bm25, minilm]),
        _fused("B BM25 + MiniLM + Sparse", [bm25, minilm, sparse]),
    ]
    if pubmed is not None:
        fusion.append(_fused("C BM25 + PubMedBERT + Sparse", [bm25, pubmed, sparse]))

    # Dual-miss reachability under each fusion at @50 and @100
    fusion_dual = []
    for block, retrievers in (
        ("A BM25 + MiniLM", [bm25, minilm]),
        ("B BM25 + MiniLM + Sparse", [bm25, minilm, sparse]),
        *((("C BM25 + PubMedBERT + Sparse", [bm25, pubmed, sparse]),) if pubmed is not None else ()),
    ):
        hits_50 = 0
        hits_100 = 0
        for query in queries:
            needed = dual_by_query.get(passage_id_key(query.question_id)) or []
            if not needed:
                continue
            ids = _hit_ids(
                rrf_fuse_retrievers(
                    retrievers, query.question, texts=texts, top_k=100, rrf_k=RETRIEVAL.rrf_k, candidate_k=100
                )
            )
            for item in needed:
                rank = rank_of_id(ids, item.get("gold_passage_id"))
                if rank is not None and rank <= 50:
                    hits_50 += 1
                if rank is not None and rank <= 100:
                    hits_100 += 1
        fusion_dual.append(
            {
                "name": block,
                "dual_miss_hit_at_50": hits_50,
                "dual_miss_hit_at_100": hits_100,
                "dual_miss_recovery_at_100": dual_miss_recovery_rate(recovered=hits_100, denominator=n_dual),
            }
        )

    # Sparse-only among dual misses: sparse@100 hit and MiniLM dense miss and BM25 miss (already true)
    sparse_only = sum(1 for item in dual_detail if item["hit_at_100"])
    still_missing = n_dual - sparse_only

    # Recovery matrix extension for all Hybrid@50 misses from objective file
    objective_path = Path(EVALUATION.objective_failure_details_path)
    hybrid50_misses = []
    if objective_path.exists():
        payload = json.loads(objective_path.read_text(encoding="utf-8"))
        hybrid50_misses = list(payload.get("failure_rows") or [])

    matrix_rows = []
    matrix_counts = {
        "recovered_by_existing_BM25": 0,
        "recovered_by_MiniLM": 0,
        "recovered_by_PubMedBERT": 0,
        "newly_recovered_by_Sparse": 0,
        "recovered_only_by_Sparse": 0,
        "recovered_by_multiple": 0,
        "never_recovered": 0,
    }
    pubmed_available = pubmed is not None
    fused_cache: dict[str, list[Any]] = {}
    for row in hybrid50_misses:
        query_id = row.get("query_id")
        gold = row.get("gold_passage_id")
        question = row.get("question") or ""
        qkey = passage_id_key(query_id)
        bm25_hit = row.get("present_in_bm25_100")
        mini_hit = row.get("present_in_dense_100")
        pubmed_rank = None
        if pubmed_available:
            pkey = f"pubmed::{qkey}"
            ids = query_cache.get(pkey)
            if ids is None:
                ids = _hit_ids(list(pubmed.search(question, top_k=100)))
                query_cache[pkey] = ids
            pubmed_rank = rank_of_id(ids, gold)
        sids = query_cache.get(qkey)
        if sids is None:
            sids = _hit_ids(list(sparse.search(question, top_k=PROBE_K)))
            query_cache[qkey] = sids
        sparse_rank = rank_of_id(sids, gold)
        pubmed_hit = pubmed_rank is not None
        sparse_hit = sparse_rank is not None and sparse_rank <= 100
        if qkey not in fused_cache:
            fused_cache[qkey] = _hit_ids(
                rrf_fuse_retrievers(
                    [bm25, minilm, sparse], question, texts=texts, top_k=100, rrf_k=RETRIEVAL.rrf_k, candidate_k=100
                )
            )
        fused_hit = rank_of_id(fused_cache[qkey], gold) is not None
        hits = [bool(bm25_hit), bool(mini_hit), bool(pubmed_hit) if pubmed_available else False, sparse_hit]
        n_hits = sum(1 for item in hits if item)
        only_sparse = sparse_hit and not bm25_hit and not mini_hit and (not pubmed_hit if pubmed_available else True)
        if bm25_hit:
            matrix_counts["recovered_by_existing_BM25"] += 1
        if mini_hit:
            matrix_counts["recovered_by_MiniLM"] += 1
        if pubmed_hit:
            matrix_counts["recovered_by_PubMedBERT"] += 1
        if sparse_hit:
            matrix_counts["newly_recovered_by_Sparse"] += 1
        if only_sparse:
            matrix_counts["recovered_only_by_Sparse"] += 1
        if n_hits >= 2:
            matrix_counts["recovered_by_multiple"] += 1
        if n_hits == 0:
            matrix_counts["never_recovered"] += 1
        matrix_rows.append(
            {
                "query_id": _jsonable(query_id),
                "gold_passage_id": _jsonable(gold),
                "BM25": "hit" if bm25_hit else "miss",
                "MiniLM": "hit" if mini_hit else "miss",
                "PubMedBERT": "hit" if pubmed_hit else ("miss" if pubmed_available else "not_evaluated"),
                "Sparse": "hit" if sparse_hit else "miss",
                "Fused_B": "hit" if fused_hit else "miss",
                "sparse_rank": sparse_rank,
            }
        )

    medcpt_payload: dict[str, Any] = {"skipped": True, "reason": "dual-miss Sparse@100 recovery below materiality threshold"}
    if include_medcpt and n_dual and recovered_at[100] >= MATERIAL_DUAL_RECOVERY_FOR_MEDCPT:
        try:
            from src.retrieval.reranker import CrossEncoderReranker, get_shared_scorer

            reranker = CrossEncoderReranker(scorer=get_shared_scorer())
            print("MedCPT on B BM25+MiniLM+Sparse @50...", flush=True)
            med_rows = []
            for query in queries:
                started = time.perf_counter()
                pool = rrf_fuse_retrievers(
                    [bm25, minilm, sparse],
                    query.question,
                    texts=texts,
                    top_k=50,
                    rrf_k=RETRIEVAL.rrf_k,
                    candidate_k=100,
                )
                fused_ms = (time.perf_counter() - started) * 1000.0
                started = time.perf_counter()
                ranked = list(reranker.rerank(query.question, pool, top_k=10))
                rerank_ms = (time.perf_counter() - started) * 1000.0
                ids = _hit_ids(ranked)
                metrics = compute_retrieval_metrics(ids, query.relevant_passage_ids)
                med_rows.append(
                    {
                        "latency_ms": fused_ms + rerank_ms,
                        "rerank_ms": rerank_ms,
                        "candidate_recall_50": recall_at_k(_hit_ids(pool), query.relevant_passage_ids, 50),
                        **metrics,
                    }
                )
            medcpt_payload = {
                "skipped": False,
                "name": "B fused@50 → MedCPT@10",
                "recall_at_10": _mean([row["recall_at_10"] for row in med_rows]),
                "mrr_at_10": _mean([row["mrr_at_10"] for row in med_rows]),
                "ndcg_at_10": _mean([row["ndcg_at_10"] for row in med_rows]),
                "candidate_recall_50": _mean([row["candidate_recall_50"] for row in med_rows]),
                "avg_latency_ms": _mean([row["latency_ms"] for row in med_rows]),
                "avg_rerank_ms": _mean([row["rerank_ms"] for row in med_rows]),
                "baseline_medcpt_recall_at_10": 0.5082,
            }
        except Exception as exc:  # noqa: BLE001
            medcpt_payload = {"skipped": True, "reason": f"MedCPT unavailable: {exc}"[:300]}

    summary = {
        "n_queries": len(queries),
        "n_passages": len(passages),
        "n_dual_misses": n_dual,
        "sparse": sparse_info,
        "candidates_considered": list(SPARSE_CANDIDATES),
        "why_this_model": (
            "naver/splade-v3-distilbert produces learned sparse term weights, is DistilBERT-sized "
            "so 40k passages can be encoded locally, and is trained on MS MARCO rather than this "
            "BioASQ eval split (no gold-ID leakage). It is not biomedical-adapted; that is a documented tradeoff."
        ),
        "leakage_note": leakage_note,
        "production_defaults_unchanged": True,
        "dual_miss_recovery": {
            "denominator": n_dual,
            "hit_at_10": recovered_at[10],
            "hit_at_50": recovered_at[50],
            "hit_at_100": recovered_at[100],
            "hit_at_200": recovered_at[200],
            "sparse_recovery_at_100": recovery_100,
            "still_missing_at_100": still_missing,
            "newly_recovered_ids": newly_recovered_ids,
            "still_missing_ids": still_missing_ids,
            "recovery_by_query": list(recovery_by_query.values()),
            "rank_histogram": rank_histogram,
            "rank_count_recovered": len(sparse_ranks),
            "median_sparse_rank_when_hit": statistics.median(sparse_ranks) if sparse_ranks else None,
            "mean_sparse_rank_when_hit": _mean(sparse_ranks) if sparse_ranks else None,
        },
        "sparse_only_recovery_among_dual_misses": sparse_only,
        "sparse_only_among_dual_misses_not_pubmedbert": matrix_counts["recovered_only_by_Sparse"],
        "full_retrieval_metrics": [_without_rows(item) for item in comparisons],
        "fusion": [_without_rows(item) for item in fusion],
        "fusion_dual_miss": fusion_dual,
        "medcpt": medcpt_payload,
        "latency": {
            "sparse_index_seconds": sparse.index_build_seconds,
            "sparse_index_nbytes": sparse.index_nbytes(),
            "sparse_nnz": sparse.nnz,
            "sparse_vocab_size": sparse.vocab_size,
            "sparse_avg_search_ms": next(
                (item["avg_latency_ms"] for item in comparisons if item["name"] == "Learned sparse"),
                None,
            ),
        },
        "recovery_matrix_counts": matrix_counts,
        "n_hybrid50_failures": len(matrix_rows),
        "status": "EXPERIMENTAL",
    }
    details = {
        **summary,
        "dual_miss_rows": dual_detail,
        "comparison_rows": {item["name"]: item["rows"] for item in comparisons},
        "recovery_matrix_rows": matrix_rows,
    }
    details_out = _write(details_path or EVALUATION.sparse_study_details_path, details)
    summary_out = _write(summary_path or EVALUATION.sparse_study_summary_path, summary)
    print(json.dumps(summary, indent=2), flush=True)
    return {
        "details_path": details_out,
        "summary_path": summary_out,
        "summary": summary,
        "skipped": False,
        "sparse": sparse,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Learned sparse retrieval experiment")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--skip-medcpt", action="store_true")
    args = parser.parse_args()
    run_sparse_study(allow_download=not args.no_download, include_medcpt=not args.skip_medcpt)


if __name__ == "__main__":
    main()
