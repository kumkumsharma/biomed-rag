"""Observed retrieval-failure analysis. Labels are ranks, not causal stories.

Production retrieval defaults are not changed by this module.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import EVALUATION, UPGRADE
from src.data.loader import get_passage_collection
from src.data.preprocessing import Passage
from src.evaluation.dataset import EvaluationQuery, load_or_create_fixed_queries
from src.evaluation.metrics import passage_id_key, rank_of_id
from src.query.classify import ENTITY_RE, SEMANTIC_CUES, classify_query, count_aspects
from src.query.decomposition import decompose_query
from src.query.expansion import expand_query
from src.query.terminology import detect_terms, normalize_query
from src.retrieval.bm25 import BM25Retriever, tokenize
from src.retrieval.dense import DenseRetriever, Embedder
from src.retrieval.embedding_models import huggingface_model_cached
from src.retrieval.hybrid import HybridRetriever

PROBE_K = 100
HYBRID_K = 50
DEFAULT_BIOMEDICAL_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"

NOT_IN_CORPUS = "NOT_IN_CORPUS"
BM25_ONLY_MISS = "BM25_ONLY_MISS"
DENSE_ONLY_MISS = "DENSE_ONLY_MISS"
BOTH_RETRIEVERS_MISS = "BOTH_RETRIEVERS_MISS"
FUSION_CUTOFF = "FUSION_CUTOFF"
TOP50_RETRIEVED = "TOP50_RETRIEVED"

OBJECTIVE_CATEGORIES = (
    NOT_IN_CORPUS,
    TOP50_RETRIEVED,
    BM25_ONLY_MISS,
    DENSE_ONLY_MISS,
    BOTH_RETRIEVERS_MISS,
    FUSION_CUTOFF,
)

FAILURE_CATEGORIES = (
    NOT_IN_CORPUS,
    BM25_ONLY_MISS,
    DENSE_ONLY_MISS,
    BOTH_RETRIEVERS_MISS,
    FUSION_CUTOFF,
)

MECHANISM_ORDER = (
    "BM25",
    "MiniLM Dense",
    "PubMedBERT Dense",
    "Controlled lexical variant",
    "Deterministic expansion",
    "Decomposition subquery",
)


def _jsonable(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (bytes, str, bool)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def _write(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def _hit_ids(hits: Sequence[Any]) -> list[Any]:
    return [getattr(item, "passage_id") for item in hits]


def _in_top(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def objective_failure_category(
    *,
    in_corpus: bool,
    bm25_rank: int | None,
    dense_rank: int | None,
    rrf_rank: int | None,
    hybrid_k: int = HYBRID_K,
    probe_k: int = PROBE_K,
) -> str:
    """Assign one objective category from ranks only. Query text is ignored.

    Rank ``None`` means the ID was absent from the top-``probe_k`` list, not
    that it has no rank in the full collection.
    """

    del probe_k  # documented cutoff; ranks are already computed at that K
    if not in_corpus:
        return NOT_IN_CORPUS
    if _in_top(rrf_rank, hybrid_k):
        return TOP50_RETRIEVED
    in_bm25 = bm25_rank is not None
    in_dense = dense_rank is not None
    if in_dense and not in_bm25:
        return BM25_ONLY_MISS
    if in_bm25 and not in_dense:
        return DENSE_ONLY_MISS
    if not in_bm25 and not in_dense:
        return BOTH_RETRIEVERS_MISS
    return FUSION_CUTOFF


def fusion_cutoff_flag(
    *,
    bm25_rank: int | None,
    dense_rank: int | None,
    rrf_rank: int | None,
    hybrid_k: int = HYBRID_K,
) -> bool:
    either = bm25_rank is not None or dense_rank is not None
    return either and (rrf_rank is None or rrf_rank > hybrid_k)


def _token_overlap(query: str, text: str) -> float:
    q = set(tokenize(query))
    p = set(tokenize(text))
    if not q or not p:
        return 0.0
    return len(q & p) / len(q)


def _phrase_overlap(query: str, text: str) -> dict[str, Any]:
    q_tokens = tokenize(query)
    p_joined = " ".join(tokenize(text))
    if not q_tokens or not p_joined:
        return {"longest_phrase_tokens": 0, "phrase": None, "exact_normalized_query": False}
    exact = " ".join(q_tokens) in p_joined
    for length in range(min(len(q_tokens), 8), 0, -1):
        for start in range(0, len(q_tokens) - length + 1):
            gram = " ".join(q_tokens[start : start + length])
            if gram in p_joined:
                return {
                    "longest_phrase_tokens": length,
                    "phrase": gram,
                    "exact_normalized_query": exact,
                }
    return {"longest_phrase_tokens": 0, "phrase": None, "exact_normalized_query": exact}


def diagnostic_signals(
    *,
    question: str,
    passage_text: str,
    in_corpus: bool,
) -> dict[str, Any]:
    """Query/passage descriptors. Not failure causes."""

    q_tokens = tokenize(question)
    p_tokens = tokenize(passage_text) if in_corpus else []
    classification = classify_query(question)
    terms = detect_terms(question)
    q_entities = set(ENTITY_RE.findall(question))
    p_entities = set(ENTITY_RE.findall(passage_text)) if in_corpus else set()
    lowered = question.lower()
    return {
        "query_token_count": len(q_tokens),
        "passage_token_count": len(p_tokens),
        "token_overlap": _token_overlap(question, passage_text) if in_corpus else None,
        "shared_tokens": sorted(set(q_tokens) & set(p_tokens)),
        "query_tokens_absent_from_passage": sorted(set(q_tokens) - set(p_tokens)),
        "phrase_overlap": _phrase_overlap(question, passage_text) if in_corpus else None,
        "has_term_map_match": bool(terms),
        "detected_terms": list(terms),
        "abbreviation_like_tokens": [token for token in ENTITY_RE.findall(question) if 2 <= len(token) <= 6],
        "entity_overlap": sorted(q_entities & p_entities),
        "query_entities": sorted(q_entities),
        "query_type": classification.query_type,
        "multi_part_signal": classification.query_type == "multi_part",
        "semantic_query_signal": classification.query_type == "semantic"
        or any(cue in lowered for cue in SEMANTIC_CUES),
        "aspect_count": count_aspects(question),
        "in_corpus": in_corpus,
    }


def hypotheses_from_observation(
    *,
    category: str,
    signals: Mapping[str, Any],
    fusion_cutoff: bool,
) -> list[dict[str, str]]:
    """Interpretive guesses. Never overwrite the objective category."""

    items: list[dict[str, str]] = []
    overlap = signals.get("token_overlap")
    if category == BOTH_RETRIEVERS_MISS and overlap is not None and overlap < 0.15:
        items.append(
            {
                "label": "possible_lexical_mismatch",
                "evidence": f"query/passage token overlap {overlap:.4f} < 0.15",
                "confidence": "low",
            }
        )
    if category == BOTH_RETRIEVERS_MISS and signals.get("has_term_map_match"):
        items.append(
            {
                "label": "possible_terminology_gap",
                "evidence": "query matches TERM_MAP; both retrievers missed at @100",
                "confidence": "low",
            }
        )
    if category == BOTH_RETRIEVERS_MISS and signals.get("multi_part_signal"):
        items.append(
            {
                "label": "possible_query_formulation_issue",
                "evidence": "query classified multi_part; both retrievers missed at @100",
                "confidence": "low",
            }
        )
    if category == BOTH_RETRIEVERS_MISS and signals.get("semantic_query_signal") and overlap is not None and overlap >= 0.15:
        items.append(
            {
                "label": "possible_semantic_mismatch",
                "evidence": "semantic cue on query and token overlap >= 0.15; both retrievers missed",
                "confidence": "low",
            }
        )
    if category == DENSE_ONLY_MISS:
        items.append(
            {
                "label": "possible_dense_encoder_gap",
                "evidence": "BM25@100 retrieved the gold ID; MiniLM Dense@100 did not",
                "confidence": "medium",
            }
        )
    if category == BM25_ONLY_MISS:
        items.append(
            {
                "label": "possible_bm25_lexical_gap",
                "evidence": "MiniLM Dense@100 retrieved the gold ID; BM25@100 did not",
                "confidence": "medium",
            }
        )
    if category == FUSION_CUTOFF or (fusion_cutoff and category in {BM25_ONLY_MISS, DENSE_ONLY_MISS}):
        items.append(
            {
                "label": "possible_fusion_cutoff",
                "evidence": "at least one first-stage retriever had the ID at @100 but Hybrid rank > 50 or missing",
                "confidence": "medium",
            }
        )
    if category == NOT_IN_CORPUS:
        items.append(
            {
                "label": "possible_dataset_issue",
                "evidence": "gold passage ID is not in the loaded corpus",
                "confidence": "high",
            }
        )
    return items


def lexical_transformations(query: str) -> list[dict[str, str]]:
    """Deterministic query rewrites for BM25 probes. Not production search queries."""

    seen: set[str] = {query.strip().lower()}
    out: list[dict[str, str]] = []

    def _add(name: str, text: str) -> None:
        stripped = text.strip()
        key = stripped.lower()
        if not stripped or key in seen:
            return
        seen.add(key)
        out.append({"name": name, "query": stripped})

    _add("lowercase", query.lower())
    punctless = " ".join(tokenize(query))
    _add("punctuation_normalized", punctless)
    collapsed = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", query)).strip()
    _add("unicode_whitespace_normalized", collapsed)
    normalized = normalize_query(query, enabled=True)
    _add("terminology_normalized", normalized.normalized)
    expansion = expand_query(query, enabled=True, max_variants=8)
    for index, variant in enumerate(expansion.expansions):
        _add(f"synonym_substitution_{index}", variant)
    return out


def observed_facts(
    *,
    query_id: Any,
    question: str,
    gold_passage_id: Any,
    in_corpus: bool,
    bm25_rank: int | None,
    dense_rank: int | None,
    rrf_rank: int | None,
    hybrid_k: int = HYBRID_K,
    probe_k: int = PROBE_K,
) -> dict[str, Any]:
    in_hybrid_50 = _in_top(rrf_rank, hybrid_k)
    in_bm25 = bm25_rank is not None
    in_dense = dense_rank is not None
    return {
        "query_id": _jsonable(query_id),
        "question": question,
        "gold_passage_id": _jsonable(gold_passage_id),
        "in_corpus": in_corpus,
        "bm25_rank_at_100": bm25_rank,
        "dense_rank_at_100": dense_rank,
        "rrf_rank_at_100": rrf_rank,
        "present_in_hybrid_50": in_hybrid_50,
        "present_in_bm25_100": in_bm25,
        "present_in_dense_100": in_dense,
        "present_in_either_retriever": in_bm25 or in_dense,
        "present_in_both_retrievers": in_bm25 and in_dense,
        "present_in_rrf_100": rrf_rank is not None,
        "rank_movement": {
            "bm25_to_rrf": None if bm25_rank is None or rrf_rank is None else bm25_rank - rrf_rank,
            "dense_to_rrf": None if dense_rank is None or rrf_rank is None else dense_rank - rrf_rank,
        },
        "probe_k": probe_k,
        "hybrid_k": hybrid_k,
    }


def biomedical_dense_recovery_rate(
    *,
    minilm_ranks: Sequence[int | None],
    biomedical_ranks: Sequence[int | None],
    probe_k: int = PROBE_K,
) -> dict[str, Any]:
    """Recovery among gold IDs missed by MiniLM Dense@probe_k.

    Denominator = count of paired ranks where MiniLM rank is None (not in top probe_k).
    Numerator = subset of those where biomedical rank is not None.
    """

    del probe_k
    if len(minilm_ranks) != len(biomedical_ranks):
        raise ValueError("rank lists must be the same length")
    denominator = 0
    numerator = 0
    still_missed = 0
    lost = 0
    for mini, bio in zip(minilm_ranks, biomedical_ranks):
        if mini is None:
            denominator += 1
            if bio is not None:
                numerator += 1
            else:
                still_missed += 1
        elif bio is None:
            lost += 1
    return {
        "denominator_definition": (
            "Gold IDs in this set whose MiniLM Dense rank is None at probe_k "
            "(absent from MiniLM Dense@100)."
        ),
        "numerator_definition": (
            "Subset of the denominator whose PubMedBERT Dense rank is not None at probe_k."
        ),
        "minilm_dense_misses": denominator,
        "newly_recovered_by_pubmedbert": numerator,
        "still_missed_by_pubmedbert": still_missed,
        "lost_vs_minilm": lost,
        "biomedical_dense_recovery_rate": (numerator / denominator) if denominator else None,
    }


def earliest_recovery(hits: Mapping[str, bool]) -> str:
    for name in MECHANISM_ORDER:
        if hits.get(name):
            return name
    return "never_recovered"


def _search_ids(retriever: Any, query: str, top_k: int) -> list[Any]:
    return _hit_ids(list(retriever.search(query, top_k=top_k)))


def run_objective_failure_study(
    *,
    passages: Sequence[Passage] | None = None,
    queries: Sequence[EvaluationQuery] | None = None,
    bm25: Any | None = None,
    dense: Any | None = None,
    hybrid: Any | None = None,
    biomedical_dense: Any | None = None,
    biomedical_model_name: str | None = None,
    allow_biomedical: bool = True,
    hybrid_k: int = HYBRID_K,
    probe_k: int = PROBE_K,
    details_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    recovery_details_path: str | Path | None = None,
    recovery_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    queries = list(queries or load_or_create_fixed_queries())
    if hybrid is None or bm25 is None or dense is None:
        if passages is None:
            print("Loading the full BioASQ passage corpus...", flush=True)
            passages = get_passage_collection()
        print("Building BM25 + MiniLM indexes (embedding cache used when present)...", flush=True)
        bm25 = BM25Retriever(list(passages), top_k=probe_k)
        dense = DenseRetriever(
            list(passages),
            top_k=probe_k,
            cache_dir=UPGRADE.embedding_cache_dir,
        )
        hybrid = HybridRetriever(
            list(passages),
            bm25_retriever=bm25,
            dense_retriever=dense,
            top_k=probe_k,
        )
    if passages is None:
        passages = getattr(bm25, "_passages", ())
    texts: dict[Any, str] = {}
    for passage in passages:
        texts[passage.passage_id] = passage.text
        texts[passage_id_key(passage.passage_id)] = passage.text

    bio_model = biomedical_model_name or UPGRADE.biomedical_embedding_model or DEFAULT_BIOMEDICAL_MODEL
    biomedical_info: dict[str, Any] = {"model": bio_model, "available": biomedical_dense is not None}
    if biomedical_dense is None and allow_biomedical:
        if huggingface_model_cached(bio_model):
            print(f"Loading cached biomedical encoder {bio_model}...", flush=True)
            biomedical_dense = DenseRetriever(
                list(passages),
                model_name=bio_model,
                top_k=probe_k,
                cache_dir=UPGRADE.embedding_cache_dir,
            )
            biomedical_info = {"model": bio_model, "available": True, "source": "cache"}
        else:
            biomedical_info = {"model": bio_model, "available": False, "source": None}

    biomedical_hybrid = None
    if biomedical_dense is not None:
        biomedical_hybrid = HybridRetriever(
            list(passages),
            bm25_retriever=bm25,
            dense_retriever=biomedical_dense,
            top_k=probe_k,
        )

    all_gold: list[dict[str, Any]] = []
    print(f"Objective failure analysis on {len(queries)} queries...", flush=True)
    for index, query in enumerate(queries, start=1):
        bm25_ids = _search_ids(bm25, query.question, probe_k)
        dense_ids = _search_ids(dense, query.question, probe_k)
        hybrid_ids = _search_ids(hybrid, query.question, probe_k)
        bio_ids = _search_ids(biomedical_dense, query.question, probe_k) if biomedical_dense is not None else []
        bio_hybrid_ids = (
            _search_ids(biomedical_hybrid, query.question, probe_k) if biomedical_hybrid is not None else []
        )
        original_relevant_in_bm25 = {
            passage_id_key(item) for item in bm25_ids if passage_id_key(item) in {passage_id_key(r) for r in query.relevant_passage_ids}
        }
        transforms = lexical_transformations(query.question)
        transform_ids: dict[str, list[Any]] = {}
        for item in transforms:
            transform_ids[item["name"]] = _search_ids(bm25, item["query"], probe_k)
        expansion = expand_query(query.question, enabled=True)
        expansion_union: list[Any] = []
        seen_exp: set[str] = set()
        expansion_by_variant: dict[str, list[Any]] = {}
        for variant in expansion.expansions:
            ids = _search_ids(hybrid, variant, probe_k) if hasattr(hybrid, "search") else _search_ids(bm25, variant, probe_k)
            expansion_by_variant[variant] = ids
            for passage_id in ids:
                key = passage_id_key(passage_id)
                if key not in seen_exp:
                    seen_exp.add(key)
                    expansion_union.append(passage_id)
        decomposition = decompose_query(query.question, enabled=True)
        decomp_union: list[Any] = []
        seen_dec: set[str] = set()
        decomp_by_sub: dict[str, list[Any]] = {}
        if decomposition.applied:
            for sub in decomposition.subqueries:
                ids = _search_ids(hybrid, sub, probe_k)
                decomp_by_sub[sub] = ids
                for passage_id in ids:
                    key = passage_id_key(passage_id)
                    if key not in seen_dec:
                        seen_dec.add(key)
                        decomp_union.append(passage_id)

        transform_displacement: dict[str, list[Any]] = {}
        for name, ids in transform_ids.items():
            kept = {passage_id_key(item) for item in ids}
            lost = sorted(original_relevant_in_bm25 - kept)
            if lost:
                transform_displacement[name] = lost

        original_hybrid_rel = {passage_id_key(item) for item in hybrid_ids[:hybrid_k] if passage_id_key(item) in {passage_id_key(r) for r in query.relevant_passage_ids}}
        equal_hits = list(hybrid.search(query.question, top_k=hybrid_k, expansions=expansion.expansions or None))
        equal_ids = _hit_ids(equal_hits)
        equal_rel = {passage_id_key(item) for item in equal_ids if passage_id_key(item) in {passage_id_key(r) for r in query.relevant_passage_ids}}

        for gold in query.relevant_passage_ids:
            in_corpus = passage_id_key(gold) in texts or gold in texts
            text = texts.get(gold) or texts.get(passage_id_key(gold)) or ""
            bm25_rank = rank_of_id(bm25_ids, gold)
            dense_rank = rank_of_id(dense_ids, gold)
            rrf_rank = rank_of_id(hybrid_ids, gold)
            bio_rank = rank_of_id(bio_ids, gold) if bio_ids or biomedical_dense is not None else None
            bio_hybrid_rank = rank_of_id(bio_hybrid_ids, gold) if biomedical_hybrid is not None else None
            category = objective_failure_category(
                in_corpus=in_corpus,
                bm25_rank=bm25_rank,
                dense_rank=dense_rank,
                rrf_rank=rrf_rank,
                hybrid_k=hybrid_k,
                probe_k=probe_k,
            )
            facts = observed_facts(
                query_id=query.question_id,
                question=query.question,
                gold_passage_id=gold,
                in_corpus=in_corpus,
                bm25_rank=bm25_rank,
                dense_rank=dense_rank,
                rrf_rank=rrf_rank,
                hybrid_k=hybrid_k,
                probe_k=probe_k,
            )
            signals = diagnostic_signals(question=query.question, passage_text=text, in_corpus=in_corpus)
            cutoff = fusion_cutoff_flag(
                bm25_rank=bm25_rank, dense_rank=dense_rank, rrf_rank=rrf_rank, hybrid_k=hybrid_k
            )
            recovered_by_transform = None
            lexical_hit = False
            for item in transforms:
                if rank_of_id(transform_ids[item["name"]], gold) is not None:
                    lexical_hit = True
                    recovered_by_transform = item["name"]
                    break
            expansion_hit = rank_of_id(expansion_union, gold) is not None
            expansion_variant = None
            if expansion_hit:
                for variant, ids in expansion_by_variant.items():
                    if rank_of_id(ids, gold) is not None:
                        expansion_variant = variant
                        break
            decomp_hit = rank_of_id(decomp_union, gold) is not None
            decomp_sub = None
            if decomp_hit:
                for sub, ids in decomp_by_sub.items():
                    if rank_of_id(ids, gold) is not None:
                        decomp_sub = sub
                        break
            mechanism_hits = {
                "BM25": bm25_rank is not None,
                "MiniLM Dense": dense_rank is not None,
                "PubMedBERT Dense": bio_rank is not None if biomedical_dense is not None else False,
                "Controlled lexical variant": lexical_hit,
                "Deterministic expansion": expansion_hit,
                "Decomposition subquery": decomp_hit,
            }
            record = {
                **facts,
                "objective_category": category,
                "fusion_cutoff_flag": cutoff,
                "diagnostic_signals": signals,
                "hypotheses": hypotheses_from_observation(
                    category=category, signals=signals, fusion_cutoff=cutoff
                ),
                "pubmedbert_dense_rank_at_100": bio_rank,
                "pubmedbert_hybrid_rank_at_100": bio_hybrid_rank,
                "lexical_recovery": {
                    "recovered": lexical_hit,
                    "transformation": recovered_by_transform,
                },
                "expansion_recovery": {
                    "recovered": expansion_hit,
                    "variant": expansion_variant,
                    "n_variants": len(expansion.expansions),
                },
                "decomposition_recovery": {
                    "recovered": decomp_hit,
                    "subquery": decomp_sub,
                    "applied": decomposition.applied,
                },
                "mechanism_hits": mechanism_hits,
                "earliest_recovery": earliest_recovery(mechanism_hits),
                "query_expansion_displaced_ids": sorted(original_hybrid_rel - equal_rel),
                "query_expansion_new_ids": sorted(equal_rel - original_hybrid_rel),
                "bm25_transform_displaced": transform_displacement,
            }
            all_gold.append(record)
        if index == len(queries) or index % 10 == 0:
            print(f"  objective failure: {index}/{len(queries)}", flush=True)

    failures = [row for row in all_gold if row["objective_category"] != TOP50_RETRIEVED]
    counts = {name: 0 for name in OBJECTIVE_CATEGORIES}
    for row in all_gold:
        counts[row["objective_category"]] = counts.get(row["objective_category"], 0) + 1
    failure_counts = {name: counts[name] for name in FAILURE_CATEGORIES}

    overlap_values = [
        row["diagnostic_signals"]["token_overlap"]
        for row in failures
        if row["objective_category"] == BOTH_RETRIEVERS_MISS
        and row["diagnostic_signals"]["token_overlap"] is not None
    ]
    low_overlap = sum(1 for value in overlap_values if value < 0.15)

    mini_ranks_misses = [row["dense_rank_at_100"] for row in failures]
    bio_ranks_misses = [row["pubmedbert_dense_rank_at_100"] for row in failures]
    bio_recovery = biomedical_dense_recovery_rate(
        minilm_ranks=mini_ranks_misses,
        biomedical_ranks=bio_ranks_misses,
        probe_k=probe_k,
    ) if biomedical_dense is not None else {
        "skipped": True,
        "reason": "biomedical encoder unavailable",
        "biomedical_dense_recovery_rate": None,
    }

    # Dense recall on the full gold set and on Hybrid@50 misses
    n_all_gold = len(all_gold)
    n_fail = len(failures)

    def _dense_recall(rank_key: str, k: int, rows: Sequence[Mapping[str, Any]]) -> float:
        if not rows:
            return 0.0
        hits = sum(1 for row in rows if _in_top(row.get(rank_key), k))
        return hits / len(rows)

    lexical_recovered = [row for row in failures if row["lexical_recovery"]["recovered"]]
    lexical_new = [
        row
        for row in failures
        if row["bm25_rank_at_100"] is None and row["lexical_recovery"]["recovered"]
    ]
    expansion_recovered = [
        row
        for row in failures
        if row["expansion_recovery"]["recovered"] and row["objective_category"] == BOTH_RETRIEVERS_MISS
    ]
    expansion_any = [row for row in failures if row["expansion_recovery"]["recovered"]]
    decomp_recovered = [row for row in failures if row["decomposition_recovery"]["recovered"]]
    transform_counts: dict[str, int] = {}
    for row in lexical_new:
        name = row["lexical_recovery"]["transformation"] or "unknown"
        transform_counts[name] = transform_counts.get(name, 0) + 1
    n_queries_displaced_by_equal_expansion = len(
        {row["query_id"] for row in all_gold if row["query_expansion_displaced_ids"]}
    )

    matrix_counts = {name: 0 for name in (*MECHANISM_ORDER, "never_recovered")}
    for row in failures:
        matrix_counts[row["earliest_recovery"]] = matrix_counts.get(row["earliest_recovery"], 0) + 1

    both_miss = [row for row in failures if row["objective_category"] == BOTH_RETRIEVERS_MISS]
    recovered_only_pubmedbert = [
        row
        for row in both_miss
        if row["pubmedbert_dense_rank_at_100"] is not None and row["dense_rank_at_100"] is None
    ]
    recovered_only_lexical_from_both = [
        row for row in both_miss if row["lexical_recovery"]["recovered"]
    ]

    summary = {
        "n_queries": len(queries),
        "n_gold_ids": n_all_gold,
        "n_hybrid50_failures": n_fail,
        "hybrid_k": hybrid_k,
        "probe_k": probe_k,
        "notes": [
            "Objective categories use BM25/Dense/RRF ranks at probe_k only.",
            "diagnostic_signals and hypotheses are not treated as proven causes.",
            "Production retrieval defaults are unchanged.",
            "Rank None means absent from top probe_k, not unranked in the full collection.",
        ],
        "objective_counts_all_gold": counts,
        "objective_failure_counts": failure_counts,
        "objective_failure_percent_of_misses": {
            name: (failure_counts[name] / n_fail if n_fail else 0.0) for name in FAILURE_CATEGORIES
        },
        "diagnostic_signal_summary": {
            "both_retrievers_miss": len(both_miss),
            "both_miss_with_token_overlap_lt_0_15": low_overlap,
            "both_miss_with_term_map_match": sum(
                1 for row in both_miss if row["diagnostic_signals"]["has_term_map_match"]
            ),
            "both_miss_multi_part_signal": sum(
                1 for row in both_miss if row["diagnostic_signals"]["multi_part_signal"]
            ),
            "mean_token_overlap_both_miss": (
                sum(overlap_values) / len(overlap_values) if overlap_values else None
            ),
        },
        "biomedical": biomedical_info,
        "biomedical_dense_recovery": bio_recovery,
        "dense_recall_on_hybrid50_misses": {
            "minilm_dense_recall_at_50": _dense_recall("dense_rank_at_100", 50, failures),
            "minilm_dense_recall_at_100": _dense_recall("dense_rank_at_100", 100, failures),
            "pubmedbert_dense_recall_at_50": _dense_recall("pubmedbert_dense_rank_at_100", 50, failures)
            if biomedical_dense is not None
            else None,
            "pubmedbert_dense_recall_at_100": _dense_recall("pubmedbert_dense_rank_at_100", 100, failures)
            if biomedical_dense is not None
            else None,
        },
        "dense_recall_on_all_gold": {
            "minilm_dense_recall_at_50": _dense_recall("dense_rank_at_100", 50, all_gold),
            "minilm_dense_recall_at_100": _dense_recall("dense_rank_at_100", 100, all_gold),
            "pubmedbert_dense_recall_at_50": _dense_recall("pubmedbert_dense_rank_at_100", 50, all_gold)
            if biomedical_dense is not None
            else None,
            "pubmedbert_dense_recall_at_100": _dense_recall("pubmedbert_dense_rank_at_100", 100, all_gold)
            if biomedical_dense is not None
            else None,
            "pubmedbert_hybrid_recall_at_50": _dense_recall("pubmedbert_hybrid_rank_at_100", 50, all_gold)
            if biomedical_hybrid is not None
            else None,
            "pubmedbert_hybrid_recall_at_100": _dense_recall("pubmedbert_hybrid_rank_at_100", 100, all_gold)
            if biomedical_hybrid is not None
            else None,
        },
        "lexical_recovery": {
            "n_hybrid50_misses_already_in_original_bm25_also_hit_by_transform": len(lexical_recovered),
            "n_bm25_misses_newly_recovered_by_transform": len(lexical_new),
            "n_both_miss_recovered_by_transform": len(recovered_only_lexical_from_both),
            "by_transformation_for_new_bm25_recoveries": transform_counts,
            "fraction_of_hybrid50_misses_newly_recovered": (
                (len(lexical_new) / n_fail) if n_fail else None
            ),
            "note": (
                "A transform 'hit' on an ID already in original BM25@100 is not a new recovery. "
                "n_bm25_misses_newly_recovered_by_transform is the evidence-backed count."
            ),
        },
        "expansion_recovery": {
            "n_hybrid50_misses_in_any_expansion_search": len(expansion_any),
            "n_both_miss_recovered_by_expansion": len(expansion_recovered),
            "n_queries_with_equal_weight_displacement": n_queries_displaced_by_equal_expansion,
            "kept_experimental": True,
            "not_production_default": True,
        },
        "decomposition_recovery": {
            "n_hybrid50_misses_recovered_by_subquery": len(decomp_recovered),
            "kept_off": True,
        },
        "recovery_matrix_counts": matrix_counts,
        "n_both_miss_recovered_only_by_pubmedbert_dense": len(recovered_only_pubmedbert),
    }
    details = {
        **summary,
        "rows": all_gold,
        "failure_rows": failures,
    }
    details_out = _write(details_path or EVALUATION.objective_failure_details_path, details)
    summary_out = _write(summary_path or EVALUATION.objective_failure_summary_path, summary)
    recovery_details = {
        "n_hybrid50_failures": n_fail,
        "mechanisms": list(MECHANISM_ORDER),
        "counts": matrix_counts,
        "rows": [
            {
                "query_id": row["query_id"],
                "question": row["question"],
                "gold_passage_id": row["gold_passage_id"],
                "objective_category": row["objective_category"],
                "MiniLM Dense": "hit" if row["mechanism_hits"]["MiniLM Dense"] else "miss",
                "BM25": "hit" if row["mechanism_hits"]["BM25"] else "miss",
                "PubMedBERT Dense": (
                    "hit"
                    if row["mechanism_hits"]["PubMedBERT Dense"]
                    else ("miss" if biomedical_dense is not None else "not_evaluated")
                ),
                "Controlled lexical variant": "hit" if row["mechanism_hits"]["Controlled lexical variant"] else "miss",
                "Deterministic expansion": "hit" if row["mechanism_hits"]["Deterministic expansion"] else "miss",
                "Decomposition subquery": "hit" if row["mechanism_hits"]["Decomposition subquery"] else "miss",
                "earliest_recovery": row["earliest_recovery"],
                "lexical_transformation": row["lexical_recovery"]["transformation"],
            }
            for row in failures
        ],
    }
    recovery_summary = {
        "n_hybrid50_failures": n_fail,
        "counts": matrix_counts,
        "notes": [
            "hit/miss is at probe_k=100 for each mechanism.",
            "earliest_recovery uses a fixed mechanism order and is not a production ranking.",
        ],
        "biomedical_dense_recovery": bio_recovery,
        "lexical_recovery": summary["lexical_recovery"],
        "expansion_recovery": summary["expansion_recovery"],
        "decomposition_recovery": summary["decomposition_recovery"],
    }
    rec_details_out = _write(recovery_details_path or EVALUATION.recovery_matrix_details_path, recovery_details)
    rec_summary_out = _write(recovery_summary_path or EVALUATION.recovery_matrix_summary_path, recovery_summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "notes"}, indent=2), flush=True)
    return {
        "details_path": details_out,
        "summary_path": summary_out,
        "recovery_details_path": rec_details_out,
        "recovery_summary_path": rec_summary_out,
        "summary": summary,
        "rows": all_gold,
        "failures": failures,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Objective retrieval failure analysis")
    parser.add_argument("--no-biomedical", action="store_true")
    args = parser.parse_args()
    run_objective_failure_study(allow_biomedical=not args.no_biomedical)


if __name__ == "__main__":
    main()
