"""Architecture decision records grounded in measured results, not marketing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import EVALUATION


ROOT = Path(__file__).resolve().parents[2]


def _load_sparse_measured() -> str:
    path = ROOT / EVALUATION.sparse_study_summary_path
    if not path.exists():
        return (
            "NOT EVALUATED until sparse_study_summary.json exists. "
            "Production remains BM25 + MiniLM Dense + RRF → Top50 → MedCPT → Top10."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("skipped"):
        return (
            f"Skipped: {payload.get('reason')} Candidate survey is in the JSON. "
            "No metrics were fabricated. Production defaults unchanged."
        )
    rec = payload.get("dual_miss_recovery") or {}
    fusion = payload.get("fusion_dual_miss") or []
    fusion_b = next((item for item in fusion if str(item.get("name", "")).startswith("B ")), {})
    return (
        f"Sparse Recovery@100: {rec.get('hit_at_100')}/{rec.get('denominator')} dual-miss gold IDs "
        f"(rate {rec.get('sparse_recovery_at_100')}). "
        f"Sparse@100 among dual misses: {payload.get('sparse_only_recovery_among_dual_misses')}. "
        f"Sparse-only (not BM25, MiniLM, or PubMedBERT): {payload.get('sparse_only_among_dual_misses_not_pubmedbert')}. "
        f"BM25+MiniLM+Sparse dual-miss@100: {fusion_b.get('dual_miss_hit_at_100')}. "
        "Status EXPERIMENTAL; production defaults unchanged."
    )


def architecture_decisions() -> list[dict[str, Any]]:
    return [
        {
            "component": "MiniLM dense encoder",
            "decision": "Use sentence-transformers/all-MiniLM-L6-v2 as the default dense model.",
            "why_chosen": (
                "It is small (~80MB), produces 384-dimensional vectors, runs on CPU, "
                "and needs no API key. That makes it a reproducible baseline."
            ),
            "alternative_considered": (
                "A PubMed-trained sentence embedding model such as "
                "pritamdeka/S-PubMedBert-MS-MARCO (768-d)."
            ),
            "measured_result": (
                "MiniLM Hybrid@10 on the fixed 100-query set: Recall@10=0.4286, "
                "MRR@10=0.8105, nDCG@10=0.5655, ~66 ms/query. "
                "Embedding comparison (Hybrid prefixes from K=100, same BM25/RRF): "
                "MiniLM Recall@10=0.4342 vs S-PubMedBert-MS-MARCO Recall@10=0.4445, "
                "MRR@10 0.7958 vs 0.8685, nDCG@10 0.5730 vs 0.6014. "
                "Biomedical index ~326 s vs MiniLM ~45 s; 768-d uses ~124 MB vs 62 MB. "
                "MiniLM remains the production default; the biomedical encoder is not auto-swapped."
            ),
            "tradeoff": (
                "384-d is cheaper to index. S-PubMedBert measured a modest quality gain "
                "at ~7× index time and 2× embedding RAM; default stays MiniLM until that cost is accepted."
            ),
        },
        {
            "component": "MedCPT cross-encoder",
            "decision": "Optional local ncbi/MedCPT-Cross-Encoder after Hybrid RRF.",
            "why_chosen": "Biomedical cross-encoder that reranks a candidate pool without calling OpenAI.",
            "alternative_considered": "Replace Hybrid, or drop reranking because of CPU cost.",
            "measured_result": (
                "Hybrid@50 → MedCPT@10: Recall@10=0.5082, MRR@10=0.8587, nDCG@10=0.6706, "
                "rerank recovery@10=1.0 on 99 queries with a relevant candidate. "
                "Rerank ≈ 3131 ms/query on CPU. Candidate Recall@50 remains 0.6243."
            ),
            "tradeoff": "Improves ranking when the relevant passage is in the pool; does not fix missing candidates; ~3.1 s CPU.",
        },
        {
            "component": "Reciprocal Rank Fusion",
            "decision": "Fuse BM25 and dense with RRF, k=60.",
            "why_chosen": "BM25 scores and cosine similarities are on different scales.",
            "alternative_considered": "Weighted sum of raw scores.",
            "measured_result": "Hybrid RRF beats BM25-only and Dense-only on the 100-query set for Recall@10/MRR/nDCG.",
            "tradeoff": "RRF ignores raw score magnitude; ranks only.",
        },
        {
            "component": "Query expansion",
            "decision": "Keep deterministic expansion implemented but default OFF.",
            "why_chosen": "Biomedical wording often mismatches the corpus; expansion was a recall hypothesis.",
            "alternative_considered": "Always-on LLM expansion, or deleting the component after a slight drop.",
            "measured_result": (
                "Upgrade experiment E: Recall@10 0.4247 vs Hybrid@10 0.4286. "
                "Expansion study: only 8/100 queries produced variants; equal-weight fusion "
                "Recall@10 0.4247 vs original 0.4298; 1 query gained a new relevant ID and 3 lost one. "
                "Priority fusion matched original Recall@10=0.4298 and Recall@50=0.6243 (no gain, no loss). "
                "Not production default."
            ),
            "tradeoff": "Extra query variants can displace original candidates in equal-weight RRF.",
        },
        {
            "component": "Query decomposition",
            "decision": "Heuristic split only for multi-part queries; default OFF; selective if enabled.",
            "why_chosen": "Multi-aspect questions may need separate searches.",
            "alternative_considered": "LLM decomposition on every query.",
            "measured_result": (
                "Upgrade experiment F: Recall@10 0.4305 vs 0.4286 (small improvement). "
                "Decomposition study: heuristic applied to 1/100 queries; overall Recall@10 unchanged at 0.4286; "
                "that one query Recall@10 stayed 0.6 while latency rose 45 ms → 125 ms. "
                "Simple queries are not decomposed."
            ),
            "tradeoff": "Only helps complex queries; extra latency on those queries.",
        },
        {
            "component": "Candidate filtering",
            "decision": "Conservative duplicate/empty filter; default OFF.",
            "why_chosen": "MedCPT is expensive; dropping obvious junk was hypothesized to help.",
            "alternative_considered": "Aggressive lexical-overlap filters.",
            "measured_result": "Removed ~0.05 candidates/query and zero relevant passages. Did not reduce MedCPT cost.",
            "tradeoff": "Too conservative to matter; aggressive filters risk recall.",
        },
        {
            "component": "Adaptive routing",
            "decision": "Deterministic heuristics exist; default OFF.",
            "why_chosen": "Different query types may need different pools/variants.",
            "alternative_considered": "LLM agent router.",
            "measured_result": "Upgrade experiment H: Recall@10 0.4243 vs 0.4286 (slight degradation).",
            "tradeoff": "Routing extra variants can lower candidate quality without raising recall.",
        },
        {
            "component": "Failure analysis methodology",
            "decision": "Report observed ranks separately from diagnostic signals and hypotheses.",
            "why_chosen": (
                "The Hybrid@50 miss table previously assigned labels such as lexical mismatch "
                "from token overlap and TERM_MAP presence. Those are not demonstrated causes."
            ),
            "alternative_considered": "Keep heuristic failure labels as the primary UI/report categories.",
            "measured_result": (
                "On 1142 gold IDs, 445 are absent from Hybrid@50. Objective counts: "
                "BOTH_RETRIEVERS_MISS=385 (86.5%), DENSE_ONLY_MISS=37, BM25_ONLY_MISS=22, "
                "FUSION_CUTOFF=1, NOT_IN_CORPUS=0. Of the 385 dual misses, 347 have token overlap < 0.15 "
                "(hypothesis only). PubMedBERT Dense newly retrieved 21/422 MiniLM Dense@100 misses "
                "among those Hybrid@50 misses (recovery rate 0.0498). Deterministic lexical transforms "
                "newly recovered 0 BM25 misses. Expansion recovered 0 dual misses; decomposition recovered 0. "
                "376/445 Hybrid@50 misses were still absent from every probed mechanism at K=100."
            ),
            "tradeoff": "Objective labels answer which retriever missed; they do not by themselves say why.",
        },
        {
            "component": "Third first-stage retriever (learned sparse)",
            "decision": (
                "Investigate a learned-sparse first-stage retriever (SPLADE) as an experiment only. "
                "Do not enable it in production."
            ),
            "why_chosen": (
                "86.5% of Hybrid@50 misses are dual misses (385/445 gold IDs absent from both BM25@100 "
                "and MiniLM Dense@100). RRF cutoff accounts for only 1 miss. Deterministic query expansion "
                "recovered 0 dual-miss IDs; decomposition recovered 0. PubMedBERT recovered 21/422 MiniLM "
                "dense misses (~5%). A third retrieval signal is being tested because existing lexical + "
                "dense signals fail to retrieve many gold passages — not because another query rewriter is needed."
            ),
            "alternative_considered": (
                "Another generic query-expansion layer; swapping MiniLM for PubMedBERT in production; "
                "ColBERT late interaction; a biomedical SPLADE checkpoint if one had been loadable locally."
            ),
            "measured_result": _load_sparse_measured(),
            "tradeoff": (
                "Learned sparse can match latent terms BM25 never expands, at extra index/encode cost. "
                "The DistilBERT SPLADE checkpoint is MS MARCO-trained, not BioASQ-adapted. "
                "It is not production unless measured dual-miss recovery justifies it."
            ),
        },
        {
            "component": "Structured traces",
            "decision": "JSON traces in-process. No LangSmith.",
            "why_chosen": "Local reproducibility and inspectability.",
            "alternative_considered": "LangSmith or other hosted tracing.",
            "measured_result": "Not a retrieval metric; traces are serializable in last_trace / GroundedQAResult.trace.",
            "tradeoff": "No hosted dashboard; files and UI inspectors instead.",
        },
    ]
