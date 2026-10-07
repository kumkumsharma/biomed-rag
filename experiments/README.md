# Experiment and diagnostic results

This folder is the audit trail for retrieval experiments and observability reports.
Historical JSON files are **never overwritten** by later experiments.

Production stack (unchanged):

Question → BM25 + `sentence-transformers/all-MiniLM-L6-v2` → RRF (k=60) → Top 50 → `ncbi/MedCPT-Cross-Encoder` → Top 10 → grounded LLM → citation validation.

## Layout

```
experiments/
  configs/          fixed 100-query set and answer sample
  results/          historical experiment summaries (keep forever)
    diagnostics/    data quality, gold quality, index health, retrieval diagnostics
      traces/       per-query end-to-end traces
      live_evaluation.jsonl
```

## Historical experiments

| Experiment | Date | What changed | What stayed fixed | Result file | Conclusion |
|---|---|---|---|---|---|
| Production baseline | 2026-09 | — | dataset, 100 queries, gold labels | `results/retrieval_evaluation_rerank_comparison_summary.json` | Rec@50 0.6243, Rec@10 0.5082, MRR 0.8587, nDCG 0.6706, ~3133 ms. Keep. |
| SPLADE | 2026-10 | learned sparse first-stage | BM25, MiniLM, RRF, MedCPT | `results/sparse_study_summary.json` | Do not enable in production. |
| Pool 50→100 | 2026-10 | candidate pool size | MiniLM, RRF k=60, MedCPT | `results/retrieval_evaluation_rerank_comparison_pool100_summary.json` | Rec@10 worse; latency up. Rejected. |
| MedCPT bi-encoder | 2026-10 | dense encoder | BM25, RRF, pool 50, MedCPT CE | `results/retrieval_evaluation_rerank_comparison_medcpt_biencoder_summary.json` | Rec@10 slightly worse. Keep MiniLM. |
| Medical QE | 2026-10 | gpt-4o-mini expansions before retrieval | MiniLM, RRF, pool 50, MedCPT | `results/retrieval_evaluation_rerank_comparison_qe_summary.json` | Rec@10 fell. Leave QE off. |
| BiomedBERT reranker | 2026-10 | reranker model only | MiniLM pool, no QE | `results/retrieval_evaluation_rerank_comparison_biomedbert_summary.json` | Same pool, worse Top-10. Keep MedCPT. |

## Diagnostics (this observability phase)

Generate with:

```
python -m src.evaluation.run_observability
```

| Report | Path |
|---|---|
| Data quality | `results/diagnostics/data_quality_report.json` |
| Gold-label quality | `results/diagnostics/evaluation_gold_quality.json` |
| Index health | `results/diagnostics/index_health_report.json` |
| Retrieval / RRF / MedCPT diagnostics | `results/diagnostics/retrieval_diagnostics.json` |
| Query traces | `results/diagnostics/traces/query_<id>.json` |
| Live query eval | `results/diagnostics/live_evaluation.jsonl` |

Raw/original 100-query metrics stay in the historical comparison files. The valid-gold view is diagnostic only and does not replace the benchmark.

## Query traces

Traces identify the first loss stage (`DATA_QUALITY`, `BM25`, `DENSE`, `RRF`, `RERANKING`, `GENERATION`, `EVALUATION`). Example pair: query 2 (Papilin success) and query 5 (metformin gold text is `"nan"`).
