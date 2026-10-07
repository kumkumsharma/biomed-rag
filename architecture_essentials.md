# Architecture Essentials

## Core Flow
Question -> classification / optional terminology / optional expansion or decomposition -> BM25 + Dense -> Hybrid Fusion -> optional candidate filter -> optional MedCPT Rerank -> Evidence -> Grounded LLM -> Citation Validation -> UI

Disabled upgrade flags keep the previous Hybrid RRF baseline.

## Rules
- BM25 and dense remain independently callable.
- Preserve passage IDs.
- External model calls use clear interfaces.
- Final LLM sees selected evidence only.
- Never invent evidence/citations.
- Return insufficient evidence when support is inadequate.
- Evaluation uses the same fixed 100 queries.
- MedCPT comparison is Hybrid@10 vs Hybrid@50 vs MedCPT@10 on those queries.
- Candidate recall@50 is the current first-stage ceiling (~0.624); extra query layers are not assumed helpful. Hybrid@100 recovered Recall@100=0.6494 in the pool study — diminishing returns after 50.
- Failure analysis must separate observed ranks from diagnostic signals and hypotheses. Do not treat token-overlap labels as causes.
- Biomedical MiniLM vs S-PubMedBert comparison is in `embedding_model_comparison_*.json`. MiniLM stays the default despite a modest measured PubMedBERT gain because of index time and 768-d RAM.
- Record configuration, scores, latency and cost where applicable. Do not claim a feature is better without a result file.

## Current Scope
Required: BM25, Dense, Hybrid, grounded QA, citations, UI, 100-query retrieval evaluation.

Optional: Query Expansion, LLM judge, MedCPT cross-encoder reranking after RRF, and the feature-flagged upgrade path (terminology, deterministic expansion, decomposition, biomedical embeddings, adaptive routing, candidate filtering, staged pipeline). All upgrade flags default off.

Experimental (not production): learned sparse first-stage retrieval (`src/retrieval/sparse.py`, `python -m src.evaluation.sparse_study`). Production remains BM25 + MiniLM + RRF → Top50 → MedCPT → Top10.

Observability: data-quality, gold-label, index-health, and per-query traces live under `experiments/results/diagnostics/`. They do not change retrieval models. `python -m src.evaluation.run_observability` writes those files.

Later: ColBERT, knowledge graph, and production deployment.
