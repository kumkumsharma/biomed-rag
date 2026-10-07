# AI Coding Agent Instructions

## Source of Truth
Read `PRD.md`, `architecture.md` and `architecture_essentials.md` before architectural changes.

## Rules
- Do not invent major components without checking the architecture.
- Do not delete PRD/architecture documents.
- Keep BM25, dense and hybrid independently testable.
- Optional MedCPT rerank wraps Hybrid RRF; do not fold it into hybrid fusion.
- Optional upgrade stages (terminology, expansion, decomposition, biomedical embeddings, routing, filtering, learned sparse) are feature-flagged and default off.
- Learned sparse (SPLADE) is an experiment for dual-miss recovery. Do not enable it in production unless measured results justify it.
- Do not treat heuristic failure labels (lexical mismatch, terminology mismatch) as demonstrated causes. Use objective rank categories in `src/evaluation/objective_failure.py`.
- Preserve passage IDs.
- Isolate external LLM/model calls.
- Keep evaluation separate from retrieval implementations.
- Never hard-code secrets.
- Prefer small focused modules.
- Update tests with behaviour changes.

## Deadline Priority
1. BM25
2. Dense
3. Hybrid
4. Grounded answer + citations
5. UI
6. Fixed 100-query evaluation

Only add optional features after the required path works.
