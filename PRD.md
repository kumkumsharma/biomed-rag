# PRD — Biomedical Hybrid Search & QA

## Customer Problem
Users need relevant biomedical evidence from a large passage corpus and a grounded answer. The system must show evidence and avoid unsupported answers.

## 1. System Boundary
**Own:** passage preparation/indexing, BM25, dense retrieval, hybrid retrieval, optional query expansion, evidence selection, grounded answer generation, citation validation, offline evaluation, basic UI.

**External:** Hugging Face `rag-datasets/rag-mini-bioasq`, embedding model, LLM provider/model, local runtime.

## 2. User Workflow
1. User enters a biomedical question.
2. Preserve the original query.
3. Optionally generate query expansions.
4. Run BM25 and dense retrieval.
5. Combine results with hybrid retrieval.
6. Select evidence.
7. Generate an answer only from selected evidence.
8. Cite passage IDs.
9. Show query, expansions, evidence and answer.
10. If evidence is insufficient, show an explicit insufficient-evidence response.

## 3. Required Capabilities
- Load/prepare passage corpus with stable passage IDs.
- BM25 retrieval with IDs and scores.
- Dense retrieval with IDs and scores.
- Hybrid BM25 + dense retrieval.
- Optional query expansion retaining the original query.
- Grounded final answer generation.
- Citation validation.
- Fixed 100-query offline evaluation.
- Recall@5, Recall@10, MRR@10, nDCG@10, latency and cost logging.
- LLM judge and small manual review.

## 4. Deterministic vs Agentic Behaviour
**Deterministic:** data preparation, BM25, dense retrieval, hybrid fusion, top-k selection, ID tracking, citation validation, metric calculation and experiment logging.

**LLM-assisted:** query expansion and final answer wording.

The LLM must not invent evidence, passage IDs or directly control indexes.

## 5. Scope and Non-goals
**Friday scope:** BM25, dense, hybrid, grounded QA, citations, insufficient-evidence handling, basic UI, fixed 100-query evaluation, required metrics, basic LLM judge, localhost.

**Not required Friday:** cloud deployment, authentication, SPLADE, ColBERT, cross-encoder reranking, knowledge graphs, adaptive retrieval or production-scale infrastructure.

## 6. Delivery Phases
1. Retrieval core: data, BM25, dense, hybrid.
2. Grounded QA: evidence, answer, citations.
3. UI.
4. Offline evaluation.
5. Final validation and demo preparation.

## 7. Definition of Done
- Runs locally.
- UI accepts a biomedical question.
- BM25, dense and hybrid return IDs/scores.
- Hybrid combines lexical and dense evidence.
- Final answer uses supplied evidence only.
- Citations reference passage IDs.
- Insufficient evidence is handled explicitly.
- Same fixed 100 queries reproduce evaluation.
- Recall@5, Recall@10, MRR@10 and nDCG@10 are reported.
- Latency/cost are recorded where applicable.
- LLM judge and small manual review are completed.
