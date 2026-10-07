# Examples Index

This folder contains reference implementations and coding patterns for the
Biomedical Hybrid Search & QA project.

Examples are not necessarily production code.

The purpose of an example is to show the coding pattern, interface,
naming, validation, error handling, and implementation style that the
AI coding agent should follow when implementing the corresponding
production component.

## How the AI Coding Agent Should Use This Folder

Before implementing a component:

1. Check this index for a relevant example.
2. Open the referenced example.
3. Follow its demonstrated pattern.
4. Adapt the example to the production requirement in `PRD.md` and `architecture.md`.
5. Do not blindly copy an example if the production requirements differ.
6. If no relevant example exists, implement according to `architecture.md` and create an example only when a reusable implementation pattern is established.

## Example Map

| Example | Location | Refer to when |
|---|---|---|
| BM25 retrieval | `examples/retrieval/bm25_example.py` | Implementing or modifying BM25/lexical retrieval |
| Dense retrieval | `examples/retrieval/dense_example.py` | Implementing or modifying embedding-based retrieval |
| Hybrid retrieval | `examples/retrieval/hybrid_example.py` | Implementing BM25 + dense fusion |
| Cross-encoder reranking | `examples/retrieval/rerank_example.py` | Adding MedCPT rerank after RRF |
| Multi-stage pipeline | `examples/retrieval/pipeline_example.py` | Query understanding, filtering, and staged retrieval |
| Grounded answer generation | `examples/generation/grounded_answer_example.py` | Implementing the final LLM answer generation |
| Query expansion | `examples/generation/query_expansion_example.py` | Implementing optional query expansion before retrieval |
| Retrieval evaluation | `examples/evaluation/retrieval_evaluation_example.py` | Implementing the offline retrieval evaluation pipeline |
| Answer evaluation | `examples/evaluation/answer_evaluation_example.py` | Implementing the LLM judge and final-answer evaluation |

## BM25 example

`examples/retrieval/bm25_example.py` is the current reusable pattern. It shows:

- building a BM25 index from `prepare_corpus` / `Passage` records
- querying with `BM25Retriever.search`
- returning `passage_id`, `score`, and `text`
- keeping BM25 independently callable

Refer to it when implementing or changing lexical retrieval. Production code lives in `src/retrieval/bm25.py` and must still follow `PRD.md` and `architecture.md`.

## Dense example

`examples/retrieval/dense_example.py` is the current reusable pattern. It shows:

- building a local in-memory dense index from `prepare_corpus` / `Passage` records
- embedding passages and the query with the same sentence-transformer
- querying with `DenseRetriever.search`
- returning `passage_id`, cosine similarity `score`, and `text`
- keeping dense retrieval independently callable from BM25

Refer to it when implementing or changing embedding-based retrieval. Production code lives in `src/retrieval/dense.py` and must still follow `PRD.md` and `architecture.md`.

## Hybrid example

`examples/retrieval/hybrid_example.py` is the current reusable pattern. It shows:

- hybrid retrieval combining BM25 (lexical) and dense (embedding) retrievers
- Reciprocal Rank Fusion (RRF) as the fusion method
- injecting existing `BM25Retriever` and `DenseRetriever` instances
- returning `passage_id`, hybrid RRF `score`, and `text`

Refer to it when implementing or changing hybrid retrieval. Production code lives in `src/retrieval/hybrid.py` and must still follow `PRD.md` and `architecture.md`.

## Cross-encoder rerank example

`examples/retrieval/rerank_example.py` is the current reusable pattern. It shows:

- wrapping the existing Hybrid RRF retriever rather than duplicating retrieval
- disabling rerank for the baseline BM25 + Dense + RRF pipeline
- enabling a local MedCPT cross-encoder over the RRF candidate pool
- keeping `passage_id`, RRF score, and text while adding `reranker_score` and ranks

Refer to it when implementing or changing second-stage reranking. Production code lives in `src/retrieval/reranker.py` and must still follow `PRD.md` and `architecture.md`.

## Grounded answer example

`examples/generation/grounded_answer_example.py` is the current reusable pattern. It shows:

- generating an answer only from selected evidence passages
- citing supporting IDs as `[PASSAGE_ID: <id>]`
- validating citations deterministically against the evidence sent to the LLM
- skipping the LLM when retrieval returns insufficient evidence

Refer to it when implementing or changing grounded answer generation. The final answer must be grounded in supplied evidence. Citation validation is deterministic application code, not an LLM self-check. Production code lives in `src/generation/answer_generator.py`, `src/generation/rag.py`, and `src/validation/citations.py`, and must still follow `PRD.md` and `architecture.md`.

## Query expansion example

`examples/generation/query_expansion_example.py` is the current reusable pattern. It shows:

- keeping the original query unchanged
- generating a small set of biomedical search alternates
- running BM25 and dense over the original query plus expansions
- fusing those rankings with the existing Hybrid RRF implementation

Refer to it when implementing or changing query expansion. Expansion is optional and easy to disable. Production code lives in `src/generation/query_expansion.py` and must still follow `PRD.md` and `architecture.md`.

## Retrieval evaluation example

`examples/evaluation/retrieval_evaluation_example.py` is the current reusable pattern. It shows:

- scoring BM25, Dense, and Hybrid RRF on the same fixed query set
- using golden `relevant_passage_ids` as retrieval ground truth
- computing Recall@5, Recall@10, MRR@10, and nDCG@10
- reporting latency without including one-time index construction

Refer to it when implementing or changing the offline evaluation pipeline. All retrieval configurations must use the same saved query IDs. Production evaluation uses the BioASQ golden QA dataset, exactly 100 query IDs from `experiments/configs/evaluation_100.json`, and the full passage corpus — not the UI 200-passage demo subset. Production code lives in `src/evaluation/dataset.py`, `src/evaluation/metrics.py`, `src/evaluation/run_evaluation.py`, `src/evaluation/rerank_comparison.py`, and `src/evaluation/upgrade_experiments.py`, and must still follow `PRD.md` and `architecture.md`.

## Answer evaluation example

`examples/evaluation/answer_evaluation_example.py` is the current reusable pattern. It shows:

- generating a grounded answer with the existing Hybrid RRF + GroundedQA path
- checking citations deterministically with `src/validation/citations.py`
- sending the question, reference answer, generated answer, and evidence to a separate LLM judge
- scoring correctness, groundedness, and context relevance

Retrieval evaluation measures ranking quality (Recall@k, MRR, nDCG). Final-answer evaluation measures whether the generated RAG answer is correct, faithful to retrieved evidence, and supported by relevant context. The LLM judge evaluates answer quality only. It does not validate citations.

Refer to it when implementing or changing answer-quality evaluation. Production evaluation uses 20 queries from the fixed 100-query set (`experiments/configs/answer_evaluation_20.json`) and Hybrid RRF over the full passage corpus. Production code lives in `src/evaluation/answer_evaluation.py` and must still follow `PRD.md` and `architecture.md`.

## Important

The architecture and PRD remain the source of truth.

Examples define **how we prefer to implement a pattern**.

If an example conflicts with the PRD or architecture, follow the PRD and architecture and flag the conflict.

## Mental Model

PRD → What the system must do

Architecture → How the system is structured

Examples → How we write/implement a particular pattern

Tests → Whether the implementation works
