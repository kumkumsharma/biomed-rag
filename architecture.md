# Architecture — Biomedical Hybrid Search & QA

## 1. Architecture Scope
Defines retrieval, orchestration, evidence selection, answer generation, evaluation and UI for the current local submission.

## 2. Design Rules
- Verify deterministic retrieval before adding LLM behaviour.
- Keep BM25, dense and hybrid independently testable.
- Preserve passage IDs end-to-end.
- Keep external model calls behind clear interfaces.
- Final LLM receives selected evidence only.
- Use the same fixed 100-query set across experiments.
- Record configuration changes.

## 3. System Boundary
```text
Hugging Face / Embedding Model / LLM
                |
                v
Data -> Indexes -> Retrieval -> Hybrid -> optional Rerank -> Evidence
                                      |
                                      v
                               Grounded LLM
                                      |
                                      v
                              Citation Validation
                                      |
                                      v
                                     UI
```

## 4. Component Responsibilities
| Component | Responsibility |
|---|---|
| Data loader | Load QA data and passage corpus |
| Preprocessor | Prepare text and stable IDs |
| BM25 index | Lexical retrieval |
| Dense index | Embedding retrieval |
| Hybrid retriever | Combine BM25 and dense |
| Learned sparse retriever | Experimental SPLADE first-stage; default off |
| Cross-encoder reranker | Optional MedCPT rerank after RRF |
| Query expansion | Alternative query phrasings |
| Evidence selector | Select context for LLM |
| Answer generator | Grounded answer |
| Citation validator | Validate cited passage IDs |
| Evaluation runner | Fixed 100-query benchmark |
| UI | Question, evidence and answer display |

## 5. Retrieval Flow
```text
Question
   |
   +--> classification
   +--> optional terminology normalization (original query retained)
   +--> optional deterministic expansion / decomposition
   |
   +--> BM25 ----+
   |             |
   +--> Dense ---+--> Hybrid Fusion --> optional filter --> optional MedCPT --> Evidence --> Grounded LLM
                                                                              |
                                                                              v
                                                                      Citation Validation
                                                                              |
                                                                              v
                                                                             UI
```

Upgrade stages are independently flag-gated. Default flags reproduce Hybrid BM25 + Dense + RRF.

## 6. Contracts
**BM25 input:** query, top_k.  
**BM25 output:** passage_id, score, text/reference.

**Dense input:** query, top_k.  
**Dense output:** passage_id, similarity score, text/reference.

**Learned sparse input:** query, top_k. Experimental; not used by HybridRetriever.  
**Learned sparse output:** passage_id, sparse inner-product score, text, pmid, rank.

**Hybrid input:** original query, optional expansions, BM25 results, dense results, fusion configuration.  
**Hybrid output:** ranked passage IDs, fusion scores, evidence records.

**Query understanding input:** original query, UpgradeConfig flags.  
**Query understanding output:** original query, normalized query, classification, expansion variants, subqueries. The original string is never discarded.

**Router input:** query, classification, flags.  
**Router output:** deterministic RetrievalPlan (components, pool size, whether to use normalized/expanded/subqueries). Disabled routing is `fixed_hybrid`.

**Filter input:** fused candidates.  
**Filter output:** kept/removed IDs with reasons. Conservative; never empties the pool.

**Reranker input:** original query, RRF candidate pool, reranker configuration.  
**Reranker output:** reranked passage IDs with reranker_score, rrf_score, bm25_rank, dense_rank, rrf_rank, final_rank, pmid. Disabled mode returns the first-stage ranking unchanged.

**Staged pipeline:** BM25 → dense → RRF → filter → MedCPT → top-k, with a JSON-serializable last_trace. No LangSmith.

## 7. Answer Contract
Input: original question + selected evidence + passage IDs.

Output must:
- answer only from supplied evidence;
- cite supporting passage IDs;
- state insufficient evidence when support is inadequate.

## 8. Evaluation
```text
Fixed 100 Golden Queries
          |
          v
   Same Retrieval Pipeline
      /      |           BM25   Dense    Hybrid   optional MedCPT Rerank
      \      |       /
       Retrieval Metrics
 Recall@5 / Recall@10 / MRR@10 / nDCG@10
 optional MedCPT comparison: Hybrid@10 vs Hybrid@50 vs MedCPT@10
 optional upgrade comparison A–J in separate JSON files
 optional scientific studies: embedding comparison, candidate-pool curve,
 MedCPT pool size, missing-evidence categories, expansion/decomposition diagnosis,
 learned-sparse dual-miss recovery (experimental; separate JSON)
 (separate JSON; do not overwrite the Hybrid baseline)
          |
     latency/cost
     failure categories (never retrieved, ranked too low, filter, reranker, terminology, ...)
```

## 9. Testing
- Unit-test BM25, dense, hybrid, optional reranker, and experimental sparse independently.
- Test citation validation.
- Test insufficient-evidence behaviour.
- Run the fixed 100-query benchmark after retrieval changes.
- Manually inspect 2–3 representative queries and judge disagreements.

## 10. Failure Rules
- Never invent evidence or passage IDs.
- Final answer cannot use unrestricted corpus knowledge.
- Preserve IDs/scores for debugging.
- Record model/configuration changes.

## 11. Project Structure
```text
2026-09-15_biomed-hybrid-search/
├── PRD.md
├── architecture.md
├── architecture_essentials.md
├── AGENTS.md
├── README.md
├── requirements.txt
├── data/
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessing.py
│   ├── retrieval/
│   │   ├── bm25.py
│   │   ├── dense.py
│   │   ├── embedding_models.py
│   │   ├── hybrid.py
│   │   ├── sparse.py
│   │   ├── sparse_models.py
│   │   ├── reranker.py
│   │   ├── pipeline.py
│   │   ├── filter.py
│   │   ├── router.py
│   │   └── traces.py
│   ├── query/
│   │   ├── terminology.py
│   │   ├── expansion.py
│   │   ├── classify.py
│   │   ├── decomposition.py
│   │   └── understand.py
│   ├── generation/
│   │   ├── query_expansion.py
│   │   └── answer_generator.py
│   ├── validation/
│   │   └── citations.py
│   ├── inspection/
│   │   ├── prompts.py
│   │   ├── config_view.py
│   │   ├── decisions.py
│   │   └── status.py
│   └── evaluation/
│       ├── dataset.py
│       ├── metrics.py
│       ├── rerank_comparison.py
│       ├── upgrade_experiments.py
│       ├── scientific_study.py
│       ├── objective_failure.py
│       ├── sparse_study.py
│       ├── failure_analysis.py
│       └── run_evaluation.py
├── app/
│   └── ui.py
├── tests/
├── experiments/
│   ├── configs/
│   └── results/
└── notebooks/
```

## 12. Architecture Change Rules
New retrieval/generation components require a clear contract, independent tests, evaluation configuration and an architecture update.
