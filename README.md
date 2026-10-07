# Biomedical Hybrid Search & QA

Local biomedical retrieval and grounded question answering over a passage corpus. The system retrieves evidence with BM25 and dense search, fuses the results, and generates an answer only from selected passages, citing original passage IDs.

## Purpose

Users need relevant biomedical evidence and a grounded answer. This project loads the Hugging Face `rag-datasets/rag-mini-bioasq` dataset, prepares a stable passage collection, and runs BM25, dense MiniLM retrieval, hybrid RRF, optional query expansion, and grounded QA with deterministic citation checks.

## Architecture

High-level flow from the architecture documents:

```text
Question
   |
   +--> optional Query Expansion
   |
   +--> BM25 ----+
   |             |
   +--> Dense ---+--> Hybrid Fusion --> optional MedCPT Rerank --> Evidence --> Grounded LLM
                                                        |
                                                        v
                                                Citation Validation
                                                        |
                                                        v
                                                       UI
```

Code is organized under `src/` as specified in `architecture.md`: data loading, retrieval, generation, validation, and evaluation stay in separate modules. Passage IDs are preserved end to end.

This repository currently implements dataset loading, passage preparation, BM25 lexical retrieval, dense retrieval with a local MiniLM index, hybrid RRF fusion, optional MedCPT cross-encoder reranking, optional query expansion, grounded answer generation, citation validation, a local Streamlit UI over the full BioASQ passage corpus, a fixed 100-query retrieval evaluation, and a 20-query LLM-judge answer evaluation.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To use a real LLM in the UI, set `OPENAI_API_KEY` or `LLM_API_KEY` in the environment. Do not put keys in source files.

Run unit tests:

```bash
pytest -m "not network"
```

Run the Hugging Face dataset smoke test (downloads the corpus on first run):

```bash
pytest -m network
```

## How to load the dataset

The loader reads `rag-datasets/rag-mini-bioasq` from Hugging Face without modifying the source data.

```python
from src.data.loader import (
    get_passage_collection,
    load_corpus_dataset,
    load_golden_qa,
    load_qa_dataset,
)

# Unmodified Hugging Face splits
corpus = load_corpus_dataset()          # text-corpus / passages
qa = load_qa_dataset()                  # question-answer-passages / test

# Retrieval-ready collection: original passage_id + text
passages = get_passage_collection()

# Golden question-answer records with original relevant passage IDs
golden = load_golden_qa()
```

Small slices are useful for checks:

```python
sample = get_passage_collection(split="passages[:5]")
```

## How to run BM25

BM25 is the lexical retriever. It indexes a prepared passage collection and returns `passage_id`, `score`, and `text`.

```python
from src.data.preprocessing import prepare_corpus
from src.retrieval.bm25 import BM25Retriever

passages = prepare_corpus(
    [
        {"id": 20598273, "passage": "Hirschsprung disease is a congenital disorder."},
        {"id": 42, "passage": "Insulin regulates blood glucose."},
    ]
)
retriever = BM25Retriever(passages)
hits = retriever.search("Hirschsprung disease", top_k=5)
```

A reference pattern is in `examples/retrieval/bm25_example.py`.

## How to run dense retrieval

Dense retrieval uses a local sentence-transformer bi-encoder. The selected model is `sentence-transformers/all-MiniLM-L6-v2` because it is small (~80MB, 384-d), runs on CPU, needs no API key, and is a standard local RAG encoder. Passages and queries are embedded with the same model and ranked by cosine similarity in memory. No cloud vector database is used.

```python
from src.data.preprocessing import prepare_corpus
from src.retrieval.dense import DenseRetriever

passages = prepare_corpus(
    [
        {"id": 20598273, "passage": "Hirschsprung disease is a congenital disorder."},
        {"id": 42, "passage": "Insulin regulates blood glucose."},
    ]
)
retriever = DenseRetriever(passages)
hits = retriever.search("Hirschsprung disease", top_k=5)
```

A reference pattern is in `examples/retrieval/dense_example.py`.

## How to run hybrid retrieval

Hybrid retrieval combines BM25 (lexical) and dense (embedding) rankings with Reciprocal Rank Fusion (RRF). Raw BM25 scores are not added to cosine similarities because they are on different scales. Each ranking contributes `1 / (60 + rank)`, with ranks starting at 1. The default RRF constant is **60**. A passage found by both retrievers gets both contributions; a passage found by only one can still appear. The fused list is unique by `passage_id` and sorted by descending RRF score.

```python
from src.data.preprocessing import prepare_corpus
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever

passages = prepare_corpus(
    [
        {"id": 20598273, "passage": "Hirschsprung disease is a congenital disorder."},
        {"id": 42, "passage": "Insulin regulates blood glucose."},
    ]
)
bm25 = BM25Retriever(passages)
dense = DenseRetriever(passages)
retriever = HybridRetriever(
    passages,
    bm25_retriever=bm25,
    dense_retriever=dense,
)
hits = retriever.search("Hirschsprung disease", top_k=5)
```

A reference pattern is in `examples/retrieval/hybrid_example.py`.

## How to run MedCPT reranking

Reranking is an optional second stage **after** Hybrid RRF. It does not replace BM25, dense, or RRF, and it does **not** call OpenAI. `ncbi/MedCPT-Cross-Encoder` runs locally through Hugging Face Transformers / PyTorch. The model is loaded once and reused.

The reranker is **disabled by default** so the baseline pipeline stays available for A/B evaluation.

```text
baseline:  BM25 + Dense + RRF
upgraded:  BM25 + Dense + RRF + MedCPT Cross-Encoder
```

```python
from src.config import RerankerConfig
from src.retrieval.reranker import wrap_retriever

# Baseline: first-stage results are forwarded unchanged
baseline = wrap_retriever(hybrid, config=RerankerConfig(enabled=False))

# Upgraded: RRF candidate pool is reranked locally
upgraded = wrap_retriever(
    hybrid,
    config=RerankerConfig(
        enabled=True,
        candidate_pool_size=50,
        final_top_k=10,
        model_name="ncbi/MedCPT-Cross-Encoder",
        device="cpu",
    ),
)
hits = upgraded.search("Hirschsprung disease", top_k=5)
# hit.passage_id, hit.bm25_rank, hit.dense_rank, hit.rrf_score,
# hit.reranker_score, hit.final_rank, hit.pmid
```

Environment flags (used by the UI and as defaults for `--rerank`):

```bash
# Baseline (default)
unset BIOMED_RERANKER_ENABLED
streamlit run app/ui.py

# Upgraded
export BIOMED_RERANKER_ENABLED=1
export BIOMED_RERANKER_CANDIDATE_POOL_SIZE=50
export BIOMED_RERANKER_FINAL_TOP_K=10
export BIOMED_RERANKER_DEVICE=auto   # cpu, or cuda when available
streamlit run app/ui.py
```

CPU notes: MedCPT is a BERT-base cross-encoder (~110M parameters). The first enabled query downloads and loads the weights (on the order of 400MB). Later queries reuse the in-process model. Scoring a pool of 50 passages on CPU is typically a few seconds; CUDA is faster when available. `candidate_pool_size` is the main latency/memory knob. Duplicate (query, passage) pairs are cached in-process. Candidate filtering can shrink the pool before MedCPT; measure Recall before/after rather than assuming a smaller pool is safe.

A reference pattern is in `examples/retrieval/rerank_example.py`.

## Optional next-generation retrieval (feature-flagged)

The Hybrid BM25 + Dense + RRF path and MedCPT wrapper are unchanged. Additional stages default **off** so existing evaluations remain comparable.

```text
Question
  → classification
  → optional terminology normalization (original kept)
  → optional deterministic expansion / decomposition
  → BM25 + Dense (MiniLM baseline, optional PubMedBERT)
  → RRF fusion
  → optional candidate filter
  → optional MedCPT
  → top-k
```

| Flag | Default | Effect |
|---|---|---|
| `BIOMED_PIPELINE_ENABLED` | off | Use `StagedRetriever` instead of Hybrid-only wrapping |
| `BIOMED_TERMINOLOGY_ENABLED` | off | Local abbreviation/synonym normalization |
| `BIOMED_DETERMINISTIC_EXPANSION_ENABLED` | off | Local synonym variants; not concatenated into one query |
| `BIOMED_LLM_EXPANSION_ENABLED` | off | Existing OpenAI expansion remains separate and optional |
| `BIOMED_DECOMPOSITION_ENABLED` | off | Heuristic multi-aspect subqueries |
| `BIOMED_ADAPTIVE_ROUTING_ENABLED` | off | Deterministic query-type routing |
| `BIOMED_FILTERING_ENABLED` | off | Conservative duplicate/empty filtering before MedCPT |
| `BIOMED_BIOMEDICAL_EMBEDDINGS_ENABLED` | off | Dense model from `BIOMED_EMBEDDING_MODEL` |
| `BIOMED_EMBEDDING_MODEL` | `pritamdeka/S-PubMedBert-MS-MARCO` | Used only when biomedical embeddings are enabled |
| `BIOMED_EMBEDDING_CACHE_DIR` | `experiments/cache/embeddings` | Passage embedding cache |
| `BIOMED_SPARSE_ENABLED` | off | Experimental learned-sparse first-stage; **not** wired into production Hybrid |
| `BIOMED_SPARSE_MODEL` | `naver/splade-v3-distilbert` | Used only by the sparse experiment |
| `BIOMED_SPARSE_CACHE_DIR` | `experiments/cache/sparse` | Sparse CSR index cache |

LLM-based expansion/decomposition is not required for the 100-query retrieval benchmark. Deterministic preprocessing runs locally.

Traces are JSON (`src/retrieval/traces.py`). LangSmith is not used.

Failure analysis (`src/evaluation/failure_analysis.py`) labels misses as never-retrieved, ranked-too-low, terminology/semantic mismatch, overly complex query, decomposition failure, filter removal, reranker failure, or insufficient evidence.

Comparison A–J (does not overwrite baseline JSON):

```bash
python -m src.evaluation.run_evaluation --upgrade
```

Writes `experiments/results/retrieval_evaluation_upgrade_summary.json` and `..._upgrade_details.json`. Configurations that need an uncached model are skipped rather than fabricated. The later scientific embedding comparison downloaded `pritamdeka/S-PubMedBert-MS-MARCO` and wrote separate JSON.

## Why these models and stages exist

- **MiniLM (384-d)** is the default dense encoder because it is small, local, and CPU-friendly. 384 is the output width of `all-MiniLM-L6-v2`. It is a baseline, not a biomedical specialist.
- **MedCPT** was added because Hybrid ranking left relevant passages in the top-50 that were not in the top-10. Measured: Recall@10 0.4286 → 0.5082 at ~3.1 s rerank on CPU. The remaining gap is candidate Recall@50 = 0.6243.
- **Query expansion** is experimental and default OFF. Equal-weight fusion slightly lowered Recall@10 (0.4247 vs 0.4286).
- **Decomposition** is experimental, default OFF, and only applies to multi-part queries. It had a small Recall@10 gain (0.4305).
- **Biomedical embeddings:** `pritamdeka/S-PubMedBert-MS-MARCO` was downloadable from Hugging Face and compared on the same 100 queries / BM25 / RRF. Measured Hybrid prefixes from K=100: MiniLM Recall@10=0.4342 vs PubMedBERT 0.4445, MRR@10 0.7958 vs 0.8685, nDCG@10 0.5730 vs 0.6014. Index ~45 s vs ~326 s; 384-d/~62 MB vs 768-d/~124 MB. MiniLM remains the production default; enable `BIOMED_BIOMEDICAL_EMBEDDINGS_ENABLED` only if that cost is accepted.
- **Learned sparse (SPLADE)** is an experiment only. 86.5% of Hybrid@50 misses are dual misses that BM25@100 and MiniLM@100 both miss. Query expansion/decomposition recovered 0 of those IDs; PubMedBERT recovered ~5% of MiniLM dense misses. The sparse study asks whether a third first-stage signal recovers any of the 385 dual-miss gold IDs. Production remains BM25 + MiniLM + RRF. See `sparse_study_summary.json`.

## Scientific follow-up studies

These write **new** JSON files and do not overwrite the Hybrid baseline:

```bash
python -m src.evaluation.scientific_study --all
# or separately:
python -m src.evaluation.scientific_study --embedding
python -m src.evaluation.scientific_study --candidate-pool
python -m src.evaluation.scientific_study --medcpt-pool
python -m src.evaluation.scientific_study --missing-evidence
python -m src.evaluation.scientific_study --expansion
python -m src.evaluation.scientific_study --decomposition
python -m src.evaluation.objective_failure
python -m src.evaluation.sparse_study
```

Outputs:

- `experiments/results/embedding_model_comparison_{summary,details}.json`
- `experiments/results/candidate_pool_{summary,details}.json`
- `experiments/results/medcpt_pool_size_{summary,details}.json`
- `experiments/results/missing_evidence_{summary,details}.json`
- `experiments/results/expansion_study_{summary,details}.json`
- `experiments/results/decomposition_study_{summary,details}.json`
- `experiments/results/objective_failure_{summary,details}.json`
- `experiments/results/recovery_matrix_{summary,details}.json`
- `experiments/results/sparse_study_{summary,details}.json`

Heuristic labels in `missing_evidence_*.json` are **not** proven causes. Use `objective_failure_*.json` for rank-based categories.

Optional original-priority expansion (`expansion_mode="priority"`) keeps the original Hybrid ranking and appends expansion-only IDs. It is **not** the default.

A reference pattern is in `examples/retrieval/pipeline_example.py`.

## How to run query expansion

Query expansion is optional and runs **before** retrieval. The original user query is kept unchanged. An LLM proposes a small set of biomedical search alternates (default 3). BM25 and dense then search the original query plus each expansion. Existing Hybrid RRF fuses all of those rankings. Disable it with `QueryExpander(enabled=False)`, `GroundedQA.ask(..., expand=False)`, or the UI checkbox.

A reference pattern is in `examples/generation/query_expansion_example.py`.

## How to run grounded answer generation

The RAG flow is:

```text
question → HybridRetriever → selected evidence → grounded LLM → citation validation
```

The LLM sees only the original question and the selected evidence passages. It does not have access to the full corpus. Answers must use that evidence only, and supporting IDs are cited as `[PASSAGE_ID: 12345]`. Citation checks are deterministic: cited IDs are parsed from the answer and compared with the evidence that was actually supplied.

If retrieval returns fewer passages than `min_evidence_count` (default 1), the system returns an insufficient-evidence response and does **not** call the LLM.

```python
from src.generation.answer_generator import AnswerGenerator
from src.generation.rag import GroundedQA

qa = GroundedQA(hybrid_retriever, generator=AnswerGenerator(llm=my_llm))
result = qa.ask("Is papilin secreted?")
```

Set `OPENAI_API_KEY` or `LLM_API_KEY` only when making a real model call. Tests use a fake LLM. Run the local grounded-QA smoke test with:

```bash
pytest tests/test_rag_smoke.py tests/test_answer_generator.py tests/test_citations.py
```

A reference pattern is in `examples/generation/grounded_answer_example.py`.

## How to run offline retrieval evaluation

There are two separate evaluation layers:

- **100-query retrieval evaluation** scores ranking quality over the full BioASQ corpus.
- **20-query answer evaluation** scores generated RAG answers with an LLM judge. It does **not** replace the 100-query retrieval evaluation.

Retrieval evaluation uses the BioASQ golden QA split, a **fixed set of exactly 100 query IDs**, and the **full passage corpus** (40,221 passages). The 100 questions are saved in `experiments/configs/evaluation_100.json`. Later runs reuse those IDs. Do not create a new query set.

BM25, Dense, and Hybrid RRF are scored with:

```bash
source .venv/bin/activate
python -m src.evaluation.run_evaluation
```

Hybrid + Query Expansion uses the **same** 100 IDs and writes separate files so the BM25/Dense/Hybrid results are not overwritten:

```bash
export OPENAI_API_KEY="your-key-here"
python -m src.evaluation.run_evaluation --hybrid-qe
```

That Hybrid + QE run makes **100 LLM calls** (one expansion per query), about **$0.01** on `gpt-4o-mini`. Retrieval itself has no LLM cost.

Hybrid RRF + MedCPT rerank uses the **same** 100 IDs and writes separate files so baseline BM25/Dense/Hybrid results are not overwritten. This path does not call OpenAI:

```bash
python -m src.evaluation.run_evaluation --rerank
```

Results: `experiments/results/retrieval_evaluation_rerank_details.json` and `experiments/results/retrieval_evaluation_rerank_summary.json`.

To decide whether MedCPT actually improves ranking, run the three-way comparison. It does **not** overwrite the baseline BM25/Dense/Hybrid JSON:

```text
A  Hybrid RRF → Top 10
B  Hybrid RRF → Top 50          (candidate-pool ceiling)
C  Hybrid RRF → Top 50 → MedCPT → Top 10
```

```bash
python -m src.evaluation.run_evaluation --rerank-comparison
```

Results:

- `experiments/results/retrieval_evaluation_rerank_comparison_summary.json`
- `experiments/results/retrieval_evaluation_rerank_comparison_details.json`

That report includes Recall@5/10, MRR@10, nDCG@10, Candidate Recall@50, Candidate Hit@50, rerank recovery@10, rank movement, A vs C gain (including regressions), the pool-vs-top-10 ceiling gap, and a Hybrid vs MedCPT latency split. First enabled run downloads `ncbi/MedCPT-Cross-Encoder` (~400MB) and scores a 50-passage pool per query on CPU.

Reported ranking metrics, using each question's golden `relevant_passage_ids` as ground truth:

- **Recall@5** / **Recall@10**: fraction of the relevant passage set found in the top 5 / top 10
- **MRR@10**: reciprocal rank of the first relevant passage within the top 10
- **nDCG@10**: binary nDCG over the top 10
- **latency**: per-query search time (index construction excluded); average and median
- **cost**: $0 for BM25/Dense/Hybrid/MedCPT rerank; estimated expansion-LLM cost for Hybrid + QE

Measured BM25 / Dense / Hybrid RRF results on the 100-query set (full corpus):

| Configuration | Recall@5 | Recall@10 | MRR@10 | nDCG@10 | Median latency (ms) | Cost |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.3029 | 0.4185 | 0.7938 | 0.5646 | 51.84 | $0 |
| Dense | 0.2528 | 0.3432 | 0.7142 | 0.4771 | 7.08 | $0 |
| Hybrid RRF | 0.3127 | 0.4286 | 0.8105 | 0.5655 | 62.63 | $0 |

Hybrid + Query Expansion has not been scored yet. After `--hybrid-qe` finishes, results are in `experiments/results/retrieval_evaluation_hybrid_qe_summary.json` and the four-way table in `experiments/results/retrieval_evaluation_comparison.json`.

Detailed BM25/Dense/Hybrid rankings: `experiments/results/retrieval_evaluation_details.json`.

A tiny synthetic walkthrough is in `examples/evaluation/retrieval_evaluation_example.py`.

## How to run final-answer evaluation

Retrieval evaluation scores ranking quality. Final-answer evaluation is a separate, smaller pass: Hybrid RRF retrieves evidence, GroundedQA generates an answer, citations are checked deterministically, and an LLM judge scores the answer.

The judge uses a **fixed 20-query sample** from the existing 100-query evaluation set (first 20 IDs). That sample is saved to `experiments/configs/answer_evaluation_20.json`. It does not judge all 4,719 BioASQ questions.

The three judge dimensions (integers 1–5) are:

- **correctness** — agreement with the golden/reference answer
- **groundedness** — whether the generated answer is supported by the retrieved evidence
- **context relevance** — whether the retrieved evidence is relevant to the question

The judge does **not** decide citation validity. Citations are still checked by `src/validation/citations.py`.

Set `OPENAI_API_KEY` or `LLM_API_KEY` first. The generation model and judge model are configured in `src/config.py`; keys are not stored in code.

```bash
source .venv/bin/activate
python -m src.evaluation.answer_evaluation
```

Results:

- per-query records: `experiments/results/answer_evaluation_details.json`
- averages: `experiments/results/answer_evaluation_summary.json`
- 3-query manual review template: `experiments/results/answer_evaluation_manual_review.md`

The summary reports average correctness, groundedness, context relevance, citation validity rate, and generation/judge latency. It does not combine those numbers into a single winner score.

### Manual review

Open `experiments/results/answer_evaluation_manual_review.md`. Human review is **not** complete until you tick the boxes there. The file lists queries 0, 10, and 19 in full, plus query IDs 5 and 18 where citations were missing. For each example compare the question, reference answer, retrieved evidence, generated answer, deterministic citation validation, and judge scores. Tick observations such as supported answer, missing evidence, incorrect answer, citation problem, or retrieval problem. Do not convert those observations into an automatic score.

A fake-LLM walkthrough is in `examples/evaluation/answer_evaluation_example.py`.

## How to start the local UI

The Streamlit UI is a presentation layer over `GroundedQA`. It does not reimplement retrieval, embeddings, RRF, or the LLM.

From the project root:

```bash
source .venv/bin/activate
streamlit run app/ui.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

The UI has Explorer, Evaluation, Retrieval Pipeline, Experiments, Failure Analysis, Architecture, and Research tabs. **Explorer** is the live pipeline. **Evaluation** and **Experiments** read saved JSON; they do not rerun the 100-query benchmark or invent missing numbers.

Explorer shows:

- the original query
- generated query expansions (or a note that expansion was off)
- the grounded answer, including `[PASSAGE_ID: …]` citations
- citation validation warnings if an ID is not in the supplied evidence
- the backend insufficient-evidence message when retrieval does not return enough passages
- live per-query metrics (configuration, expansion count, evidence count, measured latency, citation validity, LLM calls, expansion cost if an expansion call ran)
- each evidence passage with ID, fusion score, and text
- reranker status in the sidebar; when reranking is on, evidence also shows reranker_score, RRF score, BM25/dense/RRF ranks, final rank, and PMID

Evaluation shows the saved 100-query retrieval comparison, the MedCPT Hybrid@10 vs Hybrid@50 vs reranked comparison when present, the 20-query LLM-judge averages, and a few manual-review examples from `answer_evaluation_details.json`.

The first start builds BM25 and a **local in-memory** MiniLM dense index over the **full** BioASQ passage corpus (`passages`, 40,221 passages). That can take about a minute. For a faster smoke index during development, set `BIOMED_UI_CORPUS_SPLIT=passages[:200]`. Set `OPENAI_API_KEY` or `LLM_API_KEY` before asking a question that should call the LLM (answer generation and query expansion).

Dataset name, configs, splits, embedding model name, RRF constant, retrieval `top_k`, generation model name, `min_evidence_count`, and the UI corpus split live in `src/config.py`. No API keys are stored in code. Hugging Face may warn about unauthenticated requests; an optional `HF_TOKEN` environment variable can raise rate limits, but it is not required for local loading.

## Current implementation status

| Area | Status |
|---|---|
| Dataset loading (`src/data/loader.py`) | Done |
| Passage preparation (`src/data/preprocessing.py`) | Done |
| Configuration (`src/config.py`) | Done |
| BM25 lexical retrieval (`src/retrieval/bm25.py`) | Done |
| Dense retrieval (`src/retrieval/dense.py`) | Done |
| Hybrid retrieval (`src/retrieval/hybrid.py`) | Done |
| Cross-encoder rerank (`src/retrieval/reranker.py`) | Done (optional; disabled by default) |
| Query understanding / staged pipeline | Done (optional; flags default off) |
| Grounded answer generation (`src/generation/answer_generator.py`) | Done |
| Citation validation (`src/validation/citations.py`) | Done |
| Local UI (`app/ui.py`) | Done (full corpus) |
| Query expansion (`src/generation/query_expansion.py`) | Done |
| Evaluation pipeline (`src/evaluation/`) | Done |
| LLM judge / answer evaluation | Done (20-query sample; human review still required) |

## Project structure

The layout follows `architecture.md`. Retrieval, generation, validation, and evaluation live under `src/`. The Streamlit app in `app/ui.py` only displays results from `GroundedQA`.
