"""Multi-stage retrieval wrapping BM25, dense, RRF, optional filter, and MedCPT."""

from src.config import UpgradeConfig
from src.data.preprocessing import prepare_corpus
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.pipeline import StagedRetriever

passages = prepare_corpus(
    [
        {"id": 20598273, "passage": "Hirschsprung disease is a congenital disorder."},
        {"id": 42, "passage": "Insulin regulates blood glucose."},
    ]
)
bm25 = BM25Retriever(passages)
dense = DenseRetriever(passages)
retriever = StagedRetriever(
    bm25_retriever=bm25,
    dense_retriever=dense,
    passages=passages,
    config=UpgradeConfig(pipeline_enabled=True, terminology_enabled=True),
    rerank_enabled=False,
)
hits = retriever.search("Hirschsprung disease", top_k=5)
# retriever.last_trace is JSON-serializable via src.retrieval.traces.build_query_trace
