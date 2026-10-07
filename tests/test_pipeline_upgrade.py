from src.config import UpgradeConfig
from src.data.preprocessing import prepare_corpus
from src.retrieval.filter import filter_candidates
from src.retrieval.hybrid import RetrievalResult
from src.retrieval.pipeline import StagedRetriever
from src.retrieval.router import route_query
from src.retrieval.traces import TRACE_KEYS, build_query_trace


class FakeHit:
    def __init__(self, passage_id, score, text):
        self.passage_id = passage_id
        self.score = score
        self.text = text


class FakeRetriever:
    def __init__(self, ranking):
        self.ranking = ranking
        self.calls = []

    def search(self, query, top_k=None):
        self.calls.append((query, top_k))
        return list(self.ranking[:top_k] if top_k is not None else self.ranking)


class FakeScorer:
    def score_pairs(self, query, texts):
        return [float(len(text)) for text in texts]


def _passages():
    return prepare_corpus(
        [
            {"id": "A", "passage": "Hirschsprung disease of the intestine."},
            {"id": "B", "passage": "Insulin regulates blood glucose."},
            {"id": "C", "passage": "EGFR ligands include EGF."},
            {"id": "D", "passage": "Papilin is a secreted protein."},
            {"id": "E", "passage": "Papilin is a secreted protein."},
        ]
    )


def _retrievers():
    passages = _passages()
    texts = {item.passage_id: item.text for item in passages}
    ranking = [
        FakeHit("A", 3.0, texts["A"]),
        FakeHit("D", 2.0, texts["D"]),
        FakeHit("E", 1.5, texts["E"]),
        FakeHit("C", 1.0, texts["C"]),
        FakeHit("B", 0.5, texts["B"]),
    ]
    return passages, FakeRetriever(ranking), FakeRetriever(list(reversed(ranking))), texts


def test_router_disabled_is_fixed_hybrid():
    plan = route_query("What does EGFR do?", config=UpgradeConfig())
    assert plan.strategy == "fixed_hybrid"
    assert plan.use_bm25 is True
    assert plan.use_dense is True
    assert plan.use_expansions is False


def test_router_terminology_heavy_emphasizes_normalization():
    cfg = UpgradeConfig(adaptive_routing_enabled=True, expansion_enabled=True)
    plan = route_query("What does EGFR do?", config=cfg)
    assert plan.query_type == "terminology_heavy"
    assert plan.strategy == "lexical_normalized_hybrid"
    assert plan.use_normalized_query is True
    assert "bm25" in plan.components


def test_router_multi_part_uses_subqueries():
    cfg = UpgradeConfig(adaptive_routing_enabled=True)
    plan = route_query(
        "What are the risk factors, mechanisms, and treatment options for X?",
        config=cfg,
    )
    assert plan.strategy == "decompose_and_fuse"
    assert plan.use_subqueries is True


def test_filter_disabled_keeps_all():
    hits = [RetrievalResult("A", 1.0, "alpha"), RetrievalResult("B", 0.5, "beta")]
    result = filter_candidates("alpha", hits, config=UpgradeConfig(), enabled=False)
    assert result.before_count == 2
    assert result.after_count == 2
    assert result.removed == ()


def test_filter_drops_duplicate_text_but_not_empty_pool():
    hits = [
        RetrievalResult("A", 1.0, "same text"),
        RetrievalResult("B", 0.9, "same text"),
        RetrievalResult("C", 0.1, ""),
    ]
    cfg = UpgradeConfig(filtering_enabled=True, filter_drop_empty=True, filter_drop_duplicates=True)
    result = filter_candidates("same", hits, config=cfg, enabled=True)
    assert result.after_count == 1
    assert result.before_count == 3
    kept_ids = [item.passage_id for item in result.kept]
    assert kept_ids == ["A"]


def test_filter_recall_guard_never_empties_pool():
    hits = [RetrievalResult("A", 1.0, "")]
    cfg = UpgradeConfig(filtering_enabled=True, filter_drop_empty=True)
    result = filter_candidates("q", hits, config=cfg, enabled=True)
    assert result.after_count == 1
    assert result.kept[0].passage_id == "A"


def test_staged_pipeline_disabled_searches_original_only():
    passages, bm25, dense, texts = _retrievers()
    retriever = StagedRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        texts=texts,
        config=UpgradeConfig(pipeline_enabled=False),
        rerank_enabled=False,
    )
    hits = retriever.search("EGFR mutation", top_k=3)
    assert hits
    assert all(call[0] == "EGFR mutation" for call in bm25.calls)
    assert retriever.last_trace["pipeline_enabled"] is False
    assert retriever.last_trace["search_queries"] == ["EGFR mutation"]


def test_staged_pipeline_expansion_runs_multiple_queries():
    passages, bm25, dense, texts = _retrievers()
    retriever = StagedRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        texts=texts,
        config=UpgradeConfig(pipeline_enabled=True, expansion_enabled=True, terminology_enabled=True),
        rerank_enabled=False,
    )
    retriever.search("EGFR mutation", top_k=3)
    queries = [call[0] for call in bm25.calls]
    assert "EGFR mutation" in queries
    assert len(set(queries)) >= 2
    assert retriever.last_trace["expansion_variants"]


def test_staged_pipeline_filter_then_rerank_with_fake_scorer():
    passages, bm25, dense, texts = _retrievers()
    retriever = StagedRetriever(
        bm25_retriever=bm25,
        dense_retriever=dense,
        texts=texts,
        config=UpgradeConfig(pipeline_enabled=True, filtering_enabled=True, candidate_pool_size=5, final_top_k=2),
        scorer=FakeScorer(),
        rerank_enabled=True,
    )
    hits = retriever.search("secreted papilin", top_k=2)
    assert len(hits) == 2
    assert retriever.last_trace["reranker_enabled"] is True
    assert retriever.last_trace["candidate_count_after_filter"] <= retriever.last_trace["candidate_count_before_filter"]
    assert hits[0].final_rank == 1
    assert retriever.last_trace["stage_timings"]["rerank_ms"] is not None


def test_query_trace_is_json_serializable():
    trace = build_query_trace(
        {
            "original_query": "q",
            "normalized_query": "q",
            "query_classification": {"query_type": "simple_factual"},
            "expansion_variants": ["q2"],
            "subqueries": [],
            "search_queries": ["q"],
            "retrieval_strategy": {"strategy": "fixed_hybrid"},
            "bm25_results": ["A"],
            "dense_results": ["B"],
            "fusion_results": [{"passage_id": "A", "score": 0.1}],
            "filtering_results": {"after_count": 1},
            "reranker_results": [],
            "final_evidence": [{"passage_id": "A"}],
            "stage_timings": {"total_ms": 1.0},
        },
        query_id=1,
        final_answer="yes",
        citations={"is_valid": True},
        grounding_validation={"insufficient_evidence": False},
    )
    for key in TRACE_KEYS:
        assert key in trace
    import json

    json.dumps(trace)
