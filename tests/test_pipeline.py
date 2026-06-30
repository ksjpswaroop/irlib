from irlib import DenseRetriever, HashingEncoder, SearchPipeline


class SpyReranker:
    def __init__(self):
        self.seen_passages = []

    def rerank(self, query, passages, doc_ids, top_k=10):
        self.seen_passages = list(passages)
        scores = {doc_id: (10.0 if "semantic" in passage.lower() else 1.0) for doc_id, passage in zip(doc_ids, passages)}
        return sorted(scores.items(), key=lambda item: -item[1])[:top_k]


def test_search_pipeline_fuses_and_returns_top_k():
    pipeline = SearchPipeline(dense_retriever=DenseRetriever(encoder=HashingEncoder()))
    pipeline.index(["python retrieval", "java search", "semantic retrieval for rag"])

    results = pipeline.search("semantic retrieval", top_k=2, rerank_k=3)

    assert len(results) == 2
    assert all(isinstance(doc_id, int) for doc_id, _score in results)


def test_search_pipeline_reranks_with_candidate_text_not_ids():
    reranker = SpyReranker()
    pipeline = SearchPipeline(dense_retriever=DenseRetriever(encoder=HashingEncoder()), reranker=reranker)
    pipeline.index(["python retrieval", "java search", "semantic retrieval for rag"])

    results = pipeline.search("semantic retrieval", top_k=1, rerank_k=3)

    assert results[0][0] == 2
    assert reranker.seen_passages
    assert all(isinstance(passage, str) for passage in reranker.seen_passages)


def test_search_pipeline_rejects_unknown_fusion_strategy():
    pipeline = SearchPipeline(dense_retriever=DenseRetriever(encoder=HashingEncoder()), fusion_strategy="bad")
    pipeline.index(["python retrieval"])

    try:
        pipeline.search("python")
    except ValueError as exc:
        assert "fusion_strategy" in str(exc)
    else:
        raise AssertionError("Expected ValueError")

