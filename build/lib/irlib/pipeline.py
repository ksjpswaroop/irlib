"""Unified search pipeline for lexical, dense, fusion, and reranking stages.

Pseudocode:
    index documents in lexical and dense retrievers
    search both retrievers for a larger candidate set
    fuse candidate lists with RRF or normalized linear fusion
    rerank candidate texts when a reranker is configured

The pipeline is intended as an ergonomic orchestration layer over the smaller
retriever classes. It keeps the same public result shape: list[(doc_id, score)].
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from irlib.core import BaseRetriever, Document, normalize_scores, top_k as rank_top_k
from irlib.dense import DenseRetriever, HashingEncoder
from irlib.hybrid import ReciprocalRankFusion
from irlib.models import BM25Retriever


class SearchPipeline(BaseRetriever):
    """Orchestrate lexical retrieval, dense retrieval, fusion, and reranking.

    Use this when an application wants one object for a common RAG retrieval
    stack.

    Pseudocode:
        lexical.index(documents)
        dense.index(documents)
        candidates = fuse(lexical.search(q), dense.search(q))
        if reranker: rerank candidate document texts
        return top_k candidates

    Limitation: this pipeline evaluates retrieval/context selection only; it
    does not generate answers.
    """

    def __init__(
        self,
        lexical_retriever: BaseRetriever | None = None,
        dense_retriever: BaseRetriever | None = None,
        reranker: Any | None = None,
        fusion_strategy: str = "rrf",
        rrf_c: float = 60.0,
    ) -> None:
        super().__init__()
        self.lexical = lexical_retriever or BM25Retriever()
        self.dense = dense_retriever or DenseRetriever(encoder=HashingEncoder())
        self.reranker = reranker
        self.fusion_strategy = fusion_strategy
        self.rrf = ReciprocalRankFusion(c=rrf_c)

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        self.lexical.index(documents)
        self.dense.index(documents)
        self.documents = self.lexical.documents
        self._doc_by_id = {doc.id: doc for doc in self.documents}

    def search(self, query: str, top_k: int = 10, rerank_k: int = 50) -> list[tuple[int, float]]:
        candidate_k = max(top_k, rerank_k)
        lexical_results = self.lexical.search(query, top_k=candidate_k)
        dense_results = self.dense.search(query, top_k=candidate_k)
        merged = self._fuse(lexical_results, dense_results, top_k=candidate_k)
        if not self.reranker:
            return merged[:top_k]

        candidate_ids = [doc_id for doc_id, _score in merged[:candidate_k]]
        passages = [self.get_document(doc_id).text for doc_id in candidate_ids]
        if hasattr(self.reranker, "rerank"):
            return self.reranker.rerank(query, passages, doc_ids=candidate_ids, top_k=top_k)
        if hasattr(self.reranker, "score"):
            scores = self.reranker.score(query, passages)
            return rank_top_k(dict(zip(candidate_ids, scores)), top_k)
        raise TypeError("reranker must expose rerank(query, passages, doc_ids, top_k) or score(query, passages)")

    def _fuse(
        self,
        lexical_results: Sequence[tuple[int, float]],
        dense_results: Sequence[tuple[int, float]],
        *,
        top_k: int,
    ) -> list[tuple[int, float]]:
        if self.fusion_strategy == "rrf":
            return self.rrf.fuse([lexical_results, dense_results], top_k=top_k)
        if self.fusion_strategy != "linear":
            raise ValueError("fusion_strategy must be 'rrf' or 'linear'")

        lexical_scores = normalize_scores(dict(lexical_results))
        dense_scores = normalize_scores(dict(dense_results))
        doc_ids = set(lexical_scores) | set(dense_scores)
        scores = {
            doc_id: 0.5 * lexical_scores.get(doc_id, 0.0) + 0.5 * dense_scores.get(doc_id, 0.0)
            for doc_id in doc_ids
        }
        return rank_top_k(scores, top_k)


__all__ = ["SearchPipeline"]

