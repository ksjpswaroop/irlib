"""Hybrid sparse-dense retrieval and rank fusion.

Hybrid retrieval combines exact lexical matching with semantic vector matching.
The classes here keep the public `index` and `search` style used across irlib.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from irlib.core import BaseRetriever, Document, normalize_scores, top_k as rank_top_k
from irlib.dense import DenseRetriever
from irlib.models import BM25Retriever


class ReciprocalRankFusion:
    """Fuse ranked lists using reciprocal rank positions.

    Use when retrievers return scores on incomparable scales.

    Pseudocode:
        for each ranked list:
            for each doc at rank r:
                score[doc] += 1 / (c + r)
        sort by fused score

    Limitation: raw score magnitudes are ignored.
    """

    def __init__(self, c: float = 60.0) -> None:
        self.c = c

    def fuse(self, ranked_lists: Sequence[Sequence[tuple[int, float]]], top_k: int = 10) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for results in ranked_lists:
            for rank, (doc_id, _score) in enumerate(results, start=1):
                scores[doc_id] += 1.0 / (self.c + rank)
        return rank_top_k(scores, top_k)


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[tuple[int, float]]],
    *,
    c: float = 60.0,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    """Functional RRF helper."""

    return ReciprocalRankFusion(c=c).fuse(ranked_lists, top_k=top_k)


class HybridRetriever(BaseRetriever):
    """Combine sparse and dense retrievers.

    Use for RAG, enterprise search, and support search where exact keywords and
    semantic matches both matter.

    Pseudocode:
        sparse_results = sparse.search(query)
        dense_results = dense.search(query)
        if fusion == rrf: combine by rank
        else: normalize scores and weighted-sum them

    Limitation: hybrid quality depends on the first-stage retrievers and fusion
    settings.
    """

    def __init__(
        self,
        sparse_retriever: BaseRetriever | None = None,
        dense_retriever: BaseRetriever | None = None,
        *,
        sparse_weight: float = 0.5,
        dense_weight: float = 0.5,
        fusion: str = "rrf",
    ) -> None:
        super().__init__()
        self.sparse_retriever = sparse_retriever or BM25Retriever()
        self.dense_retriever = dense_retriever or DenseRetriever()
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight
        self.fusion = fusion
        self.rrf = ReciprocalRankFusion()

    def index(self, documents: Sequence[str | Mapping[str, Any] | Document]) -> None:
        self.sparse_retriever.index(documents)
        self.dense_retriever.index(documents)
        self.documents = self.sparse_retriever.documents
        self._doc_by_id = {doc.id: doc for doc in self.documents}

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        search_k = max(top_k * 5, top_k)
        sparse_results = self.sparse_retriever.search(query, top_k=search_k)
        dense_results = self.dense_retriever.search(query, top_k=search_k)
        if self.fusion == "rrf":
            return self.rrf.fuse([sparse_results, dense_results], top_k=top_k)

        sparse_scores = normalize_scores(dict(sparse_results))
        dense_scores = normalize_scores(dict(dense_results))
        doc_ids = set(sparse_scores) | set(dense_scores)
        scores = {
            doc_id: self.sparse_weight * sparse_scores.get(doc_id, 0.0)
            + self.dense_weight * dense_scores.get(doc_id, 0.0)
            for doc_id in doc_ids
        }
        return rank_top_k(scores, top_k)


__all__ = ["ReciprocalRankFusion", "reciprocal_rank_fusion", "HybridRetriever"]

