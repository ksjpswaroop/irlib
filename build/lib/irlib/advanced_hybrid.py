"""Advanced hybrid retrieval pipeline."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from irlib.core import Document
from irlib.hybrid import HybridRetriever
from irlib.rerank import Reranker
from irlib.utils import QueryExpander


class AdvancedHybridRetriever(HybridRetriever):
    """Hybrid retrieval with optional query expansion and reranking.

    Pseudocode:
        expanded_query = expander.expand(query)
        candidates = hybrid sparse+dense search
        if reranker exists: rerank candidate passages

    Use when a RAG/search pipeline needs recall from fusion and precision from
    reranking.
    """

    def __init__(
        self,
        *,
        expander: QueryExpander | None = None,
        reranker: Reranker | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.expander = expander
        self.reranker = reranker

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        expanded = self.expander.expand(query) if self.expander else query
        candidates = super().search(expanded, top_k=max(top_k * 5, top_k))
        if not self.reranker:
            return candidates[:top_k]
        doc_ids = [doc_id for doc_id, _score in candidates]
        passages = [self.get_document(doc_id).text for doc_id in doc_ids]
        return self.reranker.rerank(expanded, passages, doc_ids=doc_ids, top_k=top_k)

