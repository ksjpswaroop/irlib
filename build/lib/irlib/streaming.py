"""Streaming document processing helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Mapping

from irlib.core import Document, normalize_documents


class StreamingDocumentProcessor:
    """Index documents from an iterable in batches.

    Pseudocode:
        buffer documents until batch_size
        normalize each batch
        if retriever is provided, reindex accumulated documents
        yield normalized documents

    Limitation: this compact processor reindexes accumulated documents for
    retrievers that do not support incremental indexing.
    """

    def __init__(self, batch_size: int = 100) -> None:
        self.batch_size = batch_size

    def process(
        self,
        docs: Iterable[str | Mapping[str, Any] | Document],
        retriever=None,
    ) -> list[Document]:
        all_docs: list[Document] = []
        batch: list[str | Mapping[str, Any] | Document] = []
        for item in docs:
            batch.append(item)
            if len(batch) >= self.batch_size:
                all_docs.extend(normalize_documents(batch, start_id=len(all_docs)))
                batch = []
        if batch:
            all_docs.extend(normalize_documents(batch, start_id=len(all_docs)))
        if retriever is not None:
            retriever.index(all_docs)
        return all_docs

