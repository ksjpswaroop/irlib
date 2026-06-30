"""Async wrappers for synchronous retrievers."""

from __future__ import annotations

import asyncio


class AsyncRetriever:
    """Run any retriever's synchronous search method in a worker thread.

    Pseudocode:
        await to_thread(base_retriever.search, query, top_k)

    Use when a web service or application needs an awaitable retrieval API.
    """

    def __init__(self, base_retriever):
        self._base = base_retriever

    async def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        return await asyncio.to_thread(self._base.search, query, top_k)

