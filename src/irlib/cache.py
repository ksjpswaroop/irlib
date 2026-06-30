"""LRU caching for retriever search results."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Any


class LRUCache:
    """Tiny least-recently-used cache.

    Pseudocode:
        on get: move key to end
        on set: evict oldest key when capacity is exceeded

    Use for repeated query workloads where the base retriever is expensive.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self.cache: OrderedDict[Any, Any] = OrderedDict()

    def __contains__(self, key: Any) -> bool:
        return key in self.cache

    def __getitem__(self, key: Any) -> Any:
        value = self.cache[key]
        self.cache.move_to_end(key)
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)


class CachedRetriever:
    """Async cache wrapper for any retriever.

    Pseudocode:
        key = (query, top_k)
        if key exists: return cached result
        result = await base.search in worker thread
        cache and return result
    """

    def __init__(self, base_retriever, max_size: int = 1000) -> None:
        self._base = base_retriever
        self._cache = LRUCache(max_size=max_size)

    async def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        key = (query, top_k)
        if key in self._cache:
            return self._cache[key]
        result = await asyncio.to_thread(self._base.search, query, top_k)
        self._cache[key] = result
        return result

