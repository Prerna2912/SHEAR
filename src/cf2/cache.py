"""
CF2 — Response cache keyed by geometry hash.

Implemented as a thread-safe LRU dict.  Redis would be used in a multi-worker
deployment; for HuggingFace Spaces (single process) this in-process cache is
sufficient and avoids the Redis dependency.

The CF5 parametric explorer triggers up to 61 re-inference calls.
Without caching, the base-case geometry is re-computed every time.
This cache makes base-case lookups O(1).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional

from .schemas import InferenceResult


class InferenceCache:
    """
    Thread-safe LRU cache: geometry_hash → InferenceResult.

    Args:
        max_size: maximum number of cached results (default 128).
    """

    def __init__(self, max_size: int = 128):
        self._store: OrderedDict[str, InferenceResult] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, geometry_hash: str) -> Optional[InferenceResult]:
        with self._lock:
            if geometry_hash not in self._store:
                return None
            # Move to end (most recently used)
            self._store.move_to_end(geometry_hash)
            return self._store[geometry_hash]

    def put(self, geometry_hash: str, result: InferenceResult) -> None:
        with self._lock:
            if geometry_hash in self._store:
                self._store.move_to_end(geometry_hash)
            self._store[geometry_hash] = result
            if len(self._store) > self._max_size:
                self._store.popitem(last=False)   # evict oldest

    def invalidate(self, geometry_hash: str) -> bool:
        with self._lock:
            return self._store.pop(geometry_hash, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        return len(self._store)

    @property
    def keys(self):
        with self._lock:
            return list(self._store.keys())


# Module-level singleton shared across all requests
_CACHE = InferenceCache()


def get_cache() -> InferenceCache:
    return _CACHE
