"""
cache.py
--------
Simple in-memory LRU query cache for the /search endpoint.

Keys a search result by the SHA-256 hash of the uploaded image bytes
combined with the top_k value.  Avoids re-embedding the same image
multiple times during a session.

Usage:
    from cache import query_cache

    hit = query_cache.get(image_bytes, top_k)
    if hit:
        return hit          # cached SearchResponse dict

    result = ... (run search)
    query_cache.set(image_bytes, top_k, result)
"""

import hashlib
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


class LRUQueryCache:
    """
    Thread-safe-enough LRU cache for search results.

    For a single-worker uvicorn process (the typical dev setup) a plain
    OrderedDict is sufficient.  If you scale to multiple workers, replace
    this with Redis or memcached.

    Attributes:
        maxsize: Maximum number of entries before eviction begins.
    """

    def __init__(self, maxsize: int = 128) -> None:
        self.maxsize = maxsize
        self._store: OrderedDict = OrderedDict()
        self._hits = 0
        self._misses = 0

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _make_key(image_bytes: bytes, top_k: int) -> str:
        """SHA-256 of (image bytes ‖ top_k) — collision-resistant cache key."""
        h = hashlib.sha256(image_bytes)
        h.update(top_k.to_bytes(2, "big"))
        return h.hexdigest()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self, image_bytes: bytes, top_k: int):
        """Return the cached result dict, or None on a cache miss."""
        key = self._make_key(image_bytes, top_k)
        if key in self._store:
            # Move to end (most-recently-used)
            self._store.move_to_end(key)
            self._hits += 1
            logger.debug(f"Cache HIT  key={key[:12]}… (total hits={self._hits})")
            return self._store[key]
        self._misses += 1
        logger.debug(f"Cache MISS key={key[:12]}… (total misses={self._misses})")
        return None

    def set(self, image_bytes: bytes, top_k: int, result) -> None:
        """Store a search result.  Evicts LRU entry if capacity is exceeded."""
        key = self._make_key(image_bytes, top_k)
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = result
        if len(self._store) > self.maxsize:
            evicted = self._store.popitem(last=False)
            logger.debug(f"Cache evicted LRU entry key={evicted[0][:12]}…")

    def clear(self) -> None:
        """Flush the entire cache (useful for testing)."""
        self._store.clear()
        self._hits = 0
        self._misses = 0

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        total = self._hits + self._misses
        hit_rate = self._hits / total if total else 0.0
        return {
            "cache_size": len(self._store),
            "cache_maxsize": self.maxsize,
            "cache_hits": self._hits,
            "cache_misses": self._misses,
            "cache_hit_rate": round(hit_rate, 4),
        }


# ── Singleton ──────────────────────────────────────────────────────────────────
query_cache = LRUQueryCache(maxsize=128)
