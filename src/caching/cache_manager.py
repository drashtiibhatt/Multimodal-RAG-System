"""Cache manager for embeddings and query results."""

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta


class EmbeddingCache:
    """Cache for text embeddings to avoid regenerating same embeddings."""

    def __init__(self, cache_dir: str = "data/cache/embeddings", ttl_hours: int = 168):
        """
        Initialize embedding cache.

        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live in hours (default: 7 days)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        self.stats = {"hits": 0, "misses": 0}

    def _hash_text(self, text: str) -> str:
        """Create hash key from text."""
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for text.

        Args:
            text: Input text

        Returns:
            Cached embedding or None if not found/expired
        """
        cache_key = self._hash_text(text)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        if not cache_file.exists():
            self.stats["misses"] += 1
            return None

        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            # Check expiration
            if time.time() - cache_data["timestamp"] > self.ttl_seconds:
                cache_file.unlink()  # Delete expired cache
                self.stats["misses"] += 1
                return None

            self.stats["hits"] += 1
            return cache_data["embedding"]

        except Exception:
            self.stats["misses"] += 1
            return None

    def set(self, text: str, embedding: List[float]) -> None:
        """
        Cache embedding for text.

        Args:
            text: Input text
            embedding: Generated embedding
        """
        cache_key = self._hash_text(text)
        cache_file = self.cache_dir / f"{cache_key}.pkl"

        cache_data = {
            "text": text[:100],  # Store preview for debugging
            "embedding": embedding,
            "timestamp": time.time()
        }

        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)

    def clear(self) -> int:
        """
        Clear all cached embeddings.

        Returns:
            Number of files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
            count += 1
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0

        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": f"{hit_rate:.1f}%",
            "total_cached": len(list(self.cache_dir.glob("*.pkl")))
        }


class QueryCache:
    """Cache for complete query results (LLM responses)."""

    def __init__(self, cache_dir: str = "data/cache/queries", ttl_hours: int = 24):
        """
        Initialize query cache.

        Args:
            cache_dir: Directory to store cache files
            ttl_hours: Time-to-live in hours (default: 24 hours)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        self.stats = {"hits": 0, "misses": 0}

    def _hash_query(self, query: str, context_hash: str) -> str:
        """Create hash key from query and context."""
        combined = f"{query}:{context_hash}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _hash_context(self, context: str) -> str:
        """Create hash from context."""
        return hashlib.md5(context.encode()).hexdigest()[:16]

    def get(self, query: str, context: str) -> Optional[Dict[str, Any]]:
        """
        Get cached query result.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Cached result or None if not found/expired
        """
        context_hash = self._hash_context(context)
        cache_key = self._hash_query(query, context_hash)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            self.stats["misses"] += 1
            return None

        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)

            # Check expiration
            cached_time = datetime.fromisoformat(cache_data["cached_at"])
            if datetime.now() - cached_time > timedelta(seconds=self.ttl_seconds):
                cache_file.unlink()  # Delete expired cache
                self.stats["misses"] += 1
                return None

            self.stats["hits"] += 1
            return cache_data["result"]

        except Exception:
            self.stats["misses"] += 1
            return None

    def set(self, query: str, context: str, result: Dict[str, Any]) -> None:
        """
        Cache query result.

        Args:
            query: User query
            context: Retrieved context
            result: Generated result
        """
        context_hash = self._hash_context(context)
        cache_key = self._hash_query(query, context_hash)
        cache_file = self.cache_dir / f"{cache_key}.json"

        cache_data = {
            "query": query,
            "context_preview": context[:200],
            "result": result,
            "cached_at": datetime.now().isoformat()
        }

        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)

    def clear(self) -> int:
        """
        Clear all cached queries.

        Returns:
            Number of files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0

        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "hit_rate": f"{hit_rate:.1f}%",
            "total_cached": len(list(self.cache_dir.glob("*.json")))
        }


class CacheManager:
    """Central cache manager for all caching operations."""

    def __init__(
        self,
        embedding_ttl_hours: int = 168,  # 7 days
        query_ttl_hours: int = 24  # 24 hours
    ):
        """
        Initialize cache manager.

        Args:
            embedding_ttl_hours: TTL for embedding cache (default: 7 days)
            query_ttl_hours: TTL for query cache (default: 24 hours)
        """
        self.embedding_cache = EmbeddingCache(ttl_hours=embedding_ttl_hours)
        self.query_cache = QueryCache(ttl_hours=query_ttl_hours)

    def clear_all(self) -> Dict[str, int]:
        """
        Clear all caches.

        Returns:
            Dictionary with count of deleted files per cache type
        """
        return {
            "embeddings_cleared": self.embedding_cache.clear(),
            "queries_cleared": self.query_cache.clear()
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics from all caches."""
        return {
            "embedding_cache": self.embedding_cache.get_stats(),
            "query_cache": self.query_cache.get_stats()
        }
