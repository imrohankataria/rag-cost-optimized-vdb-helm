"""
Caching layer using Redis for RAG operations.
Caches embeddings and query results to reduce API calls and costs.
"""

import hashlib
import json
import logging
import os
from typing import Any, Optional, List, Dict
import redis

logger = logging.getLogger(__name__)


class CacheLayer:
    """Redis-based caching layer for RAG operations."""
    
    # Socket timeout for Redis connections (seconds)
    SOCKET_TIMEOUT = 5
    
    def __init__(self, 
                 host: str = None,
                 port: int = None,
                 ttl: int = 3600,
                 enabled: bool = True):
        """
        Initialize cache layer.
        
        Args:
            host: Redis host (defaults to env REDIS_HOST or 'localhost')
            port: Redis port (defaults to env REDIS_PORT or 6379)
            ttl: Time-to-live for cache entries in seconds
            enabled: Whether caching is enabled
        """
        self.enabled = enabled
        self.ttl = ttl
        self.cache_hits = 0
        self.cache_misses = 0
        
        if not enabled:
            logger.info("Cache is disabled")
            self.client = None
            return
        
        host = host or os.getenv("REDIS_HOST", "localhost")
        port = port or int(os.getenv("REDIS_PORT", "6379"))
        
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                decode_responses=True,
                socket_connect_timeout=self.SOCKET_TIMEOUT
            )
            # Test connection
            self.client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.warning(f"Failed to connect to Redis: {e}. Running without cache.")
            self.client = None
            self.enabled = False
    
    def _generate_key(self, prefix: str, data: str) -> str:
        """Generate cache key from data."""
        hash_value = hashlib.sha256(data.encode()).hexdigest()
        return f"{prefix}:{hash_value}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.enabled or not self.client:
            return None
        
        try:
            value = self.client.get(key)
            if value:
                self.cache_hits += 1
                logger.debug(f"Cache hit for key: {key[:20]}...")
                return json.loads(value)
            else:
                self.cache_misses += 1
                logger.debug(f"Cache miss for key: {key[:20]}...")
                return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache."""
        if not self.enabled or not self.client:
            return
        
        try:
            ttl = ttl or self.ttl
            self.client.setex(
                key,
                ttl,
                json.dumps(value)
            )
            logger.debug(f"Cached value for key: {key[:20]}... (TTL: {ttl}s)")
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def cache_embedding(self, text: str, embedding: List[float]) -> str:
        """Cache an embedding vector."""
        key = self._generate_key("embedding", text)
        self.set(key, embedding)
        return key
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding vector."""
        key = self._generate_key("embedding", text)
        return self.get(key)
    
    def cache_query_result(self, query: str, result: Any) -> str:
        """Cache a query result."""
        key = self._generate_key("query", query)
        self.set(key, result)
        return key
    
    def get_query_result(self, query: str) -> Optional[Any]:
        """Get cached query result."""
        key = self._generate_key("query", query)
        return self.get(key)
    
    def cache_retrieval(self, query: str, documents: List[Dict], metadata: Dict = None) -> str:
        """Cache retrieval results."""
        key = self._generate_key("retrieval", query)
        cache_data = {
            "documents": documents,
            "metadata": metadata or {}
        }
        self.set(key, cache_data)
        return key
    
    def get_retrieval(self, query: str) -> Optional[Dict]:
        """Get cached retrieval results."""
        key = self._generate_key("retrieval", query)
        return self.get(key)
    
    def invalidate(self, key: str):
        """Invalidate a cache entry."""
        if not self.enabled or not self.client:
            return
        
        try:
            self.client.delete(key)
            logger.debug(f"Invalidated cache key: {key[:20]}...")
        except Exception as e:
            logger.error(f"Cache invalidate error: {e}")
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern."""
        if not self.enabled or not self.client:
            return
        
        try:
            keys = self.client.keys(pattern)
            if keys:
                self.client.delete(*keys)
                logger.info(f"Invalidated {len(keys)} keys matching pattern: {pattern}")
        except Exception as e:
            logger.error(f"Cache pattern invalidate error: {e}")
    
    def clear(self):
        """Clear all cache entries."""
        if not self.enabled or not self.client:
            return
        
        try:
            self.client.flushdb()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        stats = {
            "enabled": self.enabled,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "total_requests": total_requests,
            "hit_rate_percent": hit_rate
        }
        
        if self.enabled and self.client:
            try:
                info = self.client.info("stats")
                stats["redis_stats"] = {
                    "total_connections": info.get("total_connections_received", 0),
                    "total_commands": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0)
                }
            except Exception as e:
                logger.error(f"Error getting Redis stats: {e}")
        
        return stats
    
    def reset_stats(self):
        """Reset cache statistics."""
        self.cache_hits = 0
        self.cache_misses = 0
        logger.info("Cache stats reset")
