"""
Basic tests for RAG Cost-Optimized system.
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cost_tracker import CostTracker
from cache_layer import CacheLayer
from vector_store import VectorStore, chunk_text


class TestCostTracker:
    """Test cost tracking functionality."""
    
    def test_track_embedding(self):
        tracker = CostTracker()
        cost = tracker.track_embedding("Test text", cached=False)
        assert cost > 0
        assert len(tracker.operations) == 1
        assert tracker.operations[0]["type"] == "embedding"
    
    def test_track_embedding_cached(self):
        tracker = CostTracker()
        cost = tracker.track_embedding("Test text", cached=True)
        assert cost == 0
        assert tracker.operations[0]["cached"] == True
    
    def test_track_vector_search(self):
        tracker = CostTracker()
        cost = tracker.track_vector_search(cached=False)
        assert cost == 0.001
        
        cached_cost = tracker.track_vector_search(cached=True)
        assert cached_cost == 0.0001
    
    def test_cost_breakdown(self):
        tracker = CostTracker()
        tracker.track_embedding("Test 1", cached=False)
        tracker.track_embedding("Test 2", cached=True)
        tracker.track_vector_search(cached=False)
        
        breakdown = tracker.get_breakdown()
        assert "embedding" in breakdown
        assert "vector_search" in breakdown
        assert breakdown["embedding"]["count"] == 2
        assert breakdown["embedding"]["cached_count"] == 1
        assert breakdown["embedding"]["uncached_count"] == 1
    
    def test_cache_savings(self):
        tracker = CostTracker()
        tracker.track_embedding("Test", cached=False)
        tracker.track_embedding("Test", cached=True)
        
        savings = tracker.get_cache_savings()
        assert savings["cached_operations"] == 1
        assert savings["savings"] >= 0
    
    def test_reset(self):
        tracker = CostTracker()
        tracker.track_embedding("Test")
        tracker.reset()
        assert len(tracker.operations) == 0
        assert tracker.total_cost == 0


class TestCacheLayer:
    """Test caching functionality."""
    
    def test_disabled_cache(self):
        cache = CacheLayer(enabled=False)
        assert cache.enabled == False
        cache.set("key", "value")
        assert cache.get("key") is None
    
    def test_cache_key_generation(self):
        cache = CacheLayer(enabled=False)
        key1 = cache._generate_key("prefix", "data")
        key2 = cache._generate_key("prefix", "data")
        assert key1 == key2
        
        key3 = cache._generate_key("prefix", "different")
        assert key1 != key3
    
    def test_cache_stats(self):
        cache = CacheLayer(enabled=False)
        cache.cache_hits = 10
        cache.cache_misses = 5
        
        stats = cache.get_stats()
        assert stats["hits"] == 10
        assert stats["misses"] == 5
        assert stats["total_requests"] == 15
        assert stats["hit_rate_percent"] == pytest.approx(66.67, rel=0.1)


class TestVectorStore:
    """Test vector store operations."""
    
    def test_chunk_text(self):
        text = "This is a test text that should be chunked into smaller pieces."
        chunks = chunk_text(text, chunk_size=20, overlap=5)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 20 for chunk in chunks)
    
    def test_chunk_text_small(self):
        text = "Short"
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_text_empty(self):
        chunks = chunk_text("", chunk_size=100, overlap=10)
        assert len(chunks) == 0


class TestIntegration:
    """Integration tests (without OpenAI API)."""
    
    def test_cost_tracking_workflow(self):
        """Test a complete cost tracking workflow."""
        tracker = CostTracker()
        
        # Simulate a query workflow
        tracker.track_embedding("Document 1", cached=False)
        tracker.track_embedding("Document 2", cached=False)
        tracker.track_embedding("Query text", cached=False)
        tracker.track_vector_search(cached=False)
        tracker.track_llm_generation(100, 50, "gpt-3.5-turbo")
        
        summary = tracker.get_summary()
        assert summary["total_operations"] == 5
        assert summary["total_cost"] > 0
        
        breakdown = summary["breakdown"]
        assert len(breakdown) == 3  # embedding, vector_search, llm_generation
    
    def test_caching_reduces_cost(self):
        """Test that caching reduces costs."""
        tracker1 = CostTracker()
        tracker1.track_embedding("Test query", cached=False)
        tracker1.track_vector_search(cached=False)
        uncached_cost = tracker1.total_cost
        
        tracker2 = CostTracker()
        tracker2.track_embedding("Test query", cached=True)
        tracker2.track_vector_search(cached=True)
        cached_cost = tracker2.total_cost
        
        assert cached_cost < uncached_cost


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
