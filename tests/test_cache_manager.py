# tests/test_cache_manager.py
import pytest
from semantic_cache.cache_manager import CacheManager
from semantic_cache.persistent_cache import PersistentCache
from semantic_cache.session_cache import SessionCache
from semantic_cache.vector_store import VectorStore

def test_cache_manager_set_get_invalidate():
    # Initialize caches and vector store
    persistent_cache = PersistentCache()
    session_cache = SessionCache()
    # Use an embedding dimension that matches your embedding service (default 768 for SentenceTransformers)
    vector_store = VectorStore(dim=768)
    cache_manager = CacheManager(persistent_cache, session_cache, vector_store)
    
    query = "test query"
    value = "test response"
    
    # Initially, the cache should not have the query.
    assert cache_manager.get(query) is None, "CacheManager should return None for a new query."
    
    # Set a new value
    cache_manager.set(query, value)
    retrieved = cache_manager.get(query)
    assert retrieved == value, "CacheManager should return the value that was set."
    
    # Invalidate and confirm removal.
    cache_manager.invalidate(query)
    assert cache_manager.get(query) is None, "CacheManager should return None after invalidation."
