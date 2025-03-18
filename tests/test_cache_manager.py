import pytest
from semantic_cache.persistent_cache import PersistentCache
from semantic_cache.session_cache import SessionCache
from semantic_cache.vector_store import VectorStore
from semantic_cache.cache_manager import CacheManager

def test_cache_manager_set_get_invalidate():
    persistent_cache = PersistentCache()
    session_cache = SessionCache(max_size=10, ttl=10)
    vector_store = VectorStore(use_subprocess=False)
    cache_manager = CacheManager(persistent_cache, session_cache, vector_store)
    
    query = "What is the weather today?"
    response = "It's sunny and 25°C."
    
    # Initially, the cache should be empty.
    assert cache_manager.get(query) is None, "CacheManager: Should return None for a new query."
    
    # Set the cache value.
    cache_manager.set(query, response)
    retrieved = cache_manager.get(query)
    assert retrieved == response, "CacheManager: Retrieved response should match stored response."
    
    # Invalidate the cache for the query.
    cache_manager.invalidate(query)
    assert cache_manager.get(query) is None, "CacheManager: Cache should be invalidated after deletion."
