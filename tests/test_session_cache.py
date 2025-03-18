# tests/test_session_cache.py
import time
import pytest
from semantic_cache.session_cache import SessionCache

def test_session_cache_set_get():
    cache = SessionCache(max_size=2, ttl=5)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1, "Session cache should return the correct value for key 'a'."
    assert cache.get("b") == 2, "Session cache should return the correct value for key 'b'."

def test_session_cache_expiry():
    cache = SessionCache(max_size=2, ttl=1)
    cache.set("a", 1)
    time.sleep(1.1)
    assert cache.get("a") is None, "Session cache should expire keys after TTL."

def test_session_cache_eviction():
    cache = SessionCache(max_size=2, ttl=10)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)  # This should trigger eviction (LRU)
    metrics = cache.get_metrics()
    assert metrics["evictions"] >= 1, "Session cache should evict keys when max_size is exceeded."
