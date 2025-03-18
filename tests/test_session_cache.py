import time
import pytest
from semantic_cache.session_cache import SessionCache

def test_session_cache_set_get():
    cache = SessionCache(max_size=3, ttl=5)
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1, "SessionCache: Key 'a' should be retrievable."
    assert cache.get("b") == 2, "SessionCache: Key 'b' should be retrievable."

def test_session_cache_expiration():
    cache = SessionCache(max_size=3, ttl=1)
    cache.set("a", 1)
    # Wait for the key to expire
    time.sleep(1.1)
    assert cache.get("a") is None, "SessionCache: Key 'a' should expire after TTL."

def test_session_cache_delete_and_metrics():
    cache = SessionCache(max_size=3, ttl=10)
    cache.set("a", 1)
    cache.delete("a")
    assert cache.get("a") is None, "SessionCache: Key 'a' should be deleted."
    # Test that metrics exist (assuming get_metrics is added)
    metrics = cache.get_metrics()
    assert "hits" in metrics and "misses" in metrics, "SessionCache: Metrics should include 'hits' and 'misses'."
