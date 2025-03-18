# tests/test_persistent_cache.py
import pytest
from semantic_cache.persistent_cache import PersistentCache

def test_persistent_cache_set_get_delete():
    cache = PersistentCache()
    key = "test_key"
    value = {"data": "test_value"}
    
    # Set a value
    cache.set(key, value)
    result = cache.get(key)
    assert result == value, "Persistent cache should return the value that was set."
    
    # Delete the value and verify it's gone
    cache.delete(key)
    assert cache.get(key) is None, "Persistent cache should return None for a deleted key."
