import pytest
from semantic_cache.persistent_cache import PersistentCache

def test_persistent_cache_set_get_delete():
    cache = PersistentCache()
    key = "test_key"
    value = {"data": "test_value"}
    
    # Set a value in persistent cache
    cache.set(key, value)
    result = cache.get(key)
    assert result == value, "PersistentCache: Retrieved value should match the stored value."
    
    # Delete the key and verify deletion
    cache.delete(key)
    assert cache.get(key) is None, "PersistentCache: Value should be None after deletion."
