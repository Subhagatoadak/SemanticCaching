# tests/test_vector_store.py
import numpy as np
import pytest
from semantic_cache.vector_store import VectorStore

def test_vector_store_add_search_delete():
    dim = 128
    vs = VectorStore(dim=dim)
    # Create a random vector
    vector = np.random.rand(dim).astype('float32')
    key = "test_vector"
    
    # Add the vector
    vs.add(key, vector)
    results = vs.search(vector, top_k=1)
    assert len(results) == 1, "There should be one search result."
    result_key, distance = results[0]
    assert result_key == key, "The key returned from search should match the added key."
    
    # Delete the vector
    vs.delete(key)
    results_after = vs.search(vector, top_k=1)
    # After deletion, the result should no longer contain our key.
    for r in results_after:
        assert r[0] != key, "Deleted vector should not appear in search results."
