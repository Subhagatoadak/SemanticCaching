import numpy as np
import pytest
from semantic_cache import config
from semantic_cache.vector_store import VectorStore

def create_random_vector(dim):
    return np.random.rand(dim).astype("float32")

def test_vector_store_add_search_delete():
    vs = VectorStore()
    key = "test_key"
    vector = create_random_vector(config.VECTOR_DIM)
    
    # Test adding the vector
    vs.add(key, vector)
    results = vs.search(vector, top_k=1)
    assert len(results) > 0, "VectorStore: Should return at least one search result."
    found_key, distance = results[0]
    assert found_key == key, "VectorStore: Retrieved key should match the added key."
    
    # Test deletion
    vs.delete(key)
    results_after_delete = vs.search(vector, top_k=1)
    assert len(results_after_delete) == 0, "VectorStore: No results should be found after deletion."

def test_vector_store_reset_index():
    vs = VectorStore()
    key = "test_key"
    vector = create_random_vector(config.VECTOR_DIM)
    vs.add(key, vector)
    # Ensure vector exists before reset
    results = vs.search(vector, top_k=1)
    assert len(results) > 0, "VectorStore: Vector should be found before reset."
    
    # Reset the index
    vs.reset_index()
    results_after_reset = vs.search(vector, top_k=1)
    assert len(results_after_reset) == 0, "VectorStore: No vectors should be found after reset."
