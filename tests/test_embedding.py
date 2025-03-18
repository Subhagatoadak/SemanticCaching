# tests/test_embedding.py
import numpy as np
import pytest
from semantic_cache.embedding import generate_embedding

def test_generate_embedding():
    query = "This is a test query."
    emb = generate_embedding(query)
    assert emb is not None, "Embedding should not be None."
    assert isinstance(emb, np.ndarray), "Embedding should be a numpy array."
    # Check that the embedding is normalized (norm approximately 1)
    norm = np.linalg.norm(emb)
    np.testing.assert_almost_equal(norm, 1.0, decimal=2, err_msg="Embedding vector not normalized.")
