import numpy as np
import pytest
from semantic_cache import config
from semantic_cache.embedding import generate_embedding

def test_generate_embedding_shape_and_normalization():
    query = "Test query for embedding"
    embedding = generate_embedding(query)
    # Check that the embedding has the correct dimension
    assert embedding.shape[0] == config.VECTOR_DIM, f"Expected {config.VECTOR_DIM}, got {embedding.shape[0]}"
    # Check that the embedding norm is non-zero
    norm = np.linalg.norm(embedding)
    assert norm > 0, "Embedding norm should be > 0"
