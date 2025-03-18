# semantic_cache/embedding.py
import logging
import time
from typing import Any
import numpy as np
from semantic_cache import config

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    logger.info(f"Loaded SentenceTransformer model: {config.EMBEDDING_MODEL_NAME}")
except ImportError as e:
    logger.error("SentenceTransformers not installed. Run `pip install sentence-transformers`.")
    model = None

def generate_embedding(query: str) -> np.ndarray:
    """
    Generate an embedding vector for the given query using SentenceTransformers.
    Implements retry logic in case of transient errors.
    """
    attempt = 0
    while attempt < config.EMBEDDING_RETRY_COUNT:
        try:
            if model is None:
                raise RuntimeError("Embedding model unavailable.")
            embedding = model.encode(query, batch_size=1,show_progress_bar=False, num_workers=0)
            norm = np.linalg.norm(embedding) or 1.0
            normalized_embedding = embedding / norm
            return normalized_embedding.astype('float32')
        except Exception as e:
            attempt += 1
            logger.error(f"Embedding generation failed on attempt {attempt}/{config.EMBEDDING_RETRY_COUNT}: {e}")
            time.sleep(config.EMBEDDING_RETRY_DELAY)
    raise RuntimeError("Failed to generate embedding after multiple attempts.")
