# semantic_cache/cache_manager.py
import hashlib
import threading
import logging
from typing import Any, Optional
from semantic_cache.embedding import generate_embedding
from semantic_cache.persistent_cache import PersistentCache
from semantic_cache.session_cache import SessionCache
from semantic_cache.vector_store import VectorStore
from semantic_cache import config

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self,
                 persistent_cache: PersistentCache,
                 session_cache: SessionCache,
                 vector_store: VectorStore,
                 similarity_threshold: float = config.SIMILARITY_THRESHOLD):
        self.persistent_cache = persistent_cache
        self.session_cache = session_cache
        self.vector_store = vector_store
        self.similarity_threshold = similarity_threshold
        self.lock = threading.Lock()  # Concurrency control for cache updates

    def get_cache_key(self, query: str) -> str:
        """
        Generate a unique key for the query using SHA256.
        """
        return hashlib.sha256(query.encode('utf-8')).hexdigest()

    def get(self, query: str) -> Optional[Any]:
        """
        Retrieve a cached response for the query.
        Order: session cache -> persistent cache -> semantic similarity lookup.
        """
        key = self.get_cache_key(query)
        result = self.session_cache.get(key)
        if result is not None:
            logger.info(f"CacheManager: Session cache hit for query: {query}")
            return result

        result = self.persistent_cache.get(key)
        if result is not None:
            self.session_cache.set(key, result)
            logger.info(f"CacheManager: Persistent cache hit for query: {query}")
            return result

        try:
            embedding = generate_embedding(query)
        except Exception as e:
            logger.exception(f"Failed to generate embedding for query '{query}': {e}")
            return None

        similar_entries = self.vector_store.search(embedding, top_k=1)
        for sim_key, distance in similar_entries:
            if distance < self.similarity_threshold:
                similar_result = self.persistent_cache.get(sim_key)
                if similar_result is not None:
                    self.session_cache.set(key, similar_result)
                    logger.info(f"CacheManager: Semantic cache hit for query: {query} with similar key: {sim_key}")
                    return similar_result

        logger.info(f"CacheManager: Cache miss for query: {query}")
        return None

    def set(self, query: str, value: Any):
        """
        Cache the response for the query in both caches and add the embedding to the vector store.
        """
        key = self.get_cache_key(query)
        with self.lock:
            self.session_cache.set(key, value)
            self.persistent_cache.set(key, value)
            try:
                embedding = generate_embedding(query)
                self.vector_store.add(key, embedding)
            except Exception as e:
                logger.exception(f"Failed to generate embedding or add to vector store for query '{query}': {e}")

    def invalidate(self, query: str):
        """
        Invalidate a cached response. Note: FAISS does not support in-place deletion.
        """
        key = self.get_cache_key(query)
        with self.lock:
            if self.session_cache.get(key) is not None:
                self.session_cache.cache.pop(key, None)
            self.persistent_cache.delete(key)
            logger.info(f"CacheManager: Invalidated cache for query: {query}")
            # For the vector store, consider re-indexing to remove stale entries.
