import hashlib
import logging

logger = logging.getLogger(__name__)

class CacheManager:
    """
    CacheManager handles multi-tier caching using a session (in-memory) cache,
    a persistent cache (e.g., Redis), and a semantic vector store (using FAISS).

    It uses an embedding model to generate query embeddings and performs a semantic
    search in the vector store if an exact match is not found.
    """

    def __init__(self, persistent_cache, session_cache, vector_store):
        """
        Initialize the CacheManager with the provided cache components.

        Args:
            persistent_cache: An instance of a persistent cache (e.g., Redis)
            session_cache: An instance of an in-memory session cache
            vector_store: An instance of a FAISS-based vector store
        """
        self.persistent_cache = persistent_cache
        self.session_cache = session_cache
        self.vector_store = vector_store
        logger.info("CacheManager initialized with persistent, session, and vector store caches.")

    def get_cache_key(self, query):
        """
        Generate a unique cache key for a given query using SHA256.

        Args:
            query (str): The query string.

        Returns:
            str: A hexadecimal string representing the cache key.
        """
        return hashlib.sha256(query.encode()).hexdigest()

    def get(self, query):
        """
        Retrieve a cached response for the given query using a three-tier lookup:
        1. Session (in-memory) cache.
        2. Persistent cache.
        3. FAISS vector store for semantically similar queries.

        Args:
            query (str): The query string.

        Returns:
            The cached response if found; otherwise, None.
        """
        key = self.get_cache_key(query)
        logger.debug(f"Looking up cache key: {key} for query: '{query}'")

        # Check session cache first.
        result = self.session_cache.get(key)
        if result is not None:
            logger.debug(f"Cache hit in session cache for key: {key}")
            return result

        # Check persistent cache next.
        result = self.persistent_cache.get(key)
        if result is not None:
            logger.debug(f"Cache hit in persistent cache for key: {key}. Updating session cache.")
            self.session_cache.set(key, result)
            return result

        # Compute embedding and search FAISS for semantically similar entries.
        try:
            from semantic_cache.embedding import generate_embedding
            embedding = generate_embedding(query)
        except Exception as e:
            logger.error(f"Error generating embedding for query '{query}': {e}")
            return None

        logger.debug("Performing FAISS search for semantically similar entries.")
        similar_entries = self.vector_store.search(embedding, top_k=1)
        for sim_key, distance in similar_entries:
            similar_result = self.persistent_cache.get(sim_key)
            if similar_result is not None:
                logger.debug(f"Found similar cached result for key: {sim_key} (distance: {distance}). Updating session cache.")
                self.session_cache.set(key, similar_result)
                return similar_result

        logger.debug(f"No cached result found for query: '{query}'")
        return None

    def set(self, query, value):
        """
        Cache the response for a given query in both the session and persistent caches,
        and add the query embedding to the vector store.

        Args:
            query (str): The query string.
            value: The response to be cached.
        """
        key = self.get_cache_key(query)
        logger.debug(f"Setting cache for key: {key} for query: '{query}'")

        try:
            self.session_cache.set(key, value)
            self.persistent_cache.set(key, value)
        except Exception as e:
            logger.error(f"Error setting cache for key {key}: {e}")

        try:
            from semantic_cache.embedding import generate_embedding
            embedding = generate_embedding(query)
            self.vector_store.add(key, embedding)
        except Exception as e:
            logger.error(f"Error adding vector for query '{query}' to vector store: {e}")

    def invalidate(self, query):
        """
        Invalidate (delete) the cached response for the given query from both the
        persistent and session caches.

        Args:
            query (str): The query string.
        """
        key = self.get_cache_key(query)
        logger.debug(f"Invalidating cache for key: {key} for query: '{query}'")
        try:
            self.persistent_cache.delete(key)
            self.session_cache.delete(key)
        except Exception as e:
            logger.error(f"Error invalidating cache for key {key}: {e}")
