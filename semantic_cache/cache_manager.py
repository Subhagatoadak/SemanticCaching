# semantic_cache/cache_manager.py
import hashlib

class CacheManager:
    def __init__(self, persistent_cache, session_cache, vector_store):
        self.persistent_cache = persistent_cache
        self.session_cache = session_cache
        self.vector_store = vector_store

    def get_cache_key(self, query):
        return hashlib.sha256(query.encode()).hexdigest()

    def get(self, query):
        key = self.get_cache_key(query)
        # Check session cache first:
        result = self.session_cache.get(key)
        if result is not None:
            return result
        # Check persistent cache next:
        result = self.persistent_cache.get(key)
        if result is not None:
            self.session_cache.set(key, result)
            return result
        # Finally, check FAISS for semantically similar vectors:
        from semantic_cache.embedding import generate_embedding
        try:
            embedding = generate_embedding(query)
        except Exception:
            return None
        similar_entries = self.vector_store.search(embedding, top_k=1)
        for sim_key, _ in similar_entries:
            similar_result = self.persistent_cache.get(sim_key)
            if similar_result is not None:
                self.session_cache.set(key, similar_result)
                return similar_result
        return None

    def set(self, query, value):
        key = self.get_cache_key(query)
        self.session_cache.set(key, value)
        self.persistent_cache.set(key, value)
        from semantic_cache.embedding import generate_embedding
        embedding = generate_embedding(query)
        self.vector_store.add(key, embedding)

    def invalidate(self, query):
        key = self.get_cache_key(query)
        self.persistent_cache.delete(key)
        self.session_cache.delete(key)
