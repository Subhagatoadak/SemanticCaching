import os
# Set tokenizers parallelism to false before any heavy imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import multiprocessing
from semantic_cache.persistent_cache import PersistentCache
from semantic_cache.session_cache import SessionCache
from semantic_cache.vector_store import VectorStore
from semantic_cache.cache_manager import CacheManager
from semantic_cache.embedding import generate_embedding

logging.basicConfig(level=logging.INFO)

def main():
    # Create cache components with inline execution to avoid subprocess timeouts
    persistent_cache = PersistentCache()
    session_cache = SessionCache()
    vector_store = VectorStore(use_subprocess=False)  # Run inline for testing
    cache_manager = CacheManager(persistent_cache, session_cache, vector_store)
    
    # Reset FAISS index
    cache_manager.vector_store.reset_index()
    
    query = "What is the weather today?"
    response = "It's sunny and 25°C."
    
    # Store the response
    cache_manager.set(query, response)
    print(f"Stored in cache: {query} -> {response}")
    
    # Retrieve the response
    cached_response = cache_manager.get(query)
    print(f"Retrieved from cache: {cached_response}")
    
    # Attempt to retrieve a similar query
    similar_query = "Tell me today's weather"
    cached_response_similar = cache_manager.get(similar_query)
    if cached_response_similar:
        print(f"Retrieved similar response: {cached_response_similar}")
    else:
        print(f"No similar cached response found for: {similar_query}")
    
    # Invalidate the cache for the query
    cache_manager.invalidate(query)
    print(f"Invalidated cache for query: {query}")
    
    # Confirm invalidation
    post_delete_response = cache_manager.get(query)
    if post_delete_response:
        print(f"Unexpected retrieval after deletion: {post_delete_response}")
    else:
        print(f"Cache successfully invalidated. No response found for: {query}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
