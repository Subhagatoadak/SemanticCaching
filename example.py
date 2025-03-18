import os
import logging
import multiprocessing
from semantic_cache.persistent_cache import PersistentCache
from semantic_cache.session_cache import SessionCache
from semantic_cache.vector_store import VectorStore
from semantic_cache.cache_manager import CacheManager
from semantic_cache.embedding import generate_embedding

# Disable TOKENIZERS parallelism to avoid warnings about forking after parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup logging
logging.basicConfig(level=logging.INFO)

def main():
    # Create cache components
    persistent_cache = PersistentCache()
    session_cache = SessionCache()
    vector_store = VectorStore()

    # Create CacheManager instance with all components
    cache_manager = CacheManager(persistent_cache, session_cache, vector_store)
    
    # Reset FAISS index before adding new vectors
    cache_manager.vector_store.reset_index()
    
    query = "What is the weather today?"
    response = "It's sunny and 25°C."
    
    # 1. Store the response in the cache
    cache_manager.set(query, response)
    print(f"Stored in cache: {query} -> {response}")
    
    # 2. Retrieve the response from the cache
    cached_response = cache_manager.get(query)
    print(f"Retrieved from cache: {cached_response}")
    
    # 3. Attempt to retrieve a similar query (should be a cache miss if not similar enough)
    similar_query = "Tell me today's weather"
    cached_response_similar = cache_manager.get(similar_query)
    if cached_response_similar:
        print(f"Retrieved similar response: {cached_response_similar}")
    else:
        print(f"No similar cached response found for: {similar_query}")
    
    # 4. Invalidate (delete) the cache for the original query from both caches
    cache_manager.invalidate(query)
    print(f"Invalidated cache for query: {query}")
    
    # 5. Confirm deletion by attempting to retrieve again
    post_delete_response = cache_manager.get(query)
    if post_delete_response:
        print(f"Unexpected retrieval after deletion: {post_delete_response}")
    else:
        print(f"Cache successfully invalidated. No response found for: {query}")

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Ensure multiprocessing works correctly on all platforms
    main()
