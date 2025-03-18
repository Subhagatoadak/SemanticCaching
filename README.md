# Semantic Caching for Generative AI

This repository contains a production-grade, modular implementation of semantic caching for generative AI tools. The design supports:
- **Cross-Session & Cross-User Caching:** Combines persistent caching (via Redis) with in-memory session-based caching (LRU with TTL).
- **Semantic Similarity:** Uses FAISS to perform similarity searches on query embeddings.
- **Modular Design:** Each component (embedding, persistent cache, session cache, vector store, and cache manager) is separated into its own module for reuse and easy modification.

## Project Structure

```
semantic_cache/
├── __init__.py
├── config.py           # Configuration for cache policies, TTLs, and vector dimensions.
├── embedding.py        # Dummy embedding generator (replace with production-grade model).
├── persistent_cache.py # Persistent caching backed by Redis.
├── session_cache.py    # In-memory session-based caching using an LRU strategy.
├── vector_store.py     # FAISS-based vector store for similarity search.
└── cache_manager.py    # Integration layer for managing caches and semantic search.
example.py              # Sample usage of the semantic caching system.
```

## Prerequisites

- Python 3.7+
- [Redis](https://redis.io/) server installed and running
- [FAISS](https://github.com/facebookresearch/faiss) (install via `faiss-cpu`)
- Other Python libraries: `redis`, `numpy`

Install the necessary Python dependencies with:

```bash
pip install redis faiss-cpu numpy
```

## Setup Instructions

### Redis Setup

Follow the instructions provided below to install and configure Redis:

- **On Linux (Ubuntu/Debian):**
  ```bash
  sudo apt-get update
  sudo apt-get install redis-server
  redis-cli ping  # Should return PONG
  ```

- **Using Docker:**
  ```bash
  docker run --name redis -p 6379:6379 -d redis
  # For persistence, mount a volume:
  docker run --name redis -p 6379:6379 -v /your/local/dir:/data -d redis redis-server --appendonly yes
  ```

- **On macOS (Homebrew):**
  ```bash
  brew install redis
  brew services start redis
  ```

> Refer to [Redis Documentation](https://redis.io/documentation) for detailed production configuration.

### Running the Example

Test the caching system by running the example file:

```bash
python example.py
```

The example demonstrates:
- Checking the session cache.
- Falling back to the persistent cache.
- Using the vector store to find semantically similar queries.
- Generating a new response when a cache miss occurs.

## TODO List & Known Gaps

### Embedding Module
- [ ] **Integrate a Real Embedding Service:**  
  Replace the dummy `generate_embedding` function with a production-grade model (e.g., OpenAI API, SentenceTransformers).
- [ ] **Error Handling:**  
  Add robust error handling and retries for embedding service calls.

### Persistent Cache
- [ ] **Enhanced Error Handling:**  
  Improve Redis connection handling, including reconnect strategies on failures.
- [ ] **Security Enhancements:**  
  Support Redis authentication and SSL configurations.
- [ ] **Logging & Monitoring:**  
  Integrate logging for cache operations and monitor cache hit/miss ratios.

### Session Cache
- [ ] **Thread Safety:**  
  Improve the LRU cache to be thread-safe if used in multi-threaded environments.
- [ ] **Advanced Metrics:**  
  Add monitoring for session cache performance (e.g., hit rates, evictions).

### Vector Store
- [ ] **Update/Deletion Support:**  
  Investigate strategies for removing or updating vectors in FAISS, as in-place deletion is non-trivial.
- [ ] **Performance Optimization:**  
  Tune search parameters and consider other FAISS index types for scalability.
- [ ] **Unit Testing:**  
  Add comprehensive tests for vector store functionality.

### Cache Manager
- [ ] **Concurrency Controls:**  
  Implement locking or other concurrency strategies to handle simultaneous cache updates.
- [ ] **Dynamic Similarity Threshold:**  
  Experiment with adaptive similarity thresholds based on system performance and query patterns.
- [ ] **Policy-Based Invalidation:**  
  Develop policies to automatically invalidate or refresh stale cache entries.

### Documentation & Testing
- [ ] **Comprehensive Documentation:**  
  Expand inline documentation and create usage guides for each module.
- [ ] **Unit & Integration Tests:**  
  Write tests covering all modules and integrate with CI/CD pipelines (e.g., GitHub Actions).
- [ ] **Code Comments:**  
  Improve comments and code clarity throughout the repository.

### Production Readiness
- [ ] **Logging Framework:**  
  Integrate a robust logging framework (e.g., Python’s `logging` module) across all modules.
- [ ] **Security Best Practices:**  
  Ensure endpoints (Redis, FAISS) are secured with authentication, firewalls, and proper network configurations.
- [ ] **Deployment Strategy:**  
  Consider using Docker Compose or Kubernetes for orchestrating deployment and scaling.
- [ ] **Benchmarking & Load Testing:**  
  Benchmark system performance under load and optimize cache policies as needed.

## Contributing

Contributions, suggestions, and improvements are welcome. Please open issues or submit pull requests with your enhancements.

## License

This project is licensed under the MIT License.

