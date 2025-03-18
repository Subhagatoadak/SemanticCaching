# **Semantic Caching for Generative AI**
🚀 **A Python package for caching AI-generated responses using Redis and FAISS for semantic similarity search.**  
Designed to **reduce computation time, improve response speed, and enable cross-session caching** for AI models.

---

## 🌟 **Features**
✅ **Cross-session & Cross-user Caching** – Combines an in-memory cache (session) with a persistent Redis-based cache.  
✅ **Semantic Similarity Search** – Uses **FAISS** to find semantically similar cached responses.  
✅ **Efficient Cache Management** – Supports LRU eviction, cache expiration policies, and manual invalidation.  
✅ **Configurable Storage Policies** – Define TTL, eviction policies, and dynamic cache limits.  
✅ **Robust & Production-Ready** – Includes logging, monitoring, and optimized error handling.

---

## 📦 **Folder Structure**
```
semantic_cache/
├── __init__.py
├── cache_manager.py       # Manages cache retrieval, similarity search, and storage policies
├── config.py              # Stores configurable settings (Redis, FAISS, cache policies)
├── embedding.py           # Generates embeddings using SentenceTransformers
├── persistent_cache.py    # Redis-based persistent caching implementation
├── session_cache.py       # Thread-safe in-memory session cache
├── utils.py               # Utility functions (logging setup, error handling)
├── vector_store.py        # FAISS-based vector store for similarity search
tests/                     # Unit tests for each module
├── test_cache_manager.py
├── test_embedding.py
├── test_persistent_cache.py
├── test_session_cache.py
├── test_vector_store.py
setup.py                   # Setup script for packaging
environment.yml             # Conda environment configuration
README.md                  # Project documentation
```

---

## 🛠️ **Installation**
### **1️⃣ Setup Conda Environment**
```sh
conda env create --file environment.yml
conda activate semantic_cache_env
```

### **2️⃣ Install the Package**
```sh
pip install -e .
```

### **3️⃣ Install Redis Locally (If Needed)**
```sh
# Ubuntu / Debian
sudo apt update && sudo apt install redis-server

# MacOS (Homebrew)
brew install redis
brew services start redis

# Docker (Alternative)
docker run --name redis -p 6379:6379 -d redis
```

---

## 🚀 **Usage**
### **1️⃣ Basic Example**
```python
from semantic_cache.cache_manager import CacheManager
from semantic_cache.persistent_cache import PersistentCache
from semantic_cache.session_cache import SessionCache
from semantic_cache.vector_store import VectorStore

# Initialize caches and vector store
persistent_cache = PersistentCache()
session_cache = SessionCache()
vector_store = VectorStore(dim=768)
cache_manager = CacheManager(persistent_cache, session_cache, vector_store)

# Example query
query = "What is the weather today?"
response = "It's sunny and 25°C."

# Store in cache
cache_manager.set(query, response)

# Retrieve cached response
cached_response = cache_manager.get(query)
print(cached_response)  # Output: "It's sunny and 25°C."
```

---

## 🔎 **How It Works**
### **Cache Lookup Order**
1️⃣ **Session Cache (Fastest)** → Checks if query exists in memory.  
2️⃣ **Persistent Redis Cache** → If not in memory, checks Redis.  
3️⃣ **Semantic Search via FAISS** → If not in Redis, searches for semantically similar queries.  
4️⃣ **Cache Miss → Generate Response** → If no match, call the AI model and store the response.  

---

## 🧪 **Running Tests**
```sh
pytest tests/
```

---

## 🏗 **Planned Enhancements (Next Version)**
### ✅ **1️⃣ Improved Embedding Module**
- [ ] Support multiple embedding models (OpenAI, Hugging Face, Custom LLMs).
- [ ] Dynamic embedding selection based on use case.

### ✅ **2️⃣ Advanced Cache Policies**
- [ ] Implement **dynamic TTLs** based on query frequency.
- [ ] **Pre-warm** the cache with commonly used queries.
- [ ] Support **cache sharding** for large-scale applications.

### ✅ **3️⃣ FAISS Enhancements**
- [ ] **Better handling of deletions** (re-indexing strategy).
- [ ] Support **multi-threaded FAISS searches** for faster lookups.
- [ ] Use **quantization** for memory-efficient FAISS storage.

### ✅ **4️⃣ Redis Improvements**
- [ ] Support **Redis Clustering** for distributed caching.
- [ ] Implement **auto-reconnect & failover** for Redis failures.
- [ ] Add **backup mechanisms** for persisted cache storage.

### ✅ **5️⃣ Production Optimization**
- [ ] Integrate **Prometheus/Grafana for monitoring** cache performance.
- [ ] Implement **async caching** using Celery or background workers.
- [ ] Provide **REST API endpoints** for caching operations.

### ✅ **6️⃣ Better Developer Experience**
- [ ] **Add detailed logging** for debugging cache operations.
- [ ] **Create CLI tool** to inspect & manage the cache.
- [ ] **Auto-generate API documentation** using Sphinx or MkDocs.

---

## 🏆 **Contributing**
🙌 Contributions are welcome! To contribute:
1. **Fork** the repository.
2. **Create** a feature branch (`git checkout -b feature-new-enhancement`).
3. **Commit** your changes (`git commit -m "Added new feature"`).
4. **Push** and **open a pull request**.

---

## 🛡 **License**
This project is licensed under the **MIT License**. See `LICENSE` for details.

---

## ⭐ **Support the Project**
If you find this project useful, please ⭐ **star this repository** and share your feedback! 🚀✨

---

## 🔗 **Resources**
- **FAISS Documentation**: [https://faiss.ai/](https://faiss.ai/)
- **Redis Documentation**: [https://redis.io/](https://redis.io/)
- **SentenceTransformers**: [https://www.sbert.net/](https://www.sbert.net/)

---

This **README** makes it **easy to understand the package**, install and use it, and **provides a clear roadmap** for the next version. 🚀