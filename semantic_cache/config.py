# semantic_cache/config.py
import os

# Persistent cache settings (Redis)
PERSISTENT_CACHE_HOST = os.getenv('REDIS_HOST', 'localhost')
PERSISTENT_CACHE_PORT = int(os.getenv('REDIS_PORT', 6379))
PERSISTENT_CACHE_DB = int(os.getenv('REDIS_DB', 0))
PERSISTENT_CACHE_TTL = int(os.getenv('REDIS_TTL', 3600))  # seconds
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)
REDIS_USE_SSL = os.getenv('REDIS_USE_SSL', 'False').lower() in ('true', '1', 'yes')

# Session cache settings
SESSION_CACHE_MAX_SIZE = int(os.getenv('SESSION_CACHE_MAX_SIZE', 100))
SESSION_CACHE_TTL = int(os.getenv('SESSION_CACHE_TTL', 300))  # seconds

# Vector store settings
VECTOR_DIM = 768  # typical dimension for SentenceTransformers
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', 0.5))  # adjust based on your scale

# Embedding service settings
EMBEDDING_MODEL_NAME =  "all-mpnet-base-v2" 
EMBEDDING_RETRY_COUNT = int(os.getenv('EMBEDDING_RETRY_COUNT', 3))
EMBEDDING_RETRY_DELAY = float(os.getenv('EMBEDDING_RETRY_DELAY', 1.0))  # seconds

# Logging settings
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
