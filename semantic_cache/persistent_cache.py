# semantic_cache/persistent_cache.py
import redis
import pickle
import logging
from typing import Any, Optional
from semantic_cache import config

logger = logging.getLogger(__name__)

class PersistentCache:
    def __init__(self,
                 host: str = config.PERSISTENT_CACHE_HOST,
                 port: int = config.PERSISTENT_CACHE_PORT,
                 db: int = config.PERSISTENT_CACHE_DB,
                 ttl: int = config.PERSISTENT_CACHE_TTL,
                 password: str = config.REDIS_PASSWORD,
                 use_ssl: bool = config.REDIS_USE_SSL):
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                ssl=use_ssl,
                socket_timeout=5
            )
            self.client.ping()
            logger.info("Connected to Redis successfully.")
        except Exception as e:
            logger.exception("Failed to connect to Redis.")
            raise e
        self.ttl = ttl

    def set(self, key: str, value: Any):
        """Store the value in Redis with a TTL."""
        try:
            serialized_value = pickle.dumps(value)
            self.client.setex(key, self.ttl, serialized_value)
            logger.debug(f"Set key {key} in persistent cache.")
        except Exception as e:
            logger.exception(f"Failed to set key {key} in Redis: {e}")

    def get(self, key: str) -> Optional[Any]:
        """Retrieve the value from Redis if it exists and is not expired."""
        try:
            serialized_value = self.client.get(key)
            if serialized_value:
                logger.debug(f"Cache hit for key {key} in persistent cache.")
                return pickle.loads(serialized_value)
            logger.debug(f"Cache miss for key {key} in persistent cache.")
            return None
        except Exception as e:
            logger.exception(f"Error retrieving key {key} from Redis: {e}")
            return None

    def delete(self, key: str):
        """Remove the key from the cache."""
        try:
            self.client.delete(key)
            logger.debug(f"Deleted key {key} from persistent cache.")
        except Exception as e:
            logger.exception(f"Error deleting key {key} from Redis: {e}")
