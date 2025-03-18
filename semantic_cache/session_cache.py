import time
import threading
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

class SessionCache:
    def __init__(self, max_size=100, ttl=300):
        """
        In-memory session cache with LRU eviction and TTL expiration.

        Args:
            max_size (int): Maximum number of entries in the cache.
            ttl (int): Time-to-live (in seconds) for each cache entry.
        """
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.Lock()
        self.hits = 0  # Count of cache hits
        self.misses = 0  # Count of cache misses

    def set(self, key, value):
        """
        Set a value in the cache under the given key.
        If the cache exceeds max_size, evict the least-recently-used item.
        """
        with self.lock:
            self.cache[key] = (value, time.time())
            if len(self.cache) > self.max_size:
                evicted_key, _ = self.cache.popitem(last=False)
                logger.info(f"SessionCache: Evicted key {evicted_key} due to cache size limits.")
            logger.info(f"SessionCache: Set key {key}")

    def get(self, key):
        """
        Retrieve the value for a given key if it exists and has not expired.
        Returns None if the key is missing or expired.
        """
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    self.hits += 1
                    logger.info(f"SessionCache: Hit for key {key}")
                    # Refresh the key's position for LRU policy
                    self.cache.move_to_end(key)
                    return value
                else:
                    self.misses += 1
                    logger.info(f"SessionCache: Key {key} expired, removing from cache.")
                    del self.cache[key]
            else:
                self.misses += 1
                logger.info(f"SessionCache: Miss for key {key}")
        return None

    def delete(self, key):
        """
        Delete the entry associated with the given key from the cache.
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                logger.info(f"SessionCache: Deleted key {key}")
