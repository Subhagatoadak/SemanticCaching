# semantic_cache/session_cache.py
import time
import threading
from collections import OrderedDict
from typing import Any, Optional
from semantic_cache import config
import logging

logger = logging.getLogger(__name__)

class SessionCache:
    def __init__(self,
                 max_size: int = config.SESSION_CACHE_MAX_SIZE,
                 ttl: int = config.SESSION_CACHE_TTL):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def set(self, key: str, value: Any):
        with self.lock:
            current_time = time.time()
            if key in self.cache:
                self.cache.pop(key)
            self.cache[key] = (value, current_time)
            logger.debug(f"Session cache set key: {key}")
            self._evict_if_needed()

    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            current_time = time.time()
            if key in self.cache:
                value, timestamp = self.cache[key]
                if current_time - timestamp < self.ttl:
                    self.cache.move_to_end(key)
                    self.hits += 1
                    logger.debug(f"Session cache hit for key: {key}")
                    return value
                else:
                    self.cache.pop(key)
                    self.evictions += 1
                    logger.debug(f"Session cache expired key: {key}")
            self.misses += 1
            logger.debug(f"Session cache miss for key: {key}")
            return None

    def _evict_if_needed(self):
        while len(self.cache) > self.max_size:
            evicted_key, _ = self.cache.popitem(last=False)
            self.evictions += 1
            logger.info(f"Session cache evicted key: {evicted_key}")

    def get_metrics(self):
        with self.lock:
            return {"hits": self.hits, "misses": self.misses, "evictions": self.evictions}
