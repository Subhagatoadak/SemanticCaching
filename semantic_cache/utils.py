# semantic_cache/utils.py
import logging
from semantic_cache import config

def setup_logging():
    logging.basicConfig(
        level=config.LOG_LEVEL,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
