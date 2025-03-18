# tests/conftest.py
import pytest
import fakeredis
import redis

@pytest.fixture(autouse=True)
def patch_redis(monkeypatch):
    """
    Automatically patch redis.Redis to use fakeredis.FakeRedis in all tests.
    This prevents connection errors by providing an in-memory Redis.
    """
    monkeypatch.setattr(redis, "Redis", fakeredis.FakeRedis)
