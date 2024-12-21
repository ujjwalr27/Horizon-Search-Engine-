import redis
import json
import time
import asyncio
import concurrent.futures
from typing import Dict, Any, Optional
from logging_config import get_module_logger

from settings import (
    REDIS_HOST, 
    REDIS_PORT, 
    CACHE_EXPIRY, 
    LRU_CACHE_SIZE
)

class RedisAdaptiveCache:
    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, max_size=LRU_CACHE_SIZE, default_expiry=CACHE_EXPIRY):
        self.logger = get_module_logger('redis_cache')
        
        try:
            self.redis_pool = redis.ConnectionPool(
                host=host, 
                port=port, 
                decode_responses=True,
                max_connections=10
            )
            self.redis_client = redis.Redis(connection_pool=self.redis_pool)
            self.redis_client.ping()
            
            self.max_size = max_size
            self.default_expiry = default_expiry
            self.logger.info(f"Redis cache initialized: {host}:{port}")
        except redis.ConnectionError as e:
            self.logger.error(f"Redis connection error: {e}")
            raise

    def _generate_key(self, query: str) -> str:
        return f"search:{hash(query)}"

    def put(self, query: str, data: Any, expiry: Optional[int] = None) -> bool:
        try:
            key = self._generate_key(query)
            if not data:
                self.logger.warning("Attempt to cache empty data")
                return False
            serialized_data = json.dumps(data)
            with self.redis_client.pipeline() as pipe:
                pipe.set(key, serialized_data, ex=expiry or self.default_expiry)
                pipe.zadd('cache:timestamps', {key: time.time()})
                pipe.execute()
            self._evict_if_needed()
            return True
        except Exception as e:
            self.logger.error(f"Cache put error for query {query}: {e}")
            return False

    def get(self, query: str) -> Optional[Any]:
        try:
            key = self._generate_key(query)
            cached_data = self.redis_client.get(key)
            if cached_data is not None:
                self.redis_client.zadd('cache:timestamps', {key: time.time()})
                return json.loads(cached_data)
            return None
        except json.JSONDecodeError:
            self.logger.error(f"JSON decode error for key {key}")
            return None
        except Exception as e:
            self.logger.error(f"Cache get error for query {query}: {e}")
            return None

    def _evict_if_needed(self) -> None:
        try:
            current_size = self.redis_client.zcard('cache:timestamps')
            if current_size > self.max_size:
                old_keys = self.redis_client.zrange('cache:timestamps', 0, current_size - self.max_size - 1)
                with self.redis_client.pipeline() as pipe:
                    for key in old_keys:
                        pipe.delete(key)
                        pipe.zrem('cache:timestamps', key)
                    pipe.execute()
                self.logger.info(f"Evicted {current_size - self.max_size} cache entries")
        except Exception as e:
            self.logger.error(f"Cache eviction error: {e}")

    async def async_put(self, query: str, data: Any, expiry: Optional[int] = None) -> bool:
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, self.put, query, data, expiry)

    async def async_get(self, query: str) -> Optional[Any]:
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, self.get, query)

    def clear(self) -> None:
        try:
            self.redis_client.flushdb()
            self.logger.info("Entire Redis cache cleared")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
