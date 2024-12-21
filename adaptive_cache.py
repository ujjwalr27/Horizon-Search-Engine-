import redis
import json
import time
import asyncio
import concurrent.futures
from typing import Dict, Any, Optional
from logging_config import get_module_logger

class RedisAdaptiveCache:
    def __init__(self, host='localhost', port=6379, max_size=100, default_expiry=300):
        self.logger = get_module_logger('redis_cache')
        
        try:
            self.redis_pool = redis.ConnectionPool(
                host=host, 
                port=port, 
                decode_responses=True,
                max_connections=3  # Reduced connections
            )
            self.redis_client = redis.Redis(
                connection_pool=self.redis_pool,
                socket_timeout=2,
                socket_connect_timeout=2
            )
            self.redis_client.ping()
            
            self.max_size = max_size  # Reduced cache size
            self.default_expiry = default_expiry  # Reduced expiry time
            self.logger.info(f"Redis cache initialized: {host}:{port}")
        except redis.ConnectionError as e:
            self.logger.error(f"Redis connection error: {e}")
            raise

    def _generate_key(self, query: str) -> str:
        return f"s:{hash(query)}"  # Shortened key prefix

    def put(self, query: str,  Any, expiry: Optional[int] = None) -> bool:
        try:
            key = self._generate_key(query)
            if not 
                return False
            
            # Limit data size
            if isinstance(data, list) and len(data) > 10:
                data = data[:10]  # Keep only top 10 results
                
            serialized_data = json.dumps(data, ensure_ascii=False)
            if len(serialized_data) > 50000:  # ~50KB limit
                self.logger.warning("Data too large for cache")
                return False
                
            with self.redis_client.pipeline() as pipe:
                pipe.set(key, serialized_data, ex=expiry or self.default_expiry)
                pipe.zadd('cache:ts', {key: time.time()})
                pipe.execute()
            
            self._evict_if_needed()
            return True
        except Exception as e:
            self.logger.error(f"Cache put error: {e}")
            return False

    def get(self, query: str) -> Optional[Any]:
        try:
            key = self._generate_key(query)
            cached_data = self.redis_client.get(key)
            if cached_data is not None:
                self.redis_client.zadd('cache:ts', {key: time.time()})
                return json.loads(cached_data)
            return None
        except Exception as e:
            self.logger.error(f"Cache get error: {e}")
            return None

    def _evict_if_needed(self) -> None:
        try:
            current_size = self.redis_client.zcard('cache:ts')
            if current_size > self.max_size:
                old_keys = self.redis_client.zrange('cache:ts', 0, current_size - self.max_size - 1)
                with self.redis_client.pipeline() as pipe:
                    for key in old_keys:
                        pipe.delete(key)
                        pipe.zrem('cache:ts', key)
                    pipe.execute()
        except Exception as e:
            self.logger.error(f"Cache eviction error: {e}")

    async def async_get(self, query: str) -> Optional[Any]:
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            return await loop.run_in_executor(pool, self.get, query)

    def clear(self) -> None:
        try:
            self.redis_client.flushdb()
            self.logger.info("Cache cleared")
        except Exception as e:
            self.logger.error(f"Clear cache error: {e}")
