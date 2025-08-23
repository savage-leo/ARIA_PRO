"""
Redis Caching Layer for ARIA Pro
High-performance caching for market data and model predictions
"""

import json
import logging
import asyncio
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pickle
import hashlib

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Redis cache configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 10
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    decode_responses: bool = False

class CacheKeyBuilder:
    """Build consistent cache keys for different data types"""
    
    @staticmethod
    def market_data(symbol: str, timeframe: str, timestamp: datetime = None) -> str:
        """Build cache key for market data"""
        ts_str = timestamp.strftime("%Y%m%d_%H%M") if timestamp else "latest"
        return f"market_data:{symbol}:{timeframe}:{ts_str}"
    
    @staticmethod
    def model_prediction(model_name: str, symbol: str, features_hash: str) -> str:
        """Build cache key for model predictions"""
        return f"model_pred:{model_name}:{symbol}:{features_hash}"
    
    @staticmethod
    def ai_signal(symbol: str, timestamp: datetime = None) -> str:
        """Build cache key for AI signals"""
        ts_str = timestamp.strftime("%Y%m%d_%H%M") if timestamp else "latest"
        return f"ai_signal:{symbol}:{ts_str}"
    
    @staticmethod
    def account_info(account_id: str) -> str:
        """Build cache key for account information"""
        return f"account:{account_id}"
    
    @staticmethod
    def positions(account_id: str) -> str:
        """Build cache key for positions"""
        return f"positions:{account_id}"
    
    @staticmethod
    def hash_features(features: Dict[str, Any]) -> str:
        """Create hash of features for cache key"""
        features_str = json.dumps(features, sort_keys=True)
        return hashlib.md5(features_str.encode()).hexdigest()[:12]

class RedisCache:
    """High-performance Redis cache for ARIA Pro"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.pool = None
        self.connected = False
        self._lock = asyncio.Lock()
        
        # Cache TTL settings (in seconds)
        self.ttl_settings = {
            "market_data": 60,      # 1 minute for market data
            "model_pred": 300,      # 5 minutes for model predictions
            "ai_signal": 180,       # 3 minutes for AI signals
            "account": 30,          # 30 seconds for account info
            "positions": 15,        # 15 seconds for positions
            "default": 300          # 5 minutes default
        }
        
        logger.info(f"RedisCache initialized: {self.config.host}:{self.config.port}")
    
    async def connect(self) -> bool:
        """Establish Redis connection"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not available - caching disabled")
            return False
        
        try:
            async with self._lock:
                if self.connected:
                    return True
                
                self.pool = redis.ConnectionPool(
                    host=self.config.host,
                    port=self.config.port,
                    db=self.config.db,
                    password=self.config.password,
                    max_connections=self.config.max_connections,
                    socket_timeout=self.config.socket_timeout,
                    socket_connect_timeout=self.config.socket_connect_timeout,
                    retry_on_timeout=self.config.retry_on_timeout,
                    decode_responses=self.config.decode_responses
                )
                
                # Test connection
                client = redis.Redis(connection_pool=self.pool)
                await client.ping()
                await client.close()
                
                self.connected = True
                logger.info("Redis connection established successfully")
                return True
                
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Close Redis connection"""
        if self.pool:
            await self.pool.disconnect()
            self.connected = False
            logger.info("Redis connection closed")
    
    async def _get_client(self):
        """Get Redis client from pool"""
        if not self.connected:
            await self.connect()
        
        if not self.connected:
            return None
        
        return redis.Redis(connection_pool=self.pool)
    
    def _get_ttl(self, key: str) -> int:
        """Get TTL for cache key based on prefix"""
        prefix = key.split(':')[0] if ':' in key else 'default'
        return self.ttl_settings.get(prefix, self.ttl_settings['default'])
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        client = await self._get_client()
        if not client:
            return False
        
        try:
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value)
            elif isinstance(value, (int, float, str, bool)):
                serialized_value = str(value)
            else:
                # Use pickle for complex objects
                serialized_value = pickle.dumps(value)
            
            # Use provided TTL or determine from key prefix
            cache_ttl = ttl or self._get_ttl(key)
            
            await client.setex(key, cache_ttl, serialized_value)
            logger.debug(f"Cached: {key} (TTL: {cache_ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Cache set failed for {key}: {e}")
            return False
        finally:
            await client.close()
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        client = await self._get_client()
        if not client:
            return default
        
        try:
            value = await client.get(key)
            if value is None:
                return default
            
            # Try to deserialize as JSON first
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                # Try pickle if JSON fails
                try:
                    return pickle.loads(value)
                except:
                    # Return as string if all else fails
                    return value.decode() if isinstance(value, bytes) else value
            
        except Exception as e:
            logger.error(f"Cache get failed for {key}: {e}")
            return default
        finally:
            await client.close()
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        client = await self._get_client()
        if not client:
            return False
        
        try:
            result = await client.delete(key)
            logger.debug(f"Deleted from cache: {key}")
            return bool(result)
        except Exception as e:
            logger.error(f"Cache delete failed for {key}: {e}")
            return False
        finally:
            await client.close()
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        client = await self._get_client()
        if not client:
            return False
        
        try:
            result = await client.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Cache exists check failed for {key}: {e}")
            return False
        finally:
            await client.close()
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key"""
        client = await self._get_client()
        if not client:
            return False
        
        try:
            result = await client.expire(key, ttl)
            return bool(result)
        except Exception as e:
            logger.error(f"Cache expire failed for {key}: {e}")
            return False
        finally:
            await client.close()
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern"""
        client = await self._get_client()
        if not client:
            return []
        
        try:
            keys = await client.keys(pattern)
            return [key.decode() if isinstance(key, bytes) else key for key in keys]
        except Exception as e:
            logger.error(f"Cache keys failed for pattern {pattern}: {e}")
            return []
        finally:
            await client.close()
    
    async def flush_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        keys = await self.keys(pattern)
        if not keys:
            return 0
        
        client = await self._get_client()
        if not client:
            return 0
        
        try:
            result = await client.delete(*keys)
            logger.info(f"Flushed {result} keys matching pattern: {pattern}")
            return result
        except Exception as e:
            logger.error(f"Cache flush failed for pattern {pattern}: {e}")
            return 0
        finally:
            await client.close()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        client = await self._get_client()
        if not client:
            return {"connected": False}
        
        try:
            info = await client.info()
            stats = {
                "connected": True,
                "used_memory": info.get("used_memory_human", "N/A"),
                "total_keys": info.get("db0", {}).get("keys", 0) if "db0" in info else 0,
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": 0
            }
            
            # Calculate hit rate
            total_requests = stats["hits"] + stats["misses"]
            if total_requests > 0:
                stats["hit_rate"] = round((stats["hits"] / total_requests) * 100, 2)
            
            return stats
        except Exception as e:
            logger.error(f"Cache stats failed: {e}")
            return {"connected": False, "error": str(e)}
        finally:
            await client.close()

class CacheManager:
    """High-level cache manager with domain-specific methods"""
    
    def __init__(self, cache: RedisCache):
        self.cache = cache
        self.key_builder = CacheKeyBuilder()
    
    async def cache_market_data(self, symbol: str, timeframe: str, data: Dict[str, Any], timestamp: datetime = None) -> bool:
        """Cache market data"""
        key = self.key_builder.market_data(symbol, timeframe, timestamp)
        return await self.cache.set(key, data)
    
    async def get_market_data(self, symbol: str, timeframe: str, timestamp: datetime = None) -> Optional[Dict[str, Any]]:
        """Get cached market data"""
        key = self.key_builder.market_data(symbol, timeframe, timestamp)
        return await self.cache.get(key)
    
    async def cache_model_prediction(self, model_name: str, symbol: str, features: Dict[str, Any], prediction: Any) -> bool:
        """Cache model prediction"""
        features_hash = self.key_builder.hash_features(features)
        key = self.key_builder.model_prediction(model_name, symbol, features_hash)
        
        cache_data = {
            "prediction": prediction,
            "features": features,
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "symbol": symbol
        }
        return await self.cache.set(key, cache_data)
    
    async def get_model_prediction(self, model_name: str, symbol: str, features: Dict[str, Any]) -> Optional[Any]:
        """Get cached model prediction"""
        features_hash = self.key_builder.hash_features(features)
        key = self.key_builder.model_prediction(model_name, symbol, features_hash)
        
        cached_data = await self.cache.get(key)
        if cached_data and isinstance(cached_data, dict):
            return cached_data.get("prediction")
        return None
    
    async def cache_ai_signal(self, symbol: str, signal: Dict[str, Any], timestamp: datetime = None) -> bool:
        """Cache AI signal"""
        key = self.key_builder.ai_signal(symbol, timestamp)
        return await self.cache.set(key, signal)
    
    async def get_ai_signal(self, symbol: str, timestamp: datetime = None) -> Optional[Dict[str, Any]]:
        """Get cached AI signal"""
        key = self.key_builder.ai_signal(symbol, timestamp)
        return await self.cache.get(key)
    
    async def cache_account_info(self, account_id: str, info: Dict[str, Any]) -> bool:
        """Cache account information"""
        key = self.key_builder.account_info(account_id)
        return await self.cache.set(key, info)
    
    async def get_account_info(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Get cached account information"""
        key = self.key_builder.account_info(account_id)
        return await self.cache.get(key)
    
    async def cache_positions(self, account_id: str, positions: List[Dict[str, Any]]) -> bool:
        """Cache positions"""
        key = self.key_builder.positions(account_id)
        return await self.cache.set(key, positions)
    
    async def get_positions(self, account_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached positions"""
        key = self.key_builder.positions(account_id)
        return await self.cache.get(key)
    
    async def invalidate_symbol_cache(self, symbol: str) -> int:
        """Invalidate all cache entries for a symbol"""
        patterns = [
            f"market_data:{symbol}:*",
            f"model_pred:*:{symbol}:*",
            f"ai_signal:{symbol}:*"
        ]
        
        total_deleted = 0
        for pattern in patterns:
            deleted = await self.cache.flush_pattern(pattern)
            total_deleted += deleted
        
        logger.info(f"Invalidated {total_deleted} cache entries for symbol: {symbol}")
        return total_deleted

# Global cache instances
_redis_cache: Optional[RedisCache] = None
_cache_manager: Optional[CacheManager] = None

def get_redis_cache() -> RedisCache:
    """Get or create singleton Redis cache"""
    global _redis_cache
    if _redis_cache is None:
        from backend.core.config import get_settings
        settings = get_settings()
        
        config = CacheConfig(
            host=getattr(settings, 'REDIS_HOST', 'localhost'),
            port=getattr(settings, 'REDIS_PORT', 6379),
            db=getattr(settings, 'REDIS_DB', 0),
            password=getattr(settings, 'REDIS_PASSWORD', None)
        )
        _redis_cache = RedisCache(config)
    return _redis_cache

def get_cache_manager() -> CacheManager:
    """Get or create singleton cache manager"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager(get_redis_cache())
    return _cache_manager
