"""
Cache Management API Endpoints
Provides administrative control over Redis cache
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import logging
from backend.core.redis_cache import get_redis_cache, get_cache_manager

router = APIRouter(prefix="/cache", tags=["Cache Management"])
logger = logging.getLogger(__name__)

@router.get("/stats")
async def get_cache_stats() -> Dict[str, Any]:
    """Get Redis cache statistics"""
    try:
        redis_cache = get_redis_cache()
        stats = await redis_cache.get_stats()
        return {"ok": True, "stats": stats}
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/keys")
async def get_cache_keys(pattern: str = "*", limit: int = 100) -> Dict[str, Any]:
    """Get cache keys matching pattern"""
    try:
        redis_cache = get_redis_cache()
        keys = await redis_cache.keys(pattern)
        
        # Limit results to prevent overwhelming response
        if len(keys) > limit:
            keys = keys[:limit]
            truncated = True
        else:
            truncated = False
        
        return {
            "ok": True, 
            "keys": keys, 
            "count": len(keys),
            "truncated": truncated,
            "pattern": pattern
        }
    except Exception as e:
        logger.error(f"Failed to get cache keys: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/key/{key}")
async def get_cache_value(key: str) -> Dict[str, Any]:
    """Get value for specific cache key"""
    try:
        redis_cache = get_redis_cache()
        value = await redis_cache.get(key)
        exists = await redis_cache.exists(key)
        
        return {
            "ok": True,
            "key": key,
            "value": value,
            "exists": exists
        }
    except Exception as e:
        logger.error(f"Failed to get cache value for {key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/key/{key}")
async def delete_cache_key(key: str) -> Dict[str, Any]:
    """Delete specific cache key"""
    try:
        redis_cache = get_redis_cache()
        deleted = await redis_cache.delete(key)
        
        return {
            "ok": True,
            "key": key,
            "deleted": deleted
        }
    except Exception as e:
        logger.error(f"Failed to delete cache key {key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/pattern/{pattern}")
async def flush_cache_pattern(pattern: str) -> Dict[str, Any]:
    """Delete all keys matching pattern"""
    try:
        redis_cache = get_redis_cache()
        deleted_count = await redis_cache.flush_pattern(pattern)
        
        return {
            "ok": True,
            "pattern": pattern,
            "deleted_count": deleted_count
        }
    except Exception as e:
        logger.error(f"Failed to flush cache pattern {pattern}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/symbol/{symbol}")
async def invalidate_symbol_cache(symbol: str) -> Dict[str, Any]:
    """Invalidate all cache entries for a symbol"""
    try:
        cache_manager = get_cache_manager()
        deleted_count = await cache_manager.invalidate_symbol_cache(symbol)
        
        return {
            "ok": True,
            "symbol": symbol,
            "deleted_count": deleted_count
        }
    except Exception as e:
        logger.error(f"Failed to invalidate cache for symbol {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/warmup")
async def warmup_cache(symbols: List[str] = None) -> Dict[str, Any]:
    """Warm up cache with fresh data for specified symbols"""
    try:
        if not symbols:
            symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "BTCUSD"]
        
        cache_manager = get_cache_manager()
        warmed_up = []
        
        # This would typically fetch fresh data and cache it
        # For now, just invalidate existing cache to force refresh
        for symbol in symbols:
            deleted_count = await cache_manager.invalidate_symbol_cache(symbol)
            warmed_up.append({
                "symbol": symbol,
                "invalidated_entries": deleted_count
            })
        
        return {
            "ok": True,
            "warmed_up": warmed_up,
            "message": "Cache invalidated for fresh data on next request"
        }
    except Exception as e:
        logger.error(f"Failed to warm up cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/ttl/{key}")
async def set_cache_ttl(key: str, ttl: int) -> Dict[str, Any]:
    """Set TTL for existing cache key"""
    try:
        redis_cache = get_redis_cache()
        result = await redis_cache.expire(key, ttl)
        
        return {
            "ok": True,
            "key": key,
            "ttl": ttl,
            "updated": result
        }
    except Exception as e:
        logger.error(f"Failed to set TTL for {key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def cache_health() -> Dict[str, Any]:
    """Check cache connectivity and basic health"""
    try:
        redis_cache = get_redis_cache()
        
        # Test basic operations
        test_key = "health_check_test"
        test_value = "test_value"
        
        # Set test value
        set_result = await redis_cache.set(test_key, test_value, ttl=10)
        
        # Get test value
        get_result = await redis_cache.get(test_key)
        
        # Delete test value
        delete_result = await redis_cache.delete(test_key)
        
        health_status = {
            "connected": redis_cache.connected,
            "operations": {
                "set": set_result,
                "get": get_result == test_value,
                "delete": delete_result
            }
        }
        
        # Get stats
        stats = await redis_cache.get_stats()
        health_status["stats"] = stats
        
        return {
            "ok": True,
            "health": health_status,
            "status": "healthy" if all(health_status["operations"].values()) else "degraded"
        }
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        return {
            "ok": False,
            "health": {"connected": False, "error": str(e)},
            "status": "unhealthy"
        }
