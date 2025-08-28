"""
Distributed Locking System for Multi-Instance Deployments
Provides Redis-based distributed locks for critical operations
"""

import asyncio
import time
import logging
import uuid
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from dataclasses import dataclass
import redis.asyncio as redis

logger = logging.getLogger(__name__)


@dataclass
class LockConfig:
    """Configuration for distributed locks"""
    default_timeout: int = 30  # seconds
    default_retry_delay: float = 0.1  # seconds
    max_retries: int = 100
    heartbeat_interval: int = 10  # seconds
    lock_prefix: str = "aria_lock"


class DistributedLock:
    """Redis-based distributed lock implementation"""
    
    def __init__(self, redis_client: redis.Redis, name: str, config: LockConfig = None):
        self.redis_client = redis_client
        self.name = name
        self.config = config or LockConfig()
        self.lock_key = f"{self.config.lock_prefix}:{name}"
        self.owner_id = str(uuid.uuid4())
        self.acquired = False
        self.heartbeat_task: Optional[asyncio.Task] = None
        
    async def acquire(self, timeout: Optional[int] = None, retry_delay: Optional[float] = None) -> bool:
        """Acquire the distributed lock"""
        timeout = timeout or self.config.default_timeout
        retry_delay = retry_delay or self.config.default_retry_delay
        
        start_time = time.time()
        retries = 0
        
        while time.time() - start_time < timeout and retries < self.config.max_retries:
            try:
                # Try to acquire lock with SET NX EX (atomic operation)
                acquired = await self.redis_client.set(
                    self.lock_key,
                    self.owner_id,
                    nx=True,  # Only set if key doesn't exist
                    ex=timeout  # Expiration time
                )
                
                if acquired:
                    self.acquired = True
                    logger.debug(f"Acquired lock: {self.name} (owner: {self.owner_id})")
                    
                    # Start heartbeat to extend lock
                    self.heartbeat_task = asyncio.create_task(self._heartbeat_loop(timeout))
                    return True
                
                # Lock not acquired, wait and retry
                await asyncio.sleep(retry_delay)
                retries += 1
                
            except Exception as e:
                logger.error(f"Error acquiring lock {self.name}: {e}")
                await asyncio.sleep(retry_delay)
                retries += 1
        
        logger.warning(f"Failed to acquire lock: {self.name} after {retries} retries")
        return False
    
    async def release(self) -> bool:
        """Release the distributed lock"""
        if not self.acquired:
            return True
        
        try:
            # Stop heartbeat
            if self.heartbeat_task and not self.heartbeat_task.done():
                self.heartbeat_task.cancel()
                try:
                    await self.heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            # Use Lua script to ensure atomic release (only if we own the lock)
            lua_script = """
            if redis.call("GET", KEYS[1]) == ARGV[1] then
                return redis.call("DEL", KEYS[1])
            else
                return 0
            end
            """
            
            result = await self.redis_client.eval(lua_script, 1, self.lock_key, self.owner_id)
            
            if result:
                self.acquired = False
                logger.debug(f"Released lock: {self.name} (owner: {self.owner_id})")
                return True
            else:
                logger.warning(f"Failed to release lock {self.name} - not owner or already released")
                return False
                
        except Exception as e:
            logger.error(f"Error releasing lock {self.name}: {e}")
            return False
    
    async def extend(self, additional_time: int) -> bool:
        """Extend the lock expiration time"""
        if not self.acquired:
            return False
        
        try:
            # Use Lua script to extend only if we own the lock
            lua_script = """
            if redis.call("GET", KEYS[1]) == ARGV[1] then
                return redis.call("EXPIRE", KEYS[1], ARGV[2])
            else
                return 0
            end
            """
            
            result = await self.redis_client.eval(
                lua_script, 1, self.lock_key, self.owner_id, additional_time
            )
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error extending lock {self.name}: {e}")
            return False
    
    async def _heartbeat_loop(self, initial_timeout: int):
        """Background task to extend lock periodically"""
        try:
            while self.acquired:
                await asyncio.sleep(self.config.heartbeat_interval)
                
                if not self.acquired:
                    break
                
                # Extend lock by the heartbeat interval + buffer
                extension_time = self.config.heartbeat_interval + 10
                success = await self.extend(extension_time)
                
                if not success:
                    logger.warning(f"Failed to extend lock {self.name} - may have been lost")
                    self.acquired = False
                    break
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Heartbeat error for lock {self.name}: {e}")
            self.acquired = False
    
    async def is_locked(self) -> bool:
        """Check if the lock is currently held"""
        try:
            value = await self.redis_client.get(self.lock_key)
            return value is not None
        except Exception as e:
            logger.error(f"Error checking lock status {self.name}: {e}")
            return False
    
    async def get_owner(self) -> Optional[str]:
        """Get the current owner of the lock"""
        try:
            return await self.redis_client.get(self.lock_key)
        except Exception as e:
            logger.error(f"Error getting lock owner {self.name}: {e}")
            return None
    
    async def __aenter__(self):
        """Async context manager entry"""
        success = await self.acquire()
        if not success:
            raise TimeoutError(f"Failed to acquire lock: {self.name}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.release()


class DistributedLockManager:
    """Manager for distributed locks"""
    
    def __init__(self, redis_url: str = None, config: LockConfig = None):
        self.config = config or LockConfig()
        self.redis_client = None
        self.redis_url = redis_url
        self.locks: Dict[str, DistributedLock] = {}
        
    async def connect(self) -> bool:
        """Connect to Redis"""
        try:
            if self.redis_url:
                self.redis_client = redis.from_url(self.redis_url)
            else:
                self.redis_client = redis.Redis()
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Distributed lock manager connected to Redis")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis for distributed locks: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()
    
    def get_lock(self, name: str) -> DistributedLock:
        """Get or create a distributed lock"""
        if not self.redis_client:
            raise RuntimeError("Lock manager not connected to Redis")
        
        if name not in self.locks:
            self.locks[name] = DistributedLock(self.redis_client, name, self.config)
        
        return self.locks[name]
    
    @asynccontextmanager
    async def lock(self, name: str, timeout: int = None):
        """Context manager for acquiring and releasing locks"""
        lock = self.get_lock(name)
        try:
            success = await lock.acquire(timeout)
            if not success:
                raise TimeoutError(f"Failed to acquire lock: {name}")
            yield lock
        finally:
            await lock.release()
    
    async def force_release(self, name: str) -> bool:
        """Force release a lock (admin operation)"""
        if not self.redis_client:
            return False
        
        try:
            lock_key = f"{self.config.lock_prefix}:{name}"
            result = await self.redis_client.delete(lock_key)
            logger.warning(f"Force released lock: {name}")
            return bool(result)
        except Exception as e:
            logger.error(f"Error force releasing lock {name}: {e}")
            return False
    
    async def get_all_locks(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all active locks"""
        if not self.redis_client:
            return {}
        
        try:
            pattern = f"{self.config.lock_prefix}:*"
            keys = await self.redis_client.keys(pattern)
            
            locks_info = {}
            for key in keys:
                if isinstance(key, bytes):
                    key = key.decode()
                
                lock_name = key.replace(f"{self.config.lock_prefix}:", "")
                owner = await self.redis_client.get(key)
                ttl = await self.redis_client.ttl(key)
                
                locks_info[lock_name] = {
                    "owner": owner.decode() if isinstance(owner, bytes) else owner,
                    "ttl": ttl,
                    "key": key
                }
            
            return locks_info
            
        except Exception as e:
            logger.error(f"Error getting lock information: {e}")
            return {}
    
    async def cleanup_expired_locks(self) -> int:
        """Clean up any expired locks (should be automatic with Redis TTL)"""
        locks_info = await self.get_all_locks()
        expired_count = 0
        
        for lock_name, info in locks_info.items():
            if info["ttl"] <= 0:  # Expired or no TTL
                await self.force_release(lock_name)
                expired_count += 1
        
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired locks")
        
        return expired_count


# Global lock manager instance
_lock_manager: Optional[DistributedLockManager] = None


def get_lock_manager() -> DistributedLockManager:
    """Get or create singleton lock manager"""
    global _lock_manager
    if _lock_manager is None:
        from backend.core.config import get_settings
        settings = get_settings()
        
        redis_url = getattr(settings, 'REDIS_URL', None)
        _lock_manager = DistributedLockManager(redis_url)
    
    return _lock_manager


# Convenience functions for common lock operations
async def acquire_trading_lock(symbol: str, timeout: int = 30) -> DistributedLock:
    """Acquire trading lock for a symbol"""
    manager = get_lock_manager()
    if not manager.redis_client:
        await manager.connect()
    
    lock = manager.get_lock(f"trading:{symbol}")
    success = await lock.acquire(timeout)
    if not success:
        raise TimeoutError(f"Failed to acquire trading lock for {symbol}")
    return lock


async def acquire_model_lock(model_name: str, timeout: int = 60) -> DistributedLock:
    """Acquire model training/update lock"""
    manager = get_lock_manager()
    if not manager.redis_client:
        await manager.connect()
    
    lock = manager.get_lock(f"model:{model_name}")
    success = await lock.acquire(timeout)
    if not success:
        raise TimeoutError(f"Failed to acquire model lock for {model_name}")
    return lock


@asynccontextmanager
async def trading_lock(symbol: str, timeout: int = 30):
    """Context manager for trading operations"""
    manager = get_lock_manager()
    if not manager.redis_client:
        await manager.connect()
    
    async with manager.lock(f"trading:{symbol}", timeout) as lock:
        yield lock


@asynccontextmanager
async def model_lock(model_name: str, timeout: int = 60):
    """Context manager for model operations"""
    manager = get_lock_manager()
    if not manager.redis_client:
        await manager.connect()
    
    async with manager.lock(f"model:{model_name}", timeout) as lock:
        yield lock
