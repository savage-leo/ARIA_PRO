"""
Rate Limiting Middleware for ARIA Pro
Token bucket algorithm with per-user and global limits
"""

import time
import asyncio
from typing import Dict, Optional, Tuple
from collections import defaultdict
import logging
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import hashlib

logger = logging.getLogger(__name__)

class TokenBucket:
    """Token bucket for rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float):
        """
        Args:
            capacity: Maximum number of tokens in bucket
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from bucket
        Returns True if successful, False if insufficient tokens
        """
        async with self._lock:
            now = time.time()
            
            # Refill tokens based on elapsed time
            elapsed = now - self.last_refill
            refill_amount = elapsed * self.refill_rate
            
            self.tokens = min(self.capacity, self.tokens + refill_amount)
            self.last_refill = now
            
            # Try to consume
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def time_until_refill(self, tokens: int = 1) -> float:
        """Calculate seconds until enough tokens are available"""
        if self.tokens >= tokens:
            return 0
        
        needed = tokens - self.tokens
        return needed / self.refill_rate

class RateLimiter:
    """Rate limiter with multiple strategies"""
    
    def __init__(
        self,
        requests_per_minute: int = 100,
        burst_size: int = 20,
        enable_user_limits: bool = True,
        enable_global_limit: bool = True,
        user_requests_per_minute: int = 60
    ):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.enable_user_limits = enable_user_limits
        self.enable_global_limit = enable_global_limit
        self.user_requests_per_minute = user_requests_per_minute
        
        # Global rate limit
        self.global_bucket = TokenBucket(
            capacity=burst_size,
            refill_rate=requests_per_minute / 60.0
        )
        
        # Per-user buckets
        self.user_buckets: Dict[str, TokenBucket] = {}
        
        # IP-based buckets for unauthenticated requests
        self.ip_buckets: Dict[str, TokenBucket] = {}
        
        # Endpoint-specific limits
        self.endpoint_limits = {
            "/api/execute-order": 10,  # 10 requests per minute
            "/api/institutional-ai/execute-signal": 10,
            "/api/monitoring/kill-switch": 5,
            "/api/auth/token": 5,  # Prevent brute force
            "/api/auth/refresh": 10
        }
        
        # Cleanup task
        self._cleanup_task = None
    
    def _get_user_bucket(self, user_id: str) -> TokenBucket:
        """Get or create user-specific bucket"""
        if user_id not in self.user_buckets:
            self.user_buckets[user_id] = TokenBucket(
                capacity=self.burst_size,
                refill_rate=self.user_requests_per_minute / 60.0
            )
        return self.user_buckets[user_id]
    
    def _get_ip_bucket(self, ip: str) -> TokenBucket:
        """Get or create IP-specific bucket"""
        if ip not in self.ip_buckets:
            # More restrictive for unauthenticated requests
            self.ip_buckets[ip] = TokenBucket(
                capacity=10,
                refill_rate=30 / 60.0  # 30 requests per minute
            )
        return self.ip_buckets[ip]
    
    def _get_endpoint_bucket(self, endpoint: str) -> Optional[TokenBucket]:
        """Get endpoint-specific bucket if configured"""
        if endpoint in self.endpoint_limits:
            key = f"endpoint:{endpoint}"
            if key not in self.user_buckets:
                limit = self.endpoint_limits[endpoint]
                self.user_buckets[key] = TokenBucket(
                    capacity=max(3, limit // 3),  # Allow small bursts
                    refill_rate=limit / 60.0
                )
            return self.user_buckets[key]
        return None
    
    async def check_rate_limit(
        self,
        request: Request,
        user_id: Optional[str] = None
    ) -> Tuple[bool, Optional[float]]:
        """
        Check if request should be rate limited
        Returns (allowed, retry_after_seconds)
        """
        
        # Check endpoint-specific limit first
        path = request.url.path
        endpoint_bucket = self._get_endpoint_bucket(path)
        if endpoint_bucket:
            if not await endpoint_bucket.consume():
                retry_after = endpoint_bucket.time_until_refill()
                logger.warning(
                    f"Rate limit exceeded for endpoint {path}, "
                    f"user: {user_id}, IP: {request.client.host}"
                )
                return False, retry_after
        
        # Check global limit
        if self.enable_global_limit:
            if not await self.global_bucket.consume():
                retry_after = self.global_bucket.time_until_refill()
                logger.warning(f"Global rate limit exceeded")
                return False, retry_after
        
        # Check user/IP specific limit
        if self.enable_user_limits:
            if user_id:
                # Authenticated user
                user_bucket = self._get_user_bucket(user_id)
                if not await user_bucket.consume():
                    retry_after = user_bucket.time_until_refill()
                    logger.warning(f"User rate limit exceeded for {user_id}")
                    return False, retry_after
            else:
                # Unauthenticated - use IP
                client_ip = request.client.host
                if client_ip:
                    ip_bucket = self._get_ip_bucket(client_ip)
                    if not await ip_bucket.consume():
                        retry_after = ip_bucket.time_until_refill()
                        logger.warning(f"IP rate limit exceeded for {client_ip}")
                        return False, retry_after
        
        return True, None
    
    async def cleanup_old_buckets(self):
        """Periodically clean up unused buckets to prevent memory leak"""
        while True:
            await asyncio.sleep(3600)  # Run hourly
            
            try:
                now = time.time()
                
                # Clean user buckets inactive for >1 hour
                user_buckets_to_remove = []
                for user_id, bucket in self.user_buckets.items():
                    if now - bucket.last_refill > 3600:
                        user_buckets_to_remove.append(user_id)
                
                for user_id in user_buckets_to_remove:
                    del self.user_buckets[user_id]
                
                # Clean IP buckets inactive for >30 minutes
                ip_buckets_to_remove = []
                for ip, bucket in self.ip_buckets.items():
                    if now - bucket.last_refill > 1800:
                        ip_buckets_to_remove.append(ip)
                
                for ip in ip_buckets_to_remove:
                    del self.ip_buckets[ip]
                
                if user_buckets_to_remove or ip_buckets_to_remove:
                    logger.info(
                        f"Cleaned up {len(user_buckets_to_remove)} user buckets "
                        f"and {len(ip_buckets_to_remove)} IP buckets"
                    )
                    
            except Exception as e:
                logger.error(f"Error during rate limiter cleanup: {e}")
    
    def start_cleanup_task(self):
        """Start background cleanup task"""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self.cleanup_old_buckets())
    
    def stop_cleanup_task(self):
        """Stop background cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None

class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting"""
    
    def __init__(self, app: ASGIApp, rate_limiter: RateLimiter):
        super().__init__(app)
        self.rate_limiter = rate_limiter
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting"""
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/ready", "/api/health", "/api/ready"]:
            return await call_next(request)
        
        # Try to get user from request state (set by auth middleware)
        user_id = None
        if hasattr(request.state, "user"):
            user = request.state.user
            if isinstance(user, dict):
                user_id = user.get("username") or user.get("sub")
        
        # Check rate limit
        allowed, retry_after = await self.rate_limiter.check_rate_limit(
            request, user_id
        )
        
        if not allowed:
            # Return 429 Too Many Requests
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": retry_after
                },
                headers={
                    "Retry-After": str(int(retry_after) + 1),
                    "X-RateLimit-Limit": str(self.rate_limiter.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time() + retry_after))
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers to response
        if self.rate_limiter.enable_global_limit:
            remaining = int(self.rate_limiter.global_bucket.tokens)
            response.headers["X-RateLimit-Limit"] = str(
                self.rate_limiter.requests_per_minute
            )
            response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
            response.headers["X-RateLimit-Reset"] = str(
                int(time.time() + 60)
            )
        
        return response

# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None

def get_rate_limiter() -> RateLimiter:
    """Get or create singleton rate limiter"""
    global _rate_limiter
    
    if _rate_limiter is None:
        from backend.core.config import get_settings
        settings = get_settings()
        
        _rate_limiter = RateLimiter(
            requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
            burst_size=settings.RATE_LIMIT_BURST,
            enable_user_limits=settings.RATE_LIMIT_ENABLED,
            enable_global_limit=settings.RATE_LIMIT_ENABLED
        )
    
    return _rate_limiter
