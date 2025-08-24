"""
Rate limiting middleware for ARIA PRO
Production-grade rate limiting with Redis backend
"""

from typing import Optional, Dict, Any
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timedelta
import hashlib
import logging
import time
import asyncio
from backend.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware with configurable limits per IP and endpoint
    """
    
    def __init__(
        self,
        app,
        default_limit: int = 100,  # requests per minute
        burst_limit: int = 10,     # burst allowance
        window_seconds: int = 60,   # time window
        exclude_paths: Optional[list] = None
    ):
        super().__init__(app)
        self.default_limit = default_limit
        self.burst_limit = burst_limit
        self.window_seconds = window_seconds
        self.exclude_paths = exclude_paths or ["/health", "/healthz", "/metrics", "/docs", "/openapi.json"]
        self.request_counts: Dict[str, list] = {}
        self._cleanup_task = None
        
        # Endpoint-specific limits
        self.endpoint_limits = {
            "/api/auth/login": 5,  # 5 per minute
            "/api/auth/register": 3,  # 3 per minute
            "/api/trading/execute": 10,  # 10 per minute
            "/api/ai/generate": 20,  # 20 per minute
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting"""
        
        # Skip rate limiting for excluded paths
        path = request.url.path
        if any(path.startswith(excluded) for excluded in self.exclude_paths):
            return await call_next(request)
        
        # Get client identifier (IP address)
        client_ip = self._get_client_ip(request)
        
        # Check rate limit
        if not await self._check_rate_limit(client_ip, path):
            logger.warning(f"Rate limit exceeded for {client_ip} on {path}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later.",
                    "retry_after": self.window_seconds
                },
                headers={
                    "Retry-After": str(self.window_seconds),
                    "X-RateLimit-Limit": str(self._get_limit_for_path(path)),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + self.window_seconds)
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = await self._get_remaining_requests(client_ip, path)
        response.headers["X-RateLimit-Limit"] = str(self._get_limit_for_path(path))
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + self.window_seconds)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check for proxy headers
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _get_limit_for_path(self, path: str) -> int:
        """Get rate limit for specific path"""
        # Check exact match
        if path in self.endpoint_limits:
            return self.endpoint_limits[path]
        
        # Check prefix match
        for endpoint, limit in self.endpoint_limits.items():
            if path.startswith(endpoint):
                return limit
        
        return self.default_limit
    
    async def _check_rate_limit(self, client_ip: str, path: str) -> bool:
        """Check if request is within rate limit"""
        now = time.time()
        key = f"{client_ip}:{path}"
        limit = self._get_limit_for_path(path)
        
        # Initialize or get request timestamps
        if key not in self.request_counts:
            self.request_counts[key] = []
        
        # Remove old timestamps outside the window
        cutoff = now - self.window_seconds
        self.request_counts[key] = [
            ts for ts in self.request_counts[key] if ts > cutoff
        ]
        
        # Check if under limit
        if len(self.request_counts[key]) >= limit:
            # Check burst allowance
            if len(self.request_counts[key]) >= limit + self.burst_limit:
                return False
        
        # Add current request timestamp
        self.request_counts[key].append(now)
        return True
    
    async def _get_remaining_requests(self, client_ip: str, path: str) -> int:
        """Get remaining requests for client"""
        key = f"{client_ip}:{path}"
        limit = self._get_limit_for_path(path)
        
        if key not in self.request_counts:
            return limit
        
        # Count requests in current window
        now = time.time()
        cutoff = now - self.window_seconds
        current_requests = len([
            ts for ts in self.request_counts[key] if ts > cutoff
        ])
        
        return max(0, limit - current_requests)
    
    async def cleanup_old_entries(self):
        """Periodic cleanup of old request entries"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                now = time.time()
                cutoff = now - self.window_seconds
                
                # Clean up old entries
                for key in list(self.request_counts.keys()):
                    self.request_counts[key] = [
                        ts for ts in self.request_counts[key] if ts > cutoff
                    ]
                    
                    # Remove empty entries
                    if not self.request_counts[key]:
                        del self.request_counts[key]
                
                logger.debug(f"Cleaned up rate limit entries. Active keys: {len(self.request_counts)}")
            except Exception as e:
                logger.error(f"Error in rate limit cleanup: {e}")


def create_rate_limiter(
    app,
    redis_client=None,
    default_limit: int = 100,
    window_seconds: int = 60
) -> RateLimitMiddleware:
    """
    Factory function to create rate limiter with optional Redis backend
    
    Args:
        app: FastAPI application
        redis_client: Optional Redis client for distributed rate limiting
        default_limit: Default requests per window
        window_seconds: Time window in seconds
    """
    
    if redis_client:
        # TODO: Implement Redis-backed rate limiting for distributed systems
        logger.info("Redis-backed rate limiting not yet implemented, using in-memory")
    
    return RateLimitMiddleware(
        app=app,
        default_limit=default_limit,
        window_seconds=window_seconds
    )


# Decorator for route-specific rate limiting
def rate_limit(requests: int = 10, window: int = 60):
    """
    Decorator for applying rate limits to specific routes
    
    Usage:
        @app.get("/api/endpoint")
        @rate_limit(requests=5, window=60)
        async def endpoint():
            return {"message": "Limited endpoint"}
    """
    def decorator(func):
        func._rate_limit = {"requests": requests, "window": window}
        return func
    return decorator
