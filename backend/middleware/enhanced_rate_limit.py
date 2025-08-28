"""
Enhanced Rate Limiting Middleware with Granular Controls
Provides per-endpoint, per-user, and adaptive rate limiting
"""

import time
import asyncio
import logging
from typing import Dict, Optional, Tuple, List
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RateLimitType(Enum):
    PER_IP = "per_ip"
    PER_USER = "per_user"
    PER_ENDPOINT = "per_endpoint"
    GLOBAL = "global"


@dataclass
class RateLimitRule:
    requests_per_minute: int
    burst_allowance: int
    window_size: int = 60  # seconds
    rule_type: RateLimitType = RateLimitType.PER_IP
    endpoints: List[str] = field(default_factory=list)
    priority: int = 1  # Higher priority rules are checked first


@dataclass
class RateLimitState:
    requests: deque = field(default_factory=lambda: deque(maxlen=1000))
    burst_used: int = 0
    last_reset: float = field(default_factory=time.time)
    blocked_until: Optional[float] = None
    violation_count: int = 0


class EnhancedRateLimiter:
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                logger.info("Redis rate limiting enabled")
            except Exception as e:
                logger.warning(f"Redis connection failed, using in-memory rate limiting: {e}")
        
        # In-memory fallback
        self.rate_limit_states: Dict[str, RateLimitState] = defaultdict(RateLimitState)
        
        # Default rate limit rules
        self.rules = [
            # Critical endpoints - very restrictive
            RateLimitRule(
                requests_per_minute=10,
                burst_allowance=2,
                rule_type=RateLimitType.PER_IP,
                endpoints=["/api/trading/execute", "/api/positions/close", "/admin/*"],
                priority=10
            ),
            # Authentication endpoints
            RateLimitRule(
                requests_per_minute=5,
                burst_allowance=1,
                rule_type=RateLimitType.PER_IP,
                endpoints=["/api/auth/login", "/api/auth/refresh"],
                priority=9
            ),
            # Market data endpoints
            RateLimitRule(
                requests_per_minute=60,
                burst_allowance=10,
                rule_type=RateLimitType.PER_USER,
                endpoints=["/api/market/*", "/api/signals/*"],
                priority=5
            ),
            # General API endpoints
            RateLimitRule(
                requests_per_minute=100,
                burst_allowance=20,
                rule_type=RateLimitType.PER_IP,
                endpoints=["*"],
                priority=1
            )
        ]
        
        # Adaptive rate limiting
        self.adaptive_enabled = True
        self.system_load_threshold = 0.8
        self.adaptive_multiplier = 0.5  # Reduce limits by 50% under high load

    async def _get_redis_key(self, identifier: str, rule: RateLimitRule) -> str:
        """Generate Redis key for rate limiting"""
        rule_hash = hashlib.md5(f"{rule.rule_type.value}:{rule.requests_per_minute}".encode()).hexdigest()[:8]
        return f"rate_limit:{rule.rule_type.value}:{identifier}:{rule_hash}"

    async def _get_rate_limit_state(self, key: str, rule: RateLimitRule) -> RateLimitState:
        """Get rate limit state from Redis or memory"""
        if self.redis_client:
            try:
                data = await self.redis_client.get(key)
                if data:
                    state_dict = json.loads(data)
                    state = RateLimitState()
                    state.requests = deque(state_dict.get("requests", []), maxlen=1000)
                    state.burst_used = state_dict.get("burst_used", 0)
                    state.last_reset = state_dict.get("last_reset", time.time())
                    state.blocked_until = state_dict.get("blocked_until")
                    state.violation_count = state_dict.get("violation_count", 0)
                    return state
            except Exception as e:
                logger.warning(f"Redis get failed, using memory: {e}")
        
        return self.rate_limit_states[key]

    async def _save_rate_limit_state(self, key: str, state: RateLimitState, rule: RateLimitRule):
        """Save rate limit state to Redis or memory"""
        if self.redis_client:
            try:
                state_dict = {
                    "requests": list(state.requests),
                    "burst_used": state.burst_used,
                    "last_reset": state.last_reset,
                    "blocked_until": state.blocked_until,
                    "violation_count": state.violation_count
                }
                await self.redis_client.setex(
                    key, 
                    rule.window_size * 2,  # TTL is 2x window size
                    json.dumps(state_dict)
                )
            except Exception as e:
                logger.warning(f"Redis save failed, using memory: {e}")
                self.rate_limit_states[key] = state
        else:
            self.rate_limit_states[key] = state

    def _get_applicable_rule(self, endpoint: str, user_id: Optional[str] = None) -> RateLimitRule:
        """Get the most specific applicable rule for the endpoint"""
        applicable_rules = []
        
        for rule in self.rules:
            if self._endpoint_matches(endpoint, rule.endpoints):
                applicable_rules.append(rule)
        
        # Sort by priority (highest first)
        applicable_rules.sort(key=lambda r: r.priority, reverse=True)
        return applicable_rules[0] if applicable_rules else self.rules[-1]  # Default to general rule

    def _endpoint_matches(self, endpoint: str, patterns: List[str]) -> bool:
        """Check if endpoint matches any pattern"""
        for pattern in patterns:
            if pattern == "*":
                return True
            elif pattern.endswith("*"):
                if endpoint.startswith(pattern[:-1]):
                    return True
            elif pattern == endpoint:
                return True
        return False

    def _get_identifier(self, request: Request, rule: RateLimitRule) -> str:
        """Get identifier based on rule type"""
        if rule.rule_type == RateLimitType.PER_IP:
            return self._get_client_ip(request)
        elif rule.rule_type == RateLimitType.PER_USER:
            user_id = getattr(request.state, "user_id", None)
            return user_id or self._get_client_ip(request)
        elif rule.rule_type == RateLimitType.PER_ENDPOINT:
            return f"{request.url.path}:{self._get_client_ip(request)}"
        else:  # GLOBAL
            return "global"

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check for forwarded headers first
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"

    async def _check_system_load(self) -> float:
        """Check system load for adaptive rate limiting"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            return max(cpu_percent, memory_percent) / 100.0
        except ImportError:
            return 0.0  # No psutil, assume low load

    async def is_allowed(self, request: Request) -> Tuple[bool, Dict[str, any]]:
        """Check if request is allowed under rate limits"""
        endpoint = request.url.path
        rule = self._get_applicable_rule(endpoint)
        identifier = self._get_identifier(request, rule)
        key = await self._get_redis_key(identifier, rule)
        
        # Get current state
        state = await self._get_rate_limit_state(key, rule)
        current_time = time.time()
        
        # Check if currently blocked
        if state.blocked_until and current_time < state.blocked_until:
            remaining = int(state.blocked_until - current_time)
            return False, {
                "error": "Rate limit exceeded",
                "retry_after": remaining,
                "rule_type": rule.rule_type.value,
                "limit": rule.requests_per_minute
            }
        
        # Clean old requests outside window
        window_start = current_time - rule.window_size
        while state.requests and state.requests[0] < window_start:
            state.requests.popleft()
        
        # Reset burst allowance if needed
        if current_time - state.last_reset > rule.window_size:
            state.burst_used = 0
            state.last_reset = current_time
        
        # Apply adaptive rate limiting
        effective_limit = rule.requests_per_minute
        if self.adaptive_enabled:
            system_load = await self._check_system_load()
            if system_load > self.system_load_threshold:
                effective_limit = int(effective_limit * self.adaptive_multiplier)
                logger.debug(f"Adaptive rate limiting: reduced limit to {effective_limit} due to high system load")
        
        # Check rate limit
        requests_in_window = len(state.requests)
        
        if requests_in_window >= effective_limit:
            # Check burst allowance
            if state.burst_used >= rule.burst_allowance:
                # Block for remaining window time
                state.blocked_until = current_time + (rule.window_size - (current_time - state.requests[0]))
                state.violation_count += 1
                
                # Progressive penalties for repeat violations
                if state.violation_count > 3:
                    state.blocked_until += rule.window_size * min(state.violation_count - 3, 5)
                
                await self._save_rate_limit_state(key, state, rule)
                
                return False, {
                    "error": "Rate limit exceeded",
                    "retry_after": int(state.blocked_until - current_time),
                    "rule_type": rule.rule_type.value,
                    "limit": effective_limit,
                    "violations": state.violation_count
                }
            else:
                # Use burst allowance
                state.burst_used += 1
        
        # Allow request
        state.requests.append(current_time)
        await self._save_rate_limit_state(key, state, rule)
        
        # Return success with rate limit headers
        remaining = max(0, effective_limit - len(state.requests))
        reset_time = int(current_time + rule.window_size)
        
        return True, {
            "X-RateLimit-Limit": str(effective_limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_time),
            "X-RateLimit-Type": rule.rule_type.value
        }

    async def cleanup_expired_states(self):
        """Cleanup expired rate limit states (for memory-based storage)"""
        if self.redis_client:
            return  # Redis handles TTL automatically
        
        current_time = time.time()
        expired_keys = []
        
        for key, state in self.rate_limit_states.items():
            if current_time - state.last_reset > 3600:  # 1 hour cleanup threshold
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.rate_limit_states[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired rate limit states")


# Global rate limiter instance
enhanced_rate_limiter = EnhancedRateLimiter()


async def enhanced_rate_limit_middleware(request: Request, call_next):
    """Enhanced rate limiting middleware"""
    # Skip rate limiting for health checks
    if request.url.path in ["/health", "/ready", "/metrics"]:
        return await call_next(request)
    
    # Check rate limit
    allowed, metadata = await enhanced_rate_limiter.is_allowed(request)
    
    if not allowed:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content=metadata,
            headers={k: v for k, v in metadata.items() if k.startswith("X-")}
        )
    
    # Process request
    response = await call_next(request)
    
    # Add rate limit headers
    for key, value in metadata.items():
        if key.startswith("X-"):
            response.headers[key] = value
    
    return response
