"""
Test suite for middleware components.
"""
import pytest
import time
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from backend.middleware.rate_limit import RateLimitMiddleware
from backend.middleware.live_guard import LiveGuardMiddleware
from backend.middleware.error_boundary import ErrorBoundaryMiddleware

# Test Rate Limiting
def test_rate_limit_middleware():
    """Test rate limiting middleware."""
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, default_limit=5, window_seconds=60)
    
    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}
    
    client = TestClient(app)
    
    # Should allow first 5 requests
    for i in range(5):
        response = client.get("/test")
        assert response.status_code == 200
    
    # 6th request should be rate limited
    response = client.get("/test")
    assert response.status_code == 429
    assert "rate limit exceeded" in response.json()["detail"].lower()

def test_rate_limit_headers():
    """Test rate limit headers are added."""
    app = FastAPI()
    app.add_middleware(RateLimitMiddleware, default_limit=10, window_seconds=60)
    
    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}
    
    client = TestClient(app)
    response = client.get("/test")
    
    assert response.status_code == 200
    assert "X-RateLimit-Limit" in response.headers
    assert "X-RateLimit-Remaining" in response.headers
    assert "X-RateLimit-Reset" in response.headers

# Test Live Guard
def test_live_guard_blocks_in_dev():
    """Test live guard blocks trading endpoints in development."""
    app = FastAPI()
    app.add_middleware(
        LiveGuardMiddleware,
        environment="development",
        auto_trade_enabled=False
    )
    
    @app.post("/trading/execute")
    async def execute_trade():
        return {"status": "executed"}
    
    client = TestClient(app)
    response = client.post("/trading/execute", json={"symbol": "EURUSD", "action": "buy"})
    
    assert response.status_code == 403
    assert "not allowed in development" in response.json()["detail"].lower()

def test_live_guard_maintenance_mode():
    """Test live guard maintenance mode."""
    app = FastAPI()
    middleware = LiveGuardMiddleware(
        environment="production",
        auto_trade_enabled=True
    )
    app.add_middleware(lambda app: middleware)
    
    @app.post("/trading/execute")
    async def execute_trade():
        return {"status": "executed"}
    
    # Enable maintenance mode
    middleware.maintenance_mode = True
    
    client = TestClient(app)
    response = client.post("/trading/execute", json={"symbol": "EURUSD", "action": "buy"})
    
    assert response.status_code == 503
    assert "maintenance" in response.json()["detail"].lower()

def test_live_guard_kill_switch():
    """Test live guard kill switch."""
    app = FastAPI()
    middleware = LiveGuardMiddleware(
        environment="production",
        auto_trade_enabled=True
    )
    app.add_middleware(lambda app: middleware)
    
    @app.post("/trading/execute")
    async def execute_trade():
        return {"status": "executed"}
    
    # Engage kill switch
    middleware.engage_kill_switch("Test emergency")
    
    client = TestClient(app)
    response = client.post("/trading/execute", json={"symbol": "EURUSD", "action": "buy"})
    
    assert response.status_code == 503
    assert "kill switch engaged" in response.json()["detail"].lower()

# Test Error Boundary
def test_error_boundary_catches_exceptions():
    """Test error boundary catches unhandled exceptions."""
    app = FastAPI()
    app.add_middleware(ErrorBoundaryMiddleware, debug=False)
    
    @app.get("/test")
    async def test_endpoint():
        raise ValueError("Test error")
    
    client = TestClient(app)
    response = client.get("/test")
    
    assert response.status_code == 500
    assert "internal server error" in response.json()["detail"].lower()
    # In non-debug mode, should not expose the actual error
    assert "Test error" not in response.json()["detail"]

def test_error_boundary_debug_mode():
    """Test error boundary in debug mode shows details."""
    app = FastAPI()
    app.add_middleware(ErrorBoundaryMiddleware, debug=True)
    
    @app.get("/test")
    async def test_endpoint():
        raise ValueError("Test error details")
    
    client = TestClient(app)
    response = client.get("/test")
    
    assert response.status_code == 500
    # In debug mode, should expose error details
    assert "debug_info" in response.json()
    assert "Test error details" in str(response.json()["debug_info"])

# Integration tests
def test_middleware_stack_integration():
    """Test full middleware stack integration."""
    app = FastAPI()
    
    # Add all middleware in correct order
    app.add_middleware(ErrorBoundaryMiddleware, debug=False)
    app.add_middleware(RateLimitMiddleware, default_limit=10, window_seconds=60)
    app.add_middleware(
        LiveGuardMiddleware,
        environment="production",
        auto_trade_enabled=True
    )
    
    @app.get("/health")
    async def health():
        return {"status": "healthy"}
    
    @app.post("/trading/execute")
    async def execute_trade():
        return {"status": "executed"}
    
    client = TestClient(app)
    
    # Health endpoint should work
    response = client.get("/health")
    assert response.status_code == 200
    
    # Trading endpoint should work with proper request
    response = client.post("/trading/execute", json={"symbol": "EURUSD", "action": "buy"})
    assert response.status_code in [200, 403]  # May be blocked by live guard rules
    
    # Test rate limiting
    for _ in range(15):
        client.get("/health")
    
    response = client.get("/health")
    assert response.status_code == 429  # Should be rate limited
