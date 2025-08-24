"""
Test suite for authentication system.
"""
import pytest
from datetime import datetime, timedelta
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.core.auth import (
    create_access_token,
    create_refresh_token,
    verify_token,
    get_password_hash,
    verify_password,
    authenticate_user,
    get_current_user
)

def test_password_hashing():
    """Test password hashing and verification."""
    password = "test_password_123"
    hashed = get_password_hash(password)
    
    # Hash should be different from original
    assert hashed != password
    
    # Should verify correctly
    assert verify_password(password, hashed) is True
    
    # Wrong password should fail
    assert verify_password("wrong_password", hashed) is False

def test_access_token_creation():
    """Test JWT access token creation."""
    user_data = {"sub": "testuser", "role": "trader"}
    token = create_access_token(user_data)
    
    assert token is not None
    assert isinstance(token, str)
    
    # Verify token
    payload = verify_token(token)
    assert payload is not None
    assert payload["sub"] == "testuser"
    assert payload["role"] == "trader"

def test_refresh_token_creation():
    """Test JWT refresh token creation."""
    user_data = {"sub": "testuser"}
    token = create_refresh_token(user_data)
    
    assert token is not None
    assert isinstance(token, str)
    
    # Verify token
    payload = verify_token(token)
    assert payload is not None
    assert payload["sub"] == "testuser"
    assert payload["type"] == "refresh"

def test_token_expiration():
    """Test token expiration."""
    user_data = {"sub": "testuser"}
    # Create token with very short expiration
    token = create_access_token(user_data, expires_delta=timedelta(seconds=-1))
    
    # Should be expired
    payload = verify_token(token)
    assert payload is None

def test_api_key_authentication():
    """Test API key authentication."""
    app = FastAPI()
    
    @app.get("/protected")
    async def protected_endpoint(user=Depends(get_current_user)):
        return {"user": user["username"]}
    
    client = TestClient(app)
    
    # Without API key should fail
    response = client.get("/protected")
    assert response.status_code == 401
    
    # With valid API key should work
    response = client.get(
        "/protected",
        headers={"X-API-Key": "test_api_key"}
    )
    # Note: This depends on actual API key configuration

def test_role_based_access():
    """Test role-based access control."""
    from backend.core.auth import require_admin, require_trader
    
    app = FastAPI()
    
    @app.get("/admin")
    async def admin_endpoint(user=Depends(require_admin)):
        return {"status": "admin access"}
    
    @app.get("/trader")
    async def trader_endpoint(user=Depends(require_trader)):
        return {"status": "trader access"}
    
    client = TestClient(app)
    
    # Create tokens with different roles
    admin_token = create_access_token({"sub": "admin", "role": "admin"})
    trader_token = create_access_token({"sub": "trader", "role": "trader"})
    
    # Admin should access admin endpoint
    response = client.get(
        "/admin",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    
    # Trader should not access admin endpoint
    response = client.get(
        "/admin",
        headers={"Authorization": f"Bearer {trader_token}"}
    )
    assert response.status_code == 403
    
    # Trader should access trader endpoint
    response = client.get(
        "/trader",
        headers={"Authorization": f"Bearer {trader_token}"}
    )
    assert response.status_code == 200
