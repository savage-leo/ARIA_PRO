"""
Authentication endpoints for ARIA Pro
JWT token management with refresh token support
"""

from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional
import logging

from backend.core.auth import (
    authenticate_user, 
    create_tokens_for_user, 
    refresh_access_token,
    verify_refresh_token,
    get_current_active_user,
    User,
    Token
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["Authentication"])

class RefreshTokenRequest(BaseModel):
    refresh_token: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int = 3600

@router.post("/token", response_model=TokenResponse)
async def login_for_access_token(
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends()
):
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    tokens = create_tokens_for_user(user)
    logger.info(f"User {user.username} authenticated successfully")
    
    # Set secure HTTP-only cookies for tokens
    response.set_cookie(
        key="refresh_token",
        value=tokens.refresh_token,
        httponly=True,
        secure=True,  # HTTPS only
        samesite="strict",
        max_age=7 * 24 * 3600  # 7 days
    )
    
    return TokenResponse(
        access_token=tokens.access_token,
        refresh_token=tokens.refresh_token,
        token_type="bearer",
        expires_in=3600
    )

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshTokenRequest):
    """
    Refresh access token using refresh token
    """
    try:
        new_tokens = refresh_access_token(request.refresh_token)
        if not new_tokens:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Verify the refresh token to get user info for logging
        payload = verify_refresh_token(request.refresh_token)
        username = payload.get("sub") if payload else "unknown"
        logger.info(f"Access token refreshed for user: {username}")
        
        return TokenResponse(
            access_token=new_tokens.access_token,
            refresh_token=new_tokens.refresh_token,
            token_type="bearer",
            expires_in=3600
        )
        
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    Get current user information
    """
    return current_user

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_active_user)):
    """
    Logout user (in a full implementation, this would invalidate tokens)
    """
    logger.info(f"User {current_user.username} logged out")
    return {"message": "Successfully logged out"}

@router.get("/status")
async def auth_status():
    """
    Get authentication system status
    """
    from backend.core.config import get_settings
    settings = get_settings()
    
    return {
        "jwt_enabled": getattr(settings, 'JWT_ENABLED', False),
        "algorithm": getattr(settings, 'JWT_ALGORITHM', 'HS256'),
        "access_token_expire_minutes": getattr(settings, 'jwt_access_expire', 60).total_seconds() / 60,
        "refresh_token_expire_days": getattr(settings, 'jwt_refresh_expire', 7).days if hasattr(getattr(settings, 'jwt_refresh_expire', None), 'days') else 7
    }
