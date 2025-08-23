"""
JWT Authentication and Authorization for ARIA Pro
Production-grade authentication with role-based access control
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum
import jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import logging

from backend.core.config import get_settings

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)

# User roles
class UserRole(str, Enum):
    ADMIN = "admin"
    TRADER = "trader"
    VIEWER = "viewer"
    API = "api"

# Token models
class Token(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"

class TokenData(BaseModel):
    username: Optional[str] = None
    roles: List[str] = []
    scopes: List[str] = []

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: bool = False
    roles: List[UserRole] = [UserRole.VIEWER]
    hashed_password: str

class UserInDB(User):
    hashed_password: str

# Mock user database for production (replace with real DB)
fake_users_db = {
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "email": "admin@aria.pro",
        "hashed_password": pwd_context.hash("changeme_admin_2024"),
        "disabled": False,
        "roles": [UserRole.ADMIN, UserRole.TRADER]
    },
    "trader": {
        "username": "trader",
        "full_name": "Trader User",
        "email": "trader@aria.pro",
        "hashed_password": pwd_context.hash("changeme_trader_2024"),
        "disabled": False,
        "roles": [UserRole.TRADER]
    }
}

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def get_user(username: str) -> Optional[UserInDB]:
    """Get user from database"""
    if username in fake_users_db:
        user_dict = fake_users_db[username]
        return UserInDB(**user_dict)
    return None

def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Authenticate user with username and password"""
    user = get_user(username)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    settings = get_settings()
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + settings.jwt_access_expire
    
    to_encode.update({"exp": expire, "type": "access"})
    
    # Use secret key from settings or generate warning
    secret_key = settings.JWT_SECRET_KEY
    if not secret_key:
        logger.warning("JWT_SECRET_KEY not configured, using default (INSECURE)")
        secret_key = "CHANGE_THIS_SECRET_KEY_IN_PRODUCTION_ARIA_2024"
    
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create JWT refresh token"""
    settings = get_settings()
    to_encode = data.copy()
    expire = datetime.utcnow() + settings.jwt_refresh_expire
    to_encode.update({"exp": expire, "type": "refresh"})
    
    secret_key = settings.JWT_SECRET_KEY or "CHANGE_THIS_SECRET_KEY_IN_PRODUCTION_ARIA_2024"
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)) -> Optional[User]:
    """Get current user from JWT token"""
    settings = get_settings()
    
    # If JWT is not enabled, return a default admin user (dev mode)
    if not settings.JWT_ENABLED:
        return User(
            username="dev_admin",
            full_name="Development Admin",
            disabled=False,
            roles=[UserRole.ADMIN, UserRole.TRADER],
            hashed_password=""
        )
    
    if not token:
        return None
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        secret_key = settings.JWT_SECRET_KEY or "CHANGE_THIS_SECRET_KEY_IN_PRODUCTION_ARIA_2024"
        payload = jwt.decode(token, secret_key, algorithms=[settings.JWT_ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(
            username=username,
            roles=payload.get("roles", []),
            scopes=payload.get("scopes", [])
        )
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Ensure user is active"""
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Role-based access control dependencies
def require_role(required_roles: List[UserRole]):
    """Create a dependency that requires specific roles"""
    async def role_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if not any(role in current_user.roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {required_roles}"
            )
        return current_user
    return role_checker

# Convenience dependencies
require_admin = require_role([UserRole.ADMIN])
require_trader = require_role([UserRole.ADMIN, UserRole.TRADER])
require_viewer = require_role([UserRole.ADMIN, UserRole.TRADER, UserRole.VIEWER])

# API key authentication (alternative to JWT)
async def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Validate API key from header or query param"""
    settings = get_settings()
    if not api_key:
        return None
    
    # Check against admin API key
    if api_key == settings.ADMIN_API_KEY and settings.ADMIN_API_KEY:
        return api_key
    
    return None

def create_tokens_for_user(user: UserInDB) -> Token:
    """Create both access and refresh tokens for a user"""
    access_token_data = {
        "sub": user.username,
        "roles": [role.value for role in user.roles],
        "scopes": []
    }
    
    access_token = create_access_token(access_token_data)
    refresh_token = create_refresh_token(access_token_data)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token
    )

def verify_refresh_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify and decode refresh token"""
    settings = get_settings()
    
    try:
        secret_key = settings.JWT_SECRET_KEY or "CHANGE_THIS_SECRET_KEY_IN_PRODUCTION_ARIA_2024"
        payload = jwt.decode(token, secret_key, algorithms=[settings.JWT_ALGORITHM])
        
        # Verify token type
        if payload.get("type") != "refresh":
            return None
        
        return payload
    except jwt.PyJWTError:
        return None

def refresh_access_token(refresh_token: str) -> Optional[Token]:
    """Generate new access token from refresh token"""
    payload = verify_refresh_token(refresh_token)
    if not payload:
        return None
    
    username = payload.get("sub")
    if not username:
        return None
    
    user = get_user(username)
    if not user or user.disabled:
        return None
    
    # Create new access token with same data
    access_token_data = {
        "sub": user.username,
        "roles": [role.value for role in user.roles],
        "scopes": []
    }
    
    new_access_token = create_access_token(access_token_data)
    
    return Token(
        access_token=new_access_token,
        refresh_token=refresh_token,  # Keep same refresh token
        token_type="bearer"
    )
