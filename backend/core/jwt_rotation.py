"""
JWT Token Rotation and Security Management
Provides automatic JWT secret rotation and token management
"""

import os
import time
import asyncio
import logging
import secrets
import hashlib
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from jose import jwt, JWTError
from cryptography.fernet import Fernet
import json
import redis.asyncio as redis

logger = logging.getLogger(__name__)


@dataclass
class JWTSecret:
    key_id: str
    secret: str
    created_at: datetime
    expires_at: datetime
    is_active: bool = True


@dataclass
class TokenBlacklist:
    token_hash: str
    expires_at: datetime
    reason: str


class JWTRotationManager:
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                logger.info("Redis JWT rotation enabled")
            except Exception as e:
                logger.warning(f"Redis connection failed, using in-memory JWT rotation: {e}")
        
        # Configuration
        self.rotation_interval = timedelta(hours=int(os.environ.get("JWT_ROTATION_HOURS", "24")))
        self.secret_overlap = timedelta(hours=int(os.environ.get("JWT_SECRET_OVERLAP_HOURS", "2")))
        self.max_secrets = int(os.environ.get("JWT_MAX_SECRETS", "3"))
        
        # In-memory storage
        self.secrets: Dict[str, JWTSecret] = {}
        self.blacklisted_tokens: Dict[str, TokenBlacklist] = {}
        
        # Encryption for secret storage
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        
        # Initialize with current secret
        self._initialize_secrets()
        
        # Background rotation task
        self.rotation_task = None

    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for secret storage"""
        key_file = "jwt_encryption.key"
        if os.path.exists(key_file):
            with open(key_file, "rb") as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, "wb") as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Restrict permissions
            return key

    def _initialize_secrets(self):
        """Initialize JWT secrets"""
        current_secret = os.environ.get("JWT_SECRET_KEY")
        if not current_secret or len(current_secret) < 32:
            logger.warning("JWT_SECRET_KEY not set or too short, generating new secret")
            current_secret = secrets.token_urlsafe(32)
        
        # Create initial secret
        key_id = self._generate_key_id()
        secret = JWTSecret(
            key_id=key_id,
            secret=current_secret,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + self.rotation_interval + self.secret_overlap,
            is_active=True
        )
        self.secrets[key_id] = secret
        logger.info(f"Initialized JWT secret with key_id: {key_id}")

    def _generate_key_id(self) -> str:
        """Generate unique key ID"""
        return f"jwt_key_{int(time.time())}_{secrets.token_hex(4)}"

    def _hash_token(self, token: str) -> str:
        """Create hash of token for blacklisting"""
        return hashlib.sha256(token.encode()).hexdigest()

    async def _save_secrets_to_redis(self):
        """Save secrets to Redis"""
        if not self.redis_client:
            return
        
        try:
            secrets_data = {}
            for key_id, secret in self.secrets.items():
                encrypted_secret = self.cipher.encrypt(secret.secret.encode()).decode()
                secrets_data[key_id] = {
                    "secret": encrypted_secret,
                    "created_at": secret.created_at.isoformat(),
                    "expires_at": secret.expires_at.isoformat(),
                    "is_active": secret.is_active
                }
            
            await self.redis_client.setex(
                "jwt_secrets",
                int(self.rotation_interval.total_seconds()) * 2,
                json.dumps(secrets_data)
            )
        except Exception as e:
            logger.error(f"Failed to save secrets to Redis: {e}")

    async def _load_secrets_from_redis(self):
        """Load secrets from Redis"""
        if not self.redis_client:
            return
        
        try:
            data = await self.redis_client.get("jwt_secrets")
            if data:
                secrets_data = json.loads(data)
                for key_id, secret_data in secrets_data.items():
                    decrypted_secret = self.cipher.decrypt(secret_data["secret"].encode()).decode()
                    self.secrets[key_id] = JWTSecret(
                        key_id=key_id,
                        secret=decrypted_secret,
                        created_at=datetime.fromisoformat(secret_data["created_at"]),
                        expires_at=datetime.fromisoformat(secret_data["expires_at"]),
                        is_active=secret_data["is_active"]
                    )
        except Exception as e:
            logger.error(f"Failed to load secrets from Redis: {e}")

    async def rotate_secret(self) -> str:
        """Rotate JWT secret and return new key_id"""
        # Generate new secret
        new_secret = secrets.token_urlsafe(32)
        key_id = self._generate_key_id()
        
        # Create new secret object
        secret = JWTSecret(
            key_id=key_id,
            secret=new_secret,
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + self.rotation_interval + self.secret_overlap,
            is_active=True
        )
        
        # Deactivate old secrets but keep them for verification
        for old_secret in self.secrets.values():
            old_secret.is_active = False
        
        # Add new secret
        self.secrets[key_id] = secret
        
        # Clean up expired secrets
        await self._cleanup_expired_secrets()
        
        # Save to Redis
        await self._save_secrets_to_redis()
        
        logger.info(f"JWT secret rotated, new key_id: {key_id}")
        return key_id

    async def _cleanup_expired_secrets(self):
        """Remove expired secrets"""
        current_time = datetime.utcnow()
        expired_keys = []
        
        for key_id, secret in self.secrets.items():
            if current_time > secret.expires_at:
                expired_keys.append(key_id)
        
        for key_id in expired_keys:
            del self.secrets[key_id]
            logger.debug(f"Removed expired secret: {key_id}")
        
        # Ensure we don't have too many secrets
        if len(self.secrets) > self.max_secrets:
            # Keep the most recent secrets
            sorted_secrets = sorted(
                self.secrets.items(),
                key=lambda x: x[1].created_at,
                reverse=True
            )
            
            for key_id, _ in sorted_secrets[self.max_secrets:]:
                del self.secrets[key_id]
                logger.debug(f"Removed excess secret: {key_id}")

    def get_active_secret(self) -> Optional[JWTSecret]:
        """Get the currently active secret"""
        for secret in self.secrets.values():
            if secret.is_active:
                return secret
        return None

    def get_secret_by_key_id(self, key_id: str) -> Optional[JWTSecret]:
        """Get secret by key_id"""
        return self.secrets.get(key_id)

    def create_token(self, payload: Dict, expires_delta: Optional[timedelta] = None) -> Tuple[str, str]:
        """Create JWT token with current active secret"""
        active_secret = self.get_active_secret()
        if not active_secret:
            raise ValueError("No active JWT secret available")
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=60)
        
        # Add metadata to payload
        payload.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "kid": active_secret.key_id  # Key ID for rotation support
        })
        
        token = jwt.encode(payload, active_secret.secret, algorithm="HS256")
        return token, active_secret.key_id

    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token with any valid secret"""
        # Check if token is blacklisted
        token_hash = self._hash_token(token)
        if token_hash in self.blacklisted_tokens:
            blacklist_entry = self.blacklisted_tokens[token_hash]
            if datetime.utcnow() < blacklist_entry.expires_at:
                logger.warning(f"Attempted use of blacklisted token: {blacklist_entry.reason}")
                return None
        
        # Try to decode with header to get key_id
        try:
            unverified_header = jwt.get_unverified_header(token)
            key_id = unverified_header.get("kid")
            
            if key_id and key_id in self.secrets:
                # Use specific secret
                secret = self.secrets[key_id]
                payload = jwt.decode(token, secret.secret, algorithms=["HS256"])
                return payload
        except JWTError:
            pass
        
        # Fallback: try all secrets
        for secret in self.secrets.values():
            try:
                payload = jwt.decode(token, secret.secret, algorithms=["HS256"])
                return payload
            except JWTError:
                continue
        
        return None

    async def blacklist_token(self, token: str, reason: str = "Manual blacklist"):
        """Add token to blacklist"""
        try:
            payload = self.verify_token(token)
            if payload:
                exp = payload.get("exp")
                if exp:
                    expires_at = datetime.fromtimestamp(exp)
                else:
                    expires_at = datetime.utcnow() + timedelta(hours=24)
                
                token_hash = self._hash_token(token)
                self.blacklisted_tokens[token_hash] = TokenBlacklist(
                    token_hash=token_hash,
                    expires_at=expires_at,
                    reason=reason
                )
                
                # Save to Redis
                if self.redis_client:
                    try:
                        await self.redis_client.setex(
                            f"blacklist:{token_hash}",
                            int((expires_at - datetime.utcnow()).total_seconds()),
                            json.dumps({
                                "reason": reason,
                                "expires_at": expires_at.isoformat()
                            })
                        )
                    except Exception as e:
                        logger.error(f"Failed to save blacklist to Redis: {e}")
                
                logger.info(f"Token blacklisted: {reason}")
        except Exception as e:
            logger.error(f"Failed to blacklist token: {e}")

    async def cleanup_blacklist(self):
        """Remove expired blacklisted tokens"""
        current_time = datetime.utcnow()
        expired_hashes = []
        
        for token_hash, blacklist_entry in self.blacklisted_tokens.items():
            if current_time > blacklist_entry.expires_at:
                expired_hashes.append(token_hash)
        
        for token_hash in expired_hashes:
            del self.blacklisted_tokens[token_hash]
        
        if expired_hashes:
            logger.debug(f"Cleaned up {len(expired_hashes)} expired blacklisted tokens")

    async def start_rotation_task(self):
        """Start background rotation task"""
        if self.rotation_task and not self.rotation_task.done():
            return
        
        self.rotation_task = asyncio.create_task(self._rotation_loop())
        logger.info("JWT rotation task started")

    async def stop_rotation_task(self):
        """Stop background rotation task"""
        if self.rotation_task and not self.rotation_task.done():
            self.rotation_task.cancel()
            try:
                await self.rotation_task
            except asyncio.CancelledError:
                pass
        logger.info("JWT rotation task stopped")

    async def _rotation_loop(self):
        """Background rotation loop"""
        while True:
            try:
                # Wait for rotation interval
                await asyncio.sleep(self.rotation_interval.total_seconds())
                
                # Rotate secret
                await self.rotate_secret()
                
                # Cleanup expired items
                await self._cleanup_expired_secrets()
                await self.cleanup_blacklist()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in JWT rotation loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    def get_rotation_status(self) -> Dict:
        """Get rotation status information"""
        active_secret = self.get_active_secret()
        return {
            "active_secret_id": active_secret.key_id if active_secret else None,
            "active_secret_created": active_secret.created_at.isoformat() if active_secret else None,
            "active_secret_expires": active_secret.expires_at.isoformat() if active_secret else None,
            "total_secrets": len(self.secrets),
            "blacklisted_tokens": len(self.blacklisted_tokens),
            "rotation_interval_hours": self.rotation_interval.total_seconds() / 3600,
            "next_rotation_in_seconds": (
                active_secret.created_at + self.rotation_interval - datetime.utcnow()
            ).total_seconds() if active_secret else 0
        }


# Global JWT rotation manager
jwt_rotation_manager = JWTRotationManager()
