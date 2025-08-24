"""
Centralized configuration management with Pydantic validation.
"""
import os
from typing import List, Optional, Dict, Any
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import Field, validator
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with validation and type checking."""
    
    # Environment
    ARIA_ENV: str = Field(default="development", env="ARIA_ENV")
    DEBUG_MODE: bool = Field(default=False, env="DEBUG_MODE")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    ARIA_LOG_DIR: str = Field(default="logs", env="ARIA_LOG_DIR")
    
    # Server
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    WORKERS: int = Field(default=1, env="WORKERS")
    RELOAD: bool = Field(default=False, env="RELOAD")
    
    # Security
    JWT_SECRET_KEY: str = Field(default="", env="JWT_SECRET_KEY")
    JWT_ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, env="REFRESH_TOKEN_EXPIRE_DAYS")
    ADMIN_API_KEY: Optional[str] = Field(default=None, env="ADMIN_API_KEY")
    
    @validator("ADMIN_API_KEY")
    def admin_key_must_be_set_in_production(cls, v, values):
        """Ensure admin API key is set in production."""
        if values.get("ARIA_ENV") == "production" and not v:
            raise ValueError("ADMIN_API_KEY must be set in production")
        if values.get("ARIA_ENV") == "production" and len(v or "") < 16:
            raise ValueError("ADMIN_API_KEY must be at least 16 characters in production")
        return v
    SECURE_COOKIES: bool = Field(default=False, env="SECURE_COOKIES")
    
    # CORS & Security Headers
    ARIA_CORS_ORIGINS: str = Field(default="", env="ARIA_CORS_ORIGINS")
    ARIA_ALLOWED_HOSTS: str = Field(default="", env="ARIA_ALLOWED_HOSTS")
    ARIA_CSP_CONNECT_SRC: str = Field(default="", env="ARIA_CSP_CONNECT_SRC")
    
    # Database
    DATABASE_URL: str = Field(default="sqlite:///./data/aria.db", env="DATABASE_URL")
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # MT5 Configuration
    ARIA_ENABLE_MT5: str = Field(default="0", env="ARIA_ENABLE_MT5")
    MT5_LOGIN: Optional[int] = Field(default=None, env="MT5_LOGIN")
    MT5_PASSWORD: str = Field(default="", env="MT5_PASSWORD")
    MT5_SERVER: str = Field(default="", env="MT5_SERVER")
    MT5_TIMEOUT: int = Field(default=60000, env="MT5_TIMEOUT")
    
    # Trading Configuration - SECURE DEFAULTS
    AUTO_TRADE_ENABLED: bool = Field(default=False, env="AUTO_TRADE_ENABLED")
    AUTO_TRADE_DRY_RUN: bool = Field(default=True, env="AUTO_TRADE_DRY_RUN")
    ARIA_ENABLE_EXEC: bool = Field(default=False, env="ARIA_ENABLE_EXEC")
    AUTO_EXEC_ENABLED: bool = Field(default=False, env="AUTO_EXEC_ENABLED")
    ALLOW_LIVE: bool = Field(default=False, env="ALLOW_LIVE")
    
    # Trading Limits
    MAX_DAILY_TRADES: int = Field(default=100, env="MAX_DAILY_TRADES")
    MAX_POSITION_SIZE: float = Field(default=0.1, env="MAX_POSITION_SIZE")
    MAX_DAILY_LOSS: float = Field(default=1000.0, env="MAX_DAILY_LOSS")
    MAX_OPEN_POSITIONS: int = Field(default=10, env="MAX_OPEN_POSITIONS")
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(default=100, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    RATE_LIMIT_BURST_SIZE: int = Field(default=10, env="RATE_LIMIT_BURST_SIZE")
    
    # AI Models
    ARIA_ENABLE_AI: bool = Field(default=True, env="ARIA_ENABLE_AI")
    AI_MODEL_PATH: str = Field(default="models", env="AI_MODEL_PATH")
    AI_INFERENCE_TIMEOUT: int = Field(default=30, env="AI_INFERENCE_TIMEOUT")
    
    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = Field(default=30, env="WS_HEARTBEAT_INTERVAL")
    WS_MAX_CONNECTIONS: int = Field(default=100, env="WS_MAX_CONNECTIONS")
    WS_MESSAGE_QUEUE_SIZE: int = Field(default=1000, env="WS_MESSAGE_QUEUE_SIZE")
    
    # Monitoring
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    ENABLE_TELEMETRY: bool = Field(default=True, env="ENABLE_TELEMETRY")
    
    # External Services
    ALPHA_VANTAGE_API_KEY: str = Field(default="", env="ALPHA_VANTAGE_API_KEY")
    FINNHUB_API_KEY: str = Field(default="", env="FINNHUB_API_KEY")
    NEWS_API_KEY: str = Field(default="", env="NEWS_API_KEY")
    
    # Application
    APP_NAME: str = Field(default="ARIA Institutional Pro", env="APP_NAME")
    APP_VERSION: str = Field(default="1.2.0", env="APP_VERSION")
    
    @validator("JWT_SECRET_KEY")
    def jwt_secret_must_be_set(cls, v, values):
        """Ensure JWT secret is set in production."""
        if values.get("ARIA_ENV") == "production" and not v:
            raise ValueError("JWT_SECRET_KEY must be set in production")
        if values.get("ARIA_ENV") == "production" and len(v) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters in production")
        if not v:
            # Generate a default for development
            return "dev-secret-key-change-in-production-minimum-32-chars"
        return v
    
    @validator("ARIA_CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        """Ensure CORS origins is a string."""
        if isinstance(v, list):
            return ",".join(v)
        return v
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins into a list."""
        if not self.ARIA_CORS_ORIGINS:
            return []
        return [origin.strip() for origin in self.ARIA_CORS_ORIGINS.split(",") if origin.strip()]
    
    @property
    def allowed_hosts_list(self) -> List[str]:
        """Parse allowed hosts into a list."""
        if not self.ARIA_ALLOWED_HOSTS:
            return []
        return [host.strip() for host in self.ARIA_ALLOWED_HOSTS.split(",") if host.strip()]
    
    @property
    def csp_connect_src_extra(self) -> List[str]:
        """Parse extra CSP connect sources."""
        if not self.ARIA_CSP_CONNECT_SRC:
            return []
        # Support both space and comma separation
        delimiters = [",", " "]
        tokens = []
        for delim in delimiters:
            if delim in self.ARIA_CSP_CONNECT_SRC:
                tokens = self.ARIA_CSP_CONNECT_SRC.split(delim)
                break
        if not tokens:
            tokens = [self.ARIA_CSP_CONNECT_SRC]
        return [t.strip() for t in tokens if t.strip()]
    
    @property
    def mt5_enabled(self) -> bool:
        """Check if MT5 is enabled."""
        return self.ARIA_ENABLE_MT5 in ("1", "true", "True", "TRUE", "yes", "Yes", "YES")
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.ARIA_ENV.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.ARIA_ENV.lower() in ("development", "dev")
    
    def validate_trading_config(self) -> Dict[str, Any]:
        """Validate trading configuration consistency."""
        issues = []
        
        if self.AUTO_TRADE_ENABLED and not self.mt5_enabled:
            issues.append("AUTO_TRADE_ENABLED requires MT5 to be enabled")
        
        if self.AUTO_TRADE_ENABLED and not self.AUTO_TRADE_DRY_RUN:
            if not self.ARIA_ENABLE_EXEC:
                issues.append("Live trading requires ARIA_ENABLE_EXEC=1")
            if not self.AUTO_EXEC_ENABLED:
                issues.append("Live trading requires AUTO_EXEC_ENABLED=1")
            if not self.ALLOW_LIVE:
                issues.append("Live trading requires ALLOW_LIVE=1")
        
        if self.mt5_enabled:
            if not self.MT5_LOGIN:
                issues.append("MT5_LOGIN is required when MT5 is enabled")
            if not self.MT5_PASSWORD:
                issues.append("MT5_PASSWORD is required when MT5 is enabled")
            if not self.MT5_SERVER:
                issues.append("MT5_SERVER is required when MT5 is enabled")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary, excluding sensitive values."""
        data = self.dict()
        # Redact sensitive values
        sensitive_keys = [
            "JWT_SECRET_KEY", "MT5_PASSWORD", "ADMIN_API_KEY",
            "ALPHA_VANTAGE_API_KEY", "FINNHUB_API_KEY", "NEWS_API_KEY",
            "DATABASE_URL", "REDIS_URL"
        ]
        for key in sensitive_keys:
            if key in data and data[key]:
                data[key] = "***REDACTED***"
        return data
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

def reload_settings() -> Settings:
    """Force reload settings (clears cache)."""
    get_settings.cache_clear()
    return get_settings()

# Export commonly used settings
settings = get_settings()
