"""
Production Configuration for ARIA Backend
Centralizes all production settings and validation.
"""

import os
from functools import lru_cache

# Pydantic v1/v2 compatibility shims
try:
    from pydantic_settings import BaseSettings, SettingsConfigDict  # type: ignore
    _P2 = True
except Exception:  # pragma: no cover
    _P2 = False
    SettingsConfigDict = None  # type: ignore
    try:
        from pydantic import BaseSettings  # type: ignore
    except Exception:  # pragma: no cover
        from pydantic.v1 import BaseSettings  # type: ignore

# Unify validator API: v2 uses field_validator
try:
    from pydantic import field_validator as validator  # type: ignore
except Exception:  # pragma: no cover
    try:
        from pydantic import validator  # type: ignore
    except Exception:  # pragma: no cover
        from pydantic.v1 import validator  # type: ignore

class ProductionSettings(BaseSettings):
    """Production environment settings"""
    
    # MT5 Configuration
    mt5_login: str = ""
    mt5_password: str = ""
    mt5_server: str = ""
    mt5_enabled: bool = False
    
    # Security
    admin_api_key: str = ""
    # Comma-separated list of allowed CORS origins. Should be set via ARIA_CORS_ORIGINS.
    # Leave empty by default; FastAPI app enforces CORS configuration separately.
    allowed_origins: str = ""
    
    # Trading
    auto_trade_enabled: bool = False
    auto_trade_dry_run: bool = True
    aria_enable_exec: bool = False
    allow_live: bool = False
    
    # Rate Limiting
    rate_limit_per_minute: int = 10
    rate_limit_burst: int = 20
    
    # Logging
    log_level: str = "INFO"
    log_dir: str = "./logs"
    
    # Performance
    mt5_pool_size: int = 5
    request_timeout: int = 30

    # Pydantic v2 configuration (ignored by v1)
    try:  # type: ignore
        model_config = SettingsConfigDict(env_prefix="", case_sensitive=False)  # type: ignore
    except Exception:
        pass

    # Pydantic v1 configuration (ignored by v2)
    class Config:  # type: ignore
        env_prefix = ""
        case_sensitive = False
    
    @validator('mt5_login')
    def validate_mt5_login(cls, v):
        if v and not v.isdigit():
            raise ValueError('MT5 login must be numeric')
        return v
    
    @validator('allowed_origins')
    def validate_origins(cls, v):
        # Allow empty string (means not configured here). When provided, validate schemes.
        if not v or not v.strip():
            return ""
        origins = [origin.strip() for origin in v.split(',') if origin.strip()]
        for origin in origins:
            if not origin.startswith(('http://', 'https://')):
                raise ValueError(f'Invalid origin format: {origin}')
        return ",".join(origins)
    
    def is_production_ready(self) -> tuple[bool, list[str]]:
        """Check if configuration is ready for production"""
        issues = []
        
        if self.mt5_enabled and not all([self.mt5_login, self.mt5_password, self.mt5_server]):
            issues.append("MT5 credentials incomplete")
        
        if not self.admin_api_key:
            issues.append("ADMIN_API_KEY not set")
        
        if self.aria_enable_exec and self.auto_trade_dry_run:
            issues.append("Execution enabled but dry run mode active")
        
        if self.auto_trade_enabled and not self.allow_live and not self.auto_trade_dry_run:
            issues.append("Live trading requested but not allowed")
        
        return len(issues) == 0, issues

@lru_cache()
def get_settings() -> ProductionSettings:
    """Get cached production settings"""
    return ProductionSettings(
        mt5_login=os.getenv("MT5_LOGIN", ""),
        mt5_password=os.getenv("MT5_PASSWORD", ""),
        mt5_server=os.getenv("MT5_SERVER", ""),
        mt5_enabled=os.getenv("ARIA_ENABLE_MT5", "0") == "1",
        admin_api_key=os.getenv("ADMIN_API_KEY", ""),
        # Source from ARIA_CORS_ORIGINS to align with FastAPI CORS enforcement.
        allowed_origins=os.getenv("ARIA_CORS_ORIGINS", ""),
        auto_trade_enabled=os.getenv("AUTO_TRADE_ENABLED", "0") == "1",
        auto_trade_dry_run=os.getenv("AUTO_TRADE_DRY_RUN", "1") == "1",
        aria_enable_exec=os.getenv("ARIA_ENABLE_EXEC", "0") == "1",
        allow_live=os.getenv("ALLOW_LIVE", "0") == "1",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_dir=os.getenv("ARIA_LOG_DIR", "./logs"),
    )

def validate_production_config() -> None:
    """Validate production configuration and log warnings"""
    import logging
    logger = logging.getLogger(__name__)
    
    settings = get_settings()
    is_ready, issues = settings.is_production_ready()
    
    if not is_ready:
        logger.warning("Production configuration issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("✅ Production configuration validated successfully")
