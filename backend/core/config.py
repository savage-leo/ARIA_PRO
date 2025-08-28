# Centralized configuration using Pydantic Settings (v1 or v2 compatible)
# - Loads env from ARIA_PRO/production.env or .env once
# - Exposes typed settings and convenience accessors

from __future__ import annotations

import os
import logging
from typing import List, Optional
from datetime import timedelta

logger = logging.getLogger(__name__)

# Try Pydantic v2 (pydantic-settings) first, then fallback to v1 BaseSettings
try:
    from pydantic import BaseSettings as _BaseSettings, Field, field_validator  # type: ignore
    _V2 = False
except Exception:
    try:
        from pydantic_settings import BaseSettings as _BaseSettings  # type: ignore
        from pydantic import Field, field_validator  # type: ignore
        _V2 = True
    except Exception:
        _BaseSettings = None  # type: ignore
        Field = None  # type: ignore
        field_validator = None  # type: ignore
        _V2 = False


def _load_env_file(path: str) -> int:
    count = 0
    try:
        if not os.path.exists(path):
            return 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip()
                if k and (k not in os.environ or os.environ.get(k, "") == ""):
                    os.environ[k] = v
                    count += 1
    except Exception:
        return count
    return count


_ENV_BOOTSTRAPPED = False

def _bootstrap_env_from_files_once() -> None:
    global _ENV_BOOTSTRAPPED
    if _ENV_BOOTSTRAPPED:
        return
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))  # .../backend
        project_dir = os.path.dirname(base_dir)  # .../ARIA_PRO
        # Determine environment before loading files so we can pick the right source
        pre_env = (os.environ.get("ARIA_ENV", "") or "").strip().lower()
        dev_like = pre_env in ("dev", "development", "local", "test", "testing", "ci")
        if dev_like:
            candidates = [os.path.join(project_dir, ".env")]
        else:
            # In production, prefer production.env and then .env to fill missing values
            candidates = [
                os.path.join(project_dir, "production.env"),
                os.path.join(project_dir, ".env"),
            ]
        for candidate in candidates:
            if os.path.exists(candidate):
                loaded = _load_env_file(candidate)
                try:
                    logger.debug(f"Loaded {loaded} env vars from {os.path.basename(candidate)}")
                except Exception:
                    pass
    except Exception:
        pass
    # Backward-compatibility: map legacy monitoring env names to PROMETHEUS_*
    # This lets existing setups with ENABLE_METRICS/METRICS_PORT work seamlessly.
    try:
        if "PROMETHEUS_PORT" not in os.environ and os.environ.get("METRICS_PORT"):
            os.environ["PROMETHEUS_PORT"] = os.environ["METRICS_PORT"]
        if "PROMETHEUS_ENABLED" not in os.environ and os.environ.get("ENABLE_METRICS"):
            os.environ["PROMETHEUS_ENABLED"] = os.environ["ENABLE_METRICS"]
        if (
            "PROMETHEUS_PUSH_GATEWAY" not in os.environ
            and os.environ.get("PUSH_GATEWAY_URL")
        ):
            os.environ["PROMETHEUS_PUSH_GATEWAY"] = os.environ["PUSH_GATEWAY_URL"]
    except Exception:
        # Never fail bootstrap due to env mapping
        pass
    _ENV_BOOTSTRAPPED = True


if _BaseSettings is not None:

    # INTENTIONAL_DEFAULT_TAG (2025-08-19):
    # MT5 live execution is enabled by default per USER request.
    # The following defaults are intentionally set to True:
    #   - ARIA_ENABLE_MT5
    #   - ARIA_ENABLE_EXEC
    #   - AUTO_EXEC_ENABLED
    #   - ALLOW_LIVE
    # This tag documents the deliberate nature of these defaults for future reviewers.

    class Settings(_BaseSettings):  # type: ignore
        # Logging
        LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
        ARIA_LOG_DIR: str = Field(default="./logs", env="ARIA_LOG_DIR")

        # Core security and CORS
        ARIA_ENV: str = Field(default="production", env="ARIA_ENV")
        ARIA_CORS_ORIGINS: str = Field(default="", env="ARIA_CORS_ORIGINS")
        ARIA_ALLOWED_HOSTS: str = Field(default="", env="ARIA_ALLOWED_HOSTS")
        ARIA_CSP_CONNECT_SRC: str = Field(default="", env="ARIA_CSP_CONNECT_SRC")

        # MT5 / Trading flags - SECURE DEFAULTS
        ARIA_ENABLE_MT5: bool = Field(default=False, env="ARIA_ENABLE_MT5")
        MT5_LOGIN: Optional[int] = Field(default=None, env="MT5_LOGIN")
        MT5_PASSWORD: Optional[str] = Field(default=None, env="MT5_PASSWORD")
        MT5_SERVER: Optional[str] = Field(default=None, env="MT5_SERVER")

        # Data/AI config
        ARIA_SYMBOLS: str = Field(default="", env="ARIA_SYMBOLS")
        AUTO_TRADE_SYMBOLS: str = Field(default="", env="AUTO_TRADE_SYMBOLS")
        ARIA_MT5_BAR_SECONDS: Optional[int] = Field(default=None, env="ARIA_MT5_BAR_SECONDS")
        ARIA_FEED_BAR_SECONDS: Optional[int] = Field(default=None, env="ARIA_FEED_BAR_SECONDS")
        ARIA_INCLUDE_XGB: bool = Field(default=True, env="ARIA_INCLUDE_XGB")

        # Auto-trade + execution - SECURE DEFAULTS
        AUTO_TRADE_ENABLED: bool = Field(default=False, env="AUTO_TRADE_ENABLED")
        AUTO_TRADE_DRY_RUN: bool = Field(default=True, env="AUTO_TRADE_DRY_RUN")
        AUTO_TRADE_PRIMARY_MODEL: str = Field(default="xgb", env="AUTO_TRADE_PRIMARY_MODEL")
        ARIA_ENABLE_EXEC: bool = Field(default=False, env="ARIA_ENABLE_EXEC")
        AUTO_EXEC_ENABLED: bool = Field(default=False, env="AUTO_EXEC_ENABLED")
        ALLOW_LIVE: bool = Field(default=False, env="ALLOW_LIVE")

        # Quick profit strategy toggles
        ENABLE_QUICK_PROFIT: bool = Field(default=False, env="ENABLE_QUICK_PROFIT")
        ENABLE_ARBITRAGE: bool = Field(default=False, env="ENABLE_ARBITRAGE")
        ENABLE_NEWS_TRADING: bool = Field(default=False, env="ENABLE_NEWS_TRADING")

        # Admin/API - REQUIRED for production
        ADMIN_API_KEY: str = Field(..., env="ADMIN_API_KEY", min_length=16)
        ARIA_ENABLE_CPP_SMC: str = Field(default="auto", env="ARIA_ENABLE_CPP_SMC")

        # WebSocket - Token required for production
        ARIA_WS_HEARTBEAT_SEC: int = Field(default=20, env="ARIA_WS_HEARTBEAT_SEC")
        ARIA_WS_TOKEN: str = Field(..., env="ARIA_WS_TOKEN", min_length=16)

        # LLM Monitor / Tuning
        LLM_MONITOR_ENABLED: bool = Field(default=False, env="LLM_MONITOR_ENABLED")
        LLM_MONITOR_INTERVAL_SEC: int = Field(default=60, env="LLM_MONITOR_INTERVAL_SEC")
        LLM_MONITOR_DAN_URL: str = Field(default="http://127.0.0.1:8101", env="LLM_MONITOR_DAN_URL")
        LLM_TUNING_ENABLED: bool = Field(default=False, env="LLM_TUNING_ENABLED")
        LLM_TUNING_MAX_REL_DELTA: float = Field(default=0.2, env="LLM_TUNING_MAX_REL_DELTA")
        LLM_TUNING_COOLDOWN_SEC: int = Field(default=300, env="LLM_TUNING_COOLDOWN_SEC")
        
        # JWT Authentication settings - REQUIRED for production
        JWT_SECRET_KEY: str = Field(default="", env="JWT_SECRET_KEY")
        JWT_ALGORITHM: str = Field(default="HS256", env="JWT_ALGORITHM")
        JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, env="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
        JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(default=7, env="JWT_REFRESH_TOKEN_EXPIRE_DAYS")
        JWT_ENABLED: bool = Field(default=True, env="JWT_ENABLED")
        
        # Rate limiting - Production defaults
        RATE_LIMIT_ENABLED: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
        RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(default=200, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
        RATE_LIMIT_BURST: int = Field(default=50, env="RATE_LIMIT_BURST")
        RATE_LIMIT_ADMIN_PER_MINUTE: int = Field(default=60, env="RATE_LIMIT_ADMIN_PER_MINUTE")
        RATE_LIMIT_TRADING_PER_MINUTE: int = Field(default=30, env="RATE_LIMIT_TRADING_PER_MINUTE")
        
        # Prometheus / Monitoring
        PROMETHEUS_ENABLED: bool = Field(default=True, env="PROMETHEUS_ENABLED")
        PROMETHEUS_PORT: int = Field(default=8001, env="PROMETHEUS_PORT")
        PROMETHEUS_PUSH_GATEWAY: Optional[str] = Field(default=None, env="PROMETHEUS_PUSH_GATEWAY")
        PROMETHEUS_JOB_NAME: str = Field(default="aria_pro", env="PROMETHEUS_JOB_NAME")
        PROMETHEUS_INSTANCE: str = Field(default="aria_backend", env="PROMETHEUS_INSTANCE")
        
        # Auto-trade thresholds
        AUTO_TRADE_PROB_THRESHOLD: float = Field(default=0.75, env="AUTO_TRADE_PROB_THRESHOLD")
        AUTO_TRADE_STOP_LOSS_ATR: float = Field(default=1.5, env="AUTO_TRADE_STOP_LOSS_ATR")
        AUTO_TRADE_TAKE_PROFIT_ATR: float = Field(default=3.0, env="AUTO_TRADE_TAKE_PROFIT_ATR")
        AUTO_TRADE_MAX_RISK_PERCENT: float = Field(default=0.5, env="AUTO_TRADE_MAX_RISK_PERCENT")
        
        # Validators for production safety
        @field_validator('AUTO_TRADE_PROB_THRESHOLD')
        @classmethod
        def validate_prob_threshold(cls, v):
            if not 0 <= v <= 1:
                raise ValueError('AUTO_TRADE_PROB_THRESHOLD must be between 0 and 1')
            return v
        
        @field_validator('AUTO_TRADE_STOP_LOSS_ATR', 'AUTO_TRADE_TAKE_PROFIT_ATR')
        @classmethod
        def validate_atr_multipliers(cls, v):
            if not 0.1 <= v <= 10:
                raise ValueError('ATR multipliers must be between 0.1 and 10')
            return v
        
        @field_validator('AUTO_TRADE_MAX_RISK_PERCENT')
        @classmethod
        def validate_max_risk(cls, v):
            if not 0.01 <= v <= 5:
                raise ValueError('MAX_RISK_PERCENT must be between 0.01 and 5')
            return v
        
        @field_validator('LLM_TUNING_MAX_REL_DELTA')
        @classmethod
        def validate_tuning_delta(cls, v):
            if not 0 <= v <= 1:
                raise ValueError('LLM_TUNING_MAX_REL_DELTA must be between 0 and 1')
            return v
        
        @field_validator('RATE_LIMIT_REQUESTS_PER_MINUTE')
        @classmethod
        def validate_rate_limit(cls, v):
            if not 1 <= v <= 10000:
                raise ValueError('RATE_LIMIT_REQUESTS_PER_MINUTE must be between 1 and 10000')
            return v
        
        @field_validator('JWT_ACCESS_TOKEN_EXPIRE_MINUTES')
        @classmethod
        def validate_jwt_expiry(cls, v):
            if not 1 <= v <= 1440:  # Max 24 hours
                raise ValueError('JWT_ACCESS_TOKEN_EXPIRE_MINUTES must be between 1 and 1440')
            return v
        
        @field_validator('JWT_SECRET_KEY')
        @classmethod
        def validate_jwt_secret(cls, v):
            if not v or len(v) < 32:
                raise ValueError('JWT_SECRET_KEY must be at least 32 characters for production security')
            return v
        
        @field_validator('ADMIN_API_KEY')
        @classmethod
        def validate_admin_key(cls, v):
            if not v or len(v) < 16:
                raise ValueError('ADMIN_API_KEY must be at least 16 characters for production security')
            return v
        
        @field_validator('ARIA_WS_TOKEN')
        @classmethod
        def validate_ws_token(cls, v):
            if not v or len(v) < 16:
                raise ValueError('ARIA_WS_TOKEN must be at least 16 characters for production security')
            return v
        
        @field_validator('MT5_LOGIN')
        @classmethod
        def validate_mt5_login(cls, v, info):
            values = info.data if hasattr(info, 'data') else {}
            if values.get('ARIA_ENABLE_MT5') and not v:
                raise ValueError('MT5_LOGIN is required when ARIA_ENABLE_MT5=1')
            return v
        
        @field_validator('MT5_PASSWORD')
        @classmethod
        def validate_mt5_password(cls, v, info):
            values = info.data if hasattr(info, 'data') else {}
            if values.get('ARIA_ENABLE_MT5') and not v:
                raise ValueError('MT5_PASSWORD is required when ARIA_ENABLE_MT5=1')
            return v
        
        @field_validator('MT5_SERVER')
        @classmethod
        def validate_mt5_server(cls, v, info):
            values = info.data if hasattr(info, 'data') else {}
            if values.get('ARIA_ENABLE_MT5') and not v:
                raise ValueError('MT5_SERVER is required when ARIA_ENABLE_MT5=1')
            return v

        # Convenience accessors
        @property
        def mt5_enabled(self) -> bool:
            return bool(self.ARIA_ENABLE_MT5)

        @property
        def cors_origins_list(self) -> List[str]:
            vals = [o.strip() for o in self.ARIA_CORS_ORIGINS.split(",") if o.strip()]
            return vals

        @property
        def allowed_hosts_list(self) -> List[str]:
            vals = [h.strip() for h in self.ARIA_ALLOWED_HOSTS.split(",") if h.strip()]
            return vals

        @property
        def csp_connect_src_extra(self) -> List[str]:
            extra = self.ARIA_CSP_CONNECT_SRC.replace(",", " ").strip()
            return [tok for tok in extra.split() if tok]

        @property
        def symbols_list(self) -> List[str]:
            raw = self.AUTO_TRADE_SYMBOLS or self.ARIA_SYMBOLS
            if not raw:
                return []
            parts = raw.replace(";", ",").split(",")
            return [p.strip().upper() for p in parts if p.strip()]

        @property
        def mt5_bar_seconds(self) -> int:
            if isinstance(self.ARIA_MT5_BAR_SECONDS, int):
                return self.ARIA_MT5_BAR_SECONDS
            if isinstance(self.ARIA_FEED_BAR_SECONDS, int):
                return self.ARIA_FEED_BAR_SECONDS
            try:
                # Handle strings just in case
                s1 = int(os.environ.get("ARIA_MT5_BAR_SECONDS", ""))
                return s1
            except Exception:
                pass
            try:
                s2 = int(os.environ.get("ARIA_FEED_BAR_SECONDS", ""))
                return s2
            except Exception:
                pass
            return 60

        @property
        def include_xgb(self) -> bool:
            return bool(self.ARIA_INCLUDE_XGB)

        # Environment helpers (fallback)
        @property
        def environment(self) -> str:
            try:
                env = (self.ARIA_ENV or "production").strip().lower()
            except Exception:
                env = "production"
            return env

        @property
        def is_development(self) -> bool:
            return self.environment in ("dev", "development", "local")

        @property
        def is_test(self) -> bool:
            return self.environment in ("test", "testing", "ci")

        @property
        def is_production(self) -> bool:
            return not (self.is_development or self.is_test)
        
        @property
        def jwt_access_expire(self) -> timedelta:
            return timedelta(minutes=self.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        
        @property
        def jwt_refresh_expire(self) -> timedelta:
            return timedelta(days=self.JWT_REFRESH_TOKEN_EXPIRE_DAYS)

        # Environment helpers
        @property
        def environment(self) -> str:
            try:
                env = (self.ARIA_ENV or "production").strip().lower()
            except Exception:
                env = "production"
            return env

        @property
        def is_development(self) -> bool:
            return self.environment in ("dev", "development", "local")

        @property
        def is_test(self) -> bool:
            return self.environment in ("test", "testing", "ci")

        @property
        def is_production(self) -> bool:
            return not (self.is_development or self.is_test)

else:

    class Settings:  # Fallback minimal settings if Pydantic not available
        def __init__(self) -> None:
            # Logging
            self.LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
            self.ARIA_LOG_DIR = os.environ.get("ARIA_LOG_DIR", "./logs")

            # Security/CORS
            self.ARIA_ENV = os.environ.get("ARIA_ENV", "production")
            self.ARIA_CORS_ORIGINS = os.environ.get("ARIA_CORS_ORIGINS", "")
            self.ARIA_ALLOWED_HOSTS = os.environ.get("ARIA_ALLOWED_HOSTS", "")
            self.ARIA_CSP_CONNECT_SRC = os.environ.get("ARIA_CSP_CONNECT_SRC", "")

            # MT5 (INTENTIONAL_DEFAULT_TAG 2025-08-19: enabled by default)
            self.ARIA_ENABLE_MT5 = os.environ.get("ARIA_ENABLE_MT5", "1") in ("1", "true", "True")
            try:
                self.MT5_LOGIN = int(os.environ.get("MT5_LOGIN", "0")) if os.environ.get("MT5_LOGIN") else None
            except Exception:
                self.MT5_LOGIN = None
            self.MT5_PASSWORD = os.environ.get("MT5_PASSWORD")
            self.MT5_SERVER = os.environ.get("MT5_SERVER")

            # Data/AI
            self.ARIA_SYMBOLS = os.environ.get("ARIA_SYMBOLS", "")
            self.AUTO_TRADE_SYMBOLS = os.environ.get("AUTO_TRADE_SYMBOLS", "")
            self.ARIA_MT5_BAR_SECONDS = os.environ.get("ARIA_MT5_BAR_SECONDS")
            self.ARIA_FEED_BAR_SECONDS = os.environ.get("ARIA_FEED_BAR_SECONDS")
            self.ARIA_INCLUDE_XGB = os.environ.get("ARIA_INCLUDE_XGB", "1") in ("1", "true", "True")

            # Auto-trade
            self.AUTO_TRADE_ENABLED = os.environ.get("AUTO_TRADE_ENABLED", "0") in ("1", "true", "True")
            self.AUTO_TRADE_DRY_RUN = os.environ.get("AUTO_TRADE_DRY_RUN", "1") in ("1", "true", "True")
            self.AUTO_TRADE_PRIMARY_MODEL = os.environ.get("AUTO_TRADE_PRIMARY_MODEL", "xgb")
            # Execution flags (INTENTIONAL_DEFAULT_TAG 2025-08-19: enabled by default)
            self.ARIA_ENABLE_EXEC = os.environ.get("ARIA_ENABLE_EXEC", "1") in ("1", "true", "True")
            self.AUTO_EXEC_ENABLED = os.environ.get("AUTO_EXEC_ENABLED", "1") in ("1", "true", "True")
            self.ALLOW_LIVE = os.environ.get("ALLOW_LIVE", "1") in ("1", "true", "True")

            # Quick profit strategy toggles
            self.ENABLE_QUICK_PROFIT = os.environ.get("ENABLE_QUICK_PROFIT", "0") in ("1", "true", "True")
            self.ENABLE_ARBITRAGE = os.environ.get("ENABLE_ARBITRAGE", "0") in ("1", "true", "True")
            self.ENABLE_NEWS_TRADING = os.environ.get("ENABLE_NEWS_TRADING", "0") in ("1", "true", "True")

            # Admin/API
            self.ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "")
            self.ARIA_ENABLE_CPP_SMC = os.environ.get("ARIA_ENABLE_CPP_SMC", "auto")

            # WebSocket
            try:
                self.ARIA_WS_HEARTBEAT_SEC = int(os.environ.get("ARIA_WS_HEARTBEAT_SEC", "20"))
            except Exception:
                self.ARIA_WS_HEARTBEAT_SEC = 20
            self.ARIA_WS_TOKEN = os.environ.get("ARIA_WS_TOKEN", "")

            # LLM Monitor / Tuning
            self.LLM_MONITOR_ENABLED = os.environ.get("LLM_MONITOR_ENABLED", "0") in ("1", "true", "True")
            try:
                self.LLM_MONITOR_INTERVAL_SEC = int(os.environ.get("LLM_MONITOR_INTERVAL_SEC", "60"))
            except Exception:
                self.LLM_MONITOR_INTERVAL_SEC = 60
            self.LLM_MONITOR_DAN_URL = os.environ.get("LLM_MONITOR_DAN_URL", "http://127.0.0.1:8101")
            self.LLM_TUNING_ENABLED = os.environ.get("LLM_TUNING_ENABLED", "0") in ("1", "true", "True")
            try:
                self.LLM_TUNING_MAX_REL_DELTA = float(os.environ.get("LLM_TUNING_MAX_REL_DELTA", "0.2"))
            except Exception:
                self.LLM_TUNING_MAX_REL_DELTA = 0.2
            try:
                self.LLM_TUNING_COOLDOWN_SEC = int(os.environ.get("LLM_TUNING_COOLDOWN_SEC", "300"))
            except Exception:
                self.LLM_TUNING_COOLDOWN_SEC = 300

            # JWT Authentication
            self.JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "")
            self.JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
            try:
                self.JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
            except Exception:
                self.JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
            try:
                self.JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.environ.get("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))
            except Exception:
                self.JWT_REFRESH_TOKEN_EXPIRE_DAYS = 7
            self.JWT_ENABLED = os.environ.get("JWT_ENABLED", "1") in ("1", "true", "True")

            # Rate Limiting
            self.RATE_LIMIT_ENABLED = os.environ.get("RATE_LIMIT_ENABLED", "1") in ("1", "true", "True")
            try:
                self.RATE_LIMIT_REQUESTS_PER_MINUTE = int(os.environ.get("RATE_LIMIT_REQUESTS_PER_MINUTE", "200"))
            except Exception:
                self.RATE_LIMIT_REQUESTS_PER_MINUTE = 200
            try:
                self.RATE_LIMIT_BURST = int(os.environ.get("RATE_LIMIT_BURST", "50"))
            except Exception:
                self.RATE_LIMIT_BURST = 50

            # Prometheus / Monitoring
            self.PROMETHEUS_ENABLED = os.environ.get("PROMETHEUS_ENABLED", "1") in ("1", "true", "True")
            try:
                self.PROMETHEUS_PORT = int(os.environ.get("PROMETHEUS_PORT", "8001"))
            except Exception:
                self.PROMETHEUS_PORT = 8001
            self.PROMETHEUS_PUSH_GATEWAY = os.environ.get("PROMETHEUS_PUSH_GATEWAY")
            self.PROMETHEUS_JOB_NAME = os.environ.get("PROMETHEUS_JOB_NAME", "aria_pro")
            self.PROMETHEUS_INSTANCE = os.environ.get("PROMETHEUS_INSTANCE", "aria_backend")

        @property
        def mt5_enabled(self) -> bool:
            return bool(self.ARIA_ENABLE_MT5)

        @property
        def cors_origins_list(self) -> List[str]:
            return [o.strip() for o in self.ARIA_CORS_ORIGINS.split(",") if o.strip()]

        @property
        def allowed_hosts_list(self) -> List[str]:
            return [h.strip() for h in self.ARIA_ALLOWED_HOSTS.split(",") if h.strip()]

        @property
        def csp_connect_src_extra(self) -> List[str]:
            extra = self.ARIA_CSP_CONNECT_SRC.replace(",", " ").strip()
            return [tok for tok in extra.split() if tok]

        @property
        def symbols_list(self) -> List[str]:
            raw = self.AUTO_TRADE_SYMBOLS or self.ARIA_SYMBOLS
            if not raw:
                return []
            parts = raw.replace(";", ",").split(",")
            return [p.strip().upper() for p in parts if p.strip()]

        @property
        def mt5_bar_seconds(self) -> int:
            try:
                if self.ARIA_MT5_BAR_SECONDS is not None:
                    return int(self.ARIA_MT5_BAR_SECONDS)
            except Exception:
                pass
            try:
                if self.ARIA_FEED_BAR_SECONDS is not None:
                    return int(self.ARIA_FEED_BAR_SECONDS)
            except Exception:
                pass
            return 60

        @property
        def include_xgb(self) -> bool:
            return bool(self.ARIA_INCLUDE_XGB)


_SETTINGS: Optional[Settings] = None

def get_settings() -> Settings:
    global _SETTINGS
    if _SETTINGS is None:
        _bootstrap_env_from_files_once()
        try:
            _SETTINGS = Settings()  # type: ignore
        except Exception as e:
            logger.error(f"Failed to initialize Settings: {e}")
            # Fallback to minimal object
            _SETTINGS = Settings()  # type: ignore
    return _SETTINGS
