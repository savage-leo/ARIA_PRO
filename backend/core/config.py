# Centralized configuration using Pydantic Settings (v1 or v2 compatible)
# - Loads env from ARIA_PRO/production.env or .env once
# - Exposes typed settings and convenience accessors

from __future__ import annotations

import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Try Pydantic v2 (pydantic-settings) first, then fallback to v1 BaseSettings
try:
    from pydantic_settings import BaseSettings as _BaseSettings  # type: ignore
    from pydantic import Field  # type: ignore
    _V2 = True
except Exception:
    try:
        from pydantic import BaseSettings as _BaseSettings, Field  # type: ignore
        _V2 = False
    except Exception:
        _BaseSettings = None  # type: ignore
        Field = None  # type: ignore
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
        for candidate in (
            os.path.join(project_dir, "production.env"),
            os.path.join(project_dir, ".env"),
        ):
            if os.path.exists(candidate):
                _load_env_file(candidate)
                break
    except Exception:
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
        ARIA_CORS_ORIGINS: str = Field(default="", env="ARIA_CORS_ORIGINS")
        ARIA_ALLOWED_HOSTS: str = Field(default="", env="ARIA_ALLOWED_HOSTS")
        ARIA_CSP_CONNECT_SRC: str = Field(default="", env="ARIA_CSP_CONNECT_SRC")

        # MT5 / Trading flags
        ARIA_ENABLE_MT5: bool = Field(default=True, env="ARIA_ENABLE_MT5")
        MT5_LOGIN: Optional[int] = Field(default=None, env="MT5_LOGIN")
        MT5_PASSWORD: Optional[str] = Field(default=None, env="MT5_PASSWORD")
        MT5_SERVER: Optional[str] = Field(default=None, env="MT5_SERVER")

        # Data/AI config
        ARIA_SYMBOLS: str = Field(default="", env="ARIA_SYMBOLS")
        AUTO_TRADE_SYMBOLS: str = Field(default="", env="AUTO_TRADE_SYMBOLS")
        ARIA_MT5_BAR_SECONDS: Optional[int] = Field(default=None, env="ARIA_MT5_BAR_SECONDS")
        ARIA_FEED_BAR_SECONDS: Optional[int] = Field(default=None, env="ARIA_FEED_BAR_SECONDS")
        ARIA_INCLUDE_XGB: bool = Field(default=True, env="ARIA_INCLUDE_XGB")

        # Auto-trade + execution
        AUTO_TRADE_ENABLED: bool = Field(default=False, env="AUTO_TRADE_ENABLED")
        AUTO_TRADE_DRY_RUN: bool = Field(default=True, env="AUTO_TRADE_DRY_RUN")
        AUTO_TRADE_PRIMARY_MODEL: str = Field(default="xgb", env="AUTO_TRADE_PRIMARY_MODEL")
        ARIA_ENABLE_EXEC: bool = Field(default=True, env="ARIA_ENABLE_EXEC")
        AUTO_EXEC_ENABLED: bool = Field(default=True, env="AUTO_EXEC_ENABLED")
        ALLOW_LIVE: bool = Field(default=True, env="ALLOW_LIVE")

        # Quick profit strategy toggles
        ENABLE_QUICK_PROFIT: bool = Field(default=False, env="ENABLE_QUICK_PROFIT")
        ENABLE_ARBITRAGE: bool = Field(default=False, env="ENABLE_ARBITRAGE")
        ENABLE_NEWS_TRADING: bool = Field(default=False, env="ENABLE_NEWS_TRADING")

        # Admin/API
        ADMIN_API_KEY: str = Field(default="", env="ADMIN_API_KEY")
        ARIA_ENABLE_CPP_SMC: str = Field(default="auto", env="ARIA_ENABLE_CPP_SMC")

        # WebSocket
        ARIA_WS_HEARTBEAT_SEC: int = Field(default=20, env="ARIA_WS_HEARTBEAT_SEC")
        ARIA_WS_TOKEN: str = Field(default="", env="ARIA_WS_TOKEN")

        # LLM Monitor / Tuning
        LLM_MONITOR_ENABLED: bool = Field(default=False, env="LLM_MONITOR_ENABLED")
        LLM_MONITOR_INTERVAL_SEC: int = Field(default=60, env="LLM_MONITOR_INTERVAL_SEC")
        LLM_MONITOR_DAN_URL: str = Field(default="http://127.0.0.1:8101", env="LLM_MONITOR_DAN_URL")
        LLM_TUNING_ENABLED: bool = Field(default=False, env="LLM_TUNING_ENABLED")
        LLM_TUNING_MAX_REL_DELTA: float = Field(default=0.2, env="LLM_TUNING_MAX_REL_DELTA")
        LLM_TUNING_COOLDOWN_SEC: int = Field(default=300, env="LLM_TUNING_COOLDOWN_SEC")

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

else:

    class Settings:  # Fallback minimal settings if Pydantic not available
        def __init__(self) -> None:
            # Logging
            self.LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
            self.ARIA_LOG_DIR = os.environ.get("ARIA_LOG_DIR", "./logs")

            # Security/CORS
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
