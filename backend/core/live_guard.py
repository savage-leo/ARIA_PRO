"""
Live-only guard to enforce MT5 as the sole data source and block any simulation/mock flags.
"""
from __future__ import annotations

import os
import logging
from typing import Optional

from backend.core.config import get_settings, Settings

logger = logging.getLogger(__name__)


def enforce_live_only(settings: Optional[Settings] = None) -> None:
    """Enforce MT5-only live mode.

    - Requires ARIA_ENABLE_MT5=1 in production
    - Blocks known simulation/mock/alpha-vantage flags if present and enabled
    - Skips enforcement in development/test environments
    """
    s = settings or get_settings()

    # Skip live-only enforcement in development/test environments
    if getattr(s, 'is_development', False) or getattr(s, 'is_test', False):
        logger.info("Development/Test mode: skipping live-only MT5 enforcement")
        return

    if not s.mt5_enabled:
        raise RuntimeError(
            "Live-only mode: ARIA_ENABLE_MT5=1 is required; simulated or mock feeds are forbidden."
        )

    # Detect banned toggles that indicate non-live feeds
    banned_vars = [
        "SIM_MODE",
        "SIMULATED",
        "USE_SIM",
        "MOCK_DATA",
        "FAKE_DATA",
        "USE_SIMULATED_FEED",
        "USE_SIM_FEED",
        "BACKTEST_MODE",
        "PAPER_TRADING",
        # Legacy Alpha Vantage toggles
        "ALPHA_VANTAGE_API_KEY",
        "AV_API_KEY",
        "ENABLE_ALPHA_VANTAGE",
        "USE_AV",
    ]

    def _is_enabled(val: str) -> bool:
        v = val.strip().lower()
        return v not in ("", "0", "false", "no", "off", "none")

    violations = []
    for name in banned_vars:
        val = os.environ.get(name)
        if val is not None and _is_enabled(val):
            violations.append(name)

    if violations:
        raise RuntimeError(
            f"Live-only guard: banned non-live flags present: {', '.join(sorted(violations))}"
        )
