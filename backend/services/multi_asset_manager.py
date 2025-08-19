# -*- coding: utf-8 -*-
"""
Multi-Asset Manager: Asset-agnostic framework for hedge fund expansion
Supports Forex, Commodities, Indices, Crypto with unified position management
"""
from __future__ import annotations
import os, time, math, logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class AssetClass(Enum):
    FOREX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"
    CRYPTO = "crypto"
    BOND = "bond"
    STOCK = "stock"


@dataclass
class AssetConfig:
    """Configuration for each asset class"""

    symbol: str
    asset_class: AssetClass
    tick_size: float
    contract_size: float
    margin_requirement: float
    spread_typical: float
    trading_hours: Dict[str, str]
    volatility_factor: float = 1.0
    liquidity_tier: int = 1  # 1=highest, 5=lowest


class MultiAssetManager:
    """
    Multi-asset framework for institutional trading
    Provides asset-agnostic position sizing, risk management, and execution
    """

    def __init__(self):
        self.asset_configs = {}
        self.position_multipliers = {}
        self.correlation_matrix = {}

        # Load asset configurations
        self._initialize_asset_configs()

        # Risk parameters per asset class
        self.risk_limits = {
            AssetClass.FOREX: {"max_leverage": 30, "max_position_pct": 0.10},
            AssetClass.COMMODITY: {"max_leverage": 10, "max_position_pct": 0.08},
            AssetClass.INDEX: {"max_leverage": 20, "max_position_pct": 0.12},
            AssetClass.CRYPTO: {"max_leverage": 5, "max_position_pct": 0.05},
            AssetClass.BOND: {"max_leverage": 50, "max_position_pct": 0.15},
            AssetClass.STOCK: {"max_leverage": 4, "max_position_pct": 0.06},
        }

        logger.info(
            f"Multi-Asset Manager initialized with {len(self.asset_configs)} assets"
        )

    def _initialize_asset_configs(self):
        """Initialize configurations for all supported assets"""

        # Major Forex Pairs
        forex_pairs = {
            "EURUSD": AssetConfig(
                "EURUSD",
                AssetClass.FOREX,
                0.00001,
                100000,
                0.033,
                0.8,
                {"start": "22:00", "end": "22:00"},
                1.0,
                1,
            ),
            "GBPUSD": AssetConfig(
                "GBPUSD",
                AssetClass.FOREX,
                0.00001,
                100000,
                0.033,
                1.2,
                {"start": "22:00", "end": "22:00"},
                1.2,
                1,
            ),
            "USDJPY": AssetConfig(
                "USDJPY",
                AssetClass.FOREX,
                0.001,
                100000,
                0.033,
                0.9,
                {"start": "22:00", "end": "22:00"},
                1.1,
                1,
            ),
            "USDCHF": AssetConfig(
                "USDCHF",
                AssetClass.FOREX,
                0.00001,
                100000,
                0.033,
                1.1,
                {"start": "22:00", "end": "22:00"},
                1.0,
                1,
            ),
            "AUDUSD": AssetConfig(
                "AUDUSD",
                AssetClass.FOREX,
                0.00001,
                100000,
                0.033,
                1.0,
                {"start": "22:00", "end": "22:00"},
                1.3,
                2,
            ),
            "NZDUSD": AssetConfig(
                "NZDUSD",
                AssetClass.FOREX,
                0.00001,
                100000,
                0.033,
                1.5,
                {"start": "22:00", "end": "22:00"},
                1.4,
                2,
            ),
            "EURGBP": AssetConfig(
                "EURGBP",
                AssetClass.FOREX,
                0.00001,
                100000,
                0.033,
                1.3,
                {"start": "22:00", "end": "22:00"},
                0.8,
                2,
            ),
            "EURJPY": AssetConfig(
                "EURJPY",
                AssetClass.FOREX,
                0.001,
                100000,
                0.033,
                1.5,
                {"start": "22:00", "end": "22:00"},
                1.4,
                2,
            ),
        }

        # Commodities
        commodities = {
            "XAUUSD": AssetConfig(
                "XAUUSD",
                AssetClass.COMMODITY,
                0.01,
                100,
                0.05,
                0.30,
                {"start": "01:00", "end": "24:00"},
                1.8,
                1,
            ),
            "XAGUSD": AssetConfig(
                "XAGUSD",
                AssetClass.COMMODITY,
                0.001,
                5000,
                0.05,
                0.03,
                {"start": "01:00", "end": "24:00"},
                2.5,
                2,
            ),
            "USOIL": AssetConfig(
                "USOIL",
                AssetClass.COMMODITY,
                0.01,
                1000,
                0.10,
                0.05,
                {"start": "01:00", "end": "24:00"},
                2.2,
                1,
            ),
            "UKOIL": AssetConfig(
                "UKOIL",
                AssetClass.COMMODITY,
                0.01,
                1000,
                0.10,
                0.08,
                {"start": "03:00", "end": "22:00"},
                2.0,
                2,
            ),
        }

        # Major Indices
        indices = {
            "US30": AssetConfig(
                "US30",
                AssetClass.INDEX,
                1,
                5,
                0.05,
                3.0,
                {"start": "01:00", "end": "23:00"},
                1.5,
                1,
            ),
            "SPX500": AssetConfig(
                "SPX500",
                AssetClass.INDEX,
                0.1,
                50,
                0.05,
                0.5,
                {"start": "01:00", "end": "23:00"},
                1.2,
                1,
            ),
            "NAS100": AssetConfig(
                "NAS100",
                AssetClass.INDEX,
                0.1,
                20,
                0.05,
                1.0,
                {"start": "01:00", "end": "23:00"},
                1.8,
                1,
            ),
            "GER40": AssetConfig(
                "GER40",
                AssetClass.INDEX,
                0.5,
                25,
                0.05,
                2.0,
                {"start": "02:00", "end": "23:00"},
                1.4,
                2,
            ),
            "UK100": AssetConfig(
                "UK100",
                AssetClass.INDEX,
                0.1,
                10,
                0.05,
                1.5,
                {"start": "02:00", "end": "21:00"},
                1.3,
                2,
            ),
        }

        # Cryptocurrencies
        crypto = {
            "BTCUSD": AssetConfig(
                "BTCUSD",
                AssetClass.CRYPTO,
                1,
                1,
                0.20,
                10.0,
                {"start": "00:00", "end": "24:00"},
                4.0,
                1,
            ),
            "ETHUSD": AssetConfig(
                "ETHUSD",
                AssetClass.CRYPTO,
                0.1,
                1,
                0.20,
                2.0,
                {"start": "00:00", "end": "24:00"},
                4.5,
                1,
            ),
            "ADAUSD": AssetConfig(
                "ADAUSD",
                AssetClass.CRYPTO,
                0.0001,
                1,
                0.20,
                0.001,
                {"start": "00:00", "end": "24:00"},
                5.0,
                3,
            ),
            "DOTUSD": AssetConfig(
                "DOTUSD",
                AssetClass.CRYPTO,
                0.001,
                1,
                0.20,
                0.01,
                {"start": "00:00", "end": "24:00"},
                5.5,
                3,
            ),
        }

        # Combine all assets
        self.asset_configs.update(forex_pairs)
        self.asset_configs.update(commodities)
        self.asset_configs.update(indices)
        self.asset_configs.update(crypto)

        # Initialize correlation matrix (simplified)
        self._initialize_correlations()

    def _initialize_correlations(self):
        """Initialize asset correlation matrix for risk management"""
        symbols = list(self.asset_configs.keys())

        # Simplified correlation matrix (should be calculated from historical data)
        correlations = {
            # Forex majors tend to be correlated
            ("EURUSD", "GBPUSD"): 0.7,
            ("EURUSD", "EURGBP"): 0.6,
            ("GBPUSD", "EURGBP"): -0.5,
            # Commodities correlation
            ("XAUUSD", "XAGUSD"): 0.8,
            ("USOIL", "UKOIL"): 0.9,
            # Indices correlation
            ("US30", "SPX500"): 0.9,
            ("SPX500", "NAS100"): 0.8,
            # Crypto correlation
            ("BTCUSD", "ETHUSD"): 0.85,
            # Cross-asset correlations
            ("XAUUSD", "EURUSD"): 0.3,
            ("USOIL", "USDCAD"): -0.6,
        }

        # Build symmetric matrix
        self.correlation_matrix = {}
        for symbol1 in symbols:
            self.correlation_matrix[symbol1] = {}
            for symbol2 in symbols:
                if symbol1 == symbol2:
                    self.correlation_matrix[symbol1][symbol2] = 1.0
                else:
                    # Check both directions
                    corr = (
                        correlations.get((symbol1, symbol2))
                        or correlations.get((symbol2, symbol1))
                        or 0.0
                    )
                    self.correlation_matrix[symbol1][symbol2] = corr

    def get_asset_config(self, symbol: str) -> Optional[AssetConfig]:
        """Get configuration for an asset"""
        return self.asset_configs.get(symbol)

    def calculate_position_size(
        self,
        symbol: str,
        confidence: float,
        account_balance: float,
        risk_per_trade: float = 0.02,
    ) -> float:
        """
        Calculate position size with multi-asset considerations
        """
        config = self.get_asset_config(symbol)
        if not config:
            return 0.0

        # Base position size from confidence
        base_risk = account_balance * risk_per_trade

        # Adjust for asset class risk limits
        risk_limits = self.risk_limits.get(
            config.asset_class, {"max_position_pct": 0.05}
        )
        max_position_value = account_balance * risk_limits["max_position_pct"]

        # Volatility adjustment
        vol_adj = 1.0 / config.volatility_factor

        # Liquidity adjustment
        liquidity_adj = 1.0 / config.liquidity_tier

        # Confidence scaling
        confidence_adj = min(
            confidence * 2.0, 1.0
        )  # Scale up to 100% at 0.5 confidence

        # Calculate raw position size
        position_value = base_risk * vol_adj * liquidity_adj * confidence_adj

        # Cap at maximum position size
        position_value = min(position_value, max_position_value)

        # Convert to lots/units based on asset type
        if config.asset_class == AssetClass.FOREX:
            # Forex: convert to standard lots
            position_size = position_value / (
                config.contract_size * 0.0001
            )  # Rough pip value
            return max(0.01, min(position_size, 10.0))  # 0.01 to 10 lots

        elif config.asset_class == AssetClass.COMMODITY:
            # Commodities: typically in ounces or barrels
            if "XAU" in symbol or "XAG" in symbol:  # Gold/Silver
                return max(0.01, min(position_value / 1000, 100.0))  # Max 100 oz
            else:  # Oil
                return max(0.1, min(position_value / 100, 1000.0))  # Max 1000 barrels

        elif config.asset_class == AssetClass.INDEX:
            # Indices: typically in index points
            return max(0.1, min(position_value / 1000, 100.0))  # Max 100 lots

        elif config.asset_class == AssetClass.CRYPTO:
            # Crypto: in coins
            if "BTC" in symbol:
                return max(0.001, min(position_value / 50000, 10.0))  # Max 10 BTC
            elif "ETH" in symbol:
                return max(0.01, min(position_value / 3000, 100.0))  # Max 100 ETH
            else:
                return max(0.1, min(position_value / 1, 10000.0))  # Max 10k altcoins

        return 0.01  # Default minimal size

    def check_correlation_risk(
        self, new_symbol: str, existing_positions: Dict[str, float]
    ) -> float:
        """
        Check correlation risk when adding a new position
        Returns correlation risk factor (0.0 = no risk, 1.0 = high risk)
        """
        if not existing_positions:
            return 0.0

        max_correlation = 0.0
        total_correlation_exposure = 0.0

        for existing_symbol, existing_size in existing_positions.items():
            correlation = self.correlation_matrix.get(new_symbol, {}).get(
                existing_symbol, 0.0
            )

            # Weight correlation by position size
            weighted_correlation = abs(correlation) * abs(existing_size)
            total_correlation_exposure += weighted_correlation
            max_correlation = max(max_correlation, abs(correlation))

        # Normalize correlation exposure
        normalized_exposure = min(
            total_correlation_exposure / len(existing_positions), 1.0
        )

        return max(max_correlation, normalized_exposure)

    def get_asset_class_exposure(
        self, positions: Dict[str, float]
    ) -> Dict[AssetClass, float]:
        """Calculate exposure by asset class"""
        exposure = {asset_class: 0.0 for asset_class in AssetClass}

        for symbol, size in positions.items():
            config = self.get_asset_config(symbol)
            if config:
                exposure[config.asset_class] += abs(size)

        return exposure

    def validate_new_position(
        self, symbol: str, size: float, existing_positions: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Validate if new position should be allowed
        Returns validation result with recommendations
        """
        config = self.get_asset_config(symbol)
        if not config:
            return {"allowed": False, "reason": "Unknown asset", "recommendations": []}

        # Check correlation risk
        correlation_risk = self.check_correlation_risk(symbol, existing_positions)

        # Check asset class exposure
        exposure = self.get_asset_class_exposure(existing_positions)
        current_class_exposure = exposure.get(config.asset_class, 0.0)

        # Risk limits
        risk_limits = self.risk_limits.get(
            config.asset_class, {"max_position_pct": 0.05}
        )

        recommendations = []
        warnings = []

        # Correlation warnings
        if correlation_risk > 0.7:
            warnings.append(f"High correlation risk: {correlation_risk:.2f}")
            recommendations.append("Consider reducing position size due to correlation")

        # Asset class concentration
        if current_class_exposure > 0.5:  # 50% in one asset class
            warnings.append(
                f"High {config.asset_class.value} exposure: {current_class_exposure:.2f}"
            )
            recommendations.append(
                f"Consider diversifying away from {config.asset_class.value}"
            )

        # Position size warnings
        if abs(size) > 5.0:  # Arbitrary large position
            warnings.append("Large position size")
            recommendations.append("Consider scaling into position gradually")

        # Liquidity concerns
        if config.liquidity_tier > 2:
            warnings.append("Low liquidity asset")
            recommendations.append("Monitor for execution slippage")

        # Overall decision
        allowed = True
        if correlation_risk > 0.9:
            allowed = False
            recommendations.insert(0, "BLOCKED: Excessive correlation risk")

        return {
            "allowed": allowed,
            "correlation_risk": correlation_risk,
            "asset_class_exposure": current_class_exposure,
            "warnings": warnings,
            "recommendations": recommendations,
            "asset_config": config,
        }

    def get_supported_symbols(self) -> Dict[str, List[str]]:
        """Get all supported symbols grouped by asset class"""
        grouped = {}
        for symbol, config in self.asset_configs.items():
            asset_class = config.asset_class.value
            if asset_class not in grouped:
                grouped[asset_class] = []
            grouped[asset_class].append(symbol)

        return grouped

    def get_trading_sessions(self) -> Dict[str, Dict[str, str]]:
        """Get trading sessions for all assets"""
        sessions = {}
        for symbol, config in self.asset_configs.items():
            sessions[symbol] = {
                "asset_class": config.asset_class.value,
                "trading_hours": config.trading_hours,
                "liquidity_tier": config.liquidity_tier,
            }
        return sessions


# Global multi-asset manager
multi_asset_manager = MultiAssetManager()

