import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class RiskConfig:
    max_position_size: float  # Maximum position size as % of account
    max_daily_loss: float  # Maximum daily loss as % of account
    max_drawdown: float  # Maximum drawdown as % of account
    max_exposure: float  # Maximum total exposure as % of account
    max_slippage: float  # Maximum allowed slippage in pips
    position_timeout: int  # Position timeout in minutes


class RiskEngine:
    def __init__(self):
        self.risk_configs = {
            RiskLevel.CONSERVATIVE: RiskConfig(
                max_position_size=1.0,
                max_daily_loss=2.0,
                max_drawdown=5.0,
                max_exposure=10.0,
                max_slippage=5.0,
                position_timeout=1440,  # 24 hours
            ),
            RiskLevel.MODERATE: RiskConfig(
                max_position_size=2.0,
                max_daily_loss=3.0,
                max_drawdown=8.0,
                max_exposure=15.0,
                max_slippage=8.0,
                position_timeout=720,  # 12 hours
            ),
            RiskLevel.AGGRESSIVE: RiskConfig(
                max_position_size=3.0,
                max_daily_loss=5.0,
                max_drawdown=12.0,
                max_exposure=25.0,
                max_slippage=12.0,
                position_timeout=480,  # 8 hours
            ),
        }

        self.current_risk_level = RiskLevel.MODERATE
        self.daily_pnl = 0.0
        self.max_equity = 0.0
        self.positions_history = []
        self.slippage_log = []

    def set_risk_level(self, level: RiskLevel):
        """Set the current risk level"""
        self.current_risk_level = level
        logger.info(f"Risk level set to: {level.value}")

    def get_risk_config(self) -> RiskConfig:
        """Get current risk configuration"""
        return self.risk_configs[self.current_risk_level]

    def calculate_position_size(
        self,
        account_balance: float,
        symbol: str,
        entry_price: float,
        stop_loss: Optional[float] = None,
    ) -> float:
        """Calculate safe position size based on risk parameters"""
        config = self.get_risk_config()

        # Base position size as percentage of account
        max_position_value = account_balance * (config.max_position_size / 100.0)

        # If stop loss is provided, calculate position size based on risk
        if stop_loss and entry_price:
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share > 0:
                max_risk_amount = account_balance * (config.max_daily_loss / 100.0)
                risk_based_size = max_risk_amount / risk_per_share
                return min(max_position_value / entry_price, risk_based_size)

        # Default to percentage-based sizing
        return max_position_value / entry_price if entry_price > 0 else 0.0

    def validate_order(
        self,
        symbol: str,
        volume: float,
        order_type: str,
        account_info: Dict[str, Any],
        current_positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Validate order against risk parameters"""
        config = self.get_risk_config()
        validation_result = {
            "approved": True,
            "warnings": [],
            "errors": [],
            "adjusted_volume": volume,
        }

        # Check account balance
        if volume * account_info.get("balance", 0) > account_info.get("balance", 0) * (
            config.max_position_size / 100.0
        ):
            validation_result["warnings"].append(
                f"Position size exceeds {config.max_position_size}% of account"
            )

        # Check daily loss limit
        if self.daily_pnl < -(
            account_info.get("balance", 0) * (config.max_daily_loss / 100.0)
        ):
            validation_result["approved"] = False
            validation_result["errors"].append("Daily loss limit exceeded")

        # Check drawdown
        current_equity = account_info.get("equity", 0)
        if current_equity > self.max_equity:
            self.max_equity = current_equity

        drawdown = (
            (self.max_equity - current_equity) / self.max_equity * 100
            if self.max_equity > 0
            else 0
        )
        if drawdown > config.max_drawdown:
            validation_result["approved"] = False
            validation_result["errors"].append(
                f"Maximum drawdown exceeded: {drawdown:.2f}%"
            )

        # Check total exposure
        total_exposure = sum(
            pos.get("volume", 0) * pos.get("price_current", 0)
            for pos in current_positions
        )
        new_exposure = total_exposure + (volume * account_info.get("balance", 0))
        exposure_percentage = (
            (new_exposure / account_info.get("balance", 0)) * 100
            if account_info.get("balance", 0) > 0
            else 0
        )

        if exposure_percentage > config.max_exposure:
            validation_result["approved"] = False
            validation_result["errors"].append(
                f"Total exposure limit exceeded: {exposure_percentage:.2f}%"
            )

        return validation_result

    def track_slippage(
        self, expected_price: float, executed_price: float, symbol: str
    ) -> Dict[str, Any]:
        """Track slippage for risk monitoring"""
        slippage_pips = abs(executed_price - expected_price) * 10000  # Convert to pips
        config = self.get_risk_config()

        slippage_record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "expected_price": expected_price,
            "executed_price": executed_price,
            "slippage_pips": slippage_pips,
            "within_limit": slippage_pips <= config.max_slippage,
        }

        self.slippage_log.append(slippage_record)

        # Keep only last 1000 records
        if len(self.slippage_log) > 1000:
            self.slippage_log = self.slippage_log[-1000:]

        if slippage_pips > config.max_slippage:
            logger.warning(
                f"High slippage detected: {slippage_pips:.2f} pips for {symbol}"
            )

        return slippage_record

    def update_daily_pnl(self, pnl: float):
        """Update daily P&L tracking"""
        self.daily_pnl += pnl

        # Reset daily P&L at midnight
        now = datetime.now()
        if now.hour == 0 and now.minute == 0:
            self.daily_pnl = 0.0

    def check_position_timeout(
        self, positions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Check for positions that have exceeded timeout"""
        config = self.get_risk_config()
        timeout_positions = []

        for position in positions:
            position_time = datetime.fromtimestamp(position.get("time", 0))
            time_diff = datetime.now() - position_time

            if time_diff.total_seconds() > config.position_timeout * 60:
                timeout_positions.append(
                    {
                        "ticket": position.get("ticket"),
                        "symbol": position.get("symbol"),
                        "time_open": position_time.isoformat(),
                        "timeout_minutes": config.position_timeout,
                    }
                )

        return timeout_positions

    def get_risk_metrics(self, account_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get current risk metrics"""
        config = self.get_risk_config()
        current_equity = account_info.get("equity", 0)

        # Calculate drawdown
        drawdown = (
            (self.max_equity - current_equity) / self.max_equity * 100
            if self.max_equity > 0
            else 0
        )

        # Calculate average slippage
        avg_slippage = (
            sum(record["slippage_pips"] for record in self.slippage_log)
            / len(self.slippage_log)
            if self.slippage_log
            else 0
        )

        return {
            "risk_level": self.current_risk_level.value,
            "daily_pnl": self.daily_pnl,
            "max_equity": self.max_equity,
            "current_drawdown": drawdown,
            "max_drawdown_allowed": config.max_drawdown,
            "daily_loss_limit": account_info.get("balance", 0)
            * (config.max_daily_loss / 100.0),
            "position_size_limit": account_info.get("balance", 0)
            * (config.max_position_size / 100.0),
            "exposure_limit": account_info.get("balance", 0)
            * (config.max_exposure / 100.0),
            "max_slippage_allowed": config.max_slippage,
            "average_slippage": avg_slippage,
            "slippage_violations": len(
                [r for r in self.slippage_log if not r["within_limit"]]
            ),
            "total_trades_tracked": len(self.slippage_log),
        }

    def emergency_stop(self, account_info: Dict[str, Any]) -> bool:
        """Check if emergency stop conditions are met"""
        config = self.get_risk_config()
        current_equity = account_info.get("equity", 0)

        # Emergency stop conditions
        conditions = [
            self.daily_pnl
            < -(account_info.get("balance", 0) * (config.max_daily_loss * 1.5 / 100.0)),
            (
                (self.max_equity - current_equity) / self.max_equity * 100
                > config.max_drawdown * 1.2
                if self.max_equity > 0
                else False
            ),
        ]

        if any(conditions):
            logger.critical("EMERGENCY STOP CONDITIONS MET - TRADING HALTED")
            return True

        return False


# Global risk engine instance
risk_engine = RiskEngine()
