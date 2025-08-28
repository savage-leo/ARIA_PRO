import logging
import asyncio
import threading
import os
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
from collections import deque
from scipy import stats

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
    var_confidence: float = 0.95  # VaR confidence level
    cvar_confidence: float = 0.99  # CVaR confidence level
    correlation_threshold: float = 0.7  # Correlation risk threshold
    kelly_fraction: float = 0.25  # Kelly criterion fraction
    max_leverage: float = 10.0  # Maximum leverage
    stress_test_scenarios: int = 1000  # Monte Carlo scenarios

@dataclass
class PositionMetrics:
    symbol: str
    entry_time: datetime
    entry_price: float
    current_price: float
    position_size: float
    unrealized_pnl: float
    realized_pnl: float
    duration_minutes: int
    max_favorable_excursion: float
    max_adverse_excursion: float

@dataclass
class RiskMetrics:
    var_1d: float  # 1-day Value at Risk
    cvar_1d: float  # 1-day Conditional VaR
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    current_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    correlation_risk: float
    kelly_optimal_size: float


class RiskEngine:
    def __init__(self):
        # Thread-safe locks for shared state
        self._state_lock = threading.Lock()
        self._async_lock = asyncio.Lock()
        self._kill_switch_lock = threading.Lock()
        
        # Load configuration from environment
        self._load_config_from_env()
        
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

        # Protected shared state
        self.current_risk_level = RiskLevel.MODERATE
        self.daily_pnl = 0.0
        self.max_equity = 0.0
        self.positions_history = []
        self.slippage_log = []
        self.kill_switch_engaged = False
        self.kill_switch_reason = ""
        self._last_update = datetime.now()
        
        # Advanced analytics state
        self.pnl_history = deque(maxlen=1000)  # Rolling PnL history
        self.returns_history = deque(maxlen=252)  # Daily returns for 1 year
        self.position_metrics: Dict[str, PositionMetrics] = {}
        self.correlation_matrix = {}
        self.stress_test_results = {}
        self._last_risk_calculation = datetime.now()
        self._risk_calculation_interval = timedelta(minutes=5)

    def _load_config_from_env(self):
        """Load risk configuration from environment variables"""
        self.env_config = {
            'max_position_size_conservative': float(os.environ.get('RISK_MAX_POS_SIZE_CONSERVATIVE', '1.0')),
            'max_position_size_moderate': float(os.environ.get('RISK_MAX_POS_SIZE_MODERATE', '2.0')),
            'max_position_size_aggressive': float(os.environ.get('RISK_MAX_POS_SIZE_AGGRESSIVE', '3.0')),
            'max_daily_loss_conservative': float(os.environ.get('RISK_MAX_DAILY_LOSS_CONSERVATIVE', '2.0')),
            'max_daily_loss_moderate': float(os.environ.get('RISK_MAX_DAILY_LOSS_MODERATE', '3.0')),
            'max_daily_loss_aggressive': float(os.environ.get('RISK_MAX_DAILY_LOSS_AGGRESSIVE', '5.0')),
            'var_confidence': float(os.environ.get('RISK_VAR_CONFIDENCE', '0.95')),
            'cvar_confidence': float(os.environ.get('RISK_CVAR_CONFIDENCE', '0.99')),
            'kelly_fraction': float(os.environ.get('RISK_KELLY_FRACTION', '0.25')),
            'correlation_threshold': float(os.environ.get('RISK_CORRELATION_THRESHOLD', '0.7')),
        }

    def set_risk_level(self, level: RiskLevel):
        """Set the current risk level"""
        with self._state_lock:
            self.current_risk_level = level
            logger.info(f"Risk level set to: {level.value}")

    def get_risk_config(self) -> RiskConfig:
        """Get current risk configuration"""
        with self._state_lock:
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

        # Check daily loss limit (thread-safe)
        with self._state_lock:
            if self.daily_pnl < -(
                account_info.get("balance", 0) * (config.max_daily_loss / 100.0)
            ):
                validation_result["approved"] = False
                validation_result["errors"].append("Daily loss limit exceeded")
                self._trigger_kill_switch("Daily loss limit exceeded")

        # Check drawdown (thread-safe)
        with self._state_lock:
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
                self._trigger_kill_switch(f"Maximum drawdown exceeded: {drawdown:.2f}%")

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

        with self._state_lock:
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
        with self._state_lock:
            self.daily_pnl += pnl
            self._last_update = datetime.now()

            # Reset daily P&L at midnight
            now = datetime.now()
            if now.hour == 0 and now.minute == 0:
                self.daily_pnl = 0.0
                logger.info("Daily P&L reset at midnight")

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
        with self._state_lock:
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
                "kill_switch_engaged": self.kill_switch_engaged,
                "kill_switch_reason": self.kill_switch_reason,
                "last_update": self._last_update.isoformat(),
            }

    def _trigger_kill_switch(self, reason: str) -> None:
        """Trigger the kill switch to halt all trading"""
        with self._kill_switch_lock:
            if not self.kill_switch_engaged:
                self.kill_switch_engaged = True
                self.kill_switch_reason = reason
                logger.critical(f"KILL SWITCH ENGAGED: {reason}")
    
    def reset_kill_switch(self) -> None:
        """Reset the kill switch (admin only)"""
        with self._kill_switch_lock:
            self.kill_switch_engaged = False
            self.kill_switch_reason = ""
            logger.warning("Kill switch reset by admin")
    
    def is_kill_switch_engaged(self) -> bool:
        """Check if kill switch is engaged"""
        with self._kill_switch_lock:
            return self.kill_switch_engaged
    
    async def async_validate_order(
        self,
        symbol: str,
        volume: float,
        order_type: str,
        account_info: Dict[str, Any],
        current_positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Async wrapper for thread-safe order validation"""
        async with self._async_lock:
            return self.validate_order(symbol, volume, order_type, account_info, current_positions)
    
    async def async_update_pnl(self, pnl: float) -> None:
        """Async wrapper for thread-safe P&L update"""
        async with self._async_lock:
            self.update_daily_pnl(pnl)
    
    def emergency_stop(self, account_info: Dict[str, Any]) -> bool:
        """Check if emergency stop conditions are met"""
        with self._state_lock:
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
                self._trigger_kill_switch("EMERGENCY STOP CONDITIONS MET")
                return True

            return False

    def calculate_var(self, confidence: float = None) -> float:
        """Calculate Value at Risk using historical simulation"""
        if not self.returns_history:
            return 0.0
        
        confidence = confidence or self.env_config.get('var_confidence', 0.95)
        returns = np.array(self.returns_history)
        
        if len(returns) < 30:  # Need minimum data
            return 0.0
        
        # Calculate VaR using percentile method
        var_percentile = (1 - confidence) * 100
        var = np.percentile(returns, var_percentile)
        return abs(var)

    def calculate_cvar(self, confidence: float = None) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if not self.returns_history:
            return 0.0
        
        confidence = confidence or self.env_config.get('cvar_confidence', 0.99)
        returns = np.array(self.returns_history)
        
        if len(returns) < 30:
            return 0.0
        
        # Calculate CVaR as mean of returns below VaR threshold
        var_threshold = np.percentile(returns, (1 - confidence) * 100)
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return 0.0
        
        return abs(np.mean(tail_returns))

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(self.returns_history) < 30:
            return 0.0
        
        returns = np.array(self.returns_history)
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(self.returns_history) < 30:
            return 0.0
        
        returns = np.array(self.returns_history)
        excess_returns = returns - (risk_free_rate / 252)
        
        # Calculate downside deviation
        negative_returns = excess_returns[excess_returns < 0]
        if len(negative_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(negative_returns)
        if downside_deviation == 0:
            return 0.0
        
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)

    def calculate_kelly_optimal_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly optimal position size"""
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            return 0.0
        
        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        b = abs(avg_win / avg_loss)
        p = win_rate
        q = 1 - win_rate
        
        kelly_fraction = (b * p - q) / b
        
        # Apply safety factor
        safety_factor = self.env_config.get('kelly_fraction', 0.25)
        return max(0, min(kelly_fraction * safety_factor, 0.25))

    def get_comprehensive_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        with self._state_lock:
            # Calculate basic metrics
            var_1d = self.calculate_var()
            cvar_1d = self.calculate_cvar()
            sharpe = self.calculate_sharpe_ratio()
            sortino = self.calculate_sortino_ratio()
            
            # Calculate win rate and profit metrics
            winning_trades = [p for p in self.pnl_history if p > 0]
            losing_trades = [p for p in self.pnl_history if p < 0]
            
            win_rate = len(winning_trades) / len(self.pnl_history) if self.pnl_history else 0.0
            avg_win = np.mean(winning_trades) if winning_trades else 0.0
            avg_loss = abs(np.mean(losing_trades)) if losing_trades else 0.0
            profit_factor = (sum(winning_trades) / abs(sum(losing_trades))) if losing_trades else 0.0
            
            # Calculate drawdown
            if self.pnl_history:
                cumulative_pnl = np.cumsum(self.pnl_history)
                running_max = np.maximum.accumulate(cumulative_pnl)
                drawdown = (cumulative_pnl - running_max) / running_max
                max_drawdown = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
                current_drawdown = abs(drawdown[-1]) if len(drawdown) > 0 else 0.0
            else:
                max_drawdown = current_drawdown = 0.0
            
            # Calculate Kelly optimal size
            kelly_optimal = self.calculate_kelly_optimal_size(win_rate, avg_win, avg_loss)
            
            return RiskMetrics(
                var_1d=var_1d,
                cvar_1d=cvar_1d,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                max_drawdown=max_drawdown,
                current_drawdown=current_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss,
                correlation_risk=0.0,
                kelly_optimal_size=kelly_optimal
            )

    def add_trade_result(self, pnl: float, return_pct: float) -> None:
        """Add trade result to history for analytics"""
        with self._state_lock:
            self.pnl_history.append(pnl)
            self.returns_history.append(return_pct)
            self.daily_pnl += pnl


# Global risk engine instance
risk_engine = RiskEngine()
