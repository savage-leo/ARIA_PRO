# -*- coding: utf-8 -*-
"""
MT5 Live Execution Harness: Complete integration with hedge fund ensemble
Real-time order management, position sizing, audit logging, risk guards
"""
from __future__ import annotations
import os, time, math, logging, threading, json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import numpy as np

# Import existing MT5 infrastructure
from backend.services.mt5_executor import MT5Executor
from backend.services.mt5_client import MT5Client

# Import hedge fund components
from backend.services.t470_pipeline_optimized import t470_pipeline
from backend.services.hedge_fund_analytics import hedge_fund_analytics
from backend.core.risk_budget_enhanced import position_sizer
from backend.core.regime_online import regime_manager
from backend.services.telemetry_monitor import telemetry_monitor

logger = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    """Order request with hedge fund attribution"""

    symbol: str
    action: str  # 'BUY' | 'SELL' | 'CLOSE'
    volume: float
    confidence: float
    model_attribution: Dict[str, float]
    regime_state: str
    vol_bucket: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: str = "ARIA-HF"
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class ExecutionResult:
    """Execution result with full audit trail"""

    order_id: Optional[int]
    success: bool
    price: float
    volume: float
    spread: float
    slippage: float
    latency_ms: float
    error_message: Optional[str] = None
    mt5_result: Optional[Dict] = None


class MT5ExecutionHarness:
    """
    Complete MT5 execution harness integrated with hedge fund analytics
    Features:
    - Real-time ensemble decision → MT5 order execution
    - Position sizing integration with confidence levels
    - Model attribution tracking for every trade
    - Risk guard integration with kill switches
    - Multi-asset framework support
    - Complete audit logging
    """

    def __init__(self):
        self.mt5_executor = MT5Executor()
        self.mt5_client = MT5Client()

        # Execution state
        self.is_live = False
        self.enabled_symbols = set()
        self.position_tracker = {}
        self.order_history = []

        # Risk parameters from environment
        self.max_risk_per_trade = float(
            os.getenv("ARIA_MAX_RISK_PER_TRADE", "0.02")
        )  # 2%
        self.max_daily_trades = int(os.getenv("ARIA_MAX_DAILY_TRADES", "50"))
        self.max_open_positions = int(os.getenv("ARIA_MAX_OPEN_POSITIONS", "10"))
        self.min_confidence = float(os.getenv("ARIA_MIN_CONFIDENCE", "0.65"))

        # Kill switch states
        self.kill_switch_active = False
        self.daily_trade_count = 0
        self.daily_pnl = 0.0
        self.max_daily_loss = float(os.getenv("ARIA_MAX_DAILY_LOSS", "0.05"))  # 5%

        # Audit logging
        self.audit_lock = threading.Lock()
        self.setup_audit_logging()

        # Performance tracking
        self.execution_stats = {
            "total_orders": 0,
            "successful_orders": 0,
            "failed_orders": 0,
            "avg_slippage": 0.0,
            "avg_latency_ms": 0.0,
        }

        logger.info("MT5 Execution Harness initialized")

    def setup_audit_logging(self):
        """Setup audit trail logging"""
        audit_dir = os.path.join("data", "audit_logs")
        os.makedirs(audit_dir, exist_ok=True)

        self.audit_file = os.path.join(
            audit_dir, f"execution_audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        )
        logger.info(f"Audit logging to: {self.audit_file}")

    def connect(self) -> bool:
        """Connect to MT5 and initialize systems"""
        try:
            # Connect MT5 executor
            if not self.mt5_executor.connect():
                logger.error("Failed to connect MT5 executor")
                return False

            # Connect MT5 client for live data
            if not self.mt5_client.connect():
                logger.error("Failed to connect MT5 client")
                return False

            # Start live data threads
            self.mt5_client.start()

            # Validate account
            account_info = self.mt5_executor.get_account_info()
            logger.info(
                f"Connected to MT5 account: {account_info['login']} "
                f"Balance: {account_info['balance']} {account_info['currency']}"
            )

            self.is_live = True
            return True

        except Exception as e:
            logger.error(f"MT5 connection failed: {e}")
            return False

    def disconnect(self):
        """Disconnect from MT5"""
        try:
            self.is_live = False
            self.mt5_client.stop()
            self.mt5_executor.disconnect()
            logger.info("MT5 execution harness disconnected")
        except Exception as e:
            logger.error(f"Disconnect error: {e}")

    def enable_symbol(self, symbol: str):
        """Enable trading for a symbol"""
        self.enabled_symbols.add(symbol)

        # Subscribe to live ticks for execution
        self.mt5_client.subscribe_tick(symbol, self._on_tick)
        logger.info(f"Enabled trading for {symbol}")

    def disable_symbol(self, symbol: str):
        """Disable trading for a symbol"""
        self.enabled_symbols.discard(symbol)
        logger.info(f"Disabled trading for {symbol}")

    def _on_tick(self, symbol: str, tick_data: Dict):
        """Process live tick and generate trading decisions"""
        if not self.is_live or symbol not in self.enabled_symbols:
            return

        if self.kill_switch_active:
            return

        try:
            # Get account info for position sizing
            account_info = self.mt5_executor.get_account_info()

            # Process tick through T470 pipeline
            result = t470_pipeline.process_tick_optimized(
                symbol=symbol,
                price=(tick_data["bid"] + tick_data["ask"]) / 2,
                account_balance=account_info["balance"],
                atr=0.001,  # Default ATR, should be calculated from historical data
                spread_pips=(tick_data["ask"] - tick_data["bid"]) / 0.0001,
            )

            # Check if we should execute
            if self._should_execute(result, symbol):
                self._execute_decision(symbol, result, tick_data, account_info)

        except Exception as e:
            logger.error(f"Tick processing error for {symbol}: {e}")

    def _should_execute(self, decision: Dict, symbol: str) -> bool:
        """Determine if we should execute based on decision and risk checks"""
        # Check confidence threshold
        if decision.get("confidence", 0) < self.min_confidence:
            return False

        # Check if action is actionable
        action = decision.get("decision", {}).get("action")
        if action == "FLAT":
            return False

        # Check daily trade limits
        if self.daily_trade_count >= self.max_daily_trades:
            logger.warning("Daily trade limit reached")
            return False

        # Check open position limits
        if len(self.position_tracker) >= self.max_open_positions:
            logger.warning("Maximum open positions reached")
            return False

        # Check if we already have a position in this symbol
        if symbol in self.position_tracker:
            # For now, skip if we already have a position
            # TODO: Implement position management (scaling, hedging)
            return False

        return True

    def _execute_decision(
        self, symbol: str, decision: Dict, tick_data: Dict, account_info: Dict
    ):
        """Execute trading decision with full audit trail"""
        start_time = time.perf_counter()

        try:
            # Extract decision details
            action = decision.get("decision", {}).get("action")
            confidence = decision.get("confidence", 0)
            ensemble_confidence = decision.get("ensemble_confidence", 0)
            model_scores = decision.get("model_scores", {})
            regime = decision.get("regime", {})

            # Calculate position size
            risk_amount = account_info["balance"] * self.max_risk_per_trade

            # Use enhanced position sizing
            volume = position_sizer.map_confidence_to_units(
                p_star=confidence,
                vol_bucket=regime.get("vol_bucket", "Med"),
                atr=0.001,  # Should be calculated from historical data
                expected_rr=1.2,
            )

            # Ensure minimum/maximum volume constraints
            volume = max(0.01, min(volume, 1.0))  # MT5 typical constraints

            # Create order request
            order_request = OrderRequest(
                symbol=symbol,
                action=action,
                volume=volume,
                confidence=confidence,
                model_attribution=model_scores,
                regime_state=regime.get("state", "T"),
                vol_bucket=regime.get("vol_bucket", "Med"),
                comment=f"ARIA-HF-{regime.get('state', 'T')}-{confidence:.3f}",
            )

            # Execute order
            execution_result = self._place_mt5_order(order_request, tick_data)

            # Calculate execution metrics
            latency_ms = (time.perf_counter() - start_time) * 1000
            execution_result.latency_ms = latency_ms

            # Track execution with telemetry
            if execution_result.success:
                expected_price = (tick_data["bid"] + tick_data["ask"]) / 2
                actual_price = execution_result.price
                
                trade_data = {
                    "symbol": symbol,
                    "action": action,
                    "volume": volume,
                    "pnl": 0.0,  # Will be updated when position closes
                    "confidence": confidence,
                    "regime": regime.get("state", "T"),
                    "execution_price": actual_price,
                    "expected_price": expected_price
                }
                
                telemetry_monitor.track_execution(
                    start_time=start_time,
                    end_time=time.perf_counter(),
                    expected_price=expected_price,
                    actual_price=actual_price,
                    trade_data=trade_data
                )
            else:
                # Track execution error
                telemetry_monitor.performance_monitor.track_error("execution_failed")

            # Update tracking
            if execution_result.success:
                self.position_tracker[symbol] = {
                    "order_id": execution_result.order_id,
                    "volume": execution_result.volume,
                    "entry_price": execution_result.price,
                    "action": action,
                    "timestamp": time.time(),
                    "model_attribution": model_scores,
                    "regime": regime,
                }
                self.daily_trade_count += 1

                # Update hedge fund analytics
                hedge_fund_analytics.update_trade(
                    symbol=symbol,
                    model_name="Ensemble",
                    entry_price=execution_result.price,
                    exit_price=None,  # Position opening
                    position_size=(
                        execution_result.volume
                        if action == "BUY"
                        else -execution_result.volume
                    ),
                    regime=regime.get("state", "T"),
                    timestamp=time.time(),
                    meta={
                        "confidence": confidence,
                        "ensemble_confidence": ensemble_confidence,
                    },
                )

                # Update model attribution
                hedge_fund_analytics.update_model_attribution(
                    model_contributions=model_scores,
                    ensemble_decision=confidence,
                    actual_pnl=None,  # Will be updated on position close
                )

            # Update execution statistics
            self._update_execution_stats(execution_result)

            # Audit log
            self._log_execution(order_request, execution_result, decision)

            logger.info(
                f"Executed {action} {symbol} volume={volume:.2f} "
                f"confidence={confidence:.3f} latency={latency_ms:.1f}ms "
                f"success={execution_result.success}"
            )

        except Exception as e:
            logger.error(f"Execution error for {symbol}: {e}")
            self._log_execution_error(symbol, decision, str(e))

    def _place_mt5_order(
        self, order_request: OrderRequest, tick_data: Dict
    ) -> ExecutionResult:
        """Place order through MT5 executor"""
        try:
            # Calculate stop loss and take profit
            price = (
                tick_data["bid"] if order_request.action == "SELL" else tick_data["ask"]
            )
            spread = tick_data["ask"] - tick_data["bid"]

            # Basic SL/TP calculation (should be enhanced with ATR)
            atr_estimate = 0.001  # Should be calculated from historical data
            if order_request.action == "BUY":
                sl = price - (2.0 * atr_estimate)
                tp = price + (3.0 * atr_estimate)
            else:
                sl = price + (2.0 * atr_estimate)
                tp = price - (3.0 * atr_estimate)

            # Place order through MT5
            mt5_result = self.mt5_executor.place_order(
                symbol=order_request.symbol,
                volume=order_request.volume,
                order_type=order_request.action.lower(),
                sl=sl,
                tp=tp,
                comment=order_request.comment,
            )

            # Calculate slippage
            expected_price = price
            actual_price = mt5_result.get("price", price)
            slippage = abs(actual_price - expected_price)

            return ExecutionResult(
                order_id=mt5_result.get("order", 0),
                success=mt5_result.get("retcode") == 10009,  # TRADE_RETCODE_DONE
                price=actual_price,
                volume=order_request.volume,
                spread=spread,
                slippage=slippage,
                latency_ms=0,  # Will be set by caller
                mt5_result=mt5_result,
            )

        except Exception as e:
            return ExecutionResult(
                order_id=None,
                success=False,
                price=0,
                volume=0,
                spread=0,
                slippage=0,
                latency_ms=0,
                error_message=str(e),
            )

    def _update_execution_stats(self, result: ExecutionResult):
        """Update execution statistics"""
        self.execution_stats["total_orders"] += 1

        if result.success:
            self.execution_stats["successful_orders"] += 1
        else:
            self.execution_stats["failed_orders"] += 1

        # Update rolling averages
        total = self.execution_stats["total_orders"]
        self.execution_stats["avg_slippage"] = (
            self.execution_stats["avg_slippage"] * (total - 1) + result.slippage
        ) / total
        self.execution_stats["avg_latency_ms"] = (
            self.execution_stats["avg_latency_ms"] * (total - 1) + result.latency_ms
        ) / total

    def _log_execution(
        self, order_request: OrderRequest, result: ExecutionResult, decision: Dict
    ):
        """Log execution to audit trail"""
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "execution",
            "order_request": asdict(order_request),
            "execution_result": asdict(result),
            "decision_context": decision,
            "execution_stats": self.execution_stats.copy(),
        }

        with self.audit_lock:
            with open(self.audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(audit_entry, separators=(",", ":")) + "\n")

    def _log_execution_error(self, symbol: str, decision: Dict, error: str):
        """Log execution error"""
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "execution_error",
            "symbol": symbol,
            "decision": decision,
            "error": error,
        }

        with self.audit_lock:
            with open(self.audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(audit_entry, separators=(",", ":")) + "\n")

    def check_kill_switch(self):
        """Check kill switch conditions"""
        # Check daily P&L
        if abs(self.daily_pnl) > self.max_daily_loss:
            self.activate_kill_switch(
                f"Daily loss limit exceeded: {self.daily_pnl:.2%}"
            )
            return True

        # Check error rate
        if self.execution_stats["total_orders"] > 10:
            error_rate = (
                self.execution_stats["failed_orders"]
                / self.execution_stats["total_orders"]
            )
            if error_rate > 0.20:  # 20% error rate
                self.activate_kill_switch(f"High error rate: {error_rate:.1%}")
                return True

        return False

    def activate_kill_switch(self, reason: str):
        """Activate kill switch and stop all trading"""
        self.kill_switch_active = True
        logger.error(f"KILL SWITCH ACTIVATED: {reason}")

        # Log kill switch activation
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "kill_switch_activation",
            "reason": reason,
            "daily_pnl": self.daily_pnl,
            "daily_trade_count": self.daily_trade_count,
            "execution_stats": self.execution_stats.copy(),
        }

        with self.audit_lock:
            with open(self.audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(audit_entry, separators=(",", ":")) + "\n")

    def deactivate_kill_switch(self):
        """Manually deactivate kill switch"""
        self.kill_switch_active = False
        logger.info("Kill switch deactivated manually")

    def get_status(self) -> Dict[str, Any]:
        """Get execution harness status"""
        return {
            "is_live": self.is_live,
            "kill_switch_active": self.kill_switch_active,
            "enabled_symbols": list(self.enabled_symbols),
            "open_positions": len(self.position_tracker),
            "daily_trade_count": self.daily_trade_count,
            "daily_pnl": self.daily_pnl,
            "execution_stats": self.execution_stats.copy(),
            "position_tracker": self.position_tracker.copy(),
        }

    def get_multi_asset_symbols(self) -> List[str]:
        """Get supported multi-asset symbols"""
        return [
            # Forex
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "USDCHF",
            "AUDUSD",
            "NZDUSD",
            "EURGBP",
            "EURJPY",
            "GBPJPY",
            "AUDJPY",
            # Commodities
            "XAUUSD",
            "XAGUSD",
            "USOIL",
            "UKOIL",
            # Indices
            "US30",
            "SPX500",
            "NAS100",
            "GER40",
            "UK100",
            # Crypto (if available)
            "BTCUSD",
            "ETHUSD",
            "ADAUSD",
            "DOTUSD",
        ]


# Global execution harness instance
mt5_execution_harness = MT5ExecutionHarness()

