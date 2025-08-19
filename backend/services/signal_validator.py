# -*- coding: utf-8 -*-
"""
Signal Validator: Real-time validation of signal generation and execution flow
Monitors for errors and ensures perfect signal generation
"""
from __future__ import annotations
import time, logging, threading
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import deque
import json

from backend.services.trading_pipeline_enhanced import trading_pipeline
from backend.core.regime_online import regime_manager
from backend.core.risk_budget_enhanced import position_sizer

logger = logging.getLogger(__name__)


@dataclass
class SignalHealth:
    """Signal health metrics"""

    total_signals: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    avg_latency_ms: float = 0.0
    last_signal_time: float = 0.0
    error_rate: float = 0.0
    kill_switch_active: bool = False


class SignalValidator:
    """Real-time signal validation and monitoring"""

    def __init__(self):
        self.health = SignalHealth()
        self.recent_signals = deque(maxlen=1000)
        self.recent_errors = deque(maxlen=100)
        self.lock = threading.Lock()

        # Test symbols for validation
        self.test_symbols = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"]
        self.test_prices = {
            "EURUSD": 1.1000,
            "GBPUSD": 1.3000,
            "USDJPY": 150.00,
            "USDCHF": 0.9000,
        }

    def validate_signal_generation(
        self, symbol: str, iterations: int = 10
    ) -> Dict[str, Any]:
        """Validate signal generation for a symbol"""
        results = []
        errors = []
        latencies = []

        base_price = self.test_prices.get(symbol, 1.0000)

        for i in range(iterations):
            try:
                # Simulate price movement
                price = base_price + (i * 0.0001)

                start_time = time.perf_counter()

                # Generate signal through full pipeline
                result = trading_pipeline.process_tick(
                    symbol=symbol,
                    price=price,
                    timestamp=time.time(),
                    account_balance=100000.0,
                    atr=0.001,
                    spread_pips=1.0,
                )

                latency = (time.perf_counter() - start_time) * 1000
                latencies.append(latency)

                # Validate result structure
                self._validate_result_structure(result)

                results.append(
                    {
                        "iteration": i,
                        "success": True,
                        "latency_ms": latency,
                        "action": result["decision"]["action"],
                        "confidence": result["decision"]["p_star"],
                        "regime": f"{result['regime']['state']}/{result['regime']['vol_bucket']}",
                        "position_size": result["sizing"].get("position_size", 0),
                    }
                )

                with self.lock:
                    self.health.successful_signals += 1

            except Exception as e:
                error_msg = f"Signal generation failed for {symbol} iteration {i}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

                results.append(
                    {"iteration": i, "success": False, "error": str(e), "latency_ms": 0}
                )

                with self.lock:
                    self.health.failed_signals += 1
                    self.recent_errors.append(
                        {"timestamp": time.time(), "symbol": symbol, "error": str(e)}
                    )

        # Calculate metrics
        success_rate = sum(1 for r in results if r["success"]) / len(results)
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        with self.lock:
            self.health.total_signals += iterations
            self.health.avg_latency_ms = avg_latency
            self.health.last_signal_time = time.time()
            self.health.error_rate = len(errors) / iterations

        return {
            "symbol": symbol,
            "iterations": iterations,
            "success_rate": success_rate,
            "avg_latency_ms": avg_latency,
            "max_latency_ms": max(latencies) if latencies else 0,
            "min_latency_ms": min(latencies) if latencies else 0,
            "errors": errors,
            "results": results[-5:],  # Last 5 results
            "timestamp": time.time(),
        }

    def _validate_result_structure(self, result: Dict[str, Any]) -> None:
        """Validate that result has required structure"""
        required_keys = ["symbol", "regime", "models", "decision", "sizing", "latency"]
        for key in required_keys:
            if key not in result:
                raise ValueError(f"Missing required key: {key}")

        # Validate regime
        regime = result["regime"]
        if regime["state"] not in ["T", "R", "B"]:
            raise ValueError(f"Invalid regime state: {regime['state']}")
        if regime["vol_bucket"] not in ["Low", "Med", "High"]:
            raise ValueError(f"Invalid vol bucket: {regime['vol_bucket']}")

        # Validate decision
        decision = result["decision"]
        if decision["action"] not in ["LONG", "SHORT", "FLAT"]:
            raise ValueError(f"Invalid action: {decision['action']}")
        if not (0 <= decision["p_star"] <= 1):
            raise ValueError(f"Invalid p_star: {decision['p_star']}")

        # Validate sizing
        sizing = result["sizing"]
        if "position_size" not in sizing:
            raise ValueError("Missing position_size in sizing")

    def monitor_system_health(self) -> Dict[str, Any]:
        """Monitor overall system health"""
        # Check kill switch
        kill_switch, kill_reason = position_sizer.check_kill_switch()

        # Get regime states
        regime_states = regime_manager.get_all_states()

        # Check for recent errors
        recent_error_count = len(
            [e for e in self.recent_errors if time.time() - e["timestamp"] < 300]
        )  # Last 5 minutes

        with self.lock:
            self.health.kill_switch_active = kill_switch

            health_status = "HEALTHY"
            if kill_switch:
                health_status = "KILL_SWITCH_ACTIVE"
            elif self.health.error_rate > 0.1:
                health_status = "HIGH_ERROR_RATE"
            elif recent_error_count > 10:
                health_status = "RECENT_ERRORS"
            elif self.health.avg_latency_ms > 100:
                health_status = "HIGH_LATENCY"

        return {
            "timestamp": time.time(),
            "status": health_status,
            "kill_switch": {"active": kill_switch, "reason": kill_reason},
            "signal_health": {
                "total_signals": self.health.total_signals,
                "success_rate": (
                    self.health.successful_signals / max(1, self.health.total_signals)
                ),
                "error_rate": self.health.error_rate,
                "avg_latency_ms": self.health.avg_latency_ms,
                "last_signal_age_s": time.time() - self.health.last_signal_time,
            },
            "regime_health": {
                "symbols_tracked": len(regime_states),
                "regime_distribution": {
                    state: len(
                        [s for s in regime_states.values() if s["state"] == state]
                    )
                    for state in ["T", "R", "B"]
                },
            },
            "recent_errors": recent_error_count,
            "error_details": list(self.recent_errors)[-5:],  # Last 5 errors
        }

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive signal validation across all test symbols"""
        logger.info("Starting comprehensive signal validation...")

        all_results = {}
        overall_errors = []

        for symbol in self.test_symbols:
            try:
                result = self.validate_signal_generation(symbol, iterations=20)
                all_results[symbol] = result

                if result["success_rate"] < 0.95:
                    overall_errors.append(
                        f"{symbol}: Low success rate {result['success_rate']:.2%}"
                    )

                if result["avg_latency_ms"] > 50:
                    overall_errors.append(
                        f"{symbol}: High latency {result['avg_latency_ms']:.1f}ms"
                    )

            except Exception as e:
                error_msg = f"{symbol}: Validation failed - {e}"
                overall_errors.append(error_msg)
                logger.error(error_msg)

        # Overall assessment
        total_success_rate = sum(r["success_rate"] for r in all_results.values()) / len(
            all_results
        )
        avg_latency = sum(r["avg_latency_ms"] for r in all_results.values()) / len(
            all_results
        )

        system_health = self.monitor_system_health()

        assessment = {
            "timestamp": time.time(),
            "overall_success_rate": total_success_rate,
            "overall_avg_latency_ms": avg_latency,
            "symbols_tested": len(self.test_symbols),
            "symbols_passed": len(
                [r for r in all_results.values() if r["success_rate"] >= 0.95]
            ),
            "critical_errors": overall_errors,
            "system_health": system_health,
            "detailed_results": all_results,
            "status": (
                "PASS"
                if total_success_rate >= 0.95 and len(overall_errors) == 0
                else "FAIL"
            ),
        }

        logger.info(
            f"Comprehensive test completed: {assessment['status']} "
            f"(Success: {total_success_rate:.2%}, Latency: {avg_latency:.1f}ms)"
        )

        return assessment


# Global instance
signal_validator = SignalValidator()
