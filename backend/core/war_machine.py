"""
ARIA War Machine - Core Signal Fusion Engine
Multi-model consensus with real-time adaptation
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
import logging

logger = logging.getLogger("WAR_MACHINE")


class WarMachine:
    """
    Core signal fusion engine.
    Combines LSTM, CNN, PPO, XGB, Bayesian signals.
    """
    
    def __init__(self):
        self.signal_weights = {
            "lstm": 0.30,
            "cnn": 0.25,
            "ppo": 0.20,
            "xgb": 0.15,
            "bayesian": 0.10
        }
        self.confidence_threshold = 0.65
        self.signal_history = []
        
        # Regime-specific performance tracking
        self.regime_performance = {
            "trend": {"wins": 0, "total": 0, "winrate": 0.5},
            "range": {"wins": 0, "total": 0, "winrate": 0.5},
            "breakout": {"wins": 0, "total": 0, "winrate": 0.5}
        }
        self.performance_tracker = {}
        
    async def fuse_signals(
        self,
        signals: Dict[str, float],
        market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fuse multiple model signals into final decision.
        
        Args:
            signals: Dict of model_name -> signal_value (-1 to 1)
            market_context: Market data including volatility, spread, volume
            
        Returns:
            Final trading signal with confidence
        """
        # Validate signals
        valid_signals = {}
        for model, signal in signals.items():
            if -1 <= signal <= 1:
                valid_signals[model] = signal
            else:
                logger.warning(f"Invalid signal from {model}: {signal}")
                
        if not valid_signals:
            return {"direction": "HOLD", "confidence": 0.0, "reason": "No valid signals"}
        
        # Calculate weighted fusion
        weighted_sum = 0
        total_weight = 0
        
        for model, signal in valid_signals.items():
            weight = self.signal_weights.get(model, 0.1)
            weighted_sum += signal * weight
            total_weight += weight
            
        # Normalize
        if total_weight > 0:
            fused_signal = weighted_sum / total_weight
        else:
            fused_signal = 0
            
        # Apply market context adjustments
        fused_signal = self._apply_market_filters(fused_signal, market_context)
        
        # Determine direction and confidence
        confidence = abs(fused_signal)
        
        if fused_signal > 0.1:
            direction = "BUY"
        elif fused_signal < -0.1:
            direction = "SELL"
        else:
            direction = "HOLD"
            
        # Boost confidence if models agree
        agreement = self._calculate_agreement(valid_signals)
        confidence = min(1.0, confidence * (1 + agreement * 0.5))
        
        result = {
            "direction": direction,
            "confidence": confidence,
            "raw_signal": fused_signal,
            "agreement": agreement,
            "models_used": len(valid_signals),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Track for adaptation
        self.signal_history.append(result)
        if len(self.signal_history) > 100:
            self.signal_history.pop(0)
            
        return result
    
    def update_regime_performance(self, regime: str, was_profitable: bool) -> None:
        """Update regime-specific performance tracking"""
        if regime in self.regime_performance:
            self.regime_performance[regime]["total"] += 1
            if was_profitable:
                self.regime_performance[regime]["wins"] += 1
            
            # Update winrate
            total = self.regime_performance[regime]["total"]
            wins = self.regime_performance[regime]["wins"]
            self.regime_performance[regime]["winrate"] = wins / total if total > 0 else 0.5
    
    def _apply_market_filters(
        self,
        signal: float,
        context: Dict[str, Any]
    ) -> float:
        """Apply market microstructure filters with regime/session adaptation"""
        
        # Reduce signal in high spread conditions
        spread_pct = context.get("spread_pct", 0)
        if spread_pct > 0.001:  # 10 pips on EURUSD
            signal *= 0.8
            
        # Boost signal in high volume
        volume_ratio = context.get("volume_ratio", 1.0)
        if volume_ratio > 1.5:
            signal *= 1.2
            
        # Reduce in low volatility (no opportunity)
        atr_pct = context.get("atr_pct", 0.001)
        if atr_pct < 0.0005:  # Very low volatility
            signal *= 0.5
            
        # Regime-specific adaptations
        regime = context.get("regime", "range")
        session = context.get("session", "medium")
        
        # In trending regimes, boost signals
        if regime == "trend":
            signal *= 1.3
        
        # In breakout regimes, be more cautious (boost only strong signals)
        elif regime == "breakout":
            if abs(signal) < 0.5:  # Weak signal in breakout
                signal *= 0.7
            else:  # Strong signal in breakout
                signal *= 1.5
                
        # In ranging regimes, reduce signals
        elif regime == "range":
            signal *= 0.8
            
        # Session-specific adaptations
        if session == "high":  # High volatility session
            signal *= 1.2
        elif session == "low":  # Low volatility session
            signal *= 0.8
            
        return np.clip(signal, -1, 1)
    
    def _calculate_agreement(self, signals: Dict[str, float]) -> float:
        """Calculate model agreement score"""
        if len(signals) < 2:
            return 0
            
        values = list(signals.values())
        
        # Count how many agree on direction
        positive = sum(1 for v in values if v > 0)
        negative = sum(1 for v in values if v < 0)
        
        # Agreement ratio
        agreement = max(positive, negative) / len(values)
        
        # Boost if strong agreement
        if agreement > 0.8:
            avg_strength = np.mean([abs(v) for v in values])
            agreement *= (1 + avg_strength)
            
        return min(1.0, agreement)
    
    async def adapt_weights(self, performance_data: Dict[str, float]):
        """
        Adapt model weights based on recent performance.
        Meta-learning component.
        """
        for model, performance in performance_data.items():
            if model in self.signal_weights:
                # Increase weight for good performers
                if performance > 0.6:
                    self.signal_weights[model] *= 1.1
                elif performance < 0.4:
                    self.signal_weights[model] *= 0.9
                    
        # Normalize weights
        total = sum(self.signal_weights.values())
        for model in self.signal_weights:
            self.signal_weights[model] /= total
            
        logger.info(f"Adapted weights: {self.signal_weights}")


class PositionSizer:
    """
    Kelly Criterion and risk-based position sizing
    """
    
    def __init__(self, kelly_fraction: float = 0.25, max_risk_pct: float = 0.02):
        self.kelly_fraction = kelly_fraction
        self.max_risk_pct = max_risk_pct
        self.trade_history = []
        
    def calculate_size(
        self,
        confidence: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        account_balance: float,
        volatility: float
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.
        
        Returns:
            Position size in lots
        """
        if avg_loss == 0 or confidence < 0.5:
            return 0.01  # Minimum size
            
        # Kelly formula: f = (p*b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        b = avg_win / abs(avg_loss)
        p = win_rate * confidence  # Adjust by confidence
        q = 1 - p
        
        kelly = (p * b - q) / b if b > 0 else 0
        
        # Apply Kelly fraction (conservative)
        kelly = kelly * self.kelly_fraction
        
        # Cap at max risk
        kelly = min(kelly, self.max_risk_pct)
        
        # Adjust for volatility
        if volatility > 0.002:  # High volatility
            kelly *= 0.7
            
        # Convert to position size
        position_value = account_balance * kelly
        
        # Standard lot calculation (100k units)
        lots = position_value / 100000
        
        # Ensure minimum and maximum
        lots = max(0.01, min(lots, 10.0))
        
        return round(lots, 2)


class ExecutionEngine:
    """
    Smart order execution with slippage control
    """
    
    def __init__(self):
        self.max_slippage = 2  # pips
        self.execution_stats = {
            "total": 0,
            "successful": 0,
            "rejected": 0,
            "avg_slippage": 0
        }
        
    async def smart_execute(
        self,
        mt5_executor,
        signal: Dict[str, Any],
        size: float
    ) -> Dict[str, Any]:
        """
        Execute with institutional-grade logic.
        
        - Check spreads
        - Time the entry
        - Set dynamic SL/TP
        - Monitor slippage
        """
        symbol = signal.get("symbol", "EURUSD")
        direction = signal["direction"]
        
        # Get current market snapshot
        market = await mt5_executor.get_tick(symbol)
        
        # Check spread
        spread = market["ask"] - market["bid"]
        if spread > 0.0002:  # 2 pips
            logger.warning(f"High spread: {spread}")
            if signal["confidence"] < 0.8:
                return {"status": "rejected", "reason": "high_spread"}
                
        # Calculate dynamic SL/TP based on ATR
        atr = market.get("atr", 0.0010)
        sl_distance = atr * 1.5
        tp_distance = atr * 2.5
        
        # Adjust for signal strength
        if signal["confidence"] > 0.8:
            tp_distance *= 1.5
            
        # Execute
        result = await mt5_executor.place_order(
            symbol=symbol,
            order_type=direction,
            volume=size,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            magic=8100,  # ARIA magic number
            comment="ARIA_WAR"
        )
        
        # Track execution stats
        self.execution_stats["total"] += 1
        if result.get("status") == "success":
            self.execution_stats["successful"] += 1
            
            # Calculate slippage
            if direction == "BUY":
                expected_price = market["ask"]
                actual_price = result.get("price", expected_price)
            else:
                expected_price = market["bid"]
                actual_price = result.get("price", expected_price)
                
            slippage = abs(actual_price - expected_price)
            
            # Update average slippage
            n = self.execution_stats["successful"]
            avg = self.execution_stats["avg_slippage"]
            self.execution_stats["avg_slippage"] = (avg * (n-1) + slippage) / n
            
            logger.info(f"Executed: {result['ticket']} | Slippage: {slippage:.5f}")
        else:
            self.execution_stats["rejected"] += 1
            logger.error(f"Execution failed: {result}")
            
        return result
