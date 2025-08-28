# orchestrator/orchestrator_cpu_friendly.py
"""
CPU-friendly orchestrator for trading signals.
Minimal dependencies: no heavy ML frameworks, no ONNX, no tensorflow.
- Light features from models/light_features.py
- Bandit for strategy selection
- Economic gating
- Simple trade intent builder
"""
import json
import time
import asyncio
from pathlib import Path
import pandas as pd
import numpy as np
import logging
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.light_features import build_features, predict_from_df, train_small_model
from models.bandit_selector import ThompsonBandit
from tools.strategy_accounting import StrategyAccounting
from tools.coach import explain_trade, explain_risk_rejection

logging.basicConfig(level=logging.INFO)

class CPUFriendlyOrchestrator:
    def __init__(self):
        self.bandit = ThompsonBandit(n_strategies=3)  # 3 strategies: trend, mean-revert, breakout
        self.strategy_names = ["trend_follow", "mean_revert", "breakout"]
        self.last_signal_time = 0
        self.min_signal_interval = 60  # seconds between signals
        self.accounting = StrategyAccounting()

    def build_intent(self, signal_prob: float, df: pd.DataFrame, strategy_id: int):
        """Build trade intent from signal and market data"""
        # Determine direction
        side = "buy" if signal_prob > 0.55 else ("sell" if signal_prob < 0.45 else "hold")
        if side == "hold":
            return None
        
        # Calculate position size using simplified Kelly
        confidence = abs(signal_prob - 0.5) * 2  # Map to 0-1 confidence
        kelly_fraction = confidence * 0.1  # Max 10% of capital
        
        # Risk management: scale by volatility
        recent_volatility = df['close'].pct_change().std() * np.sqrt(252)
        vol_scalar = 1.0 / (1.0 + recent_volatility * 10)  # Reduce size in high vol
        
        volume = round(0.01 * kelly_fraction * vol_scalar, 2)
        volume = max(0.01, min(0.10, volume))  # Clamp between 0.01 and 0.10 lots
        
        price = float(df['close'].iloc[-1])
        
        intent = {
            "type": "trade_intent",
            "idempotency_key": f"intent-{int(time.time()*1000)}",
            "symbol": "EURUSD",
            "action": side,
            "volume": volume,
            "price": price,
            "slippage": 5,
            "comment": f"aria_cpu_{self.strategy_names[strategy_id]}",
            "model_id": "light_sgd_v1",
            "model_version": "1.0",
            "edge": float(signal_prob),
            "confidence": confidence,
            "strategy": self.strategy_names[strategy_id],
            "timestamp": time.time()
        }
        
        return intent
    
    async def process_market_data(self, df: pd.DataFrame):
        """Process market data and generate trade decision"""
        
        # Check minimum time between signals
        current_time = time.time()
        if current_time - self.last_signal_time < self.min_signal_interval:
            return None
        
        # Get base signal from light model
        signal_prob = light_features.predict_signal(df)
        
        # Select strategy using bandit
        strategy_id = self.bandit.select()
        
        # Apply strategy-specific adjustments
        adjusted_prob = self._apply_strategy(signal_prob, df, strategy_id)
        
        # Build intent
        intent = self.build_intent(adjusted_prob, df, strategy_id)
        
        if not intent:
            return None
        
        # Apply economic gating
        gate_result = self.accounting.should_gate_trade(intent)
        if not gate_result['allow']:
            logging.info(f"Trade gated: {gate_result['reason']}")
            explanation = explain_risk_rejection(gate_result['reason'], intent)
            return {'status': 'gated', 'reason': gate_result['reason'], 'explanation': explanation}
        
        # Check expected value gating
        metrics = self.accounting.get_metrics()
        if metrics['expected_value'] < 0 and metrics['total_trades'] > 20:
            reason = f"Negative EV: {metrics['expected_value']:.4f}"
            print(explain_risk_rejection(reason, intent))
            return None
        
        self.last_signal_time = current_time
        print(explain_trade(intent))
        
        return intent
    
    def generate_signal(self, market_data: pd.DataFrame, symbol: str) -> dict:
        """Generate trading signal from market data using lightweight indicators"""
        try:
            # Use light features for signal
            signal_prob = predict_from_df(market_data)
            
            # Determine action based on probability
            if signal_prob > 0.6:
                action = 'buy'
                confidence = signal_prob
            elif signal_prob < 0.4:
                action = 'sell'
                confidence = 1.0 - signal_prob
            else:
                return {'action': 'hold', 'confidence': 0.0}
            
            # Calculate simple entry, SL, TP
            current_price = market_data['close'].iloc[-1]
            atr = self._calculate_atr(market_data)[-1] if len(market_data) > 14 else current_price * 0.001
            
            if action == 'buy':
                entry = current_price
                stop_loss = entry - (2 * atr)
                take_profit = entry + (3 * atr)
                reason = f"Bullish signal: probability={signal_prob:.2f}"
            else:
                entry = current_price
                stop_loss = entry + (2 * atr)
                take_profit = entry - (3 * atr)
                reason = f"Bearish signal: probability={signal_prob:.2f}"
            
            return {
                'action': action,
                'confidence': confidence,
                'size': 0.01,  # Fixed small size
                'entry': entry,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'reason': reason,
                'risk_reward': abs((take_profit - entry) / (entry - stop_loss)),
                'patterns': [],
                'analysis': {'signal_prob': signal_prob}
            }
        except Exception as e:
            logging.error(f"Error generating signal: {e}")
            return {'action': 'hold', 'confidence': 0.0}
    
    def _apply_strategy(self, base_prob: float, df: pd.DataFrame, strategy_id: int) -> float:
        """Apply strategy-specific signal adjustments"""
        
        if strategy_id == 0:  # Trend following
            # Enhance signal if aligned with trend
            ema_short = df['close'].ewm(span=8).mean().iloc[-1]
            ema_long = df['close'].ewm(span=21).mean().iloc[-1]
            
            if ema_short > ema_long and base_prob > 0.5:
                return min(0.9, base_prob * 1.2)
            elif ema_short < ema_long and base_prob < 0.5:
                return max(0.1, base_prob * 0.8)
                
        elif strategy_id == 1:  # Mean reversion
            # Enhance signal if oversold/overbought
            rsi = self._calculate_rsi(df['close'])
            
            if rsi < 30 and base_prob > 0.5:  # Oversold, enhance buy
                return min(0.9, base_prob * 1.15)
            elif rsi > 70 and base_prob < 0.5:  # Overbought, enhance sell
                return max(0.1, base_prob * 0.85)
                
        elif strategy_id == 2:  # Breakout
            # Enhance signal on volatility expansion
            atr = self._calculate_atr(df)
            atr_ma = pd.Series(atr).rolling(20).mean().iloc[-1]
            
            if atr[-1] > atr_ma * 1.5:  # Volatility expansion
                # Strengthen directional signal
                if base_prob > 0.5:
                    return min(0.9, base_prob * 1.25)
                else:
                    return max(0.1, base_prob * 0.75)
        
        return base_prob
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> np.ndarray:
        """Calculate ATR"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.maximum(high - low, 
                        np.maximum(abs(high - np.roll(close, 1)),
                                  abs(low - np.roll(close, 1))))
        
        atr = np.convolve(tr, np.ones(period)/period, mode='valid')
        return atr
    
    def update_bandit(self, strategy_id: int, pnl: float):
        """Update bandit with trade result"""
        # Normalize PnL to [-1, 1] range
        normalized_reward = np.tanh(pnl * 100)  # Scale and squash
        self.bandit.update(strategy_id, normalized_reward)
        
        # Log the update
        print(f"Updated bandit strategy {strategy_id} ({self.strategy_names[strategy_id]}) with reward {normalized_reward:.3f}")

# Global instance
orchestrator = CPUFriendlyOrchestrator()

if __name__ == "__main__":
    # Test with dummy data
    dates = pd.date_range('2024-01-01', periods=100, freq='1min')
    df = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    async def test():
        intent = await orchestrator.process_market_data(df)
        if intent:
            print(f"Generated intent: {intent}")
    
    asyncio.run(test())
