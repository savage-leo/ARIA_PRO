"""
Microstructure Alpha Engine
Extracts alpha signals from market microstructure: spread dynamics, tick data, order flow
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class MicrostructureSignals:
    """Container for microstructure alpha signals"""
    spread_ratio: float  # Current spread / average spread
    spread_volatility: float  # Spread standard deviation
    tick_momentum: float  # Tick-by-tick price momentum
    volume_imbalance: float  # Buy vs sell volume imbalance
    price_acceleration: float  # Second derivative of price
    execution_edge: float  # Expected execution advantage
    flow_toxicity: float  # Adverse selection / toxic flow indicator
    liquidity_score: float  # Current liquidity availability
    market_impact: float  # Expected price impact of trade
    confidence: float  # Overall signal confidence


class SpreadAnalyzer:
    """Analyze bid-ask spread dynamics"""
    
    def __init__(self, window_size: int = 100):
        self.spread_history = deque(maxlen=window_size)
        self.spread_ema = None
        self.spread_ema_alpha = 0.1
        
    def update(self, bid: float, ask: float):
        """Update with new bid/ask"""
        spread = ask - bid
        self.spread_history.append(spread)
        
        # Update EMA
        if self.spread_ema is None:
            self.spread_ema = spread
        else:
            self.spread_ema = self.spread_ema_alpha * spread + (1 - self.spread_ema_alpha) * self.spread_ema
            
    def get_signals(self) -> Dict[str, float]:
        """Get spread-based signals"""
        if len(self.spread_history) < 10:
            return {
                'spread_ratio': 1.0,
                'spread_volatility': 0.0,
                'spread_trend': 0.0
            }
            
        current_spread = self.spread_history[-1] if self.spread_history else 0
        avg_spread = np.mean(self.spread_history)
        std_spread = np.std(self.spread_history)
        
        # Spread ratio (lower is better for execution)
        spread_ratio = current_spread / (avg_spread + 1e-9)
        
        # Spread volatility (normalized)
        spread_volatility = std_spread / (avg_spread + 1e-9)
        
        # Spread trend (widening or tightening)
        if len(self.spread_history) >= 20:
            recent = list(self.spread_history)[-10:]
            older = list(self.spread_history)[-20:-10]
            spread_trend = (np.mean(recent) - np.mean(older)) / (np.mean(older) + 1e-9)
        else:
            spread_trend = 0.0
            
        return {
            'spread_ratio': spread_ratio,
            'spread_volatility': spread_volatility,
            'spread_trend': spread_trend
        }


class TickProcessor:
    """Process tick-level price data"""
    
    def __init__(self, window_size: int = 500):
        self.price_ticks = deque(maxlen=window_size)
        self.volume_ticks = deque(maxlen=window_size)
        self.time_ticks = deque(maxlen=window_size)
        
    def add_tick(self, price: float, volume: float, timestamp: float):
        """Add new tick"""
        self.price_ticks.append(price)
        self.volume_ticks.append(volume)
        self.time_ticks.append(timestamp)
        
    def get_tick_momentum(self) -> float:
        """Calculate tick-by-tick momentum"""
        if len(self.price_ticks) < 10:
            return 0.0
            
        # Calculate price changes weighted by volume
        momentum = 0.0
        total_volume = 0.0
        
        prices = list(self.price_ticks)
        volumes = list(self.volume_ticks)
        
        for i in range(1, min(20, len(prices))):
            price_change = (prices[-i] - prices[-i-1]) / (prices[-i-1] + 1e-9)
            vol = volumes[-i]
            momentum += price_change * vol
            total_volume += vol
            
        if total_volume > 0:
            momentum = momentum / total_volume
            
        return momentum * 100  # Scale to percentage
        
    def get_price_acceleration(self) -> float:
        """Calculate second derivative of price (acceleration)"""
        if len(self.price_ticks) < 3:
            return 0.0
            
        prices = list(self.price_ticks)[-10:]
        if len(prices) < 3:
            return 0.0
            
        # Calculate first differences (velocity)
        velocities = []
        for i in range(1, len(prices)):
            vel = prices[i] - prices[i-1]
            velocities.append(vel)
            
        # Calculate second differences (acceleration)
        accelerations = []
        for i in range(1, len(velocities)):
            acc = velocities[i] - velocities[i-1]
            accelerations.append(acc)
            
        if accelerations:
            # Normalize by price level
            avg_price = np.mean(prices)
            acceleration = np.mean(accelerations) / (avg_price + 1e-9)
            return acceleration * 10000  # Scale for readability
            
        return 0.0
        
    def get_tick_imbalance(self) -> float:
        """Calculate uptick vs downtick imbalance"""
        if len(self.price_ticks) < 10:
            return 0.0
            
        upticks = 0
        downticks = 0
        unchanged = 0
        
        prices = list(self.price_ticks)
        volumes = list(self.volume_ticks)
        
        for i in range(1, min(50, len(prices))):
            if prices[i] > prices[i-1]:
                upticks += volumes[i]
            elif prices[i] < prices[i-1]:
                downticks += volumes[i]
            else:
                unchanged += volumes[i]
                
        total = upticks + downticks + unchanged
        if total > 0:
            imbalance = (upticks - downticks) / total
            return imbalance
            
        return 0.0


class OrderFlowAnalyzer:
    """Analyze order flow and market impact"""
    
    def __init__(self):
        self.buy_volume = deque(maxlen=100)
        self.sell_volume = deque(maxlen=100)
        self.trade_sizes = deque(maxlen=100)
        self.trade_prices = deque(maxlen=100)
        
    def add_trade(self, price: float, volume: float, side: str):
        """Add executed trade"""
        if side.lower() == 'buy':
            self.buy_volume.append(volume)
            self.sell_volume.append(0)
        else:
            self.buy_volume.append(0)
            self.sell_volume.append(volume)
            
        self.trade_sizes.append(volume)
        self.trade_prices.append(price)
        
    def get_volume_imbalance(self) -> float:
        """Calculate buy/sell volume imbalance"""
        if not self.buy_volume or not self.sell_volume:
            return 0.0
            
        recent_buys = sum(list(self.buy_volume)[-20:])
        recent_sells = sum(list(self.sell_volume)[-20:])
        
        total = recent_buys + recent_sells
        if total > 0:
            imbalance = (recent_buys - recent_sells) / total
            return imbalance
            
        return 0.0
        
    def get_flow_toxicity(self) -> float:
        """
        Calculate flow toxicity (VPIN-inspired)
        High toxicity indicates informed/adverse flow
        """
        if len(self.trade_sizes) < 20:
            return 0.0
            
        # Calculate volume buckets
        volumes = list(self.trade_sizes)[-50:]
        prices = list(self.trade_prices)[-50:]
        
        if len(volumes) < 2:
            return 0.0
            
        # Calculate price volatility during high volume periods
        high_vol_threshold = np.percentile(volumes, 75)
        
        high_vol_returns = []
        for i in range(1, len(volumes)):
            if volumes[i] > high_vol_threshold and prices[i-1] > 0:
                ret = abs(prices[i] - prices[i-1]) / prices[i-1]
                high_vol_returns.append(ret)
                
        if high_vol_returns:
            # High volatility during high volume = toxic flow
            toxicity = np.mean(high_vol_returns) * 100
            return min(toxicity, 1.0)  # Cap at 1.0
            
        return 0.0
        
    def estimate_market_impact(self, order_size: float) -> float:
        """Estimate market impact of an order"""
        if not self.trade_sizes or not self.trade_prices:
            return 0.0
            
        avg_trade_size = np.mean(self.trade_sizes)
        
        if avg_trade_size > 0:
            # Relative order size
            relative_size = order_size / avg_trade_size
            
            # Square-root market impact model
            # Impact = k * sqrt(order_size / avg_daily_volume)
            impact_bps = 10 * np.sqrt(relative_size)  # 10 bps for 1x average size
            
            return impact_bps / 10000  # Convert to decimal
            
        return 0.0


class LiquidityDetector:
    """Detect liquidity conditions"""
    
    def __init__(self):
        self.spread_analyzer = SpreadAnalyzer()
        self.recent_volumes = deque(maxlen=100)
        self.recent_trade_counts = deque(maxlen=100)
        
    def update(self, bid: float, ask: float, volume: float, trade_count: int):
        """Update liquidity metrics"""
        self.spread_analyzer.update(bid, ask)
        self.recent_volumes.append(volume)
        self.recent_trade_counts.append(trade_count)
        
    def get_liquidity_score(self) -> float:
        """
        Calculate overall liquidity score (0-1)
        Higher score = better liquidity
        """
        scores = []
        
        # Spread tightness component
        spread_signals = self.spread_analyzer.get_signals()
        spread_score = 1.0 / (1.0 + spread_signals['spread_ratio'])
        scores.append(spread_score)
        
        # Volume component
        if self.recent_volumes:
            current_vol = self.recent_volumes[-1] if self.recent_volumes else 0
            avg_vol = np.mean(self.recent_volumes)
            if avg_vol > 0:
                vol_score = min(current_vol / avg_vol, 2.0) / 2.0
                scores.append(vol_score)
                
        # Trade frequency component
        if self.recent_trade_counts:
            current_trades = self.recent_trade_counts[-1] if self.recent_trade_counts else 0
            avg_trades = np.mean(self.recent_trade_counts)
            if avg_trades > 0:
                trade_score = min(current_trades / avg_trades, 2.0) / 2.0
                scores.append(trade_score)
                
        if scores:
            return np.mean(scores)
            
        return 0.5  # Neutral if no data


class MicrostructureAlpha:
    """Main microstructure alpha extraction engine"""
    
    def __init__(self):
        self.spread_analyzer = SpreadAnalyzer()
        self.tick_processor = TickProcessor()
        self.flow_analyzer = OrderFlowAnalyzer()
        self.liquidity_detector = LiquidityDetector()
        
        # Cache for symbol-specific data
        self.symbol_cache = {}
        
    def update_market_data(self, symbol: str, data: Dict):
        """Update with latest market data"""
        # Initialize symbol cache if needed
        if symbol not in self.symbol_cache:
            self.symbol_cache[symbol] = {
                'spread_analyzer': SpreadAnalyzer(),
                'tick_processor': TickProcessor(),
                'flow_analyzer': OrderFlowAnalyzer(),
                'liquidity_detector': LiquidityDetector()
            }
            
        cache = self.symbol_cache[symbol]
        
        # Update spread analyzer
        bid = data.get('bid', 0)
        ask = data.get('ask', 0)
        if bid > 0 and ask > 0:
            cache['spread_analyzer'].update(bid, ask)
            
        # Update tick processor
        price = data.get('price', 0)
        volume = data.get('volume', 0)
        timestamp = data.get('timestamp', time.time())
        if price > 0:
            cache['tick_processor'].add_tick(price, volume, timestamp)
            
        # Update flow analyzer (if trade data available)
        if 'side' in data:
            cache['flow_analyzer'].add_trade(price, volume, data['side'])
            
        # Update liquidity detector
        trade_count = data.get('trade_count', 1)
        cache['liquidity_detector'].update(bid, ask, volume, trade_count)
        
    async def extract_alpha(self, symbol: str, market_data: Dict = None) -> MicrostructureSignals:
        """Extract microstructure alpha signals for a symbol"""
        
        # Update with latest data if provided
        if market_data:
            self.update_market_data(symbol, market_data)
            
        # Get symbol-specific analyzers
        if symbol not in self.symbol_cache:
            # Return neutral signals if no data
            return MicrostructureSignals(
                spread_ratio=1.0,
                spread_volatility=0.0,
                tick_momentum=0.0,
                volume_imbalance=0.0,
                price_acceleration=0.0,
                execution_edge=0.0,
                flow_toxicity=0.0,
                liquidity_score=0.5,
                market_impact=0.0,
                confidence=0.0
            )
            
        cache = self.symbol_cache[symbol]
        
        # Extract signals from each component
        spread_signals = cache['spread_analyzer'].get_signals()
        tick_momentum = cache['tick_processor'].get_tick_momentum()
        price_acceleration = cache['tick_processor'].get_price_acceleration()
        tick_imbalance = cache['tick_processor'].get_tick_imbalance()
        volume_imbalance = cache['flow_analyzer'].get_volume_imbalance()
        flow_toxicity = cache['flow_analyzer'].get_flow_toxicity()
        liquidity_score = cache['liquidity_detector'].get_liquidity_score()
        
        # Calculate execution edge
        # Better execution when: spread is tight, liquidity is good, toxicity is low
        execution_edge = (
            (1.0 - spread_signals['spread_ratio']) * 0.4 +
            liquidity_score * 0.4 +
            (1.0 - flow_toxicity) * 0.2
        )
        
        # Estimate market impact (placeholder - needs order size)
        market_impact = cache['flow_analyzer'].estimate_market_impact(10000)  # Assume standard size
        
        # Calculate overall confidence
        # Higher confidence when we have more data and consistent signals
        data_quality = min(len(cache['tick_processor'].price_ticks) / 100, 1.0)
        signal_consistency = 1.0 - np.std([
            abs(tick_momentum) / 10,
            abs(volume_imbalance),
            abs(tick_imbalance)
        ])
        confidence = data_quality * 0.5 + signal_consistency * 0.5
        
        return MicrostructureSignals(
            spread_ratio=spread_signals['spread_ratio'],
            spread_volatility=spread_signals['spread_volatility'],
            tick_momentum=tick_momentum,
            volume_imbalance=volume_imbalance,
            price_acceleration=price_acceleration,
            execution_edge=execution_edge,
            flow_toxicity=flow_toxicity,
            liquidity_score=liquidity_score,
            market_impact=market_impact,
            confidence=confidence
        )
        
    def get_execution_timing(self, symbol: str) -> Dict[str, Any]:
        """Get optimal execution timing signals"""
        if symbol not in self.symbol_cache:
            return {
                'execute_now': False,
                'wait_ticks': 0,
                'reason': 'no_data'
            }
            
        cache = self.symbol_cache[symbol]
        
        # Get current conditions
        spread_signals = cache['spread_analyzer'].get_signals()
        liquidity = cache['liquidity_detector'].get_liquidity_score()
        toxicity = cache['flow_analyzer'].get_flow_toxicity()
        
        # Decision logic
        if spread_signals['spread_ratio'] < 0.8 and liquidity > 0.7:
            # Excellent conditions
            return {
                'execute_now': True,
                'wait_ticks': 0,
                'reason': 'optimal_conditions'
            }
        elif spread_signals['spread_trend'] < 0 and liquidity > 0.5:
            # Improving conditions
            return {
                'execute_now': True,
                'wait_ticks': 0,
                'reason': 'improving_conditions'
            }
        elif toxicity > 0.7:
            # High toxicity - wait
            return {
                'execute_now': False,
                'wait_ticks': 10,
                'reason': 'high_toxicity'
            }
        elif spread_signals['spread_ratio'] > 1.5:
            # Wide spread - wait
            return {
                'execute_now': False,
                'wait_ticks': 5,
                'reason': 'wide_spread'
            }
        else:
            # Neutral conditions
            return {
                'execute_now': True,
                'wait_ticks': 0,
                'reason': 'neutral_conditions'
            }
            
    def calculate_smart_stops(self, symbol: str, entry_price: float, 
                            direction: str, atr: float) -> Dict[str, float]:
        """Calculate microstructure-aware stop loss and take profit"""
        
        base_sl_distance = atr * 1.5
        base_tp_distance = atr * 2.5
        
        if symbol in self.symbol_cache:
            cache = self.symbol_cache[symbol]
            
            # Adjust based on liquidity
            liquidity = cache['liquidity_detector'].get_liquidity_score()
            
            # Tighter stops in good liquidity
            if liquidity > 0.7:
                sl_multiplier = 0.9
                tp_multiplier = 1.1
            elif liquidity < 0.3:
                sl_multiplier = 1.2
                tp_multiplier = 0.9
            else:
                sl_multiplier = 1.0
                tp_multiplier = 1.0
                
            # Adjust based on flow toxicity
            toxicity = cache['flow_analyzer'].get_flow_toxicity()
            if toxicity > 0.5:
                # Wider stops in toxic flow
                sl_multiplier *= 1.1
                
            # Adjust based on spread
            spread_signals = cache['spread_analyzer'].get_signals()
            if spread_signals['spread_volatility'] > 0.5:
                # Wider stops in volatile spreads
                sl_multiplier *= 1.1
                
            base_sl_distance *= sl_multiplier
            base_tp_distance *= tp_multiplier
            
        if direction == 'long':
            stop_loss = entry_price - base_sl_distance
            take_profit = entry_price + base_tp_distance
        else:
            stop_loss = entry_price + base_sl_distance
            take_profit = entry_price - base_tp_distance
            
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'sl_distance': base_sl_distance,
            'tp_distance': base_tp_distance
        }


# Global instance
_microstructure_alpha = None


def get_microstructure_alpha() -> MicrostructureAlpha:
    """Get or create the global microstructure alpha engine"""
    global _microstructure_alpha
    if _microstructure_alpha is None:
        _microstructure_alpha = MicrostructureAlpha()
    return _microstructure_alpha
