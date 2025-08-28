# backend/signal_processor.py
"""
CPU-friendly signal processing module
Combines multiple lightweight indicators for signal generation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class Signal:
    """Trading signal data"""
    direction: str  # buy/sell/neutral
    strength: float  # 0-1 confidence
    entry_price: float
    stop_loss: float
    take_profit: float
    reason: str
    indicators: Dict[str, float]

class SignalProcessor:
    """Lightweight signal processing engine"""
    
    def __init__(self):
        self.min_confidence = 0.55
        self.indicator_weights = {
            'trend': 0.3,
            'momentum': 0.25,
            'volatility': 0.2,
            'volume': 0.15,
            'pattern': 0.1
        }
    
    def process(self, df: pd.DataFrame) -> Optional[Signal]:
        """Process market data and generate signal"""
        if len(df) < 50:
            return None
        
        # Calculate all indicators
        indicators = self._calculate_indicators(df)
        
        # Combine signals
        signal_scores = self._combine_signals(indicators)
        
        # Generate final signal
        return self._generate_signal(signal_scores, df, indicators)
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate all technical indicators"""
        indicators = {}
        
        # Trend indicators
        indicators['ema_trend'] = self._ema_crossover(df)
        indicators['adx'] = self._calculate_adx(df)
        
        # Momentum indicators
        indicators['rsi'] = self._calculate_rsi(df['close'])
        indicators['macd'] = self._calculate_macd(df)
        
        # Volatility indicators
        indicators['bb_signal'] = self._bollinger_bands(df)
        indicators['atr'] = self._calculate_atr(df)
        
        # Volume indicators
        indicators['volume_trend'] = self._volume_analysis(df)
        indicators['obv'] = self._calculate_obv(df)
        
        # Pattern recognition
        indicators['pattern'] = self._detect_pattern(df)
        
        return indicators
    
    def _ema_crossover(self, df: pd.DataFrame) -> float:
        """EMA crossover signal (-1 to 1)"""
        ema_short = df['close'].ewm(span=8, adjust=False).mean()
        ema_long = df['close'].ewm(span=21, adjust=False).mean()
        
        # Current position
        current = (ema_short.iloc[-1] - ema_long.iloc[-1]) / ema_long.iloc[-1]
        
        # Rate of change
        prev = (ema_short.iloc[-5] - ema_long.iloc[-5]) / ema_long.iloc[-5]
        momentum = current - prev
        
        # Normalize to -1 to 1
        signal = np.tanh(momentum * 100)
        return float(signal)
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Average Directional Index"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Calculate +DM and -DM
        plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                          np.maximum(high[1:] - high[:-1], 0), 0)
        minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                           np.maximum(low[:-1] - low[1:], 0), 0)
        
        # Calculate ATR
        tr = np.maximum(high - low, 
                       np.maximum(abs(high - np.roll(close, 1)),
                                 abs(low - np.roll(close, 1))))
        atr = pd.Series(tr).rolling(period).mean().values
        
        # Calculate +DI and -DI
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean().values / atr[1:]
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean().values / atr[1:]
        
        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
        adx = pd.Series(dx).rolling(period).mean().iloc[-1]
        
        # Return normalized ADX (0-1, higher = stronger trend)
        return min(1.0, adx / 50.0) if not np.isnan(adx) else 0.5
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI normalized to -1 to 1"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        # Normalize: RSI > 70 = sell signal (-1), RSI < 30 = buy signal (1)
        normalized = (50 - rsi.iloc[-1]) / 25
        return float(np.clip(normalized, -1, 1))
    
    def _calculate_macd(self, df: pd.DataFrame) -> float:
        """MACD signal"""
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        
        # Histogram
        histogram = macd_line - signal_line
        
        # Normalize based on recent range
        hist_std = histogram.tail(50).std()
        if hist_std > 0:
            normalized = histogram.iloc[-1] / (2 * hist_std)
        else:
            normalized = 0
        
        return float(np.clip(normalized, -1, 1))
    
    def _bollinger_bands(self, df: pd.DataFrame, period: int = 20) -> float:
        """Bollinger Bands signal"""
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        
        current_price = df['close'].iloc[-1]
        
        # Position within bands (-1 to 1)
        if upper_band.iloc[-1] == lower_band.iloc[-1]:
            return 0.0
        
        position = 2 * (current_price - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1]) - 1
        
        # Inverse for mean reversion
        return float(-np.clip(position, -1, 1))
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """ATR as volatility measure (0-1)"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.maximum(high - low, 
                       np.maximum(abs(high - np.roll(close, 1)),
                                 abs(low - np.roll(close, 1))))
        
        atr = pd.Series(tr).rolling(period).mean().iloc[-1]
        
        # Normalize by price
        normalized_atr = atr / close[-1]
        
        # Convert to 0-1 (higher = more volatile)
        return float(min(1.0, normalized_atr * 100))
    
    def _volume_analysis(self, df: pd.DataFrame) -> float:
        """Volume trend signal"""
        if 'tick_volume' not in df.columns:
            return 0.0
        
        vol_sma = df['tick_volume'].rolling(20).mean()
        current_vol = df['tick_volume'].iloc[-1]
        
        # Volume expansion/contraction
        vol_ratio = current_vol / (vol_sma.iloc[-1] + 1)
        
        # Price-volume correlation
        price_change = df['close'].pct_change().iloc[-1]
        
        # Strong volume on price move = stronger signal
        signal = np.tanh(price_change * vol_ratio * 100)
        
        return float(signal)
    
    def _calculate_obv(self, df: pd.DataFrame) -> float:
        """On-Balance Volume trend"""
        if 'tick_volume' not in df.columns:
            return 0.0
        
        obv = (np.sign(df['close'].diff()) * df['tick_volume']).cumsum()
        
        # OBV trend
        obv_ema = obv.ewm(span=10).mean()
        obv_signal = (obv.iloc[-1] - obv_ema.iloc[-1]) / (abs(obv_ema.iloc[-1]) + 1)
        
        return float(np.clip(obv_signal, -1, 1))
    
    def _detect_pattern(self, df: pd.DataFrame) -> float:
        """Simple pattern detection"""
        closes = df['close'].tail(20).values
        
        # Detect simple patterns
        patterns = []
        
        # Higher highs and higher lows (uptrend)
        highs = df['high'].tail(20).values
        lows = df['low'].tail(20).values
        
        if len(highs) >= 10:
            recent_highs = highs[-10::2]  # Every other high
            recent_lows = lows[-10::2]
            
            if all(recent_highs[i] <= recent_highs[i+1] for i in range(len(recent_highs)-1)):
                patterns.append(1.0)  # Uptrend
            elif all(recent_highs[i] >= recent_highs[i+1] for i in range(len(recent_highs)-1)):
                patterns.append(-1.0)  # Downtrend
        
        # Support/resistance break
        resistance = df['high'].tail(50).max()
        support = df['low'].tail(50).min()
        current = df['close'].iloc[-1]
        
        if current > resistance * 0.995:
            patterns.append(0.8)  # Resistance break
        elif current < support * 1.005:
            patterns.append(-0.8)  # Support break
        
        # Average patterns
        if patterns:
            return float(np.mean(patterns))
        return 0.0
    
    def _combine_signals(self, indicators: Dict[str, float]) -> Dict[str, float]:
        """Combine indicators into signal scores"""
        
        # Group indicators by category
        trend_score = np.mean([
            indicators.get('ema_trend', 0),
            indicators.get('adx', 0) * np.sign(indicators.get('ema_trend', 0))
        ])
        
        momentum_score = np.mean([
            indicators.get('rsi', 0),
            indicators.get('macd', 0)
        ])
        
        volatility_score = np.mean([
            indicators.get('bb_signal', 0),
            -indicators.get('atr', 0) * 0.5  # Prefer lower volatility
        ])
        
        volume_score = np.mean([
            indicators.get('volume_trend', 0),
            indicators.get('obv', 0)
        ])
        
        pattern_score = indicators.get('pattern', 0)
        
        # Weighted combination
        total_score = (
            trend_score * self.indicator_weights['trend'] +
            momentum_score * self.indicator_weights['momentum'] +
            volatility_score * self.indicator_weights['volatility'] +
            volume_score * self.indicator_weights['volume'] +
            pattern_score * self.indicator_weights['pattern']
        )
        
        return {
            'trend': trend_score,
            'momentum': momentum_score,
            'volatility': volatility_score,
            'volume': volume_score,
            'pattern': pattern_score,
            'total': total_score
        }
    
    def _generate_signal(self, scores: Dict[str, float], df: pd.DataFrame, 
                        indicators: Dict[str, float]) -> Optional[Signal]:
        """Generate final trading signal"""
        
        total_score = scores['total']
        
        # Determine direction and strength
        if abs(total_score) < 0.1:
            direction = 'neutral'
            strength = 0.0
        elif total_score > 0:
            direction = 'buy'
            strength = min(1.0, abs(total_score))
        else:
            direction = 'sell'
            strength = min(1.0, abs(total_score))
        
        # Check minimum confidence
        if strength < self.min_confidence:
            return None
        
        # Calculate entry, SL, TP
        current_price = float(df['close'].iloc[-1])
        atr = indicators.get('atr', 0.0001) * current_price
        
        if direction == 'buy':
            entry = current_price
            stop_loss = entry - (2 * atr)
            take_profit = entry + (3 * atr)
            reason = f"Bullish signal: trend={scores['trend']:.2f}, momentum={scores['momentum']:.2f}"
        elif direction == 'sell':
            entry = current_price
            stop_loss = entry + (2 * atr)
            take_profit = entry - (3 * atr)
            reason = f"Bearish signal: trend={scores['trend']:.2f}, momentum={scores['momentum']:.2f}"
        else:
            return None
        
        return Signal(
            direction=direction,
            strength=strength,
            entry_price=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
            indicators=indicators
        )

# Global instance
signal_processor = SignalProcessor()
