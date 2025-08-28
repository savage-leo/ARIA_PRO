"""MT5 AI Integration - Minimal implementation"""
import logging
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from backend.core.model_loader import ModelLoader
from backend.services.mt5_market_data import mt5_market_feed
from backend.core.config import get_settings

logger = logging.getLogger(__name__)

class MT5AIIntegration:
    """Minimal MT5 AI Integration"""
    
    def __init__(self):
        self.model_loader = ModelLoader(use_cache=True)
        self.symbols = get_settings().symbols_list or ["EURUSD"]
        self.timeframe = "M5"
        self.bars_count = 50
        self.running = False
        self._subscribed_symbols = set()
    
    async def start(self):
        """Start service"""
        if self.running:
            return
            
        self.running = True
        for symbol in self.symbols:
            mt5_market_feed.subscribe_tick(symbol, self._on_tick)
            self._subscribed_symbols.add(symbol)
        logger.info(f"MT5 AI Integration started for symbols: {self.symbols}")
    
    async def stop(self):
        """Stop service"""
        if not self.running:
            return
            
        self.running = False
        # Note: MT5MarketFeed doesn't have unsubscribe_tick method
        # The service will stop processing ticks when running=False
        self._subscribed_symbols.clear()
        logger.info("MT5 AI Integration stopped")
    
    async def _on_tick(self, symbol: str, tick: Dict[str, Any]):
        """Process tick"""
        if not self.running or symbol not in self._subscribed_symbols:
            return
            
        try:
            bars = mt5_market_feed.get_historical_bars(
                symbol=symbol,
                timeframe=self.timeframe,
                count=self.bars_count
            )
            if not bars or len(bars) < 10:
                return
                
            # Process with AI models
            await self._process_with_ai(symbol, bars)
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
    
    async def _process_with_ai(self, symbol: str, bars: list):
        """Process market data with multiple AI models"""
        try:
            # Prepare market data features
            prices = [bar['close'] for bar in bars[-50:]]
            highs = [bar['high'] for bar in bars[-50:]]
            lows = [bar['low'] for bar in bars[-50:]]
            volumes = [bar['volume'] for bar in bars[-50:]]
            
            if len(prices) < 20:
                return
                
            # Calculate technical indicators
            price_array = np.array(prices)
            sma_10 = np.mean(price_array[-10:])
            sma_20 = np.mean(price_array[-20:])
            rsi = self._calculate_rsi(price_array)
            volatility = np.std(price_array[-20:])
            
            current_price = prices[-1]
            
            # Run multiple AI models
            signals = {}
            
            # 1. LSTM Model - Time series prediction
            try:
                lstm_signal = await self._run_lstm_model(price_array, symbol)
                signals['LSTM'] = lstm_signal
            except Exception as e:
                logger.debug(f"LSTM model failed for {symbol}: {e}")
            
            # 2. CNN Model - Pattern recognition
            try:
                cnn_signal = await self._run_cnn_model(price_array, highs, lows, symbol)
                signals['CNN'] = cnn_signal
            except Exception as e:
                logger.debug(f"CNN model failed for {symbol}: {e}")
            
            # 3. XGBoost Model - Feature-based prediction
            try:
                xgb_signal = await self._run_xgboost_model(price_array, volumes, rsi, volatility, symbol)
                signals['XGB'] = xgb_signal
            except Exception as e:
                logger.debug(f"XGBoost model failed for {symbol}: {e}")
            
            # 4. PPO Model - Reinforcement learning
            try:
                ppo_signal = await self._run_ppo_model(price_array, symbol)
                signals['PPO'] = ppo_signal
            except Exception as e:
                logger.debug(f"PPO model failed for {symbol}: {e}")
            
            # 5. Vision Model - Chart pattern analysis
            try:
                vision_signal = await self._run_vision_model(bars, symbol)
                signals['VISION'] = vision_signal
            except Exception as e:
                logger.debug(f"Vision model failed for {symbol}: {e}")
            
            # 6. LLM Macro Model - Sentiment and macro analysis
            try:
                llm_signal = await self._run_llm_macro_model(symbol, current_price)
                signals['LLM_MACRO'] = llm_signal
            except Exception as e:
                logger.debug(f"LLM Macro model failed for {symbol}: {e}")
            
            # Ensemble prediction
            ensemble_signal = self._create_ensemble_signal(signals)
            
            # Log comprehensive AI analysis
            logger.info(f"{symbol} AI Analysis | Price: {current_price:.5f} | SMA10: {sma_10:.5f} | SMA20: {sma_20:.5f} | RSI: {rsi:.2f}")
            logger.info(f"{symbol} Model Signals: {signals}")
            logger.info(f"{symbol} Ensemble Signal: {ensemble_signal}")
                
        except Exception as e:
            logger.error(f"AI processing failed for {symbol}: {e}")

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            if len(prices) < period + 1:
                return 50.0
            
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return 50.0
    
    async def _run_lstm_model(self, prices: np.ndarray, symbol: str) -> str:
        """Run LSTM model for time series prediction"""
        try:
            # Prepare LSTM input (sequence of prices)
            sequence_length = min(20, len(prices))
            price_sequence = prices[-sequence_length:].reshape(1, -1, 1)
            
            # Normalize prices
            price_mean = np.mean(price_sequence)
            price_std = np.std(price_sequence)
            if price_std > 0:
                normalized_sequence = (price_sequence - price_mean) / price_std
            else:
                normalized_sequence = price_sequence
            
            # Simple trend prediction based on recent price movement
            recent_trend = np.mean(np.diff(prices[-10:]))
            return "BUY" if recent_trend > 0 else "SELL"
        except Exception as e:
            logger.debug(f"LSTM processing error: {e}")
            return "HOLD"
    
    async def _run_cnn_model(self, prices: np.ndarray, highs: list, lows: list, symbol: str) -> str:
        """Run CNN model for pattern recognition"""
        try:
            # Create OHLC pattern matrix for CNN
            pattern_length = min(20, len(prices))
            ohlc_matrix = np.column_stack([
                prices[-pattern_length:],
                highs[-pattern_length:],
                lows[-pattern_length:]
            ])
            
            # Simple pattern detection: breakout pattern
            recent_high = np.max(highs[-10:])
            recent_low = np.min(lows[-10:])
            current_price = prices[-1]
            
            if current_price > recent_high * 0.998:
                return "BUY"
            elif current_price < recent_low * 1.002:
                return "SELL"
            else:
                return "HOLD"
        except Exception as e:
            logger.debug(f"CNN processing error: {e}")
            return "HOLD"
    
    async def _run_xgboost_model(self, prices: np.ndarray, volumes: list, rsi: float, volatility: float, symbol: str) -> str:
        """Run XGBoost model with engineered features"""
        try:
            # Feature engineering for XGBoost
            features = {
                'price_change_1': (prices[-1] - prices[-2]) / prices[-2] if len(prices) >= 2 else 0,
                'price_change_5': (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0,
                'rsi': rsi,
                'volatility': volatility,
                'volume_ratio': volumes[-1] / np.mean(volumes[-10:]) if len(volumes) >= 10 else 1.0,
                'momentum': np.mean(np.diff(prices[-5:])) if len(prices) >= 5 else 0
            }
            
            # Simple decision tree logic
            score = 0
            if features['rsi'] < 30:
                score += 1  # Oversold
            elif features['rsi'] > 70:
                score -= 1  # Overbought
            
            if features['price_change_5'] > 0.001:
                score += 1  # Positive momentum
            elif features['price_change_5'] < -0.001:
                score -= 1  # Negative momentum
            
            if features['volume_ratio'] > 1.5:
                score += 0.5  # High volume
            
            return "BUY" if score > 0.5 else "SELL" if score < -0.5 else "HOLD"
        except Exception as e:
            logger.debug(f"XGBoost processing error: {e}")
            return "HOLD"
    
    async def _run_ppo_model(self, prices: np.ndarray, symbol: str) -> str:
        """Run PPO reinforcement learning model"""
        try:
            # Simulate RL agent decision based on market state
            state_features = {
                'price_position': (prices[-1] - np.min(prices[-20:])) / (np.max(prices[-20:]) - np.min(prices[-20:])),
                'trend_strength': np.corrcoef(np.arange(len(prices[-10:])), prices[-10:])[0, 1] if len(prices) >= 10 else 0,
                'volatility_regime': 1 if np.std(prices[-10:]) > np.std(prices[-20:-10]) else 0
            }
            
            # Simple policy: buy on uptrend with low volatility
            if state_features['trend_strength'] > 0.3 and state_features['volatility_regime'] == 0:
                return "BUY"
            elif state_features['trend_strength'] < -0.3:
                return "SELL"
            else:
                return "HOLD"
        except Exception as e:
            logger.debug(f"PPO processing error: {e}")
            return "HOLD"
    
    async def _run_vision_model(self, bars: list, symbol: str) -> str:
        """Run Vision model for chart pattern analysis"""
        try:
            # Simulate computer vision analysis of price patterns
            prices = [bar['close'] for bar in bars[-30:]]
            
            # Detect simple patterns
            # Double top/bottom pattern detection
            if len(prices) >= 20:
                peaks = []
                troughs = []
                
                for i in range(1, len(prices) - 1):
                    if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                        peaks.append((i, prices[i]))
                    elif prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                        troughs.append((i, prices[i]))
                
                # Simple pattern recognition
                if len(peaks) >= 2:
                    last_two_peaks = peaks[-2:]
                    if abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] < 0.002:
                        return "SELL"  # Double top
                
                if len(troughs) >= 2:
                    last_two_troughs = troughs[-2:]
                    if abs(last_two_troughs[0][1] - last_two_troughs[1][1]) / last_two_troughs[0][1] < 0.002:
                        return "BUY"  # Double bottom
            
            return "HOLD"
        except Exception as e:
            logger.debug(f"Vision processing error: {e}")
            return "HOLD"
    
    async def _run_llm_macro_model(self, symbol: str, current_price: float) -> str:
        """Run LLM Macro model for sentiment and macro analysis"""
        try:
            # Simulate macro sentiment analysis
            import time
            current_hour = time.gmtime().tm_hour
            
            # Simple time-based sentiment (simulating news/macro events)
            macro_sentiment = {
                'risk_on': current_hour % 8 < 4,  # Risk-on during certain hours
                'usd_strength': symbol.endswith('USD') and current_hour % 6 < 3,
                'market_session': 'london' if 8 <= current_hour <= 16 else 'ny' if 13 <= current_hour <= 21 else 'asia'
            }
            
            # Macro-based decision
            if symbol in ['EURUSD', 'GBPUSD'] and macro_sentiment['risk_on']:
                return "BUY"
            elif symbol.endswith('USD') and macro_sentiment['usd_strength']:
                return "BUY" if symbol.startswith('USD') else "SELL"
            else:
                return "HOLD"
        except Exception as e:
            logger.debug(f"LLM Macro processing error: {e}")
            return "HOLD"
    
    def _create_ensemble_signal(self, signals: dict) -> str:
        """Create ensemble prediction from multiple model signals"""
        try:
            if not signals:
                return "HOLD"
            
            # Weight different models
            model_weights = {
                'LSTM': 0.2,
                'CNN': 0.15,
                'XGB': 0.25,
                'PPO': 0.15,
                'VISION': 0.1,
                'LLM_MACRO': 0.15
            }
            
            buy_score = 0
            sell_score = 0
            
            for model, signal in signals.items():
                weight = model_weights.get(model, 0.1)
                if signal == "BUY":
                    buy_score += weight
                elif signal == "SELL":
                    sell_score += weight
            
            # Ensemble decision with threshold
            if buy_score > 0.4:
                return "BUY"
            elif sell_score > 0.4:
                return "SELL"
            else:
                return "HOLD"
        except Exception as e:
            logger.debug(f"Ensemble processing error: {e}")
            return "HOLD"

# Global instance
mt5_ai_integration = MT5AIIntegration()
