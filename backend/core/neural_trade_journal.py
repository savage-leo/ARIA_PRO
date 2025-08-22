"""
Neural Trade Journal - Trade Memory and Reinforcement Learning
Stores trade outcomes and learns from historical patterns to improve future decisions
"""

import json
import sqlite3
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


@dataclass
class TradeMemory:
    """Individual trade memory record"""
    trade_id: str
    symbol: str
    timestamp: float
    entry_price: float
    exit_price: float
    direction: str  # 'long' or 'short'
    volume: float
    profit_loss: float
    profit_pct: float
    duration_seconds: float
    
    # Context at trade time
    regime: str
    volatility: float
    spread: float
    atr: float
    
    # Model signals
    lstm_score: float
    cnn_score: float
    ppo_score: float
    xgb_score: float
    fusion_score: float
    bias_factor: float
    
    # Outcome
    win: bool
    max_profit: float
    max_drawdown: float
    exit_reason: str
    
    # Pattern embeddings
    price_pattern: List[float]  # Last 20 bars normalized
    volume_pattern: List[float]  # Last 20 volume bars normalized
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for similarity search"""
        return np.array([
            self.profit_pct,
            self.duration_seconds / 3600,  # Hours
            self.volatility,
            self.spread / self.atr if self.atr > 0 else 0,
            self.lstm_score,
            self.cnn_score,
            self.ppo_score,
            self.xgb_score,
            self.fusion_score,
            self.bias_factor,
            1.0 if self.regime == 'TREND' else 0.0,
            1.0 if self.regime == 'RANGE' else 0.0,
            1.0 if self.regime == 'BREAKOUT' else 0.0,
            1.0 if self.direction == 'long' else -1.0
        ])


class EpisodicMemory:
    """Stores individual trade episodes"""
    
    def __init__(self, db_path: str = "data/trade_memory.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            trade_id TEXT PRIMARY KEY,
            symbol TEXT,
            timestamp REAL,
            entry_price REAL,
            exit_price REAL,
            direction TEXT,
            volume REAL,
            profit_loss REAL,
            profit_pct REAL,
            duration_seconds REAL,
            regime TEXT,
            volatility REAL,
            spread REAL,
            atr REAL,
            lstm_score REAL,
            cnn_score REAL,
            ppo_score REAL,
            xgb_score REAL,
            fusion_score REAL,
            bias_factor REAL,
            win INTEGER,
            max_profit REAL,
            max_drawdown REAL,
            exit_reason TEXT,
            price_pattern TEXT,
            volume_pattern TEXT,
            feature_vector BLOB
        )
        """)
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_symbol ON trades(symbol)
        """)
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_timestamp ON trades(timestamp)
        """)
        
        cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_profit ON trades(profit_pct)
        """)
        
        conn.commit()
        conn.close()
        
    def store(self, trade: TradeMemory):
        """Store a trade memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            feature_vector = trade.to_vector()
            
            cursor.execute("""
            INSERT OR REPLACE INTO trades VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """, (
                trade.trade_id,
                trade.symbol,
                trade.timestamp,
                trade.entry_price,
                trade.exit_price,
                trade.direction,
                trade.volume,
                trade.profit_loss,
                trade.profit_pct,
                trade.duration_seconds,
                trade.regime,
                trade.volatility,
                trade.spread,
                trade.atr,
                trade.lstm_score,
                trade.cnn_score,
                trade.ppo_score,
                trade.xgb_score,
                trade.fusion_score,
                trade.bias_factor,
                1 if trade.win else 0,
                trade.max_profit,
                trade.max_drawdown,
                trade.exit_reason,
                json.dumps(trade.price_pattern),
                json.dumps(trade.volume_pattern),
                pickle.dumps(feature_vector)
            ))
            
            conn.commit()
            logger.info(f"Stored trade memory: {trade.trade_id} ({trade.symbol}) P&L: {trade.profit_pct:.2%}")
            
        except Exception as e:
            logger.error(f"Failed to store trade memory: {e}")
            
        finally:
            conn.close()
            
    def query_similar(self, context: Dict, top_k: int = 10) -> List[TradeMemory]:
        """Find similar historical trades based on context"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        similar_trades = []
        
        try:
            # Build context vector
            context_vector = np.array([
                context.get('volatility', 0),
                context.get('spread', 0) / max(context.get('atr', 1), 1e-9),
                context.get('lstm_score', 0),
                context.get('cnn_score', 0),
                context.get('ppo_score', 0),
                context.get('xgb_score', 0),
                context.get('fusion_score', 0),
                context.get('bias_factor', 1.0),
                1.0 if context.get('regime') == 'TREND' else 0.0,
                1.0 if context.get('regime') == 'RANGE' else 0.0,
                1.0 if context.get('regime') == 'BREAKOUT' else 0.0,
                1.0 if context.get('direction') == 'long' else -1.0
            ])
            
            # Query all trades for the symbol
            symbol = context.get('symbol', '')
            cursor.execute("""
            SELECT * FROM trades 
            WHERE symbol = ? 
            ORDER BY timestamp DESC 
            LIMIT 100
            """, (symbol,))
            
            rows = cursor.fetchall()
            
            # Calculate similarity scores
            similarities = []
            for row in rows:
                trade_vector = pickle.loads(row[26])  # feature_vector column
                
                # Cosine similarity
                similarity = np.dot(context_vector, trade_vector[:len(context_vector)]) / (
                    np.linalg.norm(context_vector) * np.linalg.norm(trade_vector[:len(context_vector)]) + 1e-9
                )
                
                similarities.append((similarity, row))
                
            # Sort by similarity and take top_k
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            for similarity, row in similarities[:top_k]:
                trade = TradeMemory(
                    trade_id=row[0],
                    symbol=row[1],
                    timestamp=row[2],
                    entry_price=row[3],
                    exit_price=row[4],
                    direction=row[5],
                    volume=row[6],
                    profit_loss=row[7],
                    profit_pct=row[8],
                    duration_seconds=row[9],
                    regime=row[10],
                    volatility=row[11],
                    spread=row[12],
                    atr=row[13],
                    lstm_score=row[14],
                    cnn_score=row[15],
                    ppo_score=row[16],
                    xgb_score=row[17],
                    fusion_score=row[18],
                    bias_factor=row[19],
                    win=bool(row[20]),
                    max_profit=row[21],
                    max_drawdown=row[22],
                    exit_reason=row[23],
                    price_pattern=json.loads(row[24]),
                    volume_pattern=json.loads(row[25])
                )
                similar_trades.append(trade)
                
        except Exception as e:
            logger.error(f"Failed to query similar trades: {e}")
            
        finally:
            conn.close()
            
        return similar_trades
        
    def get_recent_trades(self, symbol: str, hours: int = 24) -> List[TradeMemory]:
        """Get recent trades for a symbol"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        trades = []
        cutoff_time = time.time() - (hours * 3600)
        
        try:
            cursor.execute("""
            SELECT * FROM trades 
            WHERE symbol = ? AND timestamp > ?
            ORDER BY timestamp DESC
            """, (symbol, cutoff_time))
            
            rows = cursor.fetchall()
            
            for row in rows:
                trade = TradeMemory(
                    trade_id=row[0],
                    symbol=row[1],
                    timestamp=row[2],
                    entry_price=row[3],
                    exit_price=row[4],
                    direction=row[5],
                    volume=row[6],
                    profit_loss=row[7],
                    profit_pct=row[8],
                    duration_seconds=row[9],
                    regime=row[10],
                    volatility=row[11],
                    spread=row[12],
                    atr=row[13],
                    lstm_score=row[14],
                    cnn_score=row[15],
                    ppo_score=row[16],
                    xgb_score=row[17],
                    fusion_score=row[18],
                    bias_factor=row[19],
                    win=bool(row[20]),
                    max_profit=row[21],
                    max_drawdown=row[22],
                    exit_reason=row[23],
                    price_pattern=json.loads(row[24]),
                    volume_pattern=json.loads(row[25])
                )
                trades.append(trade)
                
        except Exception as e:
            logger.error(f"Failed to get recent trades: {e}")
            
        finally:
            conn.close()
            
        return trades


class SemanticMemory:
    """Extract and store trading patterns"""
    
    def __init__(self):
        self.patterns = {}
        self.pattern_performance = {}
        
    def extract_patterns(self, trade: TradeMemory) -> Dict[str, Any]:
        """Extract semantic patterns from trade"""
        patterns = {}
        
        # Regime pattern
        regime_key = f"{trade.regime}_{trade.direction}"
        patterns['regime_pattern'] = regime_key
        
        # Signal consensus pattern
        signals = [trade.lstm_score, trade.cnn_score, trade.ppo_score, trade.xgb_score]
        consensus = sum(1 for s in signals if (s > 0 and trade.direction == 'long') or (s < 0 and trade.direction == 'short'))
        patterns['consensus_level'] = consensus / len(signals)
        
        # Volatility pattern
        if trade.volatility < 0.001:
            patterns['vol_bucket'] = 'low'
        elif trade.volatility < 0.005:
            patterns['vol_bucket'] = 'medium'
        else:
            patterns['vol_bucket'] = 'high'
            
        # Time pattern (hour of day)
        trade_time = datetime.fromtimestamp(trade.timestamp)
        patterns['hour'] = trade_time.hour
        patterns['day_of_week'] = trade_time.weekday()
        
        # Duration pattern
        if trade.duration_seconds < 300:  # 5 minutes
            patterns['duration_type'] = 'scalp'
        elif trade.duration_seconds < 3600:  # 1 hour
            patterns['duration_type'] = 'short'
        elif trade.duration_seconds < 14400:  # 4 hours
            patterns['duration_type'] = 'medium'
        else:
            patterns['duration_type'] = 'long'
            
        # Update pattern performance tracking
        pattern_key = f"{regime_key}_{patterns['vol_bucket']}_{patterns['consensus_level']:.1f}"
        
        if pattern_key not in self.pattern_performance:
            self.pattern_performance[pattern_key] = {
                'count': 0,
                'wins': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'win_rate': 0
            }
            
        perf = self.pattern_performance[pattern_key]
        perf['count'] += 1
        if trade.win:
            perf['wins'] += 1
        perf['total_pnl'] += trade.profit_pct
        perf['avg_pnl'] = perf['total_pnl'] / perf['count']
        perf['win_rate'] = perf['wins'] / perf['count']
        
        patterns['pattern_key'] = pattern_key
        patterns['pattern_performance'] = perf
        
        return patterns
        
    def get_pattern_performance(self, pattern_key: str) -> Optional[Dict]:
        """Get performance metrics for a pattern"""
        return self.pattern_performance.get(pattern_key)


class ReinforcementLearner:
    """Q-learning based trade reinforcement"""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = {}  # State-action Q-values
        self.epsilon = 0.1  # Exploration rate
        
    def _get_state_key(self, context: Dict) -> str:
        """Convert context to state key"""
        # Discretize continuous values
        vol_bucket = 'low' if context.get('volatility', 0) < 0.001 else 'medium' if context.get('volatility', 0) < 0.005 else 'high'
        signal_bucket = 'strong_bear' if context.get('fusion_score', 0) < -0.5 else 'bear' if context.get('fusion_score', 0) < 0 else 'bull' if context.get('fusion_score', 0) < 0.5 else 'strong_bull'
        regime = context.get('regime', 'UNKNOWN')
        
        return f"{regime}_{vol_bucket}_{signal_bucket}"
        
    def update_q_values(self, trade: TradeMemory, patterns: Dict):
        """Update Q-values based on trade outcome"""
        # Create state from trade context
        context = {
            'volatility': trade.volatility,
            'fusion_score': trade.fusion_score,
            'regime': trade.regime
        }
        
        state_key = self._get_state_key(context)
        action = trade.direction
        
        # Initialize Q-value if not exists
        if state_key not in self.q_table:
            self.q_table[state_key] = {'long': 0.0, 'short': 0.0, 'hold': 0.0}
            
        # Calculate reward (use profit percentage)
        reward = trade.profit_pct
        
        # Q-learning update
        old_q = self.q_table[state_key][action]
        
        # For terminal state (trade closed), next state value is 0
        new_q = old_q + self.learning_rate * (reward - old_q)
        
        self.q_table[state_key][action] = new_q
        
        logger.debug(f"Updated Q-value for {state_key}/{action}: {old_q:.4f} -> {new_q:.4f}")
        
    def compute_expected_value(self, similar_trades: List[TradeMemory]) -> float:
        """Compute expected value from similar historical trades"""
        if not similar_trades:
            return 0.0
            
        # Weight by recency and similarity
        weights = []
        values = []
        
        current_time = time.time()
        
        for i, trade in enumerate(similar_trades):
            # Recency weight (exponential decay over 30 days)
            age_days = (current_time - trade.timestamp) / 86400
            recency_weight = np.exp(-age_days / 30)
            
            # Similarity weight (based on ranking)
            similarity_weight = 1.0 / (i + 1)
            
            weight = recency_weight * similarity_weight
            weights.append(weight)
            values.append(trade.profit_pct)
            
        # Weighted average
        if sum(weights) > 0:
            expected_value = sum(w * v for w, v in zip(weights, values)) / sum(weights)
        else:
            expected_value = 0.0
            
        return expected_value
        
    def get_trade_confidence(self, context: Dict) -> float:
        """Get confidence adjustment based on Q-values"""
        state_key = self._get_state_key(context)
        
        if state_key not in self.q_table:
            return 1.0  # Neutral confidence for unknown states
            
        q_values = self.q_table[state_key]
        direction = context.get('direction', 'long')
        
        # Get Q-value for proposed action
        action_q = q_values.get(direction, 0.0)
        hold_q = q_values.get('hold', 0.0)
        
        # Confidence based on Q-value difference
        if action_q > hold_q:
            # Scale confidence up if action is better than holding
            confidence_multiplier = 1.0 + min(action_q - hold_q, 0.5)
        else:
            # Scale confidence down if holding is better
            confidence_multiplier = max(0.5, 1.0 + (action_q - hold_q))
            
        return confidence_multiplier


class NeuralTradeJournal:
    """Main neural trade journal interface"""
    
    def __init__(self, db_path: str = "data/trade_memory.db"):
        self.episodic_memory = EpisodicMemory(db_path)
        self.semantic_memory = SemanticMemory()
        self.reinforcement_engine = ReinforcementLearner()
        self.active_trades = {}  # Track active trades for later recording
        
    def start_trade(self, trade_id: str, symbol: str, entry_price: float, 
                    direction: str, volume: float, context: Dict):
        """Record trade entry"""
        self.active_trades[trade_id] = {
            'trade_id': trade_id,
            'symbol': symbol,
            'entry_price': entry_price,
            'direction': direction,
            'volume': volume,
            'entry_time': time.time(),
            'context': context,
            'max_profit': 0.0,
            'max_drawdown': 0.0
        }
        
        logger.info(f"Started tracking trade {trade_id} ({symbol} {direction} @ {entry_price})")
        
    def update_trade(self, trade_id: str, current_price: float):
        """Update trade tracking with current price"""
        if trade_id not in self.active_trades:
            return
            
        trade = self.active_trades[trade_id]
        entry_price = trade['entry_price']
        direction = trade['direction']
        
        if direction == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
            
        # Update max profit/drawdown
        trade['max_profit'] = max(trade['max_profit'], pnl_pct)
        trade['max_drawdown'] = min(trade.get('max_drawdown', 0), pnl_pct)
        
    def complete_trade(self, trade_id: str, exit_price: float, exit_reason: str = "manual"):
        """Complete and record trade"""
        if trade_id not in self.active_trades:
            logger.warning(f"Trade {trade_id} not found in active trades")
            return
            
        trade = self.active_trades[trade_id]
        entry_price = trade['entry_price']
        direction = trade['direction']
        
        # Calculate P&L
        if direction == 'long':
            profit_loss = (exit_price - entry_price) * trade['volume']
            profit_pct = (exit_price - entry_price) / entry_price
        else:
            profit_loss = (entry_price - exit_price) * trade['volume']
            profit_pct = (entry_price - exit_price) / entry_price
            
        # Create trade memory
        context = trade['context']
        
        # Extract price and volume patterns from context
        bars = context.get('bars', [])
        price_pattern = []
        volume_pattern = []
        
        if bars and len(bars) >= 20:
            closes = [float(b.get('close', 0)) for b in bars[-20:]]
            volumes = [float(b.get('volume', 0)) for b in bars[-20:]]
            
            # Normalize patterns
            if closes:
                mean_price = np.mean(closes)
                std_price = np.std(closes) + 1e-8
                price_pattern = [(p - mean_price) / std_price for p in closes]
                
            if volumes:
                mean_vol = np.mean(volumes)
                std_vol = np.std(volumes) + 1e-8
                volume_pattern = [(v - mean_vol) / std_vol for v in volumes]
                
        trade_memory = TradeMemory(
            trade_id=trade_id,
            symbol=trade['symbol'],
            timestamp=trade['entry_time'],
            entry_price=entry_price,
            exit_price=exit_price,
            direction=direction,
            volume=trade['volume'],
            profit_loss=profit_loss,
            profit_pct=profit_pct,
            duration_seconds=time.time() - trade['entry_time'],
            regime=context.get('regime', 'UNKNOWN'),
            volatility=context.get('volatility', 0),
            spread=context.get('spread', 0),
            atr=context.get('atr', 0),
            lstm_score=context.get('lstm_score', 0),
            cnn_score=context.get('cnn_score', 0),
            ppo_score=context.get('ppo_score', 0),
            xgb_score=context.get('xgb_score', 0),
            fusion_score=context.get('fusion_score', 0),
            bias_factor=context.get('bias_factor', 1.0),
            win=profit_pct > 0,
            max_profit=trade.get('max_profit', 0),
            max_drawdown=trade.get('max_drawdown', 0),
            exit_reason=exit_reason,
            price_pattern=price_pattern,
            volume_pattern=volume_pattern
        )
        
        # Store in episodic memory
        self.episodic_memory.store(trade_memory)
        
        # Extract patterns
        patterns = self.semantic_memory.extract_patterns(trade_memory)
        
        # Update reinforcement learning
        self.reinforcement_engine.update_q_values(trade_memory, patterns)
        
        # Remove from active trades
        del self.active_trades[trade_id]
        
        logger.info(f"Completed trade {trade_id}: P&L {profit_pct:.2%}, Pattern: {patterns.get('pattern_key')}")
        
        return trade_memory
        
    def get_trade_bias(self, context: Dict) -> Dict[str, float]:
        """Get historical bias and confidence adjustment for similar trades"""
        # Query similar trades
        similar_trades = self.episodic_memory.query_similar(context, top_k=20)
        
        if not similar_trades:
            return {
                'historical_bias': 0.0,
                'confidence_multiplier': 1.0,
                'expected_value': 0.0,
                'similar_trades_count': 0
            }
            
        # Compute expected value
        expected_value = self.reinforcement_engine.compute_expected_value(similar_trades)
        
        # Get Q-learning confidence
        confidence_multiplier = self.reinforcement_engine.get_trade_confidence(context)
        
        # Calculate win rate of similar trades
        wins = sum(1 for t in similar_trades if t.win)
        win_rate = wins / len(similar_trades)
        
        # Historical bias based on win rate and expected value
        historical_bias = (win_rate - 0.5) * 2.0  # Scale to [-1, 1]
        
        return {
            'historical_bias': historical_bias,
            'confidence_multiplier': confidence_multiplier,
            'expected_value': expected_value,
            'similar_trades_count': len(similar_trades),
            'win_rate': win_rate
        }
        
    def get_pattern_stats(self, symbol: str) -> Dict[str, Any]:
        """Get pattern performance statistics for a symbol"""
        recent_trades = self.episodic_memory.get_recent_trades(symbol, hours=24*7)  # Last week
        
        pattern_stats = {}
        for trade in recent_trades:
            patterns = self.semantic_memory.extract_patterns(trade)
            pattern_key = patterns.get('pattern_key')
            
            if pattern_key:
                perf = self.semantic_memory.get_pattern_performance(pattern_key)
                if perf:
                    pattern_stats[pattern_key] = perf
                    
        return pattern_stats


# Global instance
_neural_journal = None


def get_neural_journal() -> NeuralTradeJournal:
    """Get or create the global neural trade journal"""
    global _neural_journal
    if _neural_journal is None:
        _neural_journal = NeuralTradeJournal()
    return _neural_journal
