# tools/strategy_accounting.py
"""
Economic gating and strategy accounting for trade decisions.
Pure mathematics - no heavy dependencies.
"""
import numpy as np
from typing import List, Dict, Optional
from collections import deque
import json
from pathlib import Path

LEDGER_PATH = Path(__file__).resolve().parent.parent / "ledger" / "trades.json"
LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

class StrategyAccounting:
    def __init__(self, max_history: int = 1000):
        self.trade_history = deque(maxlen=max_history)
        self.cumulative_pnl = 0.0
        self.max_drawdown = 0.0
        self.high_water_mark = 0.0
        self._load_ledger()
    
    def _load_ledger(self):
        if LEDGER_PATH.exists():
            try:
                with open(LEDGER_PATH) as f:
                    data = json.load(f)
                    self.trade_history = deque(data.get('trades', []), maxlen=1000)
                    self.cumulative_pnl = data.get('cumulative_pnl', 0.0)
                    self.high_water_mark = data.get('high_water_mark', 0.0)
            except:
                pass
    
    def _save_ledger(self):
        with open(LEDGER_PATH, 'w') as f:
            json.dump({
                'trades': list(self.trade_history),
                'cumulative_pnl': self.cumulative_pnl,
                'high_water_mark': self.high_water_mark
            }, f, indent=2)
    
    def add_trade(self, pnl: float, symbol: str = "EURUSD", volume: float = 0.01):
        trade = {
            'pnl': pnl,
            'symbol': symbol,
            'volume': volume,
            'timestamp': np.datetime64('now').item()
        }
        self.trade_history.append(trade)
        self.cumulative_pnl += pnl
        
        # Update high water mark and drawdown
        if self.cumulative_pnl > self.high_water_mark:
            self.high_water_mark = self.cumulative_pnl
        
        current_dd = (self.high_water_mark - self.cumulative_pnl) / max(self.high_water_mark, 1.0)
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
        
        self._save_ledger()
    
    def get_metrics(self, fees_per_trade: float = 0.0) -> Dict[str, float]:
        if len(self.trade_history) == 0:
            return {
                'expected_value': 0.0,
                'avg_net': 0.0,
                'sharpe': 0.0,
                'win_rate': 0.5,
                'profit_factor': 1.0,
                'max_drawdown': 0.0,
                'total_trades': 0
            }
        
        pnls = [t['pnl'] for t in self.trade_history]
        avg_gross = np.mean(pnls)
        avg_net = avg_gross - fees_per_trade
        
        # Sharpe ratio (annualized, assuming daily trades)
        if np.std(pnls) > 0:
            sharpe = (avg_gross / np.std(pnls)) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Win rate
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls) if pnls else 0.5
        
        # Profit factor
        gross_wins = sum(p for p in pnls if p > 0)
        gross_losses = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else 100.0
        
        return {
            'expected_value': float(avg_net * len(pnls)),
            'avg_net': float(avg_net),
            'sharpe': float(sharpe),
            'win_rate': float(win_rate),
            'profit_factor': float(min(profit_factor, 100.0)),
            'max_drawdown': float(self.max_drawdown),
            'total_trades': len(self.trade_history)
        }
    
    def should_trade(self, confidence: float, min_confidence: float = 0.55) -> bool:
        """Economic gating decision based on account state and confidence"""
        metrics = self.get_metrics()
        
        # Don't trade if drawdown is too high
        if metrics['max_drawdown'] > 0.1:  # 10% max DD
            return False
        
        # Don't trade if confidence is too low
        if confidence < min_confidence:
            return False
        
        # Don't trade if recent performance is terrible
        if len(self.trade_history) >= 10:
            recent_pnls = [t['pnl'] for t in list(self.trade_history)[-10:]]
            recent_avg = np.mean(recent_pnls)
            if recent_avg < -0.005:  # Recent avg loss > 0.5%
                return False
        
        return True

# Global instance
strategy_accounting = StrategyAccounting()

def strategy_metrics(trade_pnls: List[float], fees_per_trade: float = 0.0, discount: float = 0.0) -> Dict[str, float]:
    """Standalone function for quick metrics calculation"""
    if len(trade_pnls) == 0:
        return {'expected_value': 0.0, 'avg_net': 0.0, 'sharpe': 0.0}
    avg_net = np.mean(trade_pnls) - fees_per_trade
    freq = len(trade_pnls)
    ev = avg_net * freq
    sharpe = (np.mean(trade_pnls) / (np.std(trade_pnls) + 1e-9)) * np.sqrt(252)
    return {'expected_value': float(ev), 'avg_net': float(avg_net), 'sharpe': float(sharpe)}
