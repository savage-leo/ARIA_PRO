# backend/risk_manager.py
"""
CPU-friendly risk management module
Pure mathematical risk calculations without heavy dependencies
"""
import numpy as np
from typing import Dict, Optional, List
from dataclasses import dataclass
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_position_size: float = 0.10  # Max lots per position
    max_daily_loss: float = 0.02  # 2% max daily loss
    max_drawdown: float = 0.05  # 5% max drawdown
    max_positions: int = 5  # Max concurrent positions
    max_correlation: float = 0.7  # Max correlation between positions
    min_win_rate: float = 0.4  # Min win rate to continue trading
    max_leverage: float = 10.0  # Max account leverage

class RiskManager:
    """Lightweight risk management system"""
    
    def __init__(self, account_balance: float = 10000.0):
        self.account_balance = account_balance
        self.initial_balance = account_balance
        self.limits = RiskLimits()
        self.daily_pnl = 0.0
        self.open_positions = []
        self.trade_history = []
        self.load_state()
    
    def load_state(self):
        """Load risk state from file"""
        state_file = Path(__file__).parent / "risk_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                    self.daily_pnl = state.get('daily_pnl', 0.0)
                    self.trade_history = state.get('trade_history', [])
            except:
                pass
    
    def save_state(self):
        """Save risk state to file"""
        state_file = Path(__file__).parent / "risk_state.json"
        with open(state_file, 'w') as f:
            json.dump({
                'daily_pnl': self.daily_pnl,
                'trade_history': self.trade_history[-100:]  # Keep last 100
            }, f)
    
    def calculate_position_size(self, 
                              confidence: float,
                              volatility: float,
                              symbol: str = "EURUSD") -> float:
        """
        Calculate optimal position size using Kelly Criterion
        with volatility adjustment
        """
        # Kelly fraction calculation
        win_rate = self._calculate_win_rate()
        avg_win = self._calculate_avg_win()
        avg_loss = abs(self._calculate_avg_loss())
        
        if avg_loss == 0:
            avg_loss = 0.001  # Prevent division by zero
        
        # Kelly formula: f = (p * b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        b = avg_win / avg_loss if avg_loss > 0 else 1.0
        p = win_rate
        q = 1 - p
        
        kelly = (p * b - q) / b if b > 0 else 0
        kelly = max(0, min(0.25, kelly))  # Cap at 25% of capital
        
        # Adjust for confidence
        kelly *= confidence
        
        # Adjust for volatility (reduce size in high vol)
        vol_adjustment = 1.0 / (1.0 + volatility * 10)
        kelly *= vol_adjustment
        
        # Convert to lot size
        risk_per_trade = self.account_balance * kelly
        
        # Assume 1 pip = $0.10 per 0.01 lot for major pairs
        pip_value = 0.10
        stop_loss_pips = 20  # Default SL
        
        lot_size = risk_per_trade / (stop_loss_pips * pip_value)
        lot_size = round(lot_size / 100, 2) * 0.01  # Round to 0.01 lots
        
        # Apply limits
        lot_size = max(0.01, min(self.limits.max_position_size, lot_size))
        
        return lot_size
    
    def check_risk_limits(self, proposed_trade: Dict) -> tuple[bool, str]:
        """
        Check if proposed trade violates risk limits
        Returns (allowed, reason)
        """
        # Check daily loss limit
        daily_loss_pct = abs(self.daily_pnl) / self.initial_balance
        if daily_loss_pct >= self.limits.max_daily_loss:
            return False, f"Daily loss limit reached: {daily_loss_pct:.1%}"
        
        # Check max positions
        if len(self.open_positions) >= self.limits.max_positions:
            return False, f"Max positions limit reached: {self.limits.max_positions}"
        
        # Check position size
        if proposed_trade.get('volume', 0) > self.limits.max_position_size:
            return False, f"Position size too large: {proposed_trade.get('volume', 0)}"
        
        # Check drawdown
        current_drawdown = (self.initial_balance - self.account_balance) / self.initial_balance
        if current_drawdown >= self.limits.max_drawdown:
            return False, f"Max drawdown reached: {current_drawdown:.1%}"
        
        # Check win rate (if enough history)
        if len(self.trade_history) >= 20:
            win_rate = self._calculate_win_rate()
            if win_rate < self.limits.min_win_rate:
                return False, f"Win rate too low: {win_rate:.1%}"
        
        # Check correlation with existing positions
        if self._check_correlation(proposed_trade):
            return False, "Position too correlated with existing trades"
        
        return True, "Risk checks passed"
    
    def _calculate_win_rate(self) -> float:
        """Calculate historical win rate"""
        if not self.trade_history:
            return 0.5  # Default assumption
        
        wins = sum(1 for t in self.trade_history if t.get('pnl', 0) > 0)
        return wins / len(self.trade_history)
    
    def _calculate_avg_win(self) -> float:
        """Calculate average winning trade"""
        wins = [t['pnl'] for t in self.trade_history if t.get('pnl', 0) > 0]
        return np.mean(wins) if wins else 0.001
    
    def _calculate_avg_loss(self) -> float:
        """Calculate average losing trade"""
        losses = [t['pnl'] for t in self.trade_history if t.get('pnl', 0) < 0]
        return np.mean(losses) if losses else -0.001
    
    def _check_correlation(self, proposed_trade: Dict) -> bool:
        """Check if trade is too correlated with existing positions"""
        # Simple correlation check based on direction and symbol
        symbol = proposed_trade.get('symbol', '')
        direction = proposed_trade.get('action', '')
        
        # Count similar positions
        similar = 0
        for pos in self.open_positions:
            if pos.get('symbol', '').startswith(symbol[:3]):  # Same base currency
                if pos.get('action') == direction:
                    similar += 1
        
        # Don't allow more than 2 correlated positions
        return similar >= 2
    
    def update_position(self, position: Dict):
        """Update position in tracking"""
        # Add or update position
        found = False
        for i, pos in enumerate(self.open_positions):
            if pos.get('ticket') == position.get('ticket'):
                self.open_positions[i] = position
                found = True
                break
        
        if not found:
            self.open_positions.append(position)
    
    def close_position(self, ticket: int, pnl: float):
        """Close position and record P&L"""
        # Remove from open positions
        self.open_positions = [p for p in self.open_positions if p.get('ticket') != ticket]
        
        # Update daily P&L
        self.daily_pnl += pnl
        
        # Record in history
        self.trade_history.append({
            'ticket': ticket,
            'pnl': pnl,
            'timestamp': np.datetime64('now').item()
        })
        
        # Update account balance
        self.account_balance += pnl
        
        self.save_state()
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.daily_pnl = 0.0
        self.save_state()
    
    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics"""
        win_rate = self._calculate_win_rate()
        avg_win = self._calculate_avg_win()
        avg_loss = abs(self._calculate_avg_loss())
        
        # Calculate Sharpe ratio
        if self.trade_history:
            returns = [t['pnl'] / self.initial_balance for t in self.trade_history]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Calculate profit factor
        total_wins = sum(t['pnl'] for t in self.trade_history if t.get('pnl', 0) > 0)
        total_losses = abs(sum(t['pnl'] for t in self.trade_history if t.get('pnl', 0) < 0))
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        return {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe,
            'profit_factor': profit_factor,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl / self.initial_balance,
            'open_positions': len(self.open_positions),
            'account_balance': self.account_balance,
            'drawdown': (self.initial_balance - self.account_balance) / self.initial_balance
        }

# Global instance
risk_manager = RiskManager()
