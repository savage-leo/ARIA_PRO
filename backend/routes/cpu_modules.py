"""
API routes for CPU-friendly modules (risk manager, signal processor)
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Import CPU-friendly modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from backend.risk_manager import RiskManager
from backend.signal_processor import SignalProcessor

router = APIRouter(prefix="/api/cpu", tags=["cpu_modules"])

# Initialize modules
risk_manager = RiskManager()
signal_processor = SignalProcessor()

@router.get("/risk/status")
async def get_risk_status() -> Dict[str, Any]:
    """Get current risk manager status and metrics"""
    win_rate = 0.0
    if risk_manager.trade_history:
        wins = sum(1 for t in risk_manager.trade_history if t.get('pnl', 0) > 0)
        win_rate = wins / len(risk_manager.trade_history)
    
    return {
        "daily_pnl": risk_manager.daily_pnl,
        "max_daily_loss": risk_manager.limits.max_daily_loss,
        "open_positions": len(risk_manager.open_positions),
        "max_positions": risk_manager.limits.max_positions,
        "win_rate": win_rate,
        "account_balance": risk_manager.account_balance,
        "initial_balance": risk_manager.initial_balance
    }

@router.post("/risk/check")
async def check_risk_for_trade(trade_params: Dict[str, Any]) -> Dict[str, Any]:
    """Check if a trade passes risk management rules"""
    symbol = trade_params.get("symbol", "EURUSD")
    side = trade_params.get("side", "buy")
    entry_price = trade_params.get("entry_price", 1.1000)
    stop_loss = trade_params.get("stop_loss", 1.0950)
    account_balance = trade_params.get("account_balance", 10000)
    confidence = trade_params.get("confidence", 0.7)
    
    # Calculate position size and risk
    stop_loss_pips = abs(entry_price - stop_loss) * 10000
    volatility = stop_loss_pips / 100.0  # Convert pips to volatility estimate
    position_size = risk_manager.calculate_position_size(
        confidence=confidence,
        volatility=volatility,
        symbol=symbol
    )
    
    # Simple risk check
    risk_amount = position_size * abs(entry_price - stop_loss) * 100000  # Convert to USD
    max_risk = account_balance * risk_manager.limits.max_daily_loss
    allowed = (risk_amount <= max_risk and 
               len(risk_manager.open_positions) < risk_manager.limits.max_positions)
    
    return {
        "allowed": allowed,
        "position_size": position_size,
        "risk_amount": risk_amount,
        "max_risk": max_risk,
        "reason": "Risk within limits" if allowed else "Risk limit exceeded or too many positions"
    }

@router.post("/risk/record")
async def record_trade_result(trade_result: Dict[str, Any]) -> Dict[str, Any]:
    """Record a closed trade result for risk tracking"""
    symbol = trade_result.get("symbol", "EURUSD")
    pnl = trade_result.get("pnl", 0)
    ticket = trade_result.get("ticket", 0)
    
    # Record trade in history
    risk_manager.trade_history.append({
        'symbol': symbol,
        'pnl': pnl,
        'ticket': ticket,
        'timestamp': datetime.now().isoformat()
    })
    risk_manager.daily_pnl += pnl
    risk_manager.save_state()
    
    # Calculate win rate
    win_rate = 0.0
    if risk_manager.trade_history:
        wins = sum(1 for t in risk_manager.trade_history if t.get('pnl', 0) > 0)
        win_rate = wins / len(risk_manager.trade_history)
    
    return {
        "success": True,
        "updated_stats": {
            "win_rate": win_rate,
            "daily_pnl": risk_manager.daily_pnl,
            "trade_count": len(risk_manager.trade_history)
        }
    }

@router.get("/signal/status")
async def get_signal_status() -> Dict[str, Any]:
    """Get signal processor status and configuration"""
    return {
        "min_confidence": signal_processor.min_confidence,
        "indicators": list(signal_processor.indicator_weights.keys()),
        "weights": signal_processor.indicator_weights,
        "ready": True
    }

@router.post("/signal/generate")
async def generate_trading_signal(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate trading signal from market data"""
    try:
        # Convert market data to DataFrame
        if "ohlcv" in market_data:
            # Expecting array of [timestamp, open, high, low, close, volume]
            ohlcv = market_data["ohlcv"]
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms' if df['timestamp'].iloc[0] > 1e10 else 's')
            df.set_index('timestamp', inplace=True)
        else:
            # Create from individual arrays
            df = pd.DataFrame({
                'open': market_data.get('open', []),
                'high': market_data.get('high', []),
                'low': market_data.get('low', []),
                'close': market_data.get('close', []),
                'volume': market_data.get('volume', [])
            })
        
        symbol = market_data.get("symbol", "EURUSD")
        
        # Call signal processor
        signal = signal_processor.process(df)
        
        if signal:
            return {
                "direction": signal.direction,
                "strength": signal.strength,
                "entry_price": signal.entry_price,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "reason": signal.reason,
                "indicators": signal.indicators
            }
        else:
            return {
                "direction": "neutral",
                "strength": 0.0,
                "reason": "No clear signal"
            }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating signal: {str(e)}")

@router.post("/signal/backtest")
async def backtest_signal_processor(backtest_params: Dict[str, Any]) -> Dict[str, Any]:
    """Run a simple backtest of the signal processor"""
    try:
        # Generate sample data if not provided
        periods = backtest_params.get("periods", 1000)
        
        # Create synthetic market data
        dates = pd.date_range('2024-01-01', periods=periods, freq='1H')
        price = 1.1000
        prices = []
        
        for _ in range(periods):
            price *= np.random.normal(1.0, 0.001)  # Random walk
            prices.append(price)
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p * 1.001 for p in prices],
            'low': [p * 0.999 for p in prices], 
            'close': [p * 1.0005 for p in prices],
            'volume': np.random.randint(1000, 10000, periods)
        }, index=dates)
        
        # Run signals through the data
        signals = []
        for i in range(100, min(len(df), 500), 10):  # Sample every 10 bars
            window = df.iloc[max(0, i-100):i]
            signal = signal_processor.process_market_data(window, "EURUSD")
            signals.append(signal)
        
        # Calculate basic stats
        buy_signals = sum(1 for s in signals if s['action'] == 'buy')
        sell_signals = sum(1 for s in signals if s['action'] == 'sell')
        neutral_signals = sum(1 for s in signals if s['action'] == 'neutral')
        avg_confidence = np.mean([s['confidence'] for s in signals if s['confidence'] > 0])
        
        return {
            "total_signals": len(signals),
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "neutral_signals": neutral_signals,
            "avg_confidence": float(avg_confidence),
            "signal_distribution": {
                "buy_pct": buy_signals / len(signals) * 100,
                "sell_pct": sell_signals / len(signals) * 100,
                "neutral_pct": neutral_signals / len(signals) * 100
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in backtest: {str(e)}")

@router.get("/orchestrator/status")
async def get_orchestrator_status() -> Dict[str, Any]:
    """Get orchestrator status"""
    try:
        from orchestrator.orchestrator_cpu_friendly import orchestrator
        return {
            "ready": True,
            "strategies": orchestrator.strategy_names,
            "last_signal_time": orchestrator.last_signal_time,
            "min_signal_interval": orchestrator.min_signal_interval
        }
    except Exception as e:
        return {
            "ready": False,
            "error": str(e)
        }

@router.post("/orchestrator/signal")
async def generate_orchestrator_signal(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate signal using the orchestrator"""
    try:
        from orchestrator.orchestrator_cpu_friendly import orchestrator
        
        # Convert market data to DataFrame
        df = pd.DataFrame({
            'open': market_data.get('open', []),
            'high': market_data.get('high', []),
            'low': market_data.get('low', []),
            'close': market_data.get('close', []),
            'volume': market_data.get('volume', [])
        })
        
        symbol = market_data.get("symbol", "EURUSD")
        
        # Generate signal through orchestrator
        signal = orchestrator.generate_signal(df, symbol)
        
        return signal
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error generating orchestrator signal: {str(e)}")
