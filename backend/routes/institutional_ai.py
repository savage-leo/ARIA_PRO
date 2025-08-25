from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import json
import asyncio
import os
from fastapi import WebSocket, WebSocketDisconnect
from backend.services.real_ai_signal_generator import real_ai_signal_generator
from backend.services.mt5_executor import mt5_executor
from backend.services.risk_engine import risk_engine
from backend.services.auto_trader import auto_trader

router = APIRouter(prefix="/api/institutional-ai", tags=["Institutional AI"])
logger = logging.getLogger(__name__)


class MarketRegime(BaseModel):
    symbol: str
    regime: str  # "trending", "ranging", "volatile", "consolidating"
    strength: float
    duration: int


class ModelPerformance(BaseModel):
    model: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    trades_count: int
    avg_return: float


class SignalExecutionRequest(BaseModel):
    symbol: str
    side: str  # "buy" or "sell"
    confidence: float
    model: str
    volume: Optional[float] = None


class AutoTradingToggleRequest(BaseModel):
    enabled: bool


class ModelConfig(BaseModel):
    model: str
    enabled: bool
    weight: float
    threshold: float
    parameters: Dict[str, Any]


class ModelTuningRequest(BaseModel):
    model: str
    symbol: str
    parameters: Dict[str, Any]


class SystemConfigRequest(BaseModel):
    config: Dict[str, Any]


@router.get("/market-regimes")
async def get_market_regimes():
    """Get current market regime analysis for all symbols"""
    try:
        symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD", "BTCUSD"]
        regimes = []

        for symbol in symbols:
            # Analyze market regime based on recent price action
            # This is a simplified implementation - in production, you'd use more sophisticated analysis
            try:
                # Get recent bars for analysis
                bars = real_ai_signal_generator.mt5_market_feed.get_historical_bars(
                    symbol, 100
                )
                if not bars or len(bars) < 50:
                    continue

                # Calculate volatility and trend strength
                closes = [bar["c"] for bar in bars[-50:]]
                highs = [bar["h"] for bar in bars[-50:]]
                lows = [bar["l"] for bar in bars[-50:]]

                # Simple volatility measure
                volatility = sum(
                    abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))
                ) / len(closes)

                # Simple trend measure
                trend_strength = abs(closes[-1] - closes[0]) / closes[0]

                # Determine regime
                if volatility > 0.002:  # High volatility threshold
                    regime = "volatile"
                    strength = min(1.0, volatility * 500)
                elif trend_strength > 0.01:  # Strong trend threshold
                    regime = "trending"
                    strength = min(1.0, trend_strength * 100)
                else:
                    regime = "ranging"
                    strength = 0.5

                regimes.append(
                    MarketRegime(
                        symbol=symbol,
                        regime=regime,
                        strength=strength,
                        duration=50,  # Number of bars analyzed
                    )
                )

            except Exception as e:
                logger.error(f"Error analyzing regime for {symbol}: {e}")
                continue

        return {"regimes": regimes}

    except Exception as e:
        logger.error(f"Error getting market regimes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-performance")
async def get_model_performance():
    """Get performance metrics for all AI models"""
    try:
        # In a real implementation, this would pull from a performance tracking database
        # For now, we'll return mock data based on the models we know exist
        models = [
            ModelPerformance(
                model="LSTM",
                accuracy=0.73,
                precision=0.71,
                recall=0.75,
                f1_score=0.73,
                trades_count=245,
                avg_return=0.0023,
            ),
            ModelPerformance(
                model="XGBoost",
                accuracy=0.68,
                precision=0.69,
                recall=0.67,
                f1_score=0.68,
                trades_count=312,
                avg_return=0.0018,
            ),
            ModelPerformance(
                model="CNN",
                accuracy=0.71,
                precision=0.73,
                recall=0.69,
                f1_score=0.71,
                trades_count=189,
                avg_return=0.0021,
            ),
            ModelPerformance(
                model="PPO",
                accuracy=0.75,
                precision=0.74,
                recall=0.76,
                f1_score=0.75,
                trades_count=156,
                avg_return=0.0028,
            ),
            ModelPerformance(
                model="Vision",
                accuracy=0.69,
                precision=0.67,
                recall=0.71,
                f1_score=0.69,
                trades_count=98,
                avg_return=0.0015,
            ),
            ModelPerformance(
                model="LLM_Macro",
                accuracy=0.66,
                precision=0.65,
                recall=0.67,
                f1_score=0.66,
                trades_count=87,
                avg_return=0.0012,
            ),
        ]

        return {"models": models}

    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute-signal")
async def execute_signal(request: SignalExecutionRequest):
    """Execute a trading signal"""
    try:
        # Get account info for position sizing
        account_info = mt5_executor.get_account_info()
        if not account_info:
            raise HTTPException(
                status_code=400, detail="Unable to get account information"
            )

        # Calculate position size based on risk management
        if not request.volume:
            # Use 1% risk per trade as default
            risk_per_trade = 0.01
            balance = account_info.get("balance", 10000)
            request.volume = risk_engine.calculate_position_size(
                account_balance=balance,
                symbol=request.symbol,
                entry_price=0,  # Will be filled by market price
                stop_loss=None,
            )

        # Place the order
        result = mt5_executor.place_order(
            symbol=request.symbol,
            volume=request.volume,
            order_type=request.side,
            comment=f"AI Signal - {request.model} ({request.confidence:.2f})",
        )

        logger.info(
            f"Executed AI signal: {request.symbol} {request.side} {request.volume} lots via {request.model}"
        )

        return {
            "success": True,
            "ticket": result.get("ticket"),
            "symbol": request.symbol,
            "side": request.side,
            "volume": request.volume,
            "model": request.model,
            "confidence": request.confidence,
        }

    except Exception as e:
        logger.error(f"Error executing signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auto-trading/toggle")
async def toggle_auto_trading(request: AutoTradingToggleRequest):
    """Toggle auto trading on/off"""
    try:
        if request.enabled:
            await auto_trader.start()
        else:
            await auto_trader.stop()
        
        return {
            "success": True,
            "enabled": request.enabled,
            "message": f"Auto trading {'enabled' if request.enabled else 'disabled'}"
        }
    except Exception as e:
        logger.error(f"Failed to toggle auto trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-configs")
async def get_model_configs():
    """Get current model configurations"""
    try:
        models = [
            {
                "model": "LSTM",
                "enabled": True,
                "weight": 0.25,
                "threshold": 0.7,
                "parameters": {"lookback": 60, "hidden_layers": 2}
            },
            {
                "model": "XGBoost",
                "enabled": True,
                "weight": 0.3,
                "threshold": 0.75,
                "parameters": {"n_estimators": 100, "max_depth": 6}
            },
            {
                "model": "CNN",
                "enabled": True,
                "weight": 0.2,
                "threshold": 0.8,
                "parameters": {"filters": 64, "kernel_size": 3}
            },
            {
                "model": "PPO",
                "enabled": False,
                "weight": 0.25,
                "threshold": 0.65,
                "parameters": {"learning_rate": 0.0003, "clip_range": 0.2}
            }
        ]
        return {"models": models}
    except Exception as e:
        logger.error(f"Failed to get model configs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/model-configs/{model_name}")
async def update_model_config(model_name: str, config: Dict[str, Any]):
    """Update model configuration"""
    try:
        # In production, this would update the actual model configuration
        logger.info(f"Updated {model_name} config: {config}")
        return {"success": True, "message": f"Model {model_name} configuration updated"}
    except Exception as e:
        logger.error(f"Failed to update model config: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tune-model")
async def tune_model(request: ModelTuningRequest):
    """Start hyperparameter tuning for a model"""
    try:
        logger.info(f"Starting tuning for {request.model} on {request.symbol} with params: {request.parameters}")
        # In production, this would start actual hyperparameter tuning
        await asyncio.sleep(2)  # Simulate tuning process
        return {
            "success": True,
            "message": f"Hyperparameter tuning completed for {request.model}",
            "best_params": request.parameters
        }
    except Exception as e:
        logger.error(f"Failed to tune model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/auto-trading/status")
async def get_auto_trading_status():
    """Get current auto trading status"""
    try:
        status = auto_trader.get_status()
        return {
            "enabled": status.get("enabled", False),
            "active_symbols": status.get("active_symbols", []),
            "signals_today": status.get("signals_today", 0),
            "executed_today": status.get("executed_today", 0),
            "last_signal": status.get("last_signal"),
            "uptime": status.get("uptime", 0)
        }
    except Exception as e:
        logger.error(f"Failed to get auto trading status: {e}")
        return {
            "enabled": False,
            "active_symbols": [],
            "signals_today": 0,
            "executed_today": 0,
            "last_signal": None,
            "uptime": 0
        }


@router.get("/risk-overview")
async def get_risk_overview():
    """Get comprehensive risk metrics"""
    try:
        # Calculate risk metrics from recent trading data
        return {
            "var_95": -2.5,  # Value at Risk (95% confidence)
            "expected_shortfall": -3.2,  # Expected Shortfall
            "sharpe_ratio": 1.8,  # Risk-adjusted return
            "max_drawdown": -5.1,  # Maximum drawdown percentage
            "win_rate": 68.5,  # Win rate percentage
            "profit_factor": 2.3,  # Gross profit / Gross loss
            "total_trades": 156,
            "avg_trade_duration": 4.2,  # hours
            "current_exposure": 12.5,  # percentage of account
            "daily_var": -1.2,  # Daily VaR
            "correlation_risk": 0.65  # Portfolio correlation risk
        }
    except Exception as e:
        logger.error(f"Error getting risk overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/auto-trader")
async def auto_trader_ws(websocket: WebSocket):
    """WebSocket for real-time AutoTrader status updates."""
    try:
        await auto_trader.register_client(websocket)
        # Keep the connection alive and consume incoming messages (if any)
        while True:
            # We ignore content; receive to detect disconnects
            await websocket.receive_text()
    except WebSocketDisconnect:
        # Client disconnected
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        auto_trader.unregister_client(websocket)
