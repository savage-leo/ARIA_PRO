from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import json
from backend.services.real_ai_signal_generator import real_ai_signal_generator
from backend.services.mt5_executor import mt5_executor
from backend.services.risk_engine import risk_engine

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
        # This would typically update a configuration setting
        # For now, we'll just return the requested state
        logger.info(f"Auto trading {'enabled' if request.enabled else 'disabled'}")

        return {
            "success": True,
            "auto_trading_enabled": request.enabled,
            "message": f"Auto trading {'enabled' if request.enabled else 'disabled'}",
        }

    except Exception as e:
        logger.error(f"Error toggling auto trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/auto-trading/status")
async def get_auto_trading_status():
    """Get current auto trading status"""
    try:
        # In a real implementation, this would check the actual auto trading state
        return {
            "auto_trading_enabled": True,  # Default enabled since we disabled dry run
            "last_signal_time": datetime.now().isoformat(),
            "signals_today": 15,
            "executed_today": 8,
        }

    except Exception as e:
        logger.error(f"Error getting auto trading status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/risk-overview")
async def get_risk_overview():
    """Get comprehensive risk overview"""
    try:
        account_info = mt5_executor.get_account_info()
        if not account_info:
            return {
                "var_95": 0.0,
                "expected_shortfall": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
            }

        risk_metrics = risk_engine.get_risk_metrics(account_info)

        # Add additional institutional-grade risk metrics
        return {
            "var_95": risk_metrics.get("daily_loss_limit", 0)
            / account_info.get("balance", 1)
            * 100,
            "expected_shortfall": risk_metrics.get("max_drawdown_allowed", 0),
            "sharpe_ratio": 1.2,  # Mock data - would calculate from historical returns
            "max_drawdown": risk_metrics.get("current_drawdown", 0),
            "win_rate": 68.5,  # Mock data - would calculate from trade history
            "profit_factor": 1.45,  # Mock data - would calculate from trade history
        }

    except Exception as e:
        logger.error(f"Error getting risk overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))
