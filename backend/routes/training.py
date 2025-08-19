# training.py
# ARIA Training API Routes
# Natural language training commands + streaming data pipeline

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging

from backend.core.training_connector import TrainingConnector, parse_training_command

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/training", tags=["training"])

# Global training connector instance
training_connector = TrainingConnector()


class TrainingRequest(BaseModel):
    """Training request model."""

    command: Optional[str] = None  # Natural language command
    model_type: Optional[str] = "lstm"
    symbol: Optional[str] = "XAUUSD"
    timeframe: Optional[str] = "M5"
    days_back: Optional[int] = 3
    continuous: Optional[bool] = False
    batch_size: Optional[int] = 32
    epochs: Optional[int] = 10


class TrainingResponse(BaseModel):
    """Training response model."""

    status: str
    message: str
    params: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None


@router.post("/start")
async def start_training(
    request: TrainingRequest, background_tasks: BackgroundTasks
) -> TrainingResponse:
    """
    Start model training with natural language or explicit parameters.

    Examples:
        POST with command: "Train XAUUSD M5 for last 3 days"
        POST with params: {"model_type": "lstm", "symbol": "XAUUSD", ...}
    """
    try:
        # Parse natural language command if provided
        if request.command:
            params = await parse_training_command(request.command)
            logger.info(f"Parsed training command: {request.command} -> {params}")
        else:
            # Use explicit parameters
            params = {
                "model_type": request.model_type,
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "days_back": request.days_back,
                "continuous": request.continuous,
                "batch_size": request.batch_size,
                "epochs": request.epochs,
            }

        # Check if training is already active
        if training_connector.training_active:
            return TrainingResponse(
                status="error", message="Training already in progress", params=params
            )

        # Start training
        if params["continuous"]:
            # Start continuous training in background
            background_tasks.add_task(
                training_connector.continuous_training,
                params["model_type"],
                params["symbol"],
                params["timeframe"],
            )

            return TrainingResponse(
                status="started",
                message=f"Continuous {params['model_type']} training started for {params['symbol']} {params['timeframe']}",
                params=params,
            )
        else:
            # Start batch training in background
            background_tasks.add_task(
                training_connector.train_model,
                params["model_type"],
                params["symbol"],
                params["timeframe"],
                params["days_back"],
                params["batch_size"],
                params["epochs"],
            )

            return TrainingResponse(
                status="started",
                message=f"Training {params['model_type']} on {params['days_back']} days of {params['symbol']} {params['timeframe']} data",
                params=params,
            )

    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_training() -> Dict[str, str]:
    """Stop active training."""
    training_connector.stop_training()
    return {"status": "stopped", "message": "Training stopped successfully"}


@router.get("/status")
async def get_training_status() -> Dict[str, Any]:
    """Get current training status and statistics."""
    return training_connector.get_training_status()


@router.post("/parse-command")
async def parse_command(command: str) -> Dict[str, Any]:
    """
    Parse a natural language training command.

    Examples:
        "Train LSTM on XAUUSD M5 for last 7 days"
        "Start continuous CNN training for EURUSD H1"
        "Train PPO on GBPUSD M15"
    """
    try:
        params = await parse_training_command(command)
        return {"command": command, "parsed_params": params, "status": "success"}
    except Exception as e:
        return {"command": command, "error": str(e), "status": "error"}


@router.get("/models")
async def list_available_models() -> Dict[str, Any]:
    """List available models for training."""
    return {
        "models": [
            {
                "type": "lstm",
                "name": "LSTM Sequence Predictor",
                "description": "Long Short-Term Memory network for time series prediction",
                "input": "Sequential OHLCV data",
                "output": "Direction probability",
            },
            {
                "type": "cnn",
                "name": "CNN Pattern Detector",
                "description": "Convolutional network for pattern recognition",
                "input": "Price chart patterns",
                "output": "Pattern classification",
            },
            {
                "type": "ppo",
                "name": "PPO Reinforcement Trader",
                "description": "Proximal Policy Optimization for trading decisions",
                "input": "Market state features",
                "output": "Trading actions",
            },
            {
                "type": "xgb",
                "name": "XGBoost Classifier",
                "description": "Gradient boosting for price direction",
                "input": "Technical indicators",
                "output": "Buy/sell signals",
            },
        ]
    }


@router.get("/symbols")
async def list_supported_symbols() -> Dict[str, Any]:
    """List supported trading symbols."""
    return {
        "symbols": [
            {"symbol": "XAUUSD", "name": "Gold/USD", "type": "commodity"},
            {"symbol": "EURUSD", "name": "Euro/USD", "type": "forex"},
            {"symbol": "GBPUSD", "name": "British Pound/USD", "type": "forex"},
            {"symbol": "USDJPY", "name": "USD/Japanese Yen", "type": "forex"},
            {"symbol": "AUDUSD", "name": "Australian Dollar/USD", "type": "forex"},
            {"symbol": "USDCHF", "name": "USD/Swiss Franc", "type": "forex"},
            {"symbol": "USDCAD", "name": "USD/Canadian Dollar", "type": "forex"},
            {"symbol": "NZDUSD", "name": "New Zealand Dollar/USD", "type": "forex"},
        ]
    }


@router.get("/timeframes")
async def list_timeframes() -> Dict[str, Any]:
    """List supported timeframes."""
    return {
        "timeframes": [
            {"code": "M1", "name": "1 Minute", "ms": 60000},
            {"code": "M5", "name": "5 Minutes", "ms": 300000},
            {"code": "M15", "name": "15 Minutes", "ms": 900000},
            {"code": "M30", "name": "30 Minutes", "ms": 1800000},
            {"code": "H1", "name": "1 Hour", "ms": 3600000},
            {"code": "H4", "name": "4 Hours", "ms": 14400000},
            {"code": "D1", "name": "1 Day", "ms": 86400000},
        ]
    }
