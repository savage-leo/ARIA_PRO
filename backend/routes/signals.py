# Replace relevant functions in backend/routes/signals.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from backend.services.data_source_manager import FeedUnavailableError

router = APIRouter()
# Use global data source manager
from backend.services.data_source_manager import data_source_manager as data_manager


class GenerateSignalReq(BaseModel):
    symbol: str
    timeframe: str = "M5"
    bars: int = 100
    features: Dict[str, Any] = {}


@router.post("/generate")
def generate_signals(req: GenerateSignalReq):
    try:
        # Enforce MT5-only; build features with timeframe and bars
        feats: Dict[str, Any] = dict(req.features or {})
        feats["timeframe"] = req.timeframe
        feats["bars"] = req.bars

        signals = data_manager.get_ai_signals(req.symbol, feats)
        return {"ok": True, "signals": signals}
    except FeedUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_signal_history(symbol: str = None, limit: int = 50):
    """Get historical AI signals"""
    try:
        # In a real implementation, this would query a database of historical signals
        # For now, return mock data that matches the expected format
        from datetime import datetime, timedelta
        import random

        symbols = (
            [symbol]
            if symbol
            else ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "XAUUSD", "BTCUSD"]
        )
        models = ["LSTM", "XGBoost", "CNN", "PPO", "Vision", "LLM_Macro"]

        signals = []
        for i in range(min(limit, 100)):
            timestamp = datetime.now() - timedelta(minutes=i * 15)
            selected_symbol = random.choice(symbols)

            signal = {
                "symbol": selected_symbol,
                "side": random.choice(["buy", "sell"]),
                "strength": random.uniform(0.3, 0.9),
                "confidence": random.uniform(0.5, 0.95),
                "model": random.choice(models),
                "timestamp": int(timestamp.timestamp() * 1000),
                "entry_price": random.uniform(1.0, 150.0),
                "stop_loss": None,
                "take_profit": None,
            }
            signals.append(signal)

        return {"signals": signals}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
