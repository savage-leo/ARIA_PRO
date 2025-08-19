# Replace relevant endpoint functions in backend/routes/market.py
from fastapi import APIRouter, HTTPException
from typing import Dict
import os
from backend.services.data_source_manager import DataSourceManager, FeedUnavailableError

router = APIRouter()

# Use global data source manager
from backend.services.data_source_manager import data_source_manager as data_manager


@router.get("/last_bar/{symbol}")
def get_last_bar(symbol: str) -> Dict:
    try:
        bar = data_manager.get_last_bar(symbol)
        return {"ok": True, "bar": bar}
    except FeedUnavailableError as e:
        # If MT5 is required, do not fallback to simulation; return 503
        raise HTTPException(
            status_code=503, detail=f"Market data unavailable: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
