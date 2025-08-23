# Replace relevant endpoint functions in backend/routes/market.py
from fastapi import APIRouter, HTTPException
from typing import Dict
import os
from datetime import datetime
from backend.services.data_source_manager import DataSourceManager, FeedUnavailableError
from backend.core.redis_cache import get_cache_manager

router = APIRouter()

# Use global data source manager
from backend.services.data_source_manager import data_source_manager as data_manager


@router.get("/last_bar/{symbol}")
async def get_last_bar(symbol: str) -> Dict:
    cache_manager = get_cache_manager()
    
    # Try cache first
    cached_data = await cache_manager.get_market_data(symbol, "M1")
    if cached_data:
        return {"ok": True, "bar": cached_data, "cached": True}
    
    try:
        bar = data_manager.get_last_bar(symbol)
        
        # Cache the result
        await cache_manager.cache_market_data(symbol, "M1", bar, datetime.now())
        
        return {"ok": True, "bar": bar, "cached": False}
    except FeedUnavailableError as e:
        # If MT5 is required, do not fallback to simulation; return 503
        raise HTTPException(
            status_code=503, detail=f"Market data unavailable: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
