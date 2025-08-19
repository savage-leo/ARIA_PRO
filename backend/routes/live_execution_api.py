# -*- coding: utf-8 -*-
"""
Live Execution API: Control the MT5 execution harness
Real-time trading control, monitoring, and risk management
"""
from __future__ import annotations
import os, time
from typing import Dict, List, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from backend.services.mt5_execution_harness import mt5_execution_harness
from backend.services.hedge_fund_analytics import hedge_fund_analytics

router = APIRouter(prefix="/live-execution", tags=["Live Execution"])


class SymbolToggle(BaseModel):
    symbol: str
    enabled: bool


class KillSwitchCommand(BaseModel):
    action: str  # 'activate' | 'deactivate'
    reason: str = ""


@router.get("/status")
def get_execution_status() -> Dict[str, Any]:
    """Get live execution system status"""
    return mt5_execution_harness.get_status()


@router.post("/connect")
def connect_mt5() -> Dict[str, str]:
    """Connect to MT5 for live trading"""
    try:
        success = mt5_execution_harness.connect()
        if success:
            return {"status": "success", "message": "Connected to MT5"}
        else:
            return {"status": "error", "message": "Failed to connect to MT5"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/disconnect")
def disconnect_mt5() -> Dict[str, str]:
    """Disconnect from MT5"""
    try:
        mt5_execution_harness.disconnect()
        return {"status": "success", "message": "Disconnected from MT5"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/symbol/toggle")
def toggle_symbol(toggle: SymbolToggle) -> Dict[str, str]:
    """Enable or disable trading for a symbol"""
    try:
        if toggle.enabled:
            mt5_execution_harness.enable_symbol(toggle.symbol)
            message = f"Enabled trading for {toggle.symbol}"
        else:
            mt5_execution_harness.disable_symbol(toggle.symbol)
            message = f"Disabled trading for {toggle.symbol}"

        return {"status": "success", "message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/symbols/supported")
def get_supported_symbols() -> Dict[str, List[str]]:
    """Get all supported multi-asset symbols"""
    symbols = mt5_execution_harness.get_multi_asset_symbols()

    # Categorize symbols
    categorized = {
        "forex": [
            s
            for s in symbols
            if any(
                curr in s
                for curr in ["USD", "EUR", "GBP", "JPY", "AUD", "NZD", "CHF", "CAD"]
            )
        ],
        "commodities": [
            s for s in symbols if any(comm in s for comm in ["XAU", "XAG", "OIL"])
        ],
        "indices": [
            s
            for s in symbols
            if any(idx in s for idx in ["US30", "SPX", "NAS", "GER", "UK"])
        ],
        "crypto": [
            s
            for s in symbols
            if any(crypto in s for crypto in ["BTC", "ETH", "ADA", "DOT"])
        ],
    }

    return categorized


@router.post("/kill-switch")
def control_kill_switch(command: KillSwitchCommand) -> Dict[str, str]:
    """Control the kill switch"""
    try:
        if command.action == "activate":
            reason = command.reason or "Manual activation"
            mt5_execution_harness.activate_kill_switch(reason)
            return {"status": "success", "message": f"Kill switch activated: {reason}"}
        elif command.action == "deactivate":
            mt5_execution_harness.deactivate_kill_switch()
            return {"status": "success", "message": "Kill switch deactivated"}
        else:
            raise ValueError("Action must be 'activate' or 'deactivate'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions")
def get_open_positions() -> Dict[str, Any]:
    """Get current open positions"""
    status = mt5_execution_harness.get_status()
    return {
        "open_positions": status["position_tracker"],
        "position_count": status["open_positions"],
    }


@router.get("/execution-stats")
def get_execution_statistics() -> Dict[str, Any]:
    """Get execution performance statistics"""
    status = mt5_execution_harness.get_status()
    return status["execution_stats"]


@router.get("/audit-summary")
def get_audit_summary() -> Dict[str, Any]:
    """Get audit trail summary"""
    hedge_fund_data = hedge_fund_analytics.get_live_dashboard_data()
    execution_status = mt5_execution_harness.get_status()

    return {
        "total_trades_today": execution_status["daily_trade_count"],
        "daily_pnl": execution_status["daily_pnl"],
        "portfolio_metrics": hedge_fund_data.get("portfolio", {}),
        "top_strategies": hedge_fund_data.get("top_strategies", []),
        "execution_performance": execution_status["execution_stats"],
    }


@router.post("/quick-start")
def quick_start_trading(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """Quick start live trading with default symbols"""
    try:
        # Connect to MT5
        if not mt5_execution_harness.connect():
            raise Exception("Failed to connect to MT5")

        # Enable major forex pairs
        major_pairs = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF"]
        for symbol in major_pairs:
            mt5_execution_harness.enable_symbol(symbol)

        return {
            "status": "success",
            "message": f"Live trading started with {len(major_pairs)} symbols: {', '.join(major_pairs)}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/emergency-stop")
def emergency_stop() -> Dict[str, str]:
    """Emergency stop - activate kill switch and disconnect"""
    try:
        mt5_execution_harness.activate_kill_switch("Emergency stop activated")
        mt5_execution_harness.disconnect()
        return {
            "status": "success",
            "message": "Emergency stop executed - all trading halted",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/live-metrics")
def get_live_metrics() -> Dict[str, Any]:
    """Get real-time execution metrics"""
    execution_status = mt5_execution_harness.get_status()
    hedge_fund_data = hedge_fund_analytics.get_live_dashboard_data()

    return {
        "timestamp": time.time(),
        "system_status": {
            "mt5_connected": execution_status["is_live"],
            "kill_switch": execution_status["kill_switch_active"],
            "active_symbols": len(execution_status["enabled_symbols"]),
            "open_positions": execution_status["open_positions"],
        },
        "performance": {
            "daily_trades": execution_status["daily_trade_count"],
            "success_rate": (
                execution_status["execution_stats"]["successful_orders"]
                / max(1, execution_status["execution_stats"]["total_orders"])
            ),
            "avg_latency_ms": execution_status["execution_stats"]["avg_latency_ms"],
            "avg_slippage": execution_status["execution_stats"]["avg_slippage"],
        },
        "portfolio": {
            "daily_pnl": execution_status["daily_pnl"],
            "total_pnl": hedge_fund_data.get("inception_pnl", 0),
            "sharpe_ratio": hedge_fund_data.get("portfolio", {}).get("sharpe_ratio", 0),
            "max_drawdown": hedge_fund_data.get("portfolio", {}).get("max_drawdown", 0),
        },
    }

