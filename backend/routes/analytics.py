from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import sqlite3
import json
import os
from backend.core.trade_memory import TradeMemory

router = APIRouter(prefix="/api/analytics", tags=["Analytics"])


class EquityCurvePoint(BaseModel):
    timestamp: str
    equity: float
    drawdown: float
    trades: int


class PerformanceMetrics(BaseModel):
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade: float
    best_trade: float
    worst_trade: float


class HeatmapData(BaseModel):
    symbol: str
    hour: int
    day_of_week: int
    avg_pnl: float
    trade_count: int
    win_rate: float


@router.get("/equity-curve")
def get_equity_curve(
    days: int = Query(30, ge=1, le=365), symbol: Optional[str] = None
) -> Dict[str, Any]:
    """Get equity curve data for performance visualization"""
    try:
        tm = TradeMemory()

        # Get trades from last N days
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"

        query = """
        SELECT ts, payload, metadata 
        FROM trade_memory 
        WHERE ts >= ? 
        ORDER BY ts ASC
        """
        params = [cutoff]

        if symbol:
            query = query.replace("WHERE ts >= ?", "WHERE ts >= ? AND symbol = ?")
            params.append(symbol.upper())

        with sqlite3.connect(tm.db_path) as conn:
            cur = conn.cursor()
            cur.execute(query, params)
            rows = cur.fetchall()

        # Calculate running equity
        equity_points = []
        running_equity = 10000.0  # Starting equity
        peak_equity = running_equity
        trade_count = 0

        for row in rows:
            try:
                metadata = json.loads(row[2]) if row[2] else {}
                outcome = metadata.get("outcome", {})
                pnl = outcome.get("pnl", 0)

                if pnl != 0:  # Only count closed trades
                    trade_count += 1
                    running_equity += pnl
                    peak_equity = max(peak_equity, running_equity)
                    drawdown = (peak_equity - running_equity) / peak_equity * 100

                    equity_points.append(
                        {
                            "timestamp": row[0],
                            "equity": running_equity,
                            "drawdown": drawdown,
                            "trades": trade_count,
                        }
                    )
            except Exception:
                continue

        return {
            "ok": True,
            "data": equity_points,
            "summary": {
                "final_equity": running_equity,
                "total_return": (running_equity - 10000) / 10000 * 100,
                "max_drawdown": max([p["drawdown"] for p in equity_points], default=0),
                "total_trades": trade_count,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance-metrics")
def get_performance_metrics(
    days: int = Query(30, ge=1, le=365), symbol: Optional[str] = None
) -> Dict[str, Any]:
    """Calculate comprehensive performance metrics"""
    try:
        tm = TradeMemory()
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"

        query = """
        SELECT payload, metadata 
        FROM trade_memory 
        WHERE ts >= ? 
        """
        params = [cutoff]

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol.upper())

        with sqlite3.connect(tm.db_path) as conn:
            cur = conn.cursor()
            cur.execute(query, params)
            rows = cur.fetchall()

        trades = []
        for row in rows:
            try:
                metadata = json.loads(row[1]) if row[1] else {}
                outcome = metadata.get("outcome", {})
                pnl = outcome.get("pnl", 0)
                if pnl != 0:
                    trades.append(pnl)
            except Exception:
                continue

        if not trades:
            return {"ok": True, "data": None, "message": "No closed trades found"}

        # Calculate metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t > 0]
        losing_trades = [t for t in trades if t < 0]

        total_return = sum(trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_trade = total_return / total_trades if total_trades > 0 else 0

        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Calculate Sharpe ratio (simplified)
        if len(trades) > 1:
            import statistics

            sharpe_ratio = (
                statistics.mean(trades) / statistics.stdev(trades) * (252**0.5)
                if statistics.stdev(trades) > 0
                else 0
            )
        else:
            sharpe_ratio = 0

        # Calculate max drawdown
        running_equity = 10000.0
        peak = running_equity
        max_dd = 0

        for trade in trades:
            running_equity += trade
            if running_equity > peak:
                peak = running_equity
            else:
                dd = (peak - running_equity) / peak * 100
                max_dd = max(max_dd, dd)

        metrics = {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": total_trades,
            "avg_trade": avg_trade,
            "best_trade": max(trades) if trades else 0,
            "worst_trade": min(trades) if trades else 0,
        }

        return {"ok": True, "data": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trading-heatmap")
def get_trading_heatmap(days: int = Query(30, ge=1, le=365)) -> Dict[str, Any]:
    """Get trading performance heatmap by hour and day of week"""
    try:
        tm = TradeMemory()
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"

        with sqlite3.connect(tm.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT ts, symbol, payload, metadata 
                FROM trade_memory 
                WHERE ts >= ? 
                ORDER BY ts ASC
            """,
                [cutoff],
            )
            rows = cur.fetchall()

        # Group by symbol, hour, and day of week
        heatmap_data = {}

        for row in rows:
            try:
                timestamp = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
                symbol = row[1]
                metadata = json.loads(row[3]) if row[3] else {}
                outcome = metadata.get("outcome", {})
                pnl = outcome.get("pnl", 0)

                if pnl == 0:  # Skip open trades
                    continue

                hour = timestamp.hour
                day_of_week = timestamp.weekday()

                key = f"{symbol}_{hour}_{day_of_week}"

                if key not in heatmap_data:
                    heatmap_data[key] = {
                        "symbol": symbol,
                        "hour": hour,
                        "day_of_week": day_of_week,
                        "trades": [],
                        "wins": 0,
                    }

                heatmap_data[key]["trades"].append(pnl)
                if pnl > 0:
                    heatmap_data[key]["wins"] += 1

            except Exception:
                continue

        # Calculate aggregated metrics
        result = []
        for data in heatmap_data.values():
            if data["trades"]:
                result.append(
                    {
                        "symbol": data["symbol"],
                        "hour": data["hour"],
                        "day_of_week": data["day_of_week"],
                        "avg_pnl": sum(data["trades"]) / len(data["trades"]),
                        "trade_count": len(data["trades"]),
                        "win_rate": data["wins"] / len(data["trades"]) * 100,
                    }
                )

        return {"ok": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/symbol-performance")
def get_symbol_performance(days: int = Query(30, ge=1, le=365)) -> Dict[str, Any]:
    """Get performance breakdown by trading symbol"""
    try:
        tm = TradeMemory()
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat() + "Z"

        with sqlite3.connect(tm.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT symbol, metadata 
                FROM trade_memory 
                WHERE ts >= ? 
            """,
                [cutoff],
            )
            rows = cur.fetchall()

        symbol_stats = {}

        for row in rows:
            try:
                symbol = row[0]
                metadata = json.loads(row[1]) if row[1] else {}
                outcome = metadata.get("outcome", {})
                pnl = outcome.get("pnl", 0)

                if pnl == 0:  # Skip open trades
                    continue

                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {"symbol": symbol, "trades": [], "wins": 0}

                symbol_stats[symbol]["trades"].append(pnl)
                if pnl > 0:
                    symbol_stats[symbol]["wins"] += 1

            except Exception:
                continue

        # Calculate metrics per symbol
        result = []
        for stats in symbol_stats.values():
            if stats["trades"]:
                total_pnl = sum(stats["trades"])
                trade_count = len(stats["trades"])
                win_rate = stats["wins"] / trade_count * 100

                result.append(
                    {
                        "symbol": stats["symbol"],
                        "total_pnl": total_pnl,
                        "trade_count": trade_count,
                        "win_rate": win_rate,
                        "avg_trade": total_pnl / trade_count,
                        "best_trade": max(stats["trades"]),
                        "worst_trade": min(stats["trades"]),
                    }
                )

        # Sort by total PnL descending
        result.sort(key=lambda x: x["total_pnl"], reverse=True)

        return {"ok": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
