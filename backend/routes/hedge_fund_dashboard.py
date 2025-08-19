# -*- coding: utf-8 -*-
"""
Hedge Fund Dashboard: Real-time monitoring endpoints for multi-strategy fund
Institutional-grade analytics optimized for T470
"""
from __future__ import annotations
import os, time, pathlib
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Response, Query
from fastapi.responses import HTMLResponse

from backend.services.hedge_fund_analytics import hedge_fund_analytics
from backend.models.model_factory import model_factory
from backend.services.lightweight_ensemble import t470_ensemble

router = APIRouter(prefix="/hedge-fund", tags=["Hedge Fund"])


@router.get("/dashboard")
async def get_dashboard() -> HTMLResponse:
    """Hedge fund style dashboard for T470"""

    # Get live data
    dashboard_data = hedge_fund_analytics.get_live_dashboard_data()
    memory_info = model_factory.get_memory_footprint()
    ensemble_status = t470_ensemble.get_performance_summary()

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>ARIA Multi-Strategy Hedge Fund</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 15px; background: #0a0e1a; color: #e1e5e9; }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            .header {{ background: linear-gradient(135deg, #1a237e 0%, #3949ab 100%); padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 15px; }}
            .card {{ background: #1e293b; padding: 18px; border-radius: 8px; border: 1px solid #334155; }}
            .metric {{ display: flex; justify-content: space-between; margin: 8px 0; }}
            .metric-label {{ font-weight: 500; color: #94a3b8; }}
            .metric-value {{ font-weight: 600; color: #f1f5f9; }}
            .positive {{ color: #10b981; }}
            .negative {{ color: #ef4444; }}
            .neutral {{ color: #f59e0b; }}
            .strategy-row {{ display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid #374151; }}
            .strategy-name {{ font-weight: 500; }}
            .pnl-value {{ font-weight: 600; }}
            .small-text {{ font-size: 0.85em; color: #9ca3af; }}
            .status-indicator {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; }}
            .status-green {{ background: #10b981; }}
            .status-red {{ background: #ef4444; }}
            .status-yellow {{ background: #f59e0b; }}
            .refresh-btn {{ background: #3b82f6; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; }}
            .refresh-btn:hover {{ background: #2563eb; }}
        </style>
        <script>
            function refreshDashboard() {{ location.reload(); }}
            setInterval(refreshDashboard, 30000); // Auto-refresh every 30 seconds
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏛️ ARIA Multi-Strategy Hedge Fund</h1>
                <p>Institutional Trading System • T470 Optimized • Real-time Analytics</p>
                <button class="refresh-btn" onclick="refreshDashboard()">🔄 Refresh</button>
            </div>
            
            <div class="grid">
                <!-- Portfolio Performance -->
                <div class="card">
                    <h3>📈 Portfolio Performance</h3>
                    <div class="metric">
                        <span class="metric-label">Total P&L:</span>
                        <span class="metric-value {'positive' if dashboard_data.get('inception_pnl', 0) >= 0 else 'negative'}">${dashboard_data.get('inception_pnl', 0):.2f}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Daily P&L:</span>
                        <span class="metric-value {'positive' if dashboard_data.get('daily_pnl', 0) >= 0 else 'negative'}">${dashboard_data.get('daily_pnl', 0):.2f}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Sharpe Ratio:</span>
                        <span class="metric-value">{dashboard_data.get('portfolio', {}).get('sharpe_ratio', 0):.3f}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Max Drawdown:</span>
                        <span class="metric-value negative">{dashboard_data.get('portfolio', {}).get('max_drawdown', 0):.1f}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Win Rate:</span>
                        <span class="metric-value">{dashboard_data.get('portfolio', {}).get('win_rate', 0):.1f}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Total Trades:</span>
                        <span class="metric-value">{dashboard_data.get('portfolio', {}).get('total_trades', 0)}</span>
                    </div>
                </div>
                
                <!-- Strategy Attribution -->
                <div class="card">
                    <h3>🎯 Strategy Attribution</h3>
                    <div class="small-text">Top Performing Models</div>
    """

    # Add top strategies
    for strategy_name, pnl in dashboard_data.get("top_strategies", [])[:6]:
        pnl_class = "positive" if pnl >= 0 else "negative"
        html_content += f"""
                    <div class="strategy-row">
                        <span class="strategy-name">{strategy_name}</span>
                        <span class="pnl-value {pnl_class}">${pnl:.2f}</span>
                    </div>
        """

    html_content += f"""
                    <div class="metric" style="margin-top: 15px;">
                        <span class="metric-label">Active Strategies:</span>
                        <span class="metric-value">{dashboard_data.get('total_strategies', 0)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Active Positions:</span>
                        <span class="metric-value">{dashboard_data.get('active_positions', 0)}</span>
                    </div>
                </div>
                
                <!-- Risk Metrics -->
                <div class="card">
                    <h3>⚠️ Risk Analytics</h3>
                    <div class="metric">
                        <span class="metric-label">VaR (99%):</span>
                        <span class="metric-value negative">${dashboard_data.get('risk_metrics', {}).get('var_1d_99', 0):.4f}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">VaR (95%):</span>
                        <span class="metric-value negative">${dashboard_data.get('risk_metrics', {}).get('var_1d_95', 0):.4f}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Expected Shortfall:</span>
                        <span class="metric-value negative">${dashboard_data.get('risk_metrics', {}).get('expected_shortfall', 0):.4f}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Current Drawdown:</span>
                        <span class="metric-value negative">{dashboard_data.get('portfolio', {}).get('current_drawdown', 0):.1f}%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Volatility (Ann.):</span>
                        <span class="metric-value">{dashboard_data.get('portfolio', {}).get('volatility_annualized', 0):.1f}%</span>
                    </div>
                </div>
                
                <!-- Regime Performance -->
                <div class="card">
                    <h3>🔄 Regime Performance</h3>
    """

    # Add regime performance
    regime_data = dashboard_data.get("regime_performance", {})
    for regime in ["T", "R", "B"]:
        regime_names = {"T": "Trend", "R": "Range", "B": "Breakout"}
        if regime in regime_data:
            data = regime_data[regime]
            pnl_class = "positive" if data.get("total_pnl", 0) >= 0 else "negative"
            html_content += f"""
                    <div class="metric">
                        <span class="metric-label">{regime_names[regime]} ({data.get('trade_count', 0)} trades):</span>
                        <span class="metric-value {pnl_class}">${data.get('total_pnl', 0):.2f}</span>
                    </div>
                    <div class="small-text" style="margin-left: 10px;">Win Rate: {data.get('win_rate', 0):.1f}% | Avg: ${data.get('avg_pnl', 0):.4f}</div>
            """

    html_content += f"""
                </div>
                
                <!-- System Health -->
                <div class="card">
                    <h3>💻 System Health</h3>
                    <div class="metric">
                        <span class="metric-label">
                            <span class="status-indicator status-green"></span>Models Available:
                        </span>
                        <span class="metric-value">{memory_info.get('total_models', 0)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Memory Usage:</span>
                        <span class="metric-value">{memory_info.get('total_memory_mb', 0):.1f}MB</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Ensemble Decisions:</span>
                        <span class="metric-value">{ensemble_status.get('total_decisions', 0)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">
                            <span class="status-indicator {'status-red' if ensemble_status.get('memory_pressure', False) else 'status-green'}"></span>Memory Pressure:
                        </span>
                        <span class="metric-value">{'HIGH' if ensemble_status.get('memory_pressure', False) else 'NORMAL'}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Data Points:</span>
                        <span class="metric-value">{dashboard_data.get('data_points', 0)}</span>
                    </div>
                </div>
                
                <!-- Model Types -->
                <div class="card">
                    <h3>🧠 Model Arsenal</h3>
    """

    # Add model types
    model_types = memory_info.get("by_type", {})
    for model_type, info in model_types.items():
        html_content += f"""
                    <div class="metric">
                        <span class="metric-label">{model_type.title()} Models:</span>
                        <span class="metric-value">{info.get('count', 0)} ({info.get('memory_kb', 0)/1024:.1f}MB)</span>
                    </div>
        """

    html_content += f"""
                    <div class="small-text" style="margin-top: 10px;">
                        Sequence: LSTM, Transformer<br/>
                        Pattern: CNN, Autoencoder<br/>
                        Tabular: XGBoost, LightGBM<br/>
                        Policy: PPO, RL<br/>
                        Probabilistic: Bayesian
                    </div>
                </div>
            </div>
            
            <div class="card" style="margin-top: 20px;">
                <h3>🔗 Quick Actions</h3>
                <p>
                    <a href="/hedge-fund/performance" style="color: #3b82f6;">Performance Report</a> | 
                    <a href="/hedge-fund/attribution" style="color: #3b82f6;">Strategy Attribution</a> | 
                    <a href="/hedge-fund/risk" style="color: #3b82f6;">Risk Analytics</a> | 
                    <a href="/monitoring/metrics" style="color: #3b82f6;">Prometheus Metrics</a>
                </p>
            </div>
        </div>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


@router.get("/performance")
def get_performance_metrics() -> Dict[str, Any]:
    """Get detailed performance metrics"""
    return hedge_fund_analytics.get_live_dashboard_data()


@router.get("/attribution")
def get_strategy_attribution() -> Dict[str, Any]:
    """Get strategy attribution analysis"""
    return hedge_fund_analytics.calculate_strategy_attribution()


@router.get("/risk")
def get_risk_analytics() -> Dict[str, Any]:
    """Get risk analytics"""
    return hedge_fund_analytics.calculate_risk_metrics()


@router.get("/regime-analysis")
def get_regime_analysis() -> Dict[str, Any]:
    """Get regime performance analysis"""
    return hedge_fund_analytics.calculate_regime_analysis()


@router.get("/models/status")
def get_models_status() -> Dict[str, Any]:
    """Get model status and memory usage"""
    return model_factory.get_memory_footprint()


@router.get("/ensemble/status")
def get_ensemble_status() -> Dict[str, Any]:
    """Get ensemble meta-learner status"""
    return t470_ensemble.get_performance_summary()


@router.post("/analytics/trade")
def record_trade(trade_data: Dict[str, Any]) -> Dict[str, str]:
    """Record a trade for analytics"""
    try:
        hedge_fund_analytics.update_trade(
            symbol=trade_data.get("symbol", ""),
            model_name=trade_data.get("model_name", ""),
            entry_price=trade_data.get("entry_price", 0.0),
            exit_price=trade_data.get("exit_price"),
            position_size=trade_data.get("position_size", 0.0),
            regime=trade_data.get("regime", "T"),
            timestamp=trade_data.get("timestamp", time.time()),
            meta=trade_data.get("meta"),
        )
        return {"status": "success", "message": "Trade recorded"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/analytics/attribution")
def record_model_attribution(attribution_data: Dict[str, Any]) -> Dict[str, str]:
    """Record model attribution for ensemble decision"""
    try:
        hedge_fund_analytics.update_model_attribution(
            model_contributions=attribution_data.get("model_contributions", {}),
            ensemble_decision=attribution_data.get("ensemble_decision", 0.0),
            actual_pnl=attribution_data.get("actual_pnl"),
        )
        return {"status": "success", "message": "Attribution recorded"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/report/export")
def export_performance_report() -> Dict[str, Any]:
    """Export comprehensive performance report"""
    return hedge_fund_analytics.export_performance_report()


@router.post("/reset-daily")
def reset_daily_metrics() -> Dict[str, str]:
    """Reset daily metrics (call at start of trading day)"""
    try:
        hedge_fund_analytics.reset_daily_metrics()
        return {"status": "success", "message": "Daily metrics reset"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
