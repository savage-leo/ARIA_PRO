# tools/coach.py
"""
Deterministic coaching / trade explanation engine.
No LLMs. Uses templates + model/strategy metadata to produce audit-friendly explanations.
"""
from datetime import datetime

TEMPLATES = {
    "standard": ("[{time}] Model {model_id} produced a {side} signal on {symbol} for {volume} lots. "
                 "Price: {price:.5f}. Expected edge: {edge:.3f}. model_version: {model_version}. "
                 "Reason: {reason}."),
    "risk_reject": ("[{time}] Trade rejected by risk_engine: {reason}. Signal: {signal}."),
    "strategy_switch": ("[{time}] Switching to strategy {strategy_id} based on Thompson sampling. "
                        "Alpha: {alpha:.2f}, Beta: {beta:.2f}"),
    "performance": ("[{time}] Performance update: Win rate {win_rate:.1%}, Sharpe {sharpe:.2f}, "
                   "Total trades: {total_trades}")
}

def explain_trade(trade: dict) -> str:
    """
    trade keys: time, model_id, side, symbol, volume, price, edge, model_version, reason
    """
    tpl = TEMPLATES.get("standard")
    return tpl.format(
        time=datetime.utcfromtimestamp(trade.get("timestamp", datetime.utcnow().timestamp())).isoformat()+"Z",
        model_id=trade.get("model_id", "unknown"),
        side=trade.get("side", "unknown"),
        symbol=trade.get("symbol", "unknown"),
        volume=trade.get("volume", 0.0),
        price=trade.get("price", 0.0),
        edge=trade.get("edge", 0.0),
        model_version=trade.get("model_version", "nv"),
        reason=trade.get("reason", "n/a")
    )

def explain_risk_rejection(reason: str, signal: dict) -> str:
    return TEMPLATES["risk_reject"].format(
        time=datetime.utcnow().isoformat()+"Z", 
        reason=reason, 
        signal=signal
    )

def explain_strategy_switch(strategy_id: int, alpha: float, beta: float) -> str:
    return TEMPLATES["strategy_switch"].format(
        time=datetime.utcnow().isoformat()+"Z",
        strategy_id=strategy_id,
        alpha=alpha,
        beta=beta
    )

def explain_performance(win_rate: float, sharpe: float, total_trades: int) -> str:
    return TEMPLATES["performance"].format(
        time=datetime.utcnow().isoformat()+"Z",
        win_rate=win_rate,
        sharpe=sharpe,
        total_trades=total_trades
    )
