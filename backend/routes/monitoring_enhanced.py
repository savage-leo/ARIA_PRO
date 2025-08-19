# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, pathlib, datetime, math
from typing import Dict, Any, List, Iterable
from fastapi import APIRouter, HTTPException, Response

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.getenv("ARIA_DATA_ROOT", PROJECT_ROOT / "data"))
GATING_PATH = pathlib.Path(
    os.getenv("ARIA_GATING_JSON", PROJECT_ROOT / "config" / "gating.default.json")
)


def _read_json(p: pathlib.Path) -> Dict[str, Any]:
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"JSON not found: {p}")
    return json.loads(p.read_text())


def _latest_live_log() -> pathlib.Path | None:
    files = sorted((DATA_ROOT / "live_logs").glob("decisions_*.jsonl"))
    return files[-1] if files else None


@router.get("/status")
def status() -> Dict[str, Any]:
    gating = _read_json(GATING_PATH)
    calib_dir = DATA_ROOT / "calibration" / "current"
    manifest = {}
    if calib_dir.exists():
        for sym_dir in calib_dir.iterdir():
            if sym_dir.is_dir():
                f = sym_dir / "fusion_lr.json"
                if f.exists():
                    j = _read_json(f)
                    manifest[sym_dir.name] = {
                        "fusion_type": j.get("type", "logreg"),
                        "version_hash": j.get("version_hash", ""),
                        "features_order": j.get("features_order", []),
                    }
    return {
        "utc": datetime.datetime.utcnow().isoformat() + "Z",
        "gating_version": gating.get("version"),
        "thresholds": gating.get("default_thresholds"),
        "symbols": sorted(list(manifest.keys())),
        "calibration_manifest": manifest,
        "data_root": str(DATA_ROOT),
    }


@router.get("/gating")
def gating() -> Dict[str, Any]:
    return _read_json(GATING_PATH)


@router.get("/latency")
def latency(n: int = 5000) -> Dict[str, Any]:
    path = _latest_live_log()
    if not path:
        return {"count": 0, "p50": None, "p95": None, "p99": None}
    vals: List[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                vals.append(float(obj.get("lat_ms", 0.0)))
            except Exception:
                continue
    if not vals:
        return {"count": 0, "p50": None, "p95": None, "p99": None}
    vals = vals[-n:] if len(vals) > n else vals
    s = sorted(vals)

    def pct(p):
        k = max(0, min(len(s) - 1, int(math.ceil(p * len(s))) - 1))
        return s[k]

    return {"count": len(s), "p50": pct(0.50), "p95": pct(0.95), "p99": pct(0.99)}


@router.get("/decisions/last")
def decisions_last(n: int = 1000, symbol: str | None = None) -> Dict[str, Any]:
    path = _latest_live_log()
    if not path:
        return {"count": 0, "items": []}
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if symbol and obj.get("symbol") != symbol:
                    continue
                items.append(obj)
            except Exception:
                continue
    return {"count": len(items), "items": items[-n:]}


@router.get("/regime/status")
def regime_status() -> Dict[str, Any]:
    """Get current regime states for all symbols"""
    from backend.core.regime_online import regime_manager

    return {
        "utc": datetime.datetime.utcnow().isoformat() + "Z",
        "regime_states": regime_manager.get_all_states(),
    }


@router.get("/risk/status")
def risk_status() -> Dict[str, Any]:
    """Get current risk management status"""
    from backend.core.risk_budget_enhanced import position_sizer

    kill_switch, kill_reason = position_sizer.check_kill_switch()

    return {
        "utc": datetime.datetime.utcnow().isoformat() + "Z",
        "kill_switch": kill_switch,
        "kill_reason": kill_reason,
        "daily_pnl": position_sizer.daily_pnl,
        "portfolio_dd": position_sizer.portfolio_dd,
        "current_var": position_sizer.current_var,
        "limits": {
            "max_daily_dd": position_sizer.limits.max_daily_dd,
            "max_portfolio_dd": position_sizer.limits.max_portfolio_dd,
            "var_cap": position_sizer.limits.var_cap,
            "min_units": position_sizer.limits.min_units,
            "max_units": position_sizer.limits.max_units,
        },
    }


@router.get("/metrics")
def metrics() -> Response:
    """
    Prometheus-style text metrics (no client lib). Emits latency and action dist from latest log.
    """
    path = _latest_live_log()
    lines: List[str] = []
    lines.append(
        f'# HELP aria_info Static info\n# TYPE aria_info gauge\naria_info{{data_root="{DATA_ROOT}"}} 1'
    )
    if not path or not path.exists():
        return Response("\n".join(lines), media_type="text/plain")
    lat, act = [], {"LONG": 0, "SHORT": 0, "FLAT": 0}
    regimes = {"T": 0, "R": 0, "B": 0}
    vols = {"Low": 0, "Med": 0, "High": 0}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                lat.append(float(obj.get("lat_ms", 0.0)))
                act[str(obj.get("action", "FLAT"))] = (
                    act.get(str(obj.get("action", "FLAT")), 0) + 1
                )
                regimes[str(obj.get("state", "T"))] = (
                    regimes.get(str(obj.get("state", "T")), 0) + 1
                )
                vols[str(obj.get("vol_bucket", "Med"))] = (
                    vols.get(str(obj.get("vol_bucket", "Med")), 0) + 1
                )
            except Exception:
                continue
    if lat:
        s = sorted(lat)

        def pct(p):
            k = max(0, min(len(s) - 1, int(math.ceil(p * len(s))) - 1))
            return s[k]

        lines += [
            "# HELP aria_latency_ms Decision latency ms",
            "# TYPE aria_latency_ms summary",
            f'aria_latency_ms{{quantile="0.5"}} {pct(0.5)}',
            f'aria_latency_ms{{quantile="0.95"}} {pct(0.95)}',
            f'aria_latency_ms{{quantile="0.99"}} {pct(0.99)}',
            f"aria_latency_ms_sum {sum(lat)}",
            f"aria_latency_ms_count {len(lat)}",
        ]
    lines += [
        "# HELP aria_actions_total Decision action counts",
        "# TYPE aria_actions_total counter",
        f'aria_actions_total{{action="LONG"}} {act.get("LONG",0)}',
        f'aria_actions_total{{action="SHORT"}} {act.get("SHORT",0)}',
        f'aria_actions_total{{action="FLAT"}} {act.get("FLAT",0)}',
    ]
    lines += [
        "# HELP aria_regimes_total Regime state counts",
        "# TYPE aria_regimes_total counter",
        f'aria_regimes_total{{state="T"}} {regimes.get("T",0)}',
        f'aria_regimes_total{{state="R"}} {regimes.get("R",0)}',
        f'aria_regimes_total{{state="B"}} {regimes.get("B",0)}',
    ]
    lines += [
        "# HELP aria_volatility_total Volatility bucket counts",
        "# TYPE aria_volatility_total counter",
        f'aria_volatility_total{{bucket="Low"}} {vols.get("Low",0)}',
        f'aria_volatility_total{{bucket="Med"}} {vols.get("Med",0)}',
        f'aria_volatility_total{{bucket="High"}} {vols.get("High",0)}',
    ]
    return Response("\n".join(lines), media_type="text/plain")
