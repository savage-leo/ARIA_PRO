import os
import sys
import time
import json
import math
import argparse
import logging
from typing import Dict, List, Optional

# Ensure project root is on sys.path if run as a script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.services.models_interface import score_and_calibrate

logger = logging.getLogger("ARIA.TestInference")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def compute_atr(ohlcv: List[List[float]], period: int = 14) -> float:
    if not ohlcv or len(ohlcv) <= period:
        return 0.0
    trs: List[float] = []
    prev_close = float(ohlcv[0][3])
    for i in range(1, len(ohlcv)):
        o, h, l, c = [float(x) for x in ohlcv[i][:4]]
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = c
    if len(trs) < period:
        return 0.0
    recent = trs[-period:]
    return sum(recent) / float(period)


def build_features(symbol: str, n: int, intraday: bool) -> Dict[str, object]:
    """Try Alpha Vantage; fallback to MT5; then synthetic series."""
    ohlcv: List[List[float]] = []
    # Alpha Vantage first
    try:
        from backend.services.alpha_vantage_client import AlphaVantageClient

        av = AlphaVantageClient()
        ohlcv = av.get_recent_ohlcv(symbol, n=max(60, n), use_intraday=intraday)
        if not ohlcv:
            logger.warning("Alpha Vantage returned no data")
    except Exception as e:
        logger.warning(f"Alpha Vantage unavailable ({e})")

    if not ohlcv:
        # MT5 fallback
        try:
            import MetaTrader5 as mt5

            if not mt5.initialize():
                raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")
            timeframe = mt5.TIMEFRAME_M5 if intraday else mt5.TIMEFRAME_D1
            count = max(60, n)
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
            try:
                mt5.shutdown()
            except Exception:
                pass
            if rates is not None and len(rates) > 0:
                try:
                    items = sorted(rates, key=lambda r: r["time"])
                except Exception:
                    items = list(rates)
                out: List[List[float]] = []
                for bar in items[-count:]:
                    try:
                        o = float(bar["open"])
                        h = float(bar["high"])
                        l = float(bar["low"])
                        c = float(bar["close"])
                        v = 0.0
                        try:
                            v = float(bar["tick_volume"])
                        except Exception:
                            try:
                                v = float(bar["real_volume"])
                            except Exception:
                                v = 0.0
                    except Exception:
                        try:
                            o = float(bar.open)
                            h = float(bar.high)
                            l = float(bar.low)
                            c = float(bar.close)
                            v = float(getattr(bar, "tick_volume", 0.0))
                        except Exception:
                            continue
                    out.append([o, h, l, c, v])
                ohlcv = out
        except Exception as e:
            logger.warning(f"MT5 fallback unavailable ({e})")

    if ohlcv:
        closes = [float(r[3]) for r in ohlcv if isinstance(r, (list, tuple))]
        return {"ohlcv": ohlcv, "series": closes}

    # Synthetic: noisy trend series
    import random

    s: List[float] = []
    price = 1.1000
    drift = 1e-4
    vol = 2e-4
    for _ in range(n):
        price = price * (1.0 + drift + random.uniform(-vol, vol))
        s.append(price)
    return {"series": s}


def main():
    parser = argparse.ArgumentParser(
        description="Test inference for active ARIA adapters (XGB/LSTM/etc.)"
    )
    parser.add_argument("--symbol", default="EURUSD", help="FX symbol, e.g., EURUSD")
    parser.add_argument(
        "--n", type=int, default=60, help="Number of bars/prices to fetch"
    )
    parser.add_argument(
        "--intraday", action="store_true", help="Use Alpha Vantage intraday (5min)"
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated active models (xgb,lstm,cnn,ppo,vision,llm_macro)",
    )
    parser.add_argument(
        "--atr-period", type=int, default=14, help="ATR period for diagnostics"
    )
    parser.add_argument("--json", action="store_true", help="Output JSON only")

    args = parser.parse_args()

    if args.models:
        os.environ["ACTIVE_MODELS"] = args.models

    adapters = build_default_adapters()
    if not adapters:
        logger.error("No adapters selected. Set ACTIVE_MODELS or --models.")
        sys.exit(2)

    # Load adapters
    load_info: Dict[str, float] = {}
    for k, a in adapters.items():
        t0 = time.time()
        try:
            a.load()
        except Exception:
            logger.exception(f"Adapter {k} load failed")
        load_info[k] = time.time() - t0

    # Build features
    feats = build_features(args.symbol, args.n, args.intraday)

    # Predict
    results: Dict[str, float] = {}
    pred_info: Dict[str, float] = {}
    for k, a in adapters.items():
        t0 = time.time()
        try:
            score = float(a.predict(feats))
        except Exception:
            logger.exception(f"Adapter {k} predict failed")
            score = 0.0
        results[k] = max(-1.0, min(1.0, score))
        pred_info[k] = time.time() - t0

    # ATR diagnostic if we have OHLCV
    atr_val = 0.0
    if isinstance(feats.get("ohlcv"), list):
        atr_val = compute_atr(feats["ohlcv"], period=args.atr_period)

    # Output
    payload = {
        "symbol": args.symbol,
        "models": list(adapters.keys()),
        "scores": results,
        "probabilities": {k: abs(v) for k, v in results.items()},
        "load_seconds": load_info,
        "predict_seconds": pred_info,
        "atr": atr_val,
        "intraday": bool(args.intraday),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("=== ARIA Test Inference ===")
        print(f"Symbol: {args.symbol} (intraday={bool(args.intraday)})")
        print(f"Active models: {', '.join(payload['models'])}")
        print("- Load times (s):", json.dumps(payload["load_seconds"]))
        print("- Predict times (s):", json.dumps(payload["predict_seconds"]))
        print(f"ATR[{args.atr_period}]: {payload['atr']:.6f}")
        print("Scores [-1..1] and |prob|:")
        for k in payload["models"]:
            s = payload["scores"].get(k, 0.0)
            p = abs(s)
            side = "BUY" if s >= 0 else "SELL"
            print(f"  {k:10s} -> score={s:+.4f}  prob={p:.2%}  side={side}")


if __name__ == "__main__":
    main()
