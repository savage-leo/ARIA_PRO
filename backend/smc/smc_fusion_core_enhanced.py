# -*- coding: utf-8 -*-
"""
SMC Fusion Core (Enhanced++): gating + calibrated fusion + probabilities + EV.
"""
from __future__ import annotations
import os, json, time, math, hashlib, threading, pathlib, datetime
from typing import Dict, List, Any, Optional

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.getenv("ARIA_DATA_ROOT", PROJECT_ROOT / "data"))
GATING_PATH = pathlib.Path(
    os.getenv("ARIA_GATING_JSON", PROJECT_ROOT / "config" / "gating.default.json")
)
LIVE_LOG_DIR = DATA_ROOT / "live_logs"


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _load_json_cached(path: pathlib.Path, cache: dict) -> dict:
    key = str(path)
    mtime = path.stat().st_mtime if path.exists() else -1
    entry = cache.get(key)
    if entry and entry["mtime"] == mtime:
        return entry["data"]
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON: {path}")
    data = json.loads(path.read_text())
    cache[key] = {"mtime": mtime, "data": data}
    return data


class _JSONLogWriter:
    __slots__ = ("lock",)

    def __init__(self):
        self.lock = threading.Lock()
        LIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)

    def write(self, rec: dict) -> None:
        ts = datetime.datetime.utcnow().strftime("%Y%m%d")
        fp = LIVE_LOG_DIR / f"decisions_{ts}.jsonl"
        line = json.dumps(rec, separators=(",", ":"))
        with self.lock:
            with fp.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


class FusionModel:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.cache: Dict[str, Any] = {}
        self.model_dir = DATA_ROOT / "calibration" / "current" / symbol
        self.model_json = self.model_dir / "fusion_lr.json"

    def ensure_loaded(self, gating_features_order: List[str]) -> None:
        if not self.model_json.exists():
            feats = gating_features_order
            W = [
                1.0 if n in ("p_LSTM", "p_PPO", "p_XGB", "p_CNN") else 0.0
                for n in feats
            ]
            b = -2.0
            payload = {"type": "logreg", "W": W, "b": b, "features_order": feats}
            payload["version_hash"] = hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest()
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.model_json.write_text(json.dumps(payload, indent=2))
        data = _load_json_cached(self.model_json, self.cache)
        self.W = data["W"]
        self.b = data["b"]
        self.features_order = data["features_order"]

    def predict_proba(self, x: List[float]) -> float:
        z = self.b
        for wi, xi in zip(self.W, x):
            z += wi * xi
        return _sigmoid(z)  # p_long


class SMCFusionCoreEnhanced:
    def __init__(self):
        self.cache = {}
        self.logger = _JSONLogWriter()
        env = os.environ
        self.theta = {
            "T": float(env.get("ARIA_THETA_T", 0.60)),
            "R": float(env.get("ARIA_THETA_R", 0.58)),
            "B": float(env.get("ARIA_THETA_B", 0.62)),
        }
        # EV fallback RR per vol bucket
        self.rr = {
            "Low": float(env.get("ARIA_EV_RR_LOW", 1.4)),
            "Med": float(env.get("ARIA_EV_RR_MED", 1.2)),
            "High": float(env.get("ARIA_EV_RR_HIGH", 1.0)),
        }
        self.spread_max = float(env.get("ARIA_SPREAD_MAX_PIPS", 2.0))

    def _gating(self) -> dict:
        return _load_json_cached(GATING_PATH, self.cache)

    @staticmethod
    def _one_hot(name: str, cats: List[str]) -> List[float]:
        return [1.0 if c == name else 0.0 for c in cats]

    def _compose_features(
        self,
        p_scaled: Dict[str, float],
        vol_bucket: str,
        state: str,
        spread_z: float,
        session: str,
        feature_order: List[str],
    ) -> List[float]:
        feats = {
            "p_LSTM": p_scaled.get("LSTM", 0.0),
            "p_PPO": p_scaled.get("PPO", 0.0),
            "p_XGB": p_scaled.get("XGB", 0.0),
            "p_CNN": p_scaled.get("CNN", 0.0),
            "vol_Low": 0.0,
            "vol_Med": 0.0,
            "vol_High": 0.0,
            "state_T": 0.0,
            "state_R": 0.0,
            "state_B": 0.0,
            "spread_z": float(spread_z),
            "sess_ASIA": 0.0,
            "sess_EU": 0.0,
            "sess_US": 0.0,
        }
        for k, v in zip(
            ["vol_Low", "vol_Med", "vol_High"],
            self._one_hot(vol_bucket, ["Low", "Med", "High"]),
        ):
            feats[k] = v
        for k, v in zip(
            ["state_T", "state_R", "state_B"], self._one_hot(state, ["T", "R", "B"])
        ):
            feats[k] = v
        for k, v in zip(
            ["sess_ASIA", "sess_EU", "sess_US"],
            self._one_hot(session, ["ASIA", "EU", "US"]),
        ):
            feats[k] = v
        return [feats[name] for name in feature_order]

    def _state_threshold(self, state: str) -> float:
        return float(self.theta.get(state, 0.6))

    def _rr(self, vol_bucket: str, meta: Optional[Dict[str, Any]]) -> float:
        # Prefer live RR from risk engine (meta['expected_rr']), else fallback by vol bucket
        if meta and "expected_rr" in meta:
            try:
                return max(0.5, float(meta["expected_rr"]))
            except Exception:
                pass
        return float(self.rr.get(vol_bucket, 1.2))

    def decide(
        self,
        symbol: str,
        *,
        state: str,  # "T" | "R" | "B"
        vol_bucket: str,  # "Low" | "Med" | "High"
        session: str,  # "ASIA" | "EU" | "US"
        spread_z: float,
        model_probs: Dict[str, float],  # calibrated p_m in [0,1]
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Returns: {
          action, p_long, p_short, p_star, EV, size_hint, theta, weights_used, lat_ms, guards:{...}, ...
        }
        """
        t0 = time.perf_counter_ns()
        gating = self._gating()

        # Regime×vol weights
        weights = gating["weights"].get(state, {}).get(vol_bucket, {})
        p_scaled = {
            m: model_probs.get(m, 0.0) * float(weights.get(m, 0.0))
            for m in ["LSTM", "PPO", "XGB", "CNN"]
        }

        # Compose features + run tiny fusion
        feat_order = gating["feature_schema"]["fusion_features_order"]
        fusion = FusionModel(symbol)
        fusion.ensure_loaded(feat_order)
        x = self._compose_features(
            p_scaled, vol_bucket, state, spread_z, session, fusion.features_order
        )
        p_long = fusion.predict_proba(x)
        p_short = 1.0 - p_long
        p_star = p_long if p_long >= 0.5 else p_short

        # Thresholding
        theta = self._state_threshold(state)
        action = "FLAT"
        if p_star >= theta:
            action = "LONG" if p_long >= 0.5 else "SHORT"

        # EV (risk-adjusted) using RR (or provided expected_rr) — symmetric cost assumed here
        rr = self._rr(vol_bucket, meta)
        ev_long = p_long * rr - p_short * (1.0 / rr)
        ev_short = p_short * rr - p_long * (1.0 / rr)
        EV = (
            ev_long
            if action == "LONG"
            else (ev_short if action == "SHORT" else max(ev_long, ev_short))
        )

        # Guards (soft; hard enforcement is in exec layer)
        guards = {}
        if meta and "spread_pips" in meta:
            spread_pips = float(meta["spread_pips"])
            guards["spread_ok"] = spread_pips <= self.spread_max
            guards["spread_pips"] = spread_pips
            guards["spread_limit"] = self.spread_max
        if meta and "news_guard" in meta:
            guards["news_ok"] = bool(meta["news_guard"])

        size_hint = p_star  # risk engine will map to units

        rec = {
            "ts": datetime.datetime.utcnow().isoformat() + "Z",
            "symbol": symbol,
            "state": state,
            "vol_bucket": vol_bucket,
            "session": session,
            "p_long": round(p_long, 6),
            "p_short": round(p_short, 6),
            "p_star": round(p_star, 6),
            "theta": theta,
            "EV": round(float(EV), 6),
            "weights_used": weights,
            "action": action,
            "size_hint": float(size_hint),
            "lat_ms": round((time.perf_counter_ns() - t0) / 1e6, 3),
            "guards": guards,
        }
        if meta:
            rec["meta"] = meta
        self.logger.write(rec)
        return rec
