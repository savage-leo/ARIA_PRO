# -*- coding: utf-8 -*-
"""
Models Interface: returns raw model scores and calibrated probabilities per model.
Reads calibration artifacts created by backend/scripts/calibrate_fuse.py:
  data/calibration/current/{SYMBOL}/cal_{MODEL}_{STATE}.json   (platt or isotonic)
"""
from __future__ import annotations
import os, json, pathlib, math, re, logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.getenv("ARIA_DATA_ROOT", PROJECT_ROOT / "data"))
CALIB_DIR = DATA_ROOT / "calibration" / "current"
MODELS = ("LSTM", "PPO", "XGB", "CNN")


def _sigmoid(z: float) -> float:
    """Numerically stable sigmoid using math.exp"""
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


class _Calibrator:
    __slots__ = ("meta",)

    def __init__(self, meta: Dict[str, Any]):
        self.meta = meta

    def apply(self, s: float) -> float:
        t = self.meta.get("type", "platt")
        if t == "platt":
            try:
                A = float(self.meta.get("A", 1.0))
                B = float(self.meta.get("B", 0.0))
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid Platt calibration parameters: {e}")
                return 0.5
            return _sigmoid(A * float(s) + B)
        elif t == "isotonic":
            # LUT with midpoints -> probs; pick nearest
            lut = self.meta.get("lut", [])
            if not lut:
                return 0.5
            xs = [float(p[0]) for p in lut]
            ys = [float(p[1]) for p in lut]
            # binary search nearest
            lo, hi = 0, len(xs) - 1
            x = float(s)
            while lo < hi:
                mid = (lo + hi) // 2
                if xs[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            idx = lo
            # clamp to nearest of idx or idx-1
            if idx > 0 and abs(xs[idx - 1] - x) < abs(xs[idx] - x):
                idx -= 1
            return ys[idx]
        else:
            logger.warning(f"Unknown calibrator type: {t}, returning neutral 0.5")
            return 0.5


class CalibratorStore:
    def __init__(self, symbol: str):
        # Validate symbol to prevent path traversal
        if not self._validate_symbol(symbol):
            raise ValueError(f"Invalid symbol format: {symbol}")
        self.symbol = symbol
        self.base = CALIB_DIR / symbol
        
        # Ensure the resolved path is within CALIB_DIR
        try:
            resolved_base = self.base.resolve()
            if not resolved_base.is_relative_to(CALIB_DIR.resolve()):
                raise ValueError(f"Symbol path escapes calibration directory: {symbol}")
        except Exception as e:
            logger.error(f"Path validation failed for symbol {symbol}: {e}")
            raise ValueError(f"Invalid symbol path: {symbol}")
            
        self.cache: Dict[Tuple[str, str], _Calibrator] = {}
    
    def _validate_symbol(self, symbol: str) -> bool:
        """Validate symbol format to prevent path traversal"""
        # Allow only alphanumeric, underscore, and limited length
        if not symbol or len(symbol) > 20:
            return False
        # Strict pattern: alphanumeric and underscore only
        pattern = r'^[A-Z0-9_]+$'
        return bool(re.match(pattern, symbol))

    def _load(self, model: str, state: str) -> _Calibrator:
        key = (model, state)
        if key in self.cache:
            return self.cache[key]
        
        # Validate model and state to prevent injection
        if model not in MODELS:
            logger.warning(f"Unknown model type: {model}")
            self.cache[key] = _Calibrator({"type": "platt", "A": 0.0, "B": 0.0})
            return self.cache[key]
        
        if not re.match(r'^[a-zA-Z0-9_]+$', state):
            logger.warning(f"Invalid state format: {state}")
            self.cache[key] = _Calibrator({"type": "platt", "A": 0.0, "B": 0.0})
            return self.cache[key]
            
        fp = self.base / f"cal_{model}_{state}.json"
        
        # Ensure resolved path stays within base
        try:
            resolved_fp = fp.resolve()
            if not resolved_fp.is_relative_to(self.base.resolve()):
                logger.error(f"Calibration file path escapes base: {fp}")
                self.cache[key] = _Calibrator({"type": "platt", "A": 0.0, "B": 0.0})
                return self.cache[key]
        except Exception as e:
            logger.error(f"Path resolution failed: {e}")
            self.cache[key] = _Calibrator({"type": "platt", "A": 0.0, "B": 0.0})
            return self.cache[key]
            
        if not fp.exists():
            # default neutral - alert that calibration is missing
            logger.warning(f"Missing calibration file: {fp}")
            self.cache[key] = _Calibrator({"type": "platt", "A": 0.0, "B": 0.0})
            return self.cache[key]
        
        try:
            meta = json.loads(fp.read_text())
            # Validate JSON structure
            if not isinstance(meta, dict) or "type" not in meta:
                logger.error(f"Invalid calibration JSON structure in {fp}")
                self.cache[key] = _Calibrator({"type": "platt", "A": 0.0, "B": 0.0})
                return self.cache[key]
        except Exception as e:
            logger.error(f"Failed to load calibration from {fp}: {e}")
            self.cache[key] = _Calibrator({"type": "platt", "A": 0.0, "B": 0.0})
            return self.cache[key]
            
        self.cache[key] = _Calibrator(meta)
        return self.cache[key]

    def calibrate(self, state: str, raw_scores: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for m in MODELS:
            s = float(raw_scores.get(m, 0.0))
            cal = self._load(m, state).apply(s)
            # clamp numeric
            if cal < 1e-6:
                cal = 1e-6
            if cal > 1 - 1e-6:
                cal = 1 - 1e-6
            out[m] = cal
        return out


# ---- Public API ----


def score_and_calibrate(
    symbol: str, state: str, raw_scores: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """
    raw_scores: {'LSTM': s, 'PPO': s, 'XGB': s, 'CNN': s}   # uncalibrated raw model scores
    returns: {
      'raw': raw_scores,
      'calibrated': {'LSTM': p, 'PPO': p, 'XGB': p, 'CNN': p}
    }
    """
    # Validate inputs
    for k in raw_scores:
        if k not in MODELS:
            logger.warning(f"Unknown model in raw_scores: {k}")
    
    try:
        cal = CalibratorStore(symbol)
        p = cal.calibrate(state, raw_scores)
        return {"raw": {k: float(raw_scores.get(k, 0.0)) for k in MODELS}, "calibrated": p}
    except ValueError as e:
        logger.error(f"Calibration failed for {symbol}: {e}")
        # Return neutral values on error
        return {
            "raw": {k: float(raw_scores.get(k, 0.0)) for k in MODELS},
            "calibrated": {k: 0.5 for k in MODELS}
        }


class ModelsInterface:
    """Compatibility interface used by TrainingConnector.

    Provides a simple API to get calibrated probabilities per model
    using calibration artifacts in `data/calibration/current/{SYMBOL}`.
    """

    def __init__(self):
        pass

    def calibrate(
        self, symbol: str, state: str, raw_scores: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Return both raw and calibrated scores."""
        return score_and_calibrate(symbol, state, raw_scores)

    def get_calibrator_store(self, symbol: str) -> CalibratorStore:
        """Access the underlying calibrator store for advanced use."""
        return CalibratorStore(symbol)
