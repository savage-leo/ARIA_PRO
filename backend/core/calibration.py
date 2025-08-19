# backend/core/calibration.py
"""
Model score calibration for CPU-only deployment.

- Supports Platt scaling (logistic) and Isotonic Regression (PAV) without sklearn.
- Loads per-model calibration params from JSON if available.
- Provides safe identity fallback mapping.
- Supports per-(model,symbol,regime) calibration with fallbacks.

Usage:
    calib = ScoreCalibrator.load_default()
    p = calib.calibrate("lstm", score)
    probs = calib.calibrate_dict({"lstm": 0.2, "xgb": -0.3})
    p = calib.calibrate("lstm", score, symbol="EURUSD", regime="trend")

File format (backend/models/calibration.json):
{
  "lstm": {"method": "platt", "A": -1.2, "B": 0.1},
  "xgb":  {"method": "isotonic", "thresholds": [...], "values": [...]}
}

Per-(model,symbol,regime) format (backend/calibration/{model}/{symbol}/{regime}.json):
{
  "method": "platt",
  "A": -1.2,
  "B": 0.1
}
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("aria.core.calibration")


@dataclass
class PlattParams:
    A: float
    B: float


@dataclass
class IsotonicParams:
    thresholds: List[float]
    values: List[float]


class ScoreCalibrator:
    def __init__(self, model_params: Dict[str, Dict[str, object]]):
        self.params = model_params or {}

    # ----------------------- Public API -----------------------
    def calibrate(
        self,
        model: str,
        score: float,
        symbol: Optional[str] = None,
        regime: Optional[str] = None,
    ) -> float:
        """Return calibrated probability in [0,1].
        Falls back to identity mapping: p = 0.5*(score+1).
        Supports per-(model,symbol,regime) calibration with fallbacks.
        """
        s = float(max(-1.0, min(1.0, score)))

        # Try per-(model,symbol,regime) calibration first
        if symbol and regime:
            spec = self._load_per_model_calibration(model, symbol, regime)
            if spec:
                return self._apply_calibration(spec, s)

        # Try per-model calibration
        spec = self.params.get(model)
        if spec:
            return self._apply_calibration(spec, s)

        # Fallback to identity mapping
        return 0.5 * (s + 1.0)

    def _apply_calibration(self, spec: Dict[str, object], score: float) -> float:
        """Apply calibration method to score."""
        method = str(spec.get("method", "identity")).lower()
        if method == "platt":
            A = float(spec.get("A", 1.0))
            B = float(spec.get("B", 0.0))
            z = A * score + B
            # numerically stable sigmoid
            if z >= 0:
                ez = math.exp(-z)
                p = 1.0 / (1.0 + ez)
            else:
                ez = math.exp(z)
                p = ez / (1.0 + ez)
            return float(max(0.0, min(1.0, p)))
        elif method == "isotonic":
            thr = [float(x) for x in spec.get("thresholds", [])]
            vals = [float(x) for x in spec.get("values", [])]
            if len(thr) >= 2 and len(thr) == len(vals):
                return float(_isotonic_eval(score, thr, vals))
            return 0.5 * (score + 1.0)
        else:
            return 0.5 * (score + 1.0)

    def _load_per_model_calibration(
        self, model: str, symbol: str, regime: str
    ) -> Optional[Dict[str, object]]:
        """Load per-(model,symbol,regime) calibration parameters."""
        try:
            calib_path = os.path.join(
                "backend", "calibration", model, symbol, f"{regime}.json"
            )
            if os.path.exists(calib_path):
                with open(calib_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.debug(
                f"Failed to load per-model calibration for {model}/{symbol}/{regime}: {e}"
            )
        return None

    def calibrate_dict(self, scores: Dict[str, float]) -> Dict[str, float]:
        return {m: self.calibrate(m, v) for m, v in (scores or {}).items()}

    def save_to_json(self, path: str) -> None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.params, f, indent=2)
        except Exception:
            logger.exception("Failed saving calibration to %s", path)

    # ----------------------- Loading -----------------------
    @classmethod
    def load_from_json(cls, path: str) -> "ScoreCalibrator":
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return cls(data)
        except Exception:
            logger.exception("Failed loading calibration from %s", path)
        return cls({})

    @classmethod
    def load_default(cls) -> "ScoreCalibrator":
        # Allow override via env; else, try default; else empty
        path = os.environ.get(
            "ARIA_CALIBRATION_FILE",
            os.path.join("backend", "models", "calibration.json"),
        )
        return cls.load_from_json(path)


# ------------------------- Helpers -------------------------


def _isotonic_fit(xs: np.ndarray, ys: np.ndarray) -> IsotonicParams:
    """Pool-Adjacent-Violators (PAV) isotonic regression.
    Returns step function defined by thresholds and fitted values.
    """
    order = np.argsort(xs)
    x = xs[order].astype(float)
    y = ys[order].astype(float)
    n = len(x)
    w = np.ones(n, dtype=float)
    v = y.copy()
    # PAV algorithm
    i = 0
    while i < n - 1:
        if v[i] <= v[i + 1] + 1e-12:
            i += 1
            continue
        j = i
        while j >= 0 and v[j] > v[j + 1] + 1e-12:
            new_w = w[j] + w[j + 1]
            new_v = (w[j] * v[j] + w[j + 1] * v[j + 1]) / new_w
            v[j] = new_v
            w[j] = new_w
            # remove j+1
            v = np.delete(v, j + 1)
            w = np.delete(w, j + 1)
            x = np.delete(x, j + 1)
            n -= 1
            j -= 1
        i = max(j, 0)
    # Build thresholds -> values (non-decreasing)
    thresholds = x.tolist()
    values = v.tolist()
    return IsotonicParams(thresholds=thresholds, values=values)


def _isotonic_eval(s: float, thresholds: List[float], values: List[float]) -> float:
    # clamp to bounds
    if s <= thresholds[0]:
        return float(max(0.0, min(1.0, values[0])))
    if s >= thresholds[-1]:
        return float(max(0.0, min(1.0, values[-1])))
    # linear interpolate between nearest breakpoints
    idx = np.searchsorted(thresholds, s, side="right")
    x0, x1 = thresholds[idx - 1], thresholds[idx]
    y0, y1 = values[idx - 1], values[idx]
    if x1 == x0:
        return float(max(0.0, min(1.0, y0)))
    t = (s - x0) / (x1 - x0)
    y = y0 + t * (y1 - y0)
    return float(max(0.0, min(1.0, y)))
