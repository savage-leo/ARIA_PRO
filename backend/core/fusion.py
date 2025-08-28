# backend/core/fusion.py
"""
Lightweight signal fusion with logistic regression and XGB fallback.

- Inputs are raw model scores in [-1, 1] per model.
- Uses ScoreCalibrator to map each score to a probability of UP move.
- Primary: logistic regression with L2(C=1) on features: [P_SMC, P_CNN, P_XGB, P_PPO, spread_pct, ATR_pct, regime_onehot(3), session_onehot(3)].
- Fallback: shallow XGB (depth≤3, n_estimators≤100), export ONNX≤1MB.
- Coefficients can be specified per regime in a JSON file, with safe fallbacks.

JSON format (backend/models/fusion.json):
{
  "default": {"b": 0.0, "w": {"lstm": 0.3, "cnn": 0.3, "ppo": 0.2, "xgb": 0.2}},
  "trend":   {"b": 0.05, "w": {"lstm": 0.35, "cnn": 0.35, "ppo": 0.15, "xgb": 0.15}},
  "range":   {"b": -0.05, "w": {"lstm": 0.25, "cnn": 0.25, "ppo": 0.25, "xgb": 0.25}},
  "breakout": {"b": 0.1, "w": {"lstm": 0.4, "cnn": 0.4, "ppo": 0.1, "xgb": 0.1}}
}

If the file is missing, equal weights are used over the provided signal keys.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.calibration import ScoreCalibrator
from core.regime import Regime, _session_from_ts

logger = logging.getLogger(__name__)


@dataclass
class FusionParams:
    b: float
    w: Dict[str, float]


class SignalFusion:
    def __init__(
        self,
        signal_keys: List[str],
        calibrator: Optional[ScoreCalibrator] = None,
        params: Optional[Dict[str, FusionParams]] = None,
        fusion_file: Optional[str] = None,
    ) -> None:
        self.signal_keys = list(signal_keys)
        self.calibrator = calibrator or ScoreCalibrator.load_default()
        self.params = params or self._load_params(fusion_file)
        self._load_fusion_model()

    def _load_fusion_model(self) -> None:
        """Load pretrained fusion model from ONNX file if available."""
        self.fusion_model = None
        self.fusion_meta = None

        try:
            import onnxruntime as ort

            model_path = os.path.join("backend", "models", "fusion.onnx")
            meta_path = os.path.join("backend", "models", "fusion_meta.json")

            if os.path.exists(model_path):
                self.fusion_model = ort.InferenceSession(model_path)

            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.fusion_meta = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load fusion model: {e}")

    # ------------------------- Public API -------------------------
    def fuse(
        self,
        raw_signals: Dict[str, float],
        regime: Regime,
        context: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        """Return fused decision dict containing:
        - p_long, p_short in [0,1]
        - direction: 'buy'|'sell'
        - margin: |p_long - p_short|
        - used_regime: str
        - contrib: per-model contribution to long logit
        """
        # Use ONNX model if available
        if self.fusion_model and context:
            try:
                return self._fuse_with_onnx(raw_signals, regime, context)
            except Exception as e:
                logger.warning(f"ONNX fusion failed, falling back to logistic: {e}")

        # Fallback to logistic regression
        reg_key = regime.value if isinstance(regime, Regime) else str(regime)
        fp = self.params.get(reg_key) or self.params.get("default")
        if fp is None:
            # Equal weights fallback
            eq_w = {k: 1.0 / max(1, len(self.signal_keys)) for k in self.signal_keys}
            fp = FusionParams(b=0.0, w=eq_w)

        # Build per-model probabilities for long/short
        p_long_i: Dict[str, float] = {}
        p_short_i: Dict[str, float] = {}
        for k in self.signal_keys:
            s = float(raw_signals.get(k, 0.0))
            # Calibrated probability of UP move given score in [-1,1]
            p_up = float(self.calibrator.calibrate(k, s))
            p_dn = float(self.calibrator.calibrate(k, -s))
            p_long_i[k] = _clip01(p_up)
            p_short_i[k] = _clip01(p_dn)

        # Compute fused logits and probabilities
        logit_long, contrib = _weighted_logit(p_long_i, fp.w, fp.b)
        logit_short, _ = _weighted_logit(p_short_i, fp.w, fp.b)
        p_long = _sigmoid(logit_long)
        p_short = _sigmoid(logit_short)
        direction = "buy" if p_long >= p_short else "sell"
        margin = abs(p_long - p_short)

        return {
            "p_long": float(p_long),
            "p_short": float(p_short),
            "direction": direction,
            "margin": float(margin),
            "used_regime": reg_key,
            "contrib": contrib,
        }

    def _fuse_with_onnx(
        self, raw_signals: Dict[str, float], regime: Regime, context: Dict[str, object]
    ) -> Dict[str, object]:
        """Fuse signals using pretrained ONNX model."""
        if not self.fusion_model:
            raise ValueError("No fusion model loaded")

        # Extract features
        features = self._extract_features(raw_signals, regime, context)

        # Run inference
        input_name = self.fusion_model.get_inputs()[0].name
        outputs = self.fusion_model.run(
            None, {input_name: features.astype(np.float32).reshape(1, -1)}
        )
        p_hat_fuse = float(outputs[0][0])

        # Convert to long/short probabilities
        p_long = p_hat_fuse
        p_short = 1.0 - p_hat_fuse
        direction = "buy" if p_long >= p_short else "sell"
        margin = abs(p_long - p_short)

        return {
            "p_long": float(p_long),
            "p_short": float(p_short),
            "direction": direction,
            "margin": float(margin),
            "used_regime": regime.value,
            "p_hat_fuse": p_hat_fuse,
            "contrib": {},
        }

    def _extract_features(
        self, raw_signals: Dict[str, float], regime: Regime, context: Dict[str, object]
    ) -> np.ndarray:
        """Extract features for fusion model: [P_SMC, P_CNN, P_XGB, P_PPO, spread_pct, ATR_pct, regime_onehot(3), session_onehot(3)]"""
        # Model probabilities
        p_smc = float(self.calibrator.calibrate("smc", raw_signals.get("smc", 0.0)))
        p_cnn = float(self.calibrator.calibrate("cnn", raw_signals.get("cnn", 0.0)))
        p_xgb = float(self.calibrator.calibrate("xgb", raw_signals.get("xgb", 0.0)))
        p_ppo = float(self.calibrator.calibrate("ppo", raw_signals.get("ppo", 0.0)))

        # Market features
        spread_pct = float(context.get("spread_pct", 0.0))
        atr_pct = float(context.get("atr_pct", 0.0))

        # Regime one-hot encoding
        regime_map = {"range": 0, "trend": 1, "breakout": 2}
        regime_idx = regime_map.get(regime.value, 0)
        regime_onehot = np.zeros(3, dtype=float)
        regime_onehot[regime_idx] = 1.0

        # Session one-hot encoding
        ts = float(context.get("timestamp", 0.0))
        session = _session_from_ts(ts)
        session_map = {"asia": 0, "london": 1, "ny": 2, "off": 3}
        session_idx = session_map.get(session, 3)
        session_onehot = np.zeros(4, dtype=float)
        session_onehot[session_idx] = 1.0

        # Combine all features
        features = np.array(
            [p_smc, p_cnn, p_xgb, p_ppo, spread_pct, atr_pct]
            + regime_onehot.tolist()
            + session_onehot.tolist(),
            dtype=float,
        )

        return features

    # ------------------------- Loading -------------------------
    def _load_params(self, fusion_file: Optional[str]) -> Dict[str, FusionParams]:
        path = fusion_file or os.environ.get(
            "ARIA_FUSION_FILE", os.path.join("backend", "models", "fusion.json")
        )
        if not os.path.exists(path):
            # Equal weights default handled in fuse()
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                blob = json.load(f)
            out: Dict[str, FusionParams] = {}
            for key, spec in (blob or {}).items():
                b = float(spec.get("b", 0.0))
                w = {str(k): float(v) for k, v in (spec.get("w", {}) or {}).items()}
                out[str(key)] = FusionParams(b=b, w=w)
            return out
        except Exception:
            return {}


# ------------------------- helpers -------------------------


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _logit(p: float) -> float:
    p = _clip01(p)
    eps = 1e-6
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def _weighted_logit(
    p_map: Dict[str, float], w_map: Dict[str, float], b: float
) -> Tuple[float, Dict[str, float]]:
    # Use only overlapping keys; fall back to equal weights if w_map empty
    keys = list(p_map.keys())
    if not w_map:
        w_map = {k: 1.0 / max(1, len(keys)) for k in keys}
    logit = float(b)
    contrib: Dict[str, float] = {}
    for k in keys:
        w = float(w_map.get(k, 0.0))
        c = w * _logit(float(p_map[k]))
        contrib[k] = c
        logit += c
    return float(logit), contrib
