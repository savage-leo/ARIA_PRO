#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARIA Enhanced SMC Fusion Core
- Integrates secret ingredient fusion as meta-weighting AI
- Real-time multi-model signal fusion (LSTM/CNN/PPO/Visual/LLM Macro)
- Regime detection, anomaly gating, volatility & liquidity context
- Risk-based sizing (ATR, max draw cap, capped Kelly), per-symbol envelopes
- Execution router (MT5) with slippage control & kill-switch
- Online self-optimization of fusion weights (SGD, no heavy deps)
- State persistence and audit logging
"""

import os
import sys
import json
import time
import math
import queue
import atexit
import signal
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any
from collections import deque

import numpy as np

# Import enhanced components
from backend.services.mt5_client import MT5Client
from backend.core.risk_engine_enhanced import RiskEngine
from backend.services.exec_arbiter import TradeArbiter, ExecPlan
from backend.services.advanced_features import (
    detect_trap,
    bias_engine,
    simple_portfolio_manager,
)

# Optional MT5 binding (graceful degrade)
try:
    import MetaTrader5 as mt5

    MT5_ENABLED = True
except Exception:
    MT5_ENABLED = False

# --------- Logging -----------------------------------------------------------

LOG_DIR = os.environ.get("ARIA_LOG_DIR", "./logs")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("ARIA.ENHANCED_FUSION")
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(LOG_DIR, "enhanced_fusion.log"), encoding="utf-8")
sh = logging.StreamHandler(sys.stdout)
fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s :: %(message)s")
fh.setFormatter(fmt)
sh.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(sh)

# --------- Config ------------------------------------------------------------


@dataclass
class EnhancedFusionConfig:
    # Risk
    account_risk_per_trade: float = float(
        os.environ.get("ARIA_RISK_PER_TRADE", 0.005)
    )  # 0.5%
    max_daily_drawdown: float = float(os.environ.get("ARIA_MAX_DD", 0.03))  # 3%
    kelly_cap: float = float(os.environ.get("ARIA_KELLY_CAP", 0.25))  # 25% cap on Kelly
    atr_lookback: int = int(os.environ.get("ARIA_ATR_LKB", 14))
    atr_stop_mult: float = float(os.environ.get("ARIA_ATR_STOP_MULT", 2.5))
    atr_tp_mult: float = float(os.environ.get("ARIA_ATR_TP_MULT", 4.0))

    # Fusion / learning
    ema_alpha: float = float(os.environ.get("ARIA_EMA_ALPHA", 0.2))
    anomaly_z: float = float(os.environ.get("ARIA_ANOMALY_Z", 3.2))
    vol_smooth_lkb: int = int(os.environ.get("ARIA_VOL_SMOOTH_LKB", 20))
    sgd_lr: float = float(os.environ.get("ARIA_SGD_LR", 0.01))
    sgd_l2: float = float(os.environ.get("ARIA_SGD_L2", 0.0005))

    # Execution
    max_slippage: float = float(
        os.environ.get("ARIA_MAX_SLIPPAGE_PIPS", 1.5)
    )  # in pips
    partial_fill_retry_sec: float = float(os.environ.get("ARIA_RETRY_SEC", 0.75))
    symbol_point_map: Dict[str, float] = field(
        default_factory=dict
    )  # override if needed

    # Controls
    allow_short: bool = os.environ.get("ARIA_ALLOW_SHORT", "1") == "1"
    enable_execution: bool = os.environ.get("ARIA_ENABLE_EXEC", "0") == "1"
    enable_mt5: bool = os.environ.get("ARIA_ENABLE_MT5", "0") == "1"

    # Persistence
    state_path: str = os.environ.get(
        "ARIA_FUSION_STATE", "./enhanced_fusion_state.json"
    )


# --------- Utilities ---------------------------------------------------------


def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return (x - med) / (1.4826 * mad)


def ema(prev: float, x: float, alpha: float) -> float:
    return alpha * x + (1 - alpha) * prev


def capped(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def multi_tf_scan(symbol: str, bars_by_tf: Dict[str, list]):
    """
    bars_by_tf: {"M1": [...], "M5":[...], "M15":[...], ...}
    Returns combined context dict with aggregated signals and confidences.
    """
    ctx = {"bias": 0.0, "patterns": [], "conf": 0.0}
    # example simple aggregator: average bias from detectors across TFs weighted by TF importance
    weights = {"M1": 0.5, "M5": 1.0, "M15": 1.5, "H1": 2.0}
    total_w = 0.0
    bias_acc = 0.0
    patterns = []
    for tf, bars in bars_by_tf.items():
        w = weights.get(tf, 1.0)
        # run detectors from smc_enhancements (order blocks, fvg, liq)
        obs = []
        try:
            from smc_enhancements import (
                detect_order_blocks,
                detect_fvg,
                detect_liquidity_zones,
            )

            obs += detect_order_blocks(bars)
            obs += detect_fvg(bars)
            obs += detect_liquidity_zones(bars)
        except Exception:
            pass
        # crude tf bias: +1 if more buy blocks, -1 if sell
        buy_count = sum(1 for o in obs if o.get("side") in ("bull", "buy"))
        sell_count = sum(1 for o in obs if o.get("side") in ("bear", "sell"))
        tf_bias = 0.0
        if buy_count + sell_count > 0:
            tf_bias = (buy_count - sell_count) / max(1, (buy_count + sell_count))
        bias_acc += w * tf_bias
        total_w += w
        patterns += obs
    if total_w > 0:
        ctx["bias"] = bias_acc / total_w
    # pattern clustering placeholder: group by side and close proximity
    clustered = {}
    for p in patterns:
        k = p.get("side", "u")
        clustered.setdefault(k, []).append(p)
    ctx["patterns"] = clustered
    ctx["conf"] = min(1.0, abs(ctx["bias"]))
    return ctx


def pips(symbol: str, points: float, symbol_point_map: Dict[str, float]) -> float:
    pt = symbol_point_map.get(symbol)
    if pt is None:
        # Sensible default: JPY-quoted pairs often use 0.01, majors 0.0001
        s = (symbol or "").upper()
        pt = 0.01 if s.endswith("JPY") else 0.0001
    return points / pt


# --------- Online Linear Fusion (SGD) ---------------------------------------


class OnlineFusionSGD:
    """
    Lightweight online linear model for meta-weighting:
      fused = w·x + b
    with L2 regularization and SGD updates on realized returns.
    """

    def __init__(self, n_inputs: int, lr: float, l2: float):
        self.w = np.zeros(n_inputs, dtype=np.float64)
        self.b = 0.0
        self.lr = lr
        self.l2 = l2

    def predict(self, x: np.ndarray) -> float:
        return float(np.dot(self.w, x) + self.b)

    def update(self, x: np.ndarray, y: float) -> None:
        pred = self.predict(x)
        err = pred - y
        grad_w = err * x + self.l2 * self.w
        grad_b = err
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b

    def state(self) -> Dict:
        return {"w": self.w.tolist(), "b": self.b}

    def load(self, s: Dict):
        # Load persisted weights, auto-resizing to current input dimensionality
        target_len = int(self.w.shape[0])
        loaded_w = np.array(s.get("w", []), dtype=np.float64)
        if loaded_w.size == 0:
            # keep initialized shape (zeros) if no weights present
            pass
        else:
            if loaded_w.shape[0] != target_len:
                old_len = int(loaded_w.shape[0])
                if old_len < target_len:
                    padded = np.zeros(target_len, dtype=np.float64)
                    padded[:old_len] = loaded_w
                    loaded_w = padded
                else:
                    loaded_w = loaded_w[:target_len]
                try:
                    logger.warning(
                        "Resized meta-model weights from %d to %d to match current inputs",
                        old_len,
                        target_len,
                    )
                except Exception:
                    pass
            # assign resized (or original) weights
            self.w = loaded_w
        self.b = float(s.get("b", self.b))


# --------- Enhanced SMC Data Structures -------------------------------------


@dataclass
class OrderBlock:
    high: float
    low: float
    type: str  # 'bullish' or 'bearish'
    strength: float
    volume: float
    timestamp: float


@dataclass
class FairValueGap:
    high: float
    low: float
    direction: str  # 'bullish' or 'bearish'
    strength: float
    timestamp: float


@dataclass
class LiquidityZone:
    level: float
    type: str  # 'equal_high', 'equal_low', 'swing_high', 'swing_low'
    strength: float
    timestamp: float


@dataclass
class MarketContext:
    vol_ewma: float = 0.0
    spread_ewma: float = 0.0
    regime: str = "neutral"  # "trend", "meanrev", "highvol", "lowvol"
    last_price: float = 0.0
    last_update_ts: float = 0.0


@dataclass
class EnhancedTradeIdea:
    symbol: str
    bias: str  # 'bullish' or 'bearish'
    confidence: float
    entry: float
    stop: float
    takeprofit: float
    order_blocks: List[OrderBlock] = None
    fair_value_gaps: List[FairValueGap] = None
    liquidity_zones: List[LiquidityZone] = None
    meta_weights: Dict[str, float] = None  # Secret ingredient weights
    regime: str = "neutral"
    anomaly_score: float = 0.0
    ts: float = None

    def __post_init__(self):
        if self.ts is None:
            self.ts = time.time()
        if self.order_blocks is None:
            self.order_blocks = []
        if self.fair_value_gaps is None:
            self.fair_value_gaps = []
        if self.liquidity_zones is None:
            self.liquidity_zones = []
        if self.meta_weights is None:
            self.meta_weights = {}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "bias": self.bias,
            "confidence": self.confidence,
            "entry": self.entry,
            "stop": self.stop,
            "takeprofit": self.takeprofit,
            "order_blocks": [
                {
                    "high": ob.high,
                    "low": ob.low,
                    "type": ob.type,
                    "strength": ob.strength,
                }
                for ob in self.order_blocks
            ],
            "fair_value_gaps": [
                {
                    "high": fvg.high,
                    "low": fvg.low,
                    "direction": fvg.direction,
                    "strength": fvg.strength,
                }
                for fvg in self.fair_value_gaps
            ],
            "liquidity_zones": [
                {"level": lz.level, "type": lz.type, "strength": lz.strength}
                for lz in self.liquidity_zones
            ],
            "meta_weights": self.meta_weights,
            "regime": self.regime,
            "anomaly_score": self.anomaly_score,
            "ts": self.ts,
        }


# --------- Enhanced SMC Fusion Core -----------------------------------------


class EnhancedSMCFusionCore:
    def __init__(self, symbol: str, cfg: EnhancedFusionConfig, signal_keys: List[str]):
        self.symbol = symbol
        self.cfg = cfg
        self.signal_keys = signal_keys

        # SMC structures
        self.bars = deque(maxlen=1000)
        self.order_blocks = deque(maxlen=50)
        self.fair_value_gaps = deque(maxlen=50)
        self.liquidity_zones = deque(maxlen=50)

        # Secret ingredient components
        self.context: MarketContext = MarketContext()
        self.meta_model: Optional[OnlineFusionSGD] = None
        self._state = {
            "pnl_day": 0.0,
            "wins": 0,
            "losses": 0,
            "last_reset": int(time.time()),
        }

        # State save guard (prevent duplicate writes/logs)
        self._save_lock = threading.Lock()
        self._state_saved = False

        # Enhanced components (DAN_LIVE_ONLY)
        self.mt5_client = MT5Client()
        self.risk_engine = RiskEngine(self.mt5_client)
        self.trade_arbiter = TradeArbiter(self.mt5_client)

        # Initialize MT5 connection
        if self.cfg.enable_mt5:
            self.mt5_client.connect()
            self.mt5_client.start()

        # Thread-safe queues for async integration
        self.signal_q: "queue.Queue[Tuple[str, Dict[str, float], Dict[str, float]]]" = (
            queue.Queue()
        )
        self.feedback_q: "queue.Queue[Tuple[str, float, Dict[str, float]]]" = (
            queue.Queue()
        )

        # Kill-switch flag
        self._halt = False
        atexit.register(self._save_state)

        # Initialize meta-model
        self._init_meta_model()
        self._load_state()

    def _init_meta_model(self):
        """Initialize the secret ingredient meta-weighting model"""
        n_inputs = len(self.signal_keys) + 4  # + context features
        self.meta_model = OnlineFusionSGD(n_inputs, self.cfg.sgd_lr, self.cfg.sgd_l2)
        logger.info(
            "Initialized meta-weighting model for %s with %d inputs",
            self.symbol,
            n_inputs,
        )

    # ------------- Persistence -----------------
    def _save_state(self):
        # One-shot, atomic save to avoid duplicate writes/logs from atexit/shutdown hooks
        tmp_path = None
        try:
            with self._save_lock:
                if getattr(self, "_state_saved", False):
                    return
                blob = {
                    "pnl_day": self._state["pnl_day"],
                    "wins": self._state["wins"],
                    "losses": self._state["losses"],
                    "meta_model": self.meta_model.state() if self.meta_model else {},
                    "context": vars(self.context),
                }
                dirpath = os.path.dirname(self.cfg.state_path) or "."
                try:
                    os.makedirs(dirpath, exist_ok=True)
                except Exception:
                    pass
                tmp_path = f"{self.cfg.state_path}.tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(blob, f)
                os.replace(tmp_path, self.cfg.state_path)
                self._state_saved = True
                logger.info("Enhanced fusion state saved -> %s", self.cfg.state_path)
        except Exception as e:
            try:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            logger.exception("Failed saving enhanced fusion state: %s", e)

    def _load_state(self):
        if not os.path.exists(self.cfg.state_path):
            logger.info("No prior enhanced fusion state found, starting fresh.")
            return
        try:
            with open(self.cfg.state_path, "r", encoding="utf-8") as f:
                blob = json.load(f)
            self._state["pnl_day"] = float(blob.get("pnl_day", 0.0))
            self._state["wins"] = int(blob.get("wins", 0))
            self._state["losses"] = int(blob.get("losses", 0))
            if self.meta_model and "meta_model" in blob:
                self.meta_model.load(blob["meta_model"])
            if "context" in blob:
                c = blob["context"]
                self.context = MarketContext(**c)
            logger.info("Loaded enhanced fusion state from %s", self.cfg.state_path)
        except Exception as e:
            logger.exception("Failed loading enhanced fusion state: %s", e)

    # ------------- Public API -----------------
    def ingest_bar(
        self,
        bar: Dict[str, Any],
        raw_signals: Optional[Dict[str, float]] = None,
        market_feats: Optional[Dict[str, float]] = None,
    ) -> Optional[EnhancedTradeIdea]:
        """
        Enhanced bar ingestion with secret ingredient integration
        """
        try:
            # Update SMC structures
            self.bars.append(bar)
            self._update_order_blocks()
            self._update_fair_value_gaps()
            self._update_liquidity_zones()

            # Generate base SMC idea
            base_idea = self._generate_base_smc_idea(bar)
            if base_idea is None:
                return None

            # Apply secret ingredient meta-weighting if signals provided
            if isinstance(raw_signals, dict) and isinstance(market_feats, dict):
                enhanced_idea = self._apply_secret_ingredient(
                    base_idea, raw_signals, market_feats
                )
                return enhanced_idea

            return base_idea

        except Exception as e:
            logger.exception("Enhanced bar ingestion failure: %s", e)
            return None

    def submit_feedback(self, realized_return: float, last_features: Dict[str, float]):
        """
        Online learning hook for meta-weighting model
        """
        try:
            if not self.meta_model:
                return
            x = self._dict_to_vector(last_features)
            self.meta_model.update(x, realized_return)
            if realized_return > 0:
                self._state["wins"] += 1
            else:
                self._state["losses"] += 1
            self._state["pnl_day"] += realized_return
        except Exception as e:
            logger.exception("submit_feedback failure: %s", e)

    def kill_switch(self, enable: bool = True):
        self._halt = enable
        # Propagate to arbiter (so orders are actually blocked)
        try:
            if hasattr(self, "trade_arbiter") and self.trade_arbiter:
                self.trade_arbiter.engage_kill(enable)
        except Exception:
            pass
        logger.warning(
            "ENHANCED FUSION KILL-SWITCH: %s", "ENGAGED" if enable else "DISENGAGED"
        )

    # ------------- SMC Analysis (existing logic) -------------
    def _update_order_blocks(self):
        """Detect and update order blocks"""
        if len(self.bars) < 3:
            return

        # Look for order blocks in recent bars
        for i in range(len(self.bars) - 2, max(0, len(self.bars) - 20), -1):
            if i < 2:
                continue

            current = self.bars[i]
            prev = self.bars[i - 1]
            next_bar = self.bars[i + 1]

            # Bullish Order Block: Strong up move after consolidation
            if (
                current["c"] > current["o"]  # Current bar is bullish
                and current["h"] - current["l"]
                > (prev["h"] - prev["l"]) * 1.5  # Strong move
                and current["v"] > prev["v"] * 1.2
            ):  # High volume

                ob = OrderBlock(
                    high=current["h"],
                    low=current["l"],
                    type="bullish",
                    strength=self._calculate_ob_strength(current, prev, next_bar),
                    volume=current["v"],
                    timestamp=current["ts"],
                )

                if ob.strength >= 0.6:
                    self.order_blocks.append(ob)

            # Bearish Order Block: Strong down move after consolidation
            elif (
                current["c"] < current["o"]  # Current bar is bearish
                and current["h"] - current["l"]
                > (prev["h"] - prev["l"]) * 1.5  # Strong move
                and current["v"] > prev["v"] * 1.2
            ):  # High volume

                ob = OrderBlock(
                    high=current["h"],
                    low=current["l"],
                    type="bearish",
                    strength=self._calculate_ob_strength(current, prev, next_bar),
                    volume=current["v"],
                    timestamp=current["ts"],
                )

                if ob.strength >= 0.6:
                    self.order_blocks.append(ob)

    def _update_fair_value_gaps(self):
        """Detect 3-candle SMC FVG (displacement)"""
        if len(self.bars) < 3:
            return

        # Use i as the center bar index: prev(i-1), cur(i), next(i+1)
        for i in range(len(self.bars) - 2, 1, -1):
            prev_bar = self.bars[i - 1]
            cur_bar = self.bars[i]
            next_bar = self.bars[i + 1] if i + 1 < len(self.bars) else None
            if next_bar is None:
                continue

            # Bullish FVG: prev.high < next.low (gap not fully rebalanced)
            if prev_bar["h"] < next_bar["l"]:
                high = next_bar["l"]
                low = prev_bar["h"]
                gap_size = high - low
                avg_range = (
                    (prev_bar["h"] - prev_bar["l"])
                    + (cur_bar["h"] - cur_bar["l"])
                    + (next_bar["h"] - next_bar["l"])
                ) / 3.0
                gap_factor = min(gap_size / max(avg_range, 1e-6), 2.0) / 2.0
                vol_factor = min(cur_bar["v"] / max(prev_bar["v"], 1.0), 2.0) / 2.0
                strength = (gap_factor * 0.65) + (vol_factor * 0.35)
                if strength >= 0.5:
                    self.fair_value_gaps.append(
                        FairValueGap(
                            high=high,
                            low=low,
                            direction="bullish",
                            strength=strength,
                            timestamp=cur_bar["ts"],
                        )
                    )

            # Bearish FVG: prev.low > next.high
            elif prev_bar["l"] > next_bar["h"]:
                high = prev_bar["l"]
                low = next_bar["h"]
                gap_size = high - low
                avg_range = (
                    (prev_bar["h"] - prev_bar["l"])
                    + (cur_bar["h"] - cur_bar["l"])
                    + (next_bar["h"] - next_bar["l"])
                ) / 3.0
                gap_factor = min(gap_size / max(avg_range, 1e-6), 2.0) / 2.0
                vol_factor = min(cur_bar["v"] / max(prev_bar["v"], 1.0), 2.0) / 2.0
                strength = (gap_factor * 0.65) + (vol_factor * 0.35)
                if strength >= 0.5:
                    self.fair_value_gaps.append(
                        FairValueGap(
                            high=high,
                            low=low,
                            direction="bearish",
                            strength=strength,
                            timestamp=cur_bar["ts"],
                        )
                    )

    def _update_liquidity_zones(self):
        """Detect and update liquidity zones"""
        if len(self.bars) < 5:
            return

        for i in range(len(self.bars) - 3, max(2, len(self.bars) - 15), -1):
            if i < 3 or i >= len(self.bars) - 2:
                continue
            current = self.bars[i]
            prev = self.bars[i - 1]
            next_bar = self.bars[i + 1]

            # Equal Highs
            if (
                abs(current["h"] - prev["h"]) <= 0.0005
                and current["h"] > current["l"]
                and current["v"] > prev["v"] * 1.1
            ):
                lz = LiquidityZone(
                    level=max(current["h"], prev["h"]),
                    type="equal_high",
                    strength=self._calculate_liquidity_strength(
                        current, prev, next_bar
                    ),
                    timestamp=current["ts"],
                )
                if lz.strength >= 0.7:
                    self.liquidity_zones.append(lz)

            # Equal Lows
            elif (
                abs(current["l"] - prev["l"]) <= 0.0005
                and current["l"] < current["h"]
                and current["v"] > prev["v"] * 1.1
            ):
                lz = LiquidityZone(
                    level=min(current["l"], prev["l"]),
                    type="equal_low",
                    strength=self._calculate_liquidity_strength(
                        current, prev, next_bar
                    ),
                    timestamp=current["ts"],
                )
                if lz.strength >= 0.7:
                    self.liquidity_zones.append(lz)

    def _calculate_ob_strength(
        self, current: Dict, prev: Dict, next_bar: Dict
    ) -> float:
        """Calculate order block strength"""
        volume_factor = min(current["v"] / max(prev["v"], 1), 3.0) / 3.0
        price_factor = (current["h"] - current["l"]) / max(
            prev["h"] - prev["l"], 0.0001
        )
        price_factor = min(price_factor, 3.0) / 3.0

        # Follow-through factor
        follow_through = 0.5
        if (
            next_bar["c"] > current["c"] and current["c"] > current["o"]
        ):  # Bullish follow-through
            follow_through = 1.0
        elif (
            next_bar["c"] < current["c"] and current["c"] < current["o"]
        ):  # Bearish follow-through
            follow_through = 1.0

        return volume_factor * 0.4 + price_factor * 0.4 + follow_through * 0.2

    def _calculate_liquidity_strength(
        self, current: Dict, prev: Dict, next_bar: Dict
    ) -> float:
        """Calculate liquidity zone strength"""
        volume_factor = min(current["v"] / max(prev["v"], 1), 2.0) / 2.0
        price_factor = (current["h"] - current["l"]) / max(
            prev["h"] - prev["l"], 0.0001
        )
        price_factor = min(price_factor, 2.0) / 2.0

        return volume_factor * 0.6 + price_factor * 0.4

    def _generate_base_smc_idea(self, current_bar: Dict) -> Optional[EnhancedTradeIdea]:
        """Generate base SMC trading idea with advanced pattern detection"""
        if not self.order_blocks and not self.fair_value_gaps:
            return None

        # Enhanced SMC analysis with advanced patterns
        current_price = current_bar["c"]
        bullish_signals = 0
        bearish_signals = 0

        # Trap detection (DAN_LIVE_ONLY)
        bars_list = []
        for bar in list(self.bars):
            bars_list.append(
                {
                    "open": bar["o"],
                    "high": bar["h"],
                    "low": bar["l"],
                    "close": bar["c"],
                    "time": bar["ts"],
                }
            )

        trap_detected = detect_trap(bars_list, lookback=50)
        if trap_detected:
            logger.warning(
                f"[DAN_LIVE] Trap detected for {self.symbol} - reducing position size"
            )
            # Reduce confidence when trap detected
            current_bar["trap_factor"] = 0.5
        else:
            current_bar["trap_factor"] = 1.0
        total_confidence = 0

        # Check order blocks
        for ob in self.order_blocks:
            if (
                ob.type == "bullish"
                and current_price >= ob.low
                and current_price <= ob.high
            ):
                bullish_signals += ob.strength
                total_confidence += ob.strength
            elif (
                ob.type == "bearish"
                and current_price >= ob.low
                and current_price <= ob.high
            ):
                bearish_signals += ob.strength
                total_confidence += ob.strength

        # Check fair value gaps
        for fvg in self.fair_value_gaps:
            if (
                fvg.direction == "bullish"
                and current_price >= fvg.low
                and current_price <= fvg.high
            ):
                bullish_signals += fvg.strength * 0.8
                total_confidence += fvg.strength * 0.8
            elif (
                fvg.direction == "bearish"
                and current_price >= fvg.low
                and current_price <= fvg.high
            ):
                bearish_signals += fvg.strength * 0.8
                total_confidence += fvg.strength * 0.8

        # Check liquidity zones
        for lz in self.liquidity_zones:
            if abs(current_price - lz.level) < 0.0005:  # Price near liquidity zone
                if lz.type in ["equal_high", "swing_high"]:
                    bearish_signals += lz.strength * 0.6
                    total_confidence += lz.strength * 0.6
                elif lz.type in ["equal_low", "swing_low"]:
                    bullish_signals += lz.strength * 0.6
                    total_confidence += lz.strength * 0.6

        if total_confidence < 0.3:
            return None

        # Determine bias
        if abs(bullish_signals - bearish_signals) <= 1e-6:
            # Tie-breaker using trend/ATR slope
            slope = self._atr_slope(10)
            if abs(slope) < 1e-8:
                return None  # truly balanced → skip
            bias = "bullish" if slope >= 0 else "bearish"
        else:
            bias = "bullish" if bullish_signals > bearish_signals else "bearish"

        confidence = min(
            max(
                (max(bullish_signals, bearish_signals) / max(total_confidence, 1e-6)),
                0.0,
            ),
            1.0,
        )
        entry = current_price
        stop = (
            self._find_bullish_stop()
            if bias == "bullish"
            else self._find_bearish_stop()
        )
        if stop is None or stop <= 0:
            return None  # invalid stop, skip

        rr = 2.0
        takeprofit = (
            entry + rr * (entry - stop)
            if bias == "bullish"
            else entry - rr * (stop - entry)
        )

        # Get relevant SMC structures
        relevant_obs = [ob for ob in self.order_blocks if ob.type == bias]
        relevant_fvgs = [fvg for fvg in self.fair_value_gaps if fvg.direction == bias]
        relevant_lzs = [
            lz
            for lz in self.liquidity_zones
            if (bias == "bullish" and lz.type in ["equal_low", "swing_low"])
            or (bias == "bearish" and lz.type in ["equal_high", "swing_high"])
        ]

        return EnhancedTradeIdea(
            symbol=self.symbol,
            bias=bias,
            confidence=confidence,
            entry=entry,
            stop=stop,
            takeprofit=takeprofit,
            order_blocks=relevant_obs[-3:],  # Last 3 relevant order blocks
            fair_value_gaps=relevant_fvgs[-3:],  # Last 3 relevant FVGs
            liquidity_zones=relevant_lzs[-3:],  # Last 3 relevant liquidity zones
        )

    def _atr_slope(self, win: int = 10) -> float:
        """Calculate ATR slope for trend detection"""
        if len(self.bars) < win + 14:
            return 0.0
        vals = []
        for i in range(-win, 0):
            atr = self._rolling_atr(14)
            vals.append(atr or 0.0)
        return (vals[-1] - vals[0]) / (win or 1)

    def _rolling_atr(self, length: int = 14) -> Optional[float]:
        """Calculate rolling ATR"""
        if len(self.bars) < length + 1:
            return None
        trs = []
        for i in range(-length, 0):
            h = self.bars[i]["h"]
            l = self.bars[i]["l"]
            pc = self.bars[i - 1]["c"]
            tr = max(h - l, abs(h - pc), abs(l - pc))
            trs.append(tr)
        return sum(trs) / len(trs) if trs else None

    def _find_bullish_stop(self) -> Optional[float]:
        """SL below nearest bullish structure"""
        bullish_obs = [ob for ob in self.order_blocks if ob.type == "bullish"]
        if bullish_obs:
            price = self.bars[-1]["c"]
            below = sorted(
                [ob for ob in bullish_obs if ob.low <= price + 0.0005],
                key=lambda x: abs(price - x.low),
            )
            if below:
                return below[0].low - 0.001
        # fallback swing low from last N bars
        if self.bars:
            lows = [b["l"] for b in list(self.bars)[-10:]]
            return min(lows) - 0.001 if lows else None
        return None

    def _find_bearish_stop(self) -> Optional[float]:
        """SL above nearest bearish structure"""
        bearish_obs = [ob for ob in self.order_blocks if ob.type == "bearish"]
        if bearish_obs:
            price = self.bars[-1]["c"]
            above = sorted(
                [ob for ob in bearish_obs if ob.high >= price - 0.0005],
                key=lambda x: abs(price - x.high),
            )
            if above:
                return above[0].high + 0.001
        if self.bars:
            highs = [b["h"] for b in list(self.bars)[-10:]]
            return max(highs) + 0.001 if highs else None
        return None

    # ------------- Secret Ingredient Integration -------------
    def _apply_secret_ingredient(
        self,
        base_idea: EnhancedTradeIdea,
        raw_signals: Dict[str, float],
        market_feats: Dict[str, float],
    ) -> EnhancedTradeIdea:
        """
        Apply secret ingredient meta-weighting to base SMC idea
        """
        try:
            # Build feature vector for meta-model
            x, ctx = self._build_feature_vector(raw_signals, market_feats)

            # Get meta-weighting prediction
            meta_score = self.meta_model.predict(x)

            # Anomaly gate
            anomaly_score = self._calculate_anomaly_score(x)
            if anomaly_score > self.cfg.anomaly_z:
                logger.warning("[%s] Anomaly gate triggered, forcing FLAT", self.symbol)
                return self._create_flat_idea(base_idea)

            # Regime detection and shaping
            self._update_regime(market_feats.get("trend_strength", 0.0), ctx.vol_ewma)
            meta_score = self._shape_by_regime(meta_score)

            # Apply meta-weighting to confidence
            enhanced_confidence = base_idea.confidence * (0.5 + 0.5 * meta_score)
            enhanced_confidence = capped(enhanced_confidence, 0.0, 1.0)

            # Create enhanced idea
            enhanced_idea = EnhancedTradeIdea(
                symbol=base_idea.symbol,
                bias=base_idea.bias,
                confidence=enhanced_confidence,
                entry=base_idea.entry,
                stop=base_idea.stop,
                takeprofit=base_idea.takeprofit,
                order_blocks=base_idea.order_blocks,
                fair_value_gaps=base_idea.fair_value_gaps,
                liquidity_zones=base_idea.liquidity_zones,
                meta_weights={
                    k: float(v) for k, v in zip(self.signal_keys, self.meta_model.w)
                },
                regime=self.context.regime,
                anomaly_score=anomaly_score,
                ts=time.time(),
            )

            return enhanced_idea

        except Exception as e:
            logger.exception("Secret ingredient application failure: %s", e)
            return base_idea

    def _build_feature_vector(
        self, raw_signals: Dict[str, float], market_feats: Dict[str, float]
    ) -> Tuple[np.ndarray, MarketContext]:
        """Build feature vector for meta-model"""
        # Normalize/clip model signals
        sig_vals = []
        for k in self.signal_keys:
            v = float(raw_signals.get(k, 0.0))
            v = capped(v, -1.0, 1.0)
            sig_vals.append(v)

        # Update context
        price = float(market_feats.get("price", self.context.last_price or 0.0))
        spread = float(market_feats.get("spread", 0.0))
        vol = float(market_feats.get("vol", market_feats.get("atr", 0.0)))
        trend = float(market_feats.get("trend_strength", 0.0))
        liquidity = float(market_feats.get("liquidity", 0.5))
        session_factor = float(market_feats.get("session_factor", 0.5))

        # EWMA smoothing
        self.context.vol_ewma = ema(
            self.context.vol_ewma or vol, vol, self.cfg.ema_alpha
        )
        self.context.spread_ewma = ema(
            self.context.spread_ewma or spread, spread, self.cfg.ema_alpha
        )

        # Build vector: model signals + [trend, vol_norm, spread_norm, liquidity*session]
        vol_norm = (
            0.0 if self.context.vol_ewma == 0 else vol / (self.context.vol_ewma + 1e-9)
        )
        spr_norm = (
            0.0
            if self.context.spread_ewma == 0
            else spread / (self.context.spread_ewma + 1e-9)
        )
        self.context.last_price = price
        self.context.last_update_ts = time.time()

        x = np.array(
            sig_vals + [trend, vol_norm, spr_norm, liquidity * session_factor],
            dtype=np.float64,
        )
        return x, self.context

    def _dict_to_vector(self, feats: Dict[str, float]) -> np.ndarray:
        """Convert feature dict to vector for feedback"""
        vals = []
        for k in self.signal_keys:
            vals.append(float(feats.get(k, 0.0)))
        vals.append(float(feats.get("trend", 0.0)))
        vals.append(float(feats.get("vol_norm", 0.0)))
        vals.append(float(feats.get("spr_norm", 0.0)))
        vals.append(float(feats.get("liq_sess", 0.0)))
        return np.array(vals, dtype=np.float64)

    def _calculate_anomaly_score(self, x: np.ndarray) -> float:
        """Calculate anomaly score for gating"""
        if not hasattr(self, "_x_cache"):
            self._x_cache = []
        self._x_cache.append(x.copy())
        if len(self._x_cache) > 200:
            self._x_cache.pop(0)
        if len(self._x_cache) < 30:
            return 0.0
        X = np.vstack(self._x_cache)
        z = np.abs(robust_z(X.flatten()))
        return float(np.max(z))

    def _update_regime(self, trend_strength: float, vol_ewma: float):
        """Update market regime"""
        if vol_ewma <= 0:
            self.context.regime = "neutral"
            return
        vt = np.tanh(vol_ewma)
        tt = np.tanh(abs(trend_strength))
        if vt > 0.7 and tt < 0.3:
            self.context.regime = "highvol"
        elif vt < 0.3 and tt < 0.3:
            self.context.regime = "lowvol"
        elif tt >= 0.3:
            self.context.regime = "trend"
        else:
            self.context.regime = "neutral"

    def _shape_by_regime(self, score: float) -> float:
        """Shape meta-score by regime"""
        f = score
        if self.context.regime == "highvol":
            f *= 0.7
        elif self.context.regime == "lowvol":
            f *= 0.9
        elif self.context.regime == "trend":
            f *= 1.15
        return capped(f, -1.0, 1.0)

    def _create_flat_idea(self, base_idea: EnhancedTradeIdea) -> EnhancedTradeIdea:
        """Create flat idea when anomaly detected"""
        return EnhancedTradeIdea(
            symbol=base_idea.symbol,
            bias="neutral",
            confidence=0.0,
            entry=base_idea.entry,
            stop=0.0,
            takeprofit=0.0,
            meta_weights={},
            regime=self.context.regime,
            anomaly_score=float("inf"),
            ts=time.time(),
        )

    def _risk_position(self, idea: EnhancedTradeIdea, current_price: float) -> float:
        """Enhanced position sizing using RiskEngine (DAN_LIVE_ONLY)"""
        if idea is None or idea.confidence <= 0.0:
            return 0.0

        # Refresh account context (equity) if available
        try:
            self.risk_engine.refresh_account()
        except Exception:
            pass

        # Calculate stop-loss distance (price units)
        sl_distance = abs(current_price - idea.stop)

        # Convert to pips using configured symbol_point_map
        sl_pips = pips(self.symbol, sl_distance, self.cfg.symbol_point_map)
        if sl_pips <= 0:
            return 0.0

        # Use enhanced risk engine for position sizing
        lot_size = self.risk_engine.size_from_sl(sl_pips, self.symbol, idea.confidence)

        # Log meta-weights for trade analysis
        logger.info(
            f"[DAN_LIVE] {self.symbol} {idea.bias} lot_size={lot_size:.4f} "
            f"conf={idea.confidence:.3f} meta_weights={idea.meta_weights}"
        )

        return lot_size

    def execute_trade(self, idea: EnhancedTradeIdea, current_price: float) -> bool:
        """Execute trade using TradeArbiter (DAN_LIVE_ONLY)"""
        # Kill-switch gate always takes precedence (even in dry-run)
        if self._halt:
            logger.warning(
                "[KILL] Kill switch engaged; skipping trade for %s", self.symbol
            )
            return False

        # Shorting permission gate
        if idea and idea.bias == "bearish" and not self.cfg.allow_short:
            logger.info(
                "Shorting disabled by config; skipping bearish idea for %s", self.symbol
            )
            return False

        # Daily drawdown gate: halt trading if exceeded
        try:
            self.risk_engine.refresh_account()
        except Exception:
            pass
        try:
            max_dd_amt = float(self.cfg.max_daily_drawdown) * float(
                self.risk_engine.account_equity
            )
            pnl_day = float(self._state.get("pnl_day", 0.0))
            if max_dd_amt > 0.0 and pnl_day <= -max_dd_amt:
                logger.error(
                    "Max daily drawdown reached for %s (pnl_day=%.2f <= -%.2f). Engaging kill switch.",
                    self.symbol,
                    pnl_day,
                    max_dd_amt,
                )
                self.kill_switch(True)
                return False
        except Exception:
            pass

        if not self.cfg.enable_execution:
            logger.info(
                f"[DRY_RUN] Would execute: {self.symbol} {idea.bias} {idea.confidence:.3f}"
            )
            return True

        # Calculate position size
        lot_size = self._risk_position(idea, current_price)
        if lot_size <= 0:
            logger.warning(f"Invalid lot size for {self.symbol}: {lot_size}")
            return False

        # Create execution plan
        direction = 1 if idea.bias == "bullish" else -1
        exec_plan = ExecPlan(
            symbol=self.symbol,
            direction=direction,
            lots=lot_size,
            price=current_price,
            sl=idea.stop,
            tp=idea.takeprofit,
            reason=f"SMC_Fusion_{idea.confidence:.3f}",
            meta={
                "meta_weights": idea.meta_weights,
                "regime": idea.regime,
                "anomaly_score": idea.anomaly_score,
            },
        )

        # Route through arbiter
        result = self.trade_arbiter.route(exec_plan)
        if result:
            logger.info(f"[DAN_LIVE] Trade executed: {self.symbol} {lot_size:.4f} lots")
            return True
        else:
            logger.error(f"[DAN_LIVE] Trade execution failed: {self.symbol}")
            return False


# --------- Factory Functions -------------------------------------------------


def get_engine(symbol: str) -> EnhancedSMCFusionCore:
    """Factory function to create SMC fusion core (enhanced version)"""
    return get_enhanced_engine(symbol)


def get_enhanced_engine(
    symbol: str, signal_keys: List[str] = None
) -> EnhancedSMCFusionCore:
    """Factory function to create enhanced SMC fusion core"""
    if signal_keys is None:
        signal_keys = ["lstm", "cnn", "ppo", "vision", "llm_macro"]
        # Optionally include XGB signal by environment toggle
        # ARIA_INCLUDE_XGB=1 (default) or ARIA_INCLUDE_XGB_SIGNAL=1
        include_xgb = (
            os.environ.get("ARIA_INCLUDE_XGB", "1") == "1"
            or os.environ.get("ARIA_INCLUDE_XGB_SIGNAL", "0") == "1"
        )
        if include_xgb and "xgb" not in signal_keys:
            signal_keys.append("xgb")

    cfg = EnhancedFusionConfig(
        symbol_point_map={"EURUSD": 0.0001, "GBPUSD": 0.0001, "XAUUSD": 0.01},
    )

    return EnhancedSMCFusionCore(symbol, cfg, signal_keys)
