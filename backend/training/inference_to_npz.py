"""
Inference to NPZ - Score all models and update NPZ with predictions
CPU-optimized batch inference for calibration pipeline
"""

import os
import pathlib
import argparse
import json
import numpy as np
import time
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore")

import onnxruntime as ort
import xgboost as xgb
import joblib

PROJECT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.getenv("ARIA_DATA_ROOT", PROJECT / "data"))


class ModelInference:
    """Unified inference for all model types"""

    def __init__(self, symbol: str, verbose: bool = True):
        self.symbol = symbol
        self.verbose = verbose
        self.models_dir = DATA_ROOT / "models" / symbol
        self.sessions = {}
        self.metadata = {}

        # ONNX providers for CPU
        self.providers = ["CPUExecutionProvider"]

    def load_models(self, model_types: List[str] = None):
        """Load specified models or all available"""
        if model_types is None:
            model_types = ["lstm", "xgb", "cnn", "ppo"]

        for model_type in model_types:
            self._load_model(model_type)

    def _load_model(self, model_type: str):
        """Load a specific model"""
        try:
            if model_type == "lstm":
                self._load_lstm()
            elif model_type == "xgb":
                self._load_xgb()
            elif model_type == "cnn":
                self._load_cnn()
            elif model_type == "ppo":
                self._load_ppo()
            else:
                if self.verbose:
                    print(f"Unknown model type: {model_type}")
        except Exception as e:
            if self.verbose:
                print(f"Failed to load {model_type}: {e}")

    def _load_lstm(self):
        """Load LSTM ONNX model"""
        onnx_path = self.models_dir / "lstm.onnx"
        meta_path = self.models_dir / "lstm_metadata.json"

        if not onnx_path.exists():
            if self.verbose:
                print(f"LSTM model not found at {onnx_path}")
            return

        # Load ONNX session
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 1

        self.sessions["lstm"] = ort.InferenceSession(
            str(onnx_path), sess_options=sess_options, providers=self.providers
        )

        # Load metadata
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.metadata["lstm"] = json.load(f)

        if self.verbose:
            print(f"Loaded LSTM model ({onnx_path.stat().st_size / 1024:.1f} KB)")

    def _load_xgb(self):
        """Load XGBoost model"""
        # Try ONNX first
        onnx_path = self.models_dir / "xgb.onnx"
        pkl_path = self.models_dir / "xgb.pkl"
        model_path = self.models_dir / "xgb.model"
        meta_path = self.models_dir / "xgb_metadata.json"

        if onnx_path.exists():
            # Load ONNX
            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = 1
            sess_options.intra_op_num_threads = 1

            self.sessions["xgb"] = ort.InferenceSession(
                str(onnx_path), sess_options=sess_options, providers=self.providers
            )
            if self.verbose:
                print(f"Loaded XGBoost ONNX ({onnx_path.stat().st_size / 1024:.1f} KB)")
        elif pkl_path.exists():
            # Load pickle
            self.sessions["xgb"] = joblib.load(pkl_path)
            if self.verbose:
                print(f"Loaded XGBoost pickle")
        elif model_path.exists():
            # Load native XGBoost
            bst = xgb.Booster()
            bst.load_model(str(model_path))
            self.sessions["xgb"] = bst
            if self.verbose:
                print(f"Loaded XGBoost native model")
        else:
            if self.verbose:
                print(f"XGBoost model not found")
            return

        # Load metadata
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.metadata["xgb"] = json.load(f)

    def _load_cnn(self):
        """Load CNN ONNX model"""
        onnx_path = self.models_dir / "cnn.onnx"
        meta_path = self.models_dir / "cnn_metadata.json"

        if not onnx_path.exists():
            if self.verbose:
                print(f"CNN model not found at {onnx_path}")
            return

        # Load ONNX session
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 1

        self.sessions["cnn"] = ort.InferenceSession(
            str(onnx_path), sess_options=sess_options, providers=self.providers
        )

        # Load metadata
        if meta_path.exists():
            with open(meta_path, "r") as f:
                self.metadata["cnn"] = json.load(f)

        if self.verbose:
            print(f"Loaded CNN model ({onnx_path.stat().st_size / 1024:.1f} KB)")

    def _load_ppo(self):
        """Load PPO model - using momentum heuristic"""
        # PPO requires environment interaction, using momentum-based heuristic
        self.sessions["ppo"] = "heuristic"
        if self.verbose:
            print("PPO using momentum-based heuristic")

    def predict_lstm(self, arr: Dict) -> np.ndarray:
        """Run LSTM inference"""
        if "lstm" not in self.sessions:
            return np.zeros(len(arr["label"]), dtype=np.float32)

        sess = self.sessions["lstm"]
        meta = self.metadata.get("lstm", {})

        # Get feature columns from metadata
        feature_cols = meta.get(
            "feature_cols", ["r", "abs_r", "ewma_sig", "atr14", "rsi", "bb_pos", "mom5"]
        )
        seq_len = meta.get("config", {}).get("seq_len", 16)

        # Prepare features
        features = []
        for col in feature_cols:
            if col in arr:
                features.append(arr[col])

        X = np.stack(features, axis=1).astype(np.float32)

        # Normalize using stored stats
        if "normalization" in meta:
            mean = np.array(meta["normalization"]["mean"])
            std = np.array(meta["normalization"]["std"])
            X = (X - mean) / std

        # Predict in batches
        n_samples = len(X) - seq_len
        batch_size = 256
        predictions = []

        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_X = []

            for j in range(i, batch_end):
                seq = X[j : j + seq_len]
                batch_X.append(seq)

            if batch_X:
                batch_X = np.array(batch_X, dtype=np.float32)

                # Run inference
                outputs = sess.run(None, {"input": batch_X})[0]

                # Get probabilities
                probs = self._softmax(outputs)[:, 1]  # P(Long)
                predictions.extend(probs)

        # Pad to match original length
        result = np.zeros(len(arr["label"]), dtype=np.float32)
        result[seq_len : seq_len + len(predictions)] = predictions

        return result

    def predict_xgb(self, arr: Dict) -> np.ndarray:
        """Run XGBoost inference"""
        if "xgb" not in self.sessions:
            return np.zeros(len(arr["label"]), dtype=np.float32)

        model = self.sessions["xgb"]
        meta = self.metadata.get("xgb", {})

        # Get feature columns
        feature_cols = meta.get(
            "feature_cols",
            [
                "r",
                "abs_r",
                "ewma_sig",
                "atr14",
                "spread_z",
                "rsi",
                "bb_pos",
                "vol_ratio",
                "mom5",
                "mom10",
                "mom20",
            ],
        )

        # Prepare features
        features = []
        for col in feature_cols:
            if col in arr:
                feat = arr[col].astype(np.float32)
                feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
                features.append(feat)

        X = np.column_stack(features)

        # Add engineered features if needed
        X = self._add_xgb_features(X, arr, meta)

        # Predict
        if isinstance(model, ort.InferenceSession):
            # ONNX inference
            outputs = model.run(None, {"float_input": X.astype(np.float32)})[0]
            if outputs.ndim == 2 and outputs.shape[1] == 2:
                # Classification output
                predictions = self._softmax(outputs)[:, 1]
            else:
                # Raw probabilities
                predictions = outputs.flatten()
        elif isinstance(model, xgb.Booster):
            # Native XGBoost
            dmat = xgb.DMatrix(X)
            predictions = model.predict(dmat)
        else:
            # Sklearn-like interface
            predictions = model.predict_proba(X)[:, 1]

        return predictions.astype(np.float32)

    def predict_cnn(self, arr: Dict) -> np.ndarray:
        """Run CNN inference"""
        if "cnn" not in self.sessions:
            return np.zeros(len(arr["label"]), dtype=np.float32)

        sess = self.sessions["cnn"]
        meta = self.metadata.get("cnn", {})

        # Get configuration
        feature_cols = meta.get(
            "feature_cols", ["r", "abs_r", "ewma_sig", "atr14", "rsi", "bb_pos"]
        )
        seq_len = meta.get("config", {}).get("seq_len", 32)

        # Prepare features
        features = []
        for col in feature_cols:
            if col in arr:
                features.append(arr[col])

        X = np.stack(features, axis=1).astype(np.float32)

        # Normalize
        if "normalization" in meta:
            mean = np.array(meta["normalization"]["mean"])
            std = np.array(meta["normalization"]["std"])
            X = (X - mean) / std

        # Predict in batches
        n_samples = len(X) - seq_len
        batch_size = 256
        predictions = []

        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_X = []

            for j in range(i, batch_end):
                seq = X[j : j + seq_len].T  # Transpose for Conv1d
                batch_X.append(seq)

            if batch_X:
                batch_X = np.array(batch_X, dtype=np.float32)

                # Run inference
                outputs = sess.run(None, {"input": batch_X})[0]

                # Get probabilities
                probs = self._softmax(outputs)[:, 1]  # P(Long)
                predictions.extend(probs)

        # Pad to match original length
        result = np.zeros(len(arr["label"]), dtype=np.float32)
        result[seq_len : seq_len + len(predictions)] = predictions

        return result

    def predict_ppo(self, arr: Dict) -> np.ndarray:
        """PPO momentum-based heuristic predictions"""
        # Momentum-based heuristic for directional prediction
        n = len(arr["label"])
        predictions = np.zeros(n, dtype=np.float32)

        if "mom5" in arr and "ewma_sig" in arr:
            # Buy when momentum positive and volatility low
            mom = arr["mom5"]
            vol = arr["ewma_sig"]
            vol_median = np.median(vol[vol > 0])

            buy_signal = (mom > 0) & (vol < vol_median)
            predictions[buy_signal] = 0.6
            predictions[~buy_signal] = 0.4
        else:
            predictions[:] = 0.5

        return predictions

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _add_xgb_features(self, X: np.ndarray, arr: Dict, meta: Dict) -> np.ndarray:
        """Add engineered features for XGBoost"""
        extra_features = []

        # Session dummies
        if "session" in arr:
            sessions = arr["session"]
            for sess in ["ASIA", "EU", "US"]:
                dummy = (sessions == sess).astype(np.float32)
                extra_features.append(dummy)

        # Volatility bucket dummies
        if "vol_bucket" in arr:
            vol_buckets = arr["vol_bucket"]
            for bucket in ["Low", "Med", "High"]:
                dummy = (vol_buckets == bucket).astype(np.float32)
                extra_features.append(dummy)

        if extra_features:
            X = np.column_stack([X] + extra_features)

        return X.astype(np.float32)


def update_npz_with_scores(
    npz_path: pathlib.Path, symbol: str, models: List[str] = None
):
    """Update NPZ file with model predictions"""
    print(f"\nProcessing {npz_path}")

    # Load original data
    arr = dict(np.load(npz_path, allow_pickle=False))

    # Initialize inference engine
    inference = ModelInference(symbol, verbose=True)

    # Load models
    if models is None:
        models = ["lstm", "xgb", "cnn", "ppo"]

    inference.load_models(models)

    # Run predictions
    print("\nGenerating predictions...")
    results = {}

    for model in models:
        start_time = time.time()

        if model == "lstm":
            scores = inference.predict_lstm(arr)
        elif model == "xgb":
            scores = inference.predict_xgb(arr)
        elif model == "cnn":
            scores = inference.predict_cnn(arr)
        elif model == "ppo":
            scores = inference.predict_ppo(arr)
        else:
            scores = np.zeros(len(arr["label"]), dtype=np.float32)

        # Store in array
        arr[f"s_{model}"] = scores

        # Compute metrics
        valid_mask = ~np.isnan(scores) & (scores > 0) & (scores < 1)
        if valid_mask.sum() > 0:
            mean_score = scores[valid_mask].mean()
            std_score = scores[valid_mask].std()
        else:
            mean_score = 0.5
            std_score = 0.0

        elapsed = time.time() - start_time

        results[model] = {
            "mean": float(mean_score),
            "std": float(std_score),
            "min": float(scores.min()),
            "max": float(scores.max()),
            "inference_time": elapsed,
            "n_predictions": int(valid_mask.sum()),
        }

        print(
            f"  {model}: mean={mean_score:.3f}, std={std_score:.3f}, time={elapsed:.2f}s"
        )

    # Save updated NPZ
    out_path = npz_path.parent / f"{npz_path.stem}_scored.npz"
    np.savez_compressed(out_path, **arr)
    print(f"\nSaved scored NPZ to {out_path}")

    # Save inference metadata
    meta_path = out_path.with_suffix(".json")
    metadata = {
        "symbol": symbol,
        "source_npz": str(npz_path),
        "models": models,
        "results": results,
        "n_samples": len(arr["label"]),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved metadata to {meta_path}")

    return out_path, metadata


def main():
    parser = argparse.ArgumentParser(description="Generate model scores for NPZ")
    parser.add_argument("--symbol", required=True, help="Symbol")
    parser.add_argument("--npz", required=True, help="Input NPZ path")
    parser.add_argument("--models", nargs="+", default=None, help="Models to run")

    args = parser.parse_args()

    npz_path = pathlib.Path(args.npz)
    if not npz_path.exists():
        print(f"Error: NPZ not found at {npz_path}")
        return

    # Update NPZ with scores
    update_npz_with_scores(npz_path, args.symbol, args.models)


if __name__ == "__main__":
    main()
