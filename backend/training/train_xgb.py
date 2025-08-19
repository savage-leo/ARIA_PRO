"""
XGBoost Trainer - CPU-optimized gradient boosting
Produces compact ONNX models with calibrated probabilities
"""

import os
import pathlib
import argparse
import json
import numpy as np
import time
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib

PROJECT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.getenv("ARIA_DATA_ROOT", PROJECT / "data"))


def prepare_features(
    arr: Dict, feature_cols: List[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare features for XGBoost"""
    if feature_cols is None:
        # Optimized feature set for XGBoost
        feature_cols = [
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
        ]

    # Build feature matrix
    features = []
    for col in feature_cols:
        if col in arr:
            feat = arr[col].astype(np.float32)
            # Handle NaN/Inf
            feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(feat)

    X = np.column_stack(features)
    y = arr["label"].astype(int)

    # Add engineered features
    X = add_engineered_features(X, arr)

    return X, y, feature_cols


def add_engineered_features(X: np.ndarray, arr: Dict) -> np.ndarray:
    """Add domain-specific engineered features"""
    extra_features = []

    # Volatility ratio
    if "ewma_sig" in arr and "atr14" in arr:
        vol_ratio = arr["ewma_sig"] / (arr["atr14"] + 1e-8)
        extra_features.append(vol_ratio)

    # Session dummies (one-hot encoding)
    if "session" in arr:
        sessions = arr["session"]
        for sess in ["ASIA", "EU", "US"]:
            dummy = (sessions == sess).astype(np.float32)
            extra_features.append(dummy)

    # Regime indicators
    if "vol_bucket" in arr:
        vol_buckets = arr["vol_bucket"]
        for bucket in ["Low", "Med", "High"]:
            dummy = (vol_buckets == bucket).astype(np.float32)
            extra_features.append(dummy)

    if extra_features:
        X = np.column_stack([X] + extra_features)

    return X.astype(np.float32)


def compute_sample_weights(y: np.ndarray, arr: Dict) -> np.ndarray:
    """Compute sample weights for balanced training"""
    weights = np.ones(len(y), dtype=np.float32)

    # Class balancing
    unique, counts = np.unique(y, return_counts=True)
    class_weights = len(y) / (len(unique) * counts)
    weight_map = dict(zip(unique, class_weights))

    for i, label in enumerate(y):
        weights[i] *= weight_map.get(label, 1.0)

    # Regime-based weighting (optional)
    if "vol_bucket" in arr:
        vol_buckets = arr["vol_bucket"]
        # Upweight high volatility samples
        high_vol_mask = vol_buckets == "High"
        weights[high_vol_mask] *= 1.2

    return weights


def train_xgboost(symbol: str, npz_path: pathlib.Path, config: Dict):
    """Main XGBoost training function"""
    print(f"Loading data from {npz_path}")
    arr = np.load(npz_path, allow_pickle=False)

    # Prepare features
    X, y, feature_cols = prepare_features(arr, config.get("features"))

    # Remove samples with missing labels
    valid_mask = ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"Training on {len(y)} samples with {X.shape[1]} features")

    # Sample weights
    weights = compute_sample_weights(y, {k: v[valid_mask] for k, v in arr.items()})

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=config.get("cv_splits", 3))

    # XGBoost parameters optimized for CPU
    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": config.get("max_depth", 3),
        "eta": config.get("learning_rate", 0.05),
        "subsample": config.get("subsample", 0.8),
        "colsample_bytree": config.get("colsample_bytree", 0.8),
        "min_child_weight": config.get("min_child_weight", 5),
        "gamma": config.get("gamma", 0.1),
        "alpha": config.get("alpha", 0.1),  # L1 regularization
        "lambda": config.get("lambda", 1.0),  # L2 regularization
        "seed": config.get("seed", 0),
        "nthread": 2,  # Use 2 threads on T470
        "verbosity": 0,
    }

    # Cross-validation results
    cv_results = []
    best_score = -np.inf
    best_model = None
    best_iteration = 0

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\nFold {fold + 1}/{tscv.n_splits}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        w_train, w_val = weights[train_idx], weights[val_idx]

        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
        dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)

        # Train with early stopping
        evals = [(dtrain, "train"), (dval, "val")]

        start_time = time.time()
        bst = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=config.get("n_estimators", 100),
            evals=evals,
            early_stopping_rounds=config.get("early_stopping", 10),
            verbose_eval=False,
        )
        train_time = time.time() - start_time

        # Evaluate
        y_pred_proba = bst.predict(dval)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Metrics
        auc = roc_auc_score(y_val, y_pred_proba) if len(np.unique(y_val)) > 1 else 0.5
        logloss = log_loss(y_val, y_pred_proba)
        accuracy = accuracy_score(y_val, y_pred)

        fold_metrics = {
            "fold": fold,
            "auc": auc,
            "logloss": logloss,
            "accuracy": accuracy,
            "best_iteration": bst.best_iteration,
            "train_time": train_time,
        }
        cv_results.append(fold_metrics)

        print(
            f"  AUC: {auc:.4f} | LogLoss: {logloss:.4f} | Acc: {accuracy:.4f} | "
            f"Trees: {bst.best_iteration} | Time: {train_time:.1f}s"
        )

        # Keep best model
        if auc > best_score:
            best_score = auc
            best_model = bst
            best_iteration = bst.best_iteration

    # Train final model on all data
    print(f"\nTraining final model with {best_iteration} trees...")
    dtrain_all = xgb.DMatrix(X, label=y, weight=weights)

    final_model = xgb.train(xgb_params, dtrain_all, num_boost_round=best_iteration)

    # Save model
    out_dir = DATA_ROOT / "models" / symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "xgb.model"
    final_model.save_model(str(model_path))

    # Convert to sklearn wrapper for ONNX export
    clf = xgb.XGBClassifier(n_estimators=best_iteration, **xgb_params)
    clf._Booster = final_model
    clf.n_features_in_ = X.shape[1]
    clf.classes_ = np.array([0, 1])

    # Save pickle
    joblib.dump(clf, out_dir / "xgb.pkl")

    # Export to ONNX
    print("Exporting to ONNX...")
    initial_type = [("float_input", FloatTensorType([None, X.shape[1]]))]

    try:
        onx = convert_sklearn(
            clf,
            initial_types=initial_type,
            target_opset=11,
            options={id(clf): {"zipmap": False}},  # Return probabilities as array
        )

        onnx_path = out_dir / "xgb.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onx.SerializeToString())

        onnx_size_kb = onnx_path.stat().st_size / 1024
        print(f"Exported ONNX to {onnx_path} ({onnx_size_kb:.1f} KB)")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        onnx_size_kb = 0

    # Feature importance
    importance = final_model.get_score(importance_type="gain")
    feature_importance = {}
    for i, col in enumerate(feature_cols):
        feat_name = f"f{i}"
        if feat_name in importance:
            feature_importance[col] = importance[feat_name]

    # Sort by importance
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    )

    # Save metadata
    metadata = {
        "symbol": symbol,
        "best_auc": float(best_score),
        "cv_results": cv_results,
        "config": config,
        "feature_cols": feature_cols,
        "n_features": X.shape[1],
        "n_samples": len(y),
        "n_trees": best_iteration,
        "feature_importance": feature_importance,
        "onnx_size_kb": onnx_size_kb,
    }

    with open(out_dir / "xgb_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost model")
    parser.add_argument("--symbol", required=True, help="Symbol to train")
    parser.add_argument("--npz", default=None, help="Path to NPZ file")
    parser.add_argument("--rounds", type=int, default=100, help="Max boosting rounds")
    parser.add_argument("--depth", type=int, default=3, help="Max tree depth")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate (eta)")
    parser.add_argument("--subsample", type=float, default=0.8, help="Subsample ratio")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)

    # NPZ path
    if args.npz:
        npz_path = pathlib.Path(args.npz)
    else:
        npz_path = DATA_ROOT / "features_cache" / args.symbol / "train_m15.npz"

    if not npz_path.exists():
        print(f"Error: NPZ file not found at {npz_path}")
        return

    # Training config
    config = {
        "n_estimators": args.rounds,
        "max_depth": args.depth,
        "learning_rate": args.lr,
        "subsample": args.subsample,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "alpha": 0.1,
        "lambda": 1.0,
        "early_stopping": 10,
        "cv_splits": 3,
        "seed": args.seed,
    }

    # Train
    metadata = train_xgboost(args.symbol, npz_path, config)

    if metadata:
        print(f"\nTraining complete!")
        print(f"Best AUC: {metadata['best_auc']:.4f}")
        print(f"Trees: {metadata['n_trees']}")
        print(f"ONNX size: {metadata['onnx_size_kb']:.1f} KB")

        # Top features
        print("\nTop 5 features:")
        for i, (feat, score) in enumerate(
            list(metadata["feature_importance"].items())[:5]
        ):
            print(f"  {i+1}. {feat}: {score:.2f}")


if __name__ == "__main__":
    main()
