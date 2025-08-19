"""
Walk-Forward Orchestrator - Rolling window training with PSR/AUC metrics
Coordinates data prep, model training, inference, calibration, and evaluation
"""

import os
import pathlib
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import subprocess
import sys
import time
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve
from scipy import stats

PROJECT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.getenv("ARIA_DATA_ROOT", PROJECT / "data"))
TRAINING_DIR = pathlib.Path(__file__).parent


class WalkForwardOrchestrator:
    """Orchestrate walk-forward training and evaluation"""

    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        train_months: int = 4,
        test_months: int = 2,
        models: List[str] = None,
        verbose: bool = True,
    ):
        self.symbol = symbol
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.train_months = train_months
        self.test_months = test_months
        self.models = models or ["lstm", "xgb", "cnn"]
        self.verbose = verbose

        # Paths
        self.data_dir = DATA_ROOT / "parquet" / symbol
        self.models_dir = DATA_ROOT / "models" / symbol
        self.features_dir = DATA_ROOT / "features_cache" / symbol
        self.results_dir = DATA_ROOT / "walk_forward" / symbol

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.window_results = []
        self.model_scores = {m: [] for m in self.models}

    def generate_windows(
        self,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """Generate train/test windows"""
        windows = []
        current = self.start_date

        while current < self.end_date:
            train_start = current
            train_end = train_start + pd.DateOffset(months=self.train_months)
            test_start = train_end
            test_end = test_start + pd.DateOffset(months=self.test_months)

            if test_end > self.end_date:
                break

            windows.append((train_start, train_end, test_start, test_end))

            # Slide by test period
            current = current + pd.DateOffset(months=self.test_months)

        return windows

    def prepare_window_data(
        self,
        window_idx: int,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
    ) -> Tuple[pathlib.Path, pathlib.Path]:
        """Prepare data for a specific window"""
        if self.verbose:
            print(f"\n=== Window {window_idx + 1} ===")
            print(f"Train: {train_start.date()} to {train_end.date()}")
            print(f"Test: {test_start.date()} to {test_end.date()}")

        # Output paths
        train_npz = self.features_dir / f"window_{window_idx:02d}_train.npz"
        test_npz = self.features_dir / f"window_{window_idx:02d}_test.npz"

        # Check if already exists
        if train_npz.exists() and test_npz.exists():
            if self.verbose:
                print(f"Using cached NPZ files")
            return train_npz, test_npz

        # Run data preparation
        cmd = [
            sys.executable,
            str(TRAINING_DIR / "prepare_m15_npz.py"),
            "--symbol",
            self.symbol,
            "--start",
            train_start.strftime("%Y-%m-%d"),
            "--end",
            test_end.strftime("%Y-%m-%d"),
            "--output",
            str(self.features_dir / f"window_{window_idx:02d}_full.npz"),
        ]

        if self.verbose:
            print(f"Preparing data...")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error preparing data: {result.stderr}")
            return None, None

        # Split into train/test
        full_npz = self.features_dir / f"window_{window_idx:02d}_full.npz"
        if not full_npz.exists():
            print(f"Failed to create NPZ")
            return None, None

        # Load and split
        data = np.load(full_npz, allow_pickle=False)
        timestamps = data["timestamp"]

        # Find split point
        test_start_ts = test_start.timestamp()
        split_idx = np.searchsorted(timestamps, test_start_ts)

        # Split arrays
        train_data = {}
        test_data = {}

        for key in data.keys():
            arr = data[key]
            train_data[key] = arr[:split_idx]
            test_data[key] = arr[split_idx:]

        # Save splits
        np.savez_compressed(train_npz, **train_data)
        np.savez_compressed(test_npz, **test_data)

        if self.verbose:
            print(f"Train samples: {len(train_data['label'])}")
            print(f"Test samples: {len(test_data['label'])}")

        return train_npz, test_npz

    def train_models(self, window_idx: int, train_npz: pathlib.Path):
        """Train all models for a window"""
        results = {}

        for model in self.models:
            if self.verbose:
                print(f"\nTraining {model.upper()}...")

            # Model-specific output path
            model_path = self.models_dir / f"window_{window_idx:02d}_{model}"

            # Build command
            if model == "lstm":
                cmd = [
                    sys.executable,
                    str(TRAINING_DIR / "train_lstm.py"),
                    "--symbol",
                    self.symbol,
                    "--npz",
                    str(train_npz),
                    "--output",
                    str(model_path),
                    "--epochs",
                    "20",
                    "--batch",
                    "64",
                    "--lr",
                    "0.001",
                ]
            elif model == "xgb":
                cmd = [
                    sys.executable,
                    str(TRAINING_DIR / "train_xgb.py"),
                    "--symbol",
                    self.symbol,
                    "--npz",
                    str(train_npz),
                    "--output",
                    str(model_path),
                    "--n_estimators",
                    "100",
                    "--max_depth",
                    "4",
                    "--learning_rate",
                    "0.05",
                ]
            elif model == "cnn":
                cmd = [
                    sys.executable,
                    str(TRAINING_DIR / "train_cnn1d.py"),
                    "--symbol",
                    self.symbol,
                    "--npz",
                    str(train_npz),
                    "--output",
                    str(model_path),
                    "--epochs",
                    "15",
                    "--batch",
                    "64",
                    "--lr",
                    "0.001",
                    "--model_type",
                    "dilated",
                ]
            else:
                continue

            # Run training
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True)
            train_time = time.time() - start_time

            if result.returncode != 0:
                print(f"Error training {model}: {result.stderr}")
                results[model] = {"status": "failed", "error": result.stderr}
            else:
                results[model] = {
                    "status": "success",
                    "model_path": str(model_path),
                    "train_time": train_time,
                }

                # Check if ONNX was created
                onnx_path = pathlib.Path(f"{model_path}.onnx")
                if onnx_path.exists():
                    results[model]["onnx_size_kb"] = onnx_path.stat().st_size / 1024

        return results

    def run_inference(self, window_idx: int, test_npz: pathlib.Path) -> pathlib.Path:
        """Run inference on test data"""
        if self.verbose:
            print(f"\nRunning inference on test data...")

        # Output path
        scored_npz = test_npz.parent / f"window_{window_idx:02d}_test_scored.npz"

        # Build command
        cmd = [
            sys.executable,
            str(TRAINING_DIR / "inference_to_npz.py"),
            "--symbol",
            self.symbol,
            "--npz",
            str(test_npz),
            "--models",
        ] + self.models

        # Run inference
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error running inference: {result.stderr}")
            return None

        # Check for scored NPZ
        expected_path = test_npz.parent / f"{test_npz.stem}_scored.npz"
        if expected_path.exists():
            return expected_path

        return None

    def evaluate_window(self, window_idx: int, scored_npz: pathlib.Path) -> Dict:
        """Evaluate model performance on test window"""
        if self.verbose:
            print(f"\nEvaluating window {window_idx + 1}...")

        # Load scored data
        data = np.load(scored_npz, allow_pickle=False)
        labels = data["label"]

        # Remove invalid samples
        valid_mask = ~np.isnan(labels)
        labels = labels[valid_mask]

        results = {"window_idx": window_idx, "n_samples": len(labels), "models": {}}

        for model in self.models:
            score_key = f"s_{model}"
            if score_key not in data:
                continue

            scores = data[score_key][valid_mask]

            # Skip if no valid predictions
            if np.isnan(scores).all():
                continue

            # Compute metrics
            metrics = self._compute_metrics(labels, scores)
            results["models"][model] = metrics

            # Store for PSR calculation
            self.model_scores[model].append(metrics)

            if self.verbose:
                print(
                    f"  {model}: AUC={metrics['auc']:.3f}, Acc={metrics['accuracy']:.3f}, PSR={metrics['psr']:.3f}"
                )

        return results

    def _compute_metrics(self, labels: np.ndarray, scores: np.ndarray) -> Dict:
        """Compute evaluation metrics"""
        # Handle NaN scores
        valid_mask = ~np.isnan(scores)
        if valid_mask.sum() == 0:
            return {
                "auc": 0.5,
                "accuracy": 0.5,
                "precision": 0.0,
                "recall": 0.0,
                "psr": 0.0,
                "sharpe": 0.0,
                "mean_score": 0.5,
                "std_score": 0.0,
            }

        labels = labels[valid_mask]
        scores = scores[valid_mask]

        # Binary predictions
        preds = (scores > 0.5).astype(int)

        # Compute metrics
        try:
            auc = roc_auc_score(labels, scores)
        except:
            auc = 0.5

        accuracy = accuracy_score(labels, preds)

        # Precision/Recall
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # PSR (Probabilistic Sharpe Ratio)
        # Based on accuracy relative to random (0.5)
        n = len(labels)
        p = accuracy
        if n > 0:
            z = (p - 0.5) * np.sqrt(n) / np.sqrt(p * (1 - p))
            psr = stats.norm.cdf(z)
        else:
            psr = 0.5

        # Sharpe-like metric
        returns = (preds == labels).astype(float) - 0.5
        if len(returns) > 1 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        return {
            "auc": float(auc),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "psr": float(psr),
            "sharpe": float(sharpe),
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std()),
            "n_valid": int(valid_mask.sum()),
        }

    def compute_gating_weights(self) -> Dict:
        """Compute optimal gating weights based on performance"""
        weights = {}

        for model in self.models:
            if model not in self.model_scores or not self.model_scores[model]:
                weights[model] = 0.0
                continue

            # Average metrics across windows
            metrics = self.model_scores[model]
            avg_auc = np.mean([m["auc"] for m in metrics])
            avg_psr = np.mean([m["psr"] for m in metrics])
            avg_sharpe = np.mean([m["sharpe"] for m in metrics])

            # Weighted score
            score = (
                0.4 * avg_auc + 0.4 * avg_psr + 0.2 * (1 / (1 + np.exp(-avg_sharpe)))
            )
            weights[model] = score

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def save_results(self):
        """Save orchestration results"""
        # Compute final metrics
        final_metrics = {}

        for model in self.models:
            if model not in self.model_scores or not self.model_scores[model]:
                continue

            metrics = self.model_scores[model]
            final_metrics[model] = {
                "mean_auc": float(np.mean([m["auc"] for m in metrics])),
                "std_auc": float(np.std([m["auc"] for m in metrics])),
                "mean_psr": float(np.mean([m["psr"] for m in metrics])),
                "std_psr": float(np.std([m["psr"] for m in metrics])),
                "mean_sharpe": float(np.mean([m["sharpe"] for m in metrics])),
                "n_windows": len(metrics),
            }

        # Compute gating weights
        gating_weights = self.compute_gating_weights()

        # Save results
        results = {
            "symbol": self.symbol,
            "config": {
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
                "train_months": self.train_months,
                "test_months": self.test_months,
                "models": self.models,
            },
            "window_results": self.window_results,
            "final_metrics": final_metrics,
            "gating_weights": gating_weights,
            "timestamp": datetime.now().isoformat(),
        }

        # Save JSON
        results_path = (
            self.results_dir
            / f"walk_forward_results_{datetime.now():%Y%m%d_%H%M%S}.json"
        )
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        if self.verbose:
            print(f"\nSaved results to {results_path}")

        # Save gating weights separately
        gating_path = self.results_dir / "gating_weights.json"
        with open(gating_path, "w") as f:
            json.dump(gating_weights, f, indent=2)

        if self.verbose:
            print(f"Saved gating weights to {gating_path}")

        return results

    def run(self):
        """Run complete walk-forward orchestration"""
        print(f"\n{'='*60}")
        print(f"Walk-Forward Orchestration for {self.symbol}")
        print(f"{'='*60}")

        # Generate windows
        windows = self.generate_windows()
        print(f"\nGenerated {len(windows)} windows")

        # Process each window
        for idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
            # Prepare data
            train_npz, test_npz = self.prepare_window_data(
                idx, train_start, train_end, test_start, test_end
            )

            if train_npz is None or test_npz is None:
                print(f"Skipping window {idx + 1} due to data preparation error")
                continue

            # Train models
            train_results = self.train_models(idx, train_npz)

            # Run inference
            scored_npz = self.run_inference(idx, test_npz)

            if scored_npz is None:
                print(
                    f"Skipping evaluation for window {idx + 1} due to inference error"
                )
                continue

            # Evaluate
            eval_results = self.evaluate_window(idx, scored_npz)
            eval_results["training"] = train_results

            self.window_results.append(eval_results)

        # Save final results
        print(f"\n{'='*60}")
        print(f"Final Results")
        print(f"{'='*60}")

        results = self.save_results()

        # Print summary
        print(f"\nModel Performance Summary:")
        for model, metrics in results["final_metrics"].items():
            print(f"\n{model.upper()}:")
            print(f"  AUC: {metrics['mean_auc']:.3f} ± {metrics['std_auc']:.3f}")
            print(f"  PSR: {metrics['mean_psr']:.3f} ± {metrics['std_psr']:.3f}")
            print(f"  Sharpe: {metrics['mean_sharpe']:.3f}")

        print(f"\nOptimal Gating Weights:")
        for model, weight in results["gating_weights"].items():
            print(f"  {model}: {weight:.3f}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward Orchestrator")
    parser.add_argument("--symbol", required=True, help="Symbol")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--train_months", type=int, default=4, help="Training months")
    parser.add_argument("--test_months", type=int, default=2, help="Test months")
    parser.add_argument("--models", nargs="+", default=None, help="Models to train")

    args = parser.parse_args()

    # Create orchestrator
    orchestrator = WalkForwardOrchestrator(
        symbol=args.symbol,
        start_date=args.start,
        end_date=args.end,
        train_months=args.train_months,
        test_months=args.test_months,
        models=args.models,
    )

    # Run orchestration
    orchestrator.run()


if __name__ == "__main__":
    main()
