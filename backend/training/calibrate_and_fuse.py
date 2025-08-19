"""
Calibration and Fusion Integration
Applies Platt/Isotonic calibration and trains fusion head
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

from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, log_loss
import joblib

PROJECT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.getenv("ARIA_DATA_ROOT", PROJECT / "data"))


class CalibrationFusion:
    """Calibrate model outputs and train fusion head"""

    def __init__(self, symbol: str, verbose: bool = True):
        self.symbol = symbol
        self.verbose = verbose
        self.calibrators = {}
        self.fusion_model = None

        # Paths
        self.models_dir = DATA_ROOT / "models" / symbol
        self.calibration_dir = DATA_ROOT / "calibration" / symbol
        self.calibration_dir.mkdir(parents=True, exist_ok=True)

    def load_scored_data(self, npz_paths: List[pathlib.Path]) -> Dict:
        """Load and concatenate scored NPZ files"""
        all_data = {}

        for path in npz_paths:
            if not path.exists():
                continue

            data = np.load(path, allow_pickle=False)

            for key in data.keys():
                if key not in all_data:
                    all_data[key] = []
                all_data[key].append(data[key])

        # Concatenate arrays
        for key in all_data:
            all_data[key] = np.concatenate(all_data[key])

        return all_data

    def calibrate_model(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        model_name: str,
        method: str = "isotonic",
    ) -> Tuple[object, np.ndarray]:
        """Calibrate a single model's outputs"""
        # Remove invalid samples
        valid_mask = ~np.isnan(scores) & ~np.isnan(labels)
        scores = scores[valid_mask]
        labels = labels[valid_mask]

        if len(scores) == 0:
            return None, np.array([])

        if method == "platt":
            # Platt scaling (sigmoid)
            calibrator = LogisticRegression(max_iter=1000)
            calibrator.fit(scores.reshape(-1, 1), labels)
            calibrated = calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]

        elif method == "isotonic":
            # Isotonic regression
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(scores, labels)
            calibrated = calibrator.transform(scores)

        else:
            # Beta calibration or histogram binning could be added
            calibrator = None
            calibrated = scores

        # Compute calibration metrics
        if self.verbose:
            self._print_calibration_metrics(model_name, scores, calibrated, labels)

        return calibrator, calibrated

    def _print_calibration_metrics(
        self,
        model_name: str,
        raw: np.ndarray,
        calibrated: np.ndarray,
        labels: np.ndarray,
    ):
        """Print calibration improvement metrics"""

        # Expected Calibration Error (ECE)
        def compute_ece(probs, labels, n_bins=10):
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (probs > bin_lower) & (probs <= bin_upper)
                prop_in_bin = in_bin.mean()

                if prop_in_bin > 0:
                    accuracy_in_bin = labels[in_bin].mean()
                    avg_confidence_in_bin = probs[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            return ece

        raw_ece = compute_ece(raw, labels)
        cal_ece = compute_ece(calibrated, labels)

        # Log loss
        raw_loss = log_loss(labels, raw)
        cal_loss = log_loss(labels, calibrated)

        print(f"\n{model_name} Calibration:")
        print(
            f"  ECE: {raw_ece:.4f} -> {cal_ece:.4f} ({(raw_ece-cal_ece)/raw_ece*100:.1f}% improvement)"
        )
        print(f"  Log Loss: {raw_loss:.4f} -> {cal_loss:.4f}")

    def train_fusion(self, data: Dict, models: List[str]) -> LogisticRegression:
        """Train fusion head on calibrated outputs"""
        if self.verbose:
            print("\nTraining fusion head...")

        # Prepare features (calibrated model outputs)
        features = []
        for model in models:
            cal_key = f"cal_{model}"
            if cal_key in data:
                features.append(data[cal_key])

        if not features:
            print("No calibrated features found")
            return None

        X = np.column_stack(features)
        y = data["label"]

        # Remove invalid samples
        valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        # Train fusion model
        fusion = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)

        # Cross-validation score
        cv_scores = cross_val_score(fusion, X, y, cv=5, scoring="roc_auc")

        # Final fit
        fusion.fit(X, y)

        # Fusion predictions
        fusion_probs = fusion.predict_proba(X)[:, 1]
        fusion_auc = roc_auc_score(y, fusion_probs)

        if self.verbose:
            print(f"  CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"  Final AUC: {fusion_auc:.3f}")
            print(f"  Fusion weights: {fusion.coef_[0]}")

        return fusion

    def save_calibrators(self):
        """Save calibration models"""
        for model_name, calibrator in self.calibrators.items():
            if calibrator is None:
                continue

            cal_path = self.calibration_dir / f"{model_name}_calibrator.pkl"
            joblib.dump(calibrator, cal_path)

            if self.verbose:
                print(f"Saved {model_name} calibrator to {cal_path}")

    def save_fusion(self):
        """Save fusion model"""
        if self.fusion_model is None:
            return

        fusion_path = self.calibration_dir / "fusion_model.pkl"
        joblib.dump(self.fusion_model, fusion_path)

        if self.verbose:
            print(f"Saved fusion model to {fusion_path}")

        # Save fusion config
        config = {
            "type": "logistic_regression",
            "input_models": list(self.calibrators.keys()),
            "coefficients": self.fusion_model.coef_[0].tolist(),
            "intercept": float(self.fusion_model.intercept_[0]),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        config_path = self.calibration_dir / "fusion_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    def run_calibration_pipeline(
        self,
        npz_paths: List[pathlib.Path],
        models: List[str] = None,
        calibration_method: str = "isotonic",
    ):
        """Run complete calibration and fusion pipeline"""
        if models is None:
            models = ["lstm", "xgb", "cnn"]

        print(f"\n{'='*60}")
        print(f"Calibration & Fusion Pipeline")
        print(f"{'='*60}")

        # Load scored data
        print("\nLoading scored data...")
        data = self.load_scored_data(npz_paths)

        if "label" not in data:
            print("Error: No labels found in data")
            return

        print(f"Loaded {len(data['label'])} samples")

        # Calibrate each model
        for model in models:
            score_key = f"s_{model}"
            if score_key not in data:
                print(f"Warning: {model} scores not found")
                continue

            calibrator, calibrated = self.calibrate_model(
                data[score_key], data["label"], model, calibration_method
            )

            if calibrator is not None:
                self.calibrators[model] = calibrator
                data[f"cal_{model}"] = calibrated

        # Train fusion
        if len(self.calibrators) >= 2:
            self.fusion_model = self.train_fusion(data, list(self.calibrators.keys()))

        # Save models
        self.save_calibrators()
        self.save_fusion()

        # Generate final report
        self._generate_report(data, models)

        return self.calibrators, self.fusion_model

    def _generate_report(self, data: Dict, models: List[str]):
        """Generate calibration report"""
        report = {"symbol": self.symbol, "n_samples": len(data["label"]), "models": {}}

        labels = data["label"]
        valid_labels = ~np.isnan(labels)
        labels = labels[valid_labels]

        print(f"\n{'='*60}")
        print(f"Performance Summary")
        print(f"{'='*60}")

        for model in models:
            raw_key = f"s_{model}"
            cal_key = f"cal_{model}"

            if raw_key not in data:
                continue

            raw_scores = data[raw_key][valid_labels]

            # Raw metrics
            valid_raw = ~np.isnan(raw_scores)
            if valid_raw.sum() > 0:
                raw_auc = roc_auc_score(labels[valid_raw], raw_scores[valid_raw])
            else:
                raw_auc = 0.5

            model_report = {"raw_auc": float(raw_auc)}

            # Calibrated metrics
            if cal_key in data:
                cal_scores = data[cal_key]
                if len(cal_scores) == len(labels):
                    cal_auc = roc_auc_score(labels, cal_scores)
                    model_report["cal_auc"] = float(cal_auc)
                    model_report["improvement"] = float(cal_auc - raw_auc)

            report["models"][model] = model_report

            print(f"\n{model.upper()}:")
            print(f"  Raw AUC: {model_report['raw_auc']:.3f}")
            if "cal_auc" in model_report:
                print(f"  Calibrated AUC: {model_report['cal_auc']:.3f}")
                print(f"  Improvement: {model_report['improvement']:+.3f}")

        # Fusion performance
        if self.fusion_model is not None:
            X_fusion = []
            for model in self.calibrators.keys():
                cal_key = f"cal_{model}"
                if cal_key in data:
                    X_fusion.append(data[cal_key])

            if X_fusion:
                X_fusion = np.column_stack(X_fusion)
                fusion_probs = self.fusion_model.predict_proba(X_fusion)[:, 1]
                fusion_auc = roc_auc_score(labels, fusion_probs)

                report["fusion"] = {
                    "auc": float(fusion_auc),
                    "weights": self.fusion_model.coef_[0].tolist(),
                }

                print(f"\nFusion Model:")
                print(f"  AUC: {fusion_auc:.3f}")
                print(f"  Weights: {report['fusion']['weights']}")

        # Save report
        report_path = self.calibration_dir / "calibration_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nSaved report to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Calibration and Fusion")
    parser.add_argument("--symbol", required=True, help="Symbol")
    parser.add_argument("--npz", nargs="+", required=True, help="Scored NPZ files")
    parser.add_argument("--models", nargs="+", default=None, help="Models to calibrate")
    parser.add_argument(
        "--method",
        default="isotonic",
        choices=["platt", "isotonic"],
        help="Calibration method",
    )

    args = parser.parse_args()

    # Convert paths
    npz_paths = [pathlib.Path(p) for p in args.npz]

    # Check paths exist
    for path in npz_paths:
        if not path.exists():
            print(f"Error: {path} not found")
            return

    # Run calibration
    calibrator = CalibrationFusion(args.symbol)
    calibrator.run_calibration_pipeline(npz_paths, args.models, args.method)


if __name__ == "__main__":
    main()
