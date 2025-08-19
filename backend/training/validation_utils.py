"""
Validation Utilities - Model validation, metrics, and artifact management
"""

import os
import pathlib
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
import onnxruntime as ort

PROJECT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.getenv("ARIA_DATA_ROOT", PROJECT / "data"))


class ModelValidator:
    """Validate model performance and generate reports"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.models_dir = DATA_ROOT / "models" / symbol
        self.validation_dir = DATA_ROOT / "validation" / symbol
        self.validation_dir.mkdir(parents=True, exist_ok=True)

    def validate_onnx_model(self, onnx_path: pathlib.Path) -> Dict:
        """Validate ONNX model properties"""
        if not onnx_path.exists():
            return {"error": "Model not found"}

        try:
            # Load session
            sess_options = ort.SessionOptions()
            sess_options.inter_op_num_threads = 1
            sess_options.intra_op_num_threads = 1

            session = ort.InferenceSession(
                str(onnx_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )

            # Get model info
            inputs = session.get_inputs()
            outputs = session.get_outputs()

            validation = {
                "file_size_kb": onnx_path.stat().st_size / 1024,
                "inputs": [
                    {"name": inp.name, "shape": inp.shape, "type": inp.type}
                    for inp in inputs
                ],
                "outputs": [
                    {"name": out.name, "shape": out.shape, "type": out.type}
                    for out in outputs
                ],
                "providers": session.get_providers(),
            }

            # Test inference speed
            input_shape = inputs[0].shape
            if isinstance(input_shape[0], str):
                batch_size = 1
            else:
                batch_size = input_shape[0] or 1

            # Create dummy input
            dummy_shape = []
            for dim in input_shape:
                if isinstance(dim, str) or dim is None:
                    dummy_shape.append(1)
                else:
                    dummy_shape.append(dim)

            dummy_input = np.random.randn(*dummy_shape).astype(np.float32)

            # Warm-up
            for _ in range(5):
                _ = session.run(None, {inputs[0].name: dummy_input})

            # Measure inference time
            import time

            n_runs = 100
            start = time.time()
            for _ in range(n_runs):
                _ = session.run(None, {inputs[0].name: dummy_input})
            elapsed = time.time() - start

            validation["inference_ms"] = (elapsed / n_runs) * 1000
            validation["status"] = "valid"

        except Exception as e:
            validation = {"error": str(e), "status": "invalid"}

        return validation

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict:
        """Compute comprehensive metrics"""
        # Remove invalid samples
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        metrics = {
            "n_samples": len(y_true),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }

        # Add AUC if probabilities provided
        if y_prob is not None:
            y_prob = y_prob[valid_mask]
            try:
                metrics["auc"] = roc_auc_score(y_true, y_prob)
            except:
                metrics["auc"] = 0.5

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Class balance
        metrics["class_balance"] = {
            "class_0": int((y_true == 0).sum()),
            "class_1": int((y_true == 1).sum()),
        }

        return metrics

    def validate_predictions(
        self, npz_path: pathlib.Path, models: List[str] = None
    ) -> Dict:
        """Validate predictions from scored NPZ"""
        if not npz_path.exists():
            return {"error": "NPZ file not found"}

        data = np.load(npz_path, allow_pickle=False)

        if models is None:
            models = ["lstm", "xgb", "cnn"]

        results = {
            "npz_file": str(npz_path),
            "n_samples": len(data["label"]),
            "models": {},
        }

        labels = data["label"]

        for model in models:
            score_key = f"s_{model}"
            if score_key not in data:
                continue

            scores = data[score_key]
            preds = (scores > 0.5).astype(int)

            metrics = self.compute_metrics(labels, preds, scores)
            results["models"][model] = metrics

        return results

    def plot_calibration_curve(
        self, y_true: np.ndarray, y_prob: np.ndarray, model_name: str, n_bins: int = 10
    ):
        """Plot calibration curve"""
        if not PLOTTING_AVAILABLE:
            print(
                f"Matplotlib not available, skipping calibration plot for {model_name}"
            )
            return None

        # Remove invalid
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
        y_true = y_true[valid_mask]
        y_prob = y_prob[valid_mask]

        # Compute calibration
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        true_probs = []
        pred_probs = []

        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if mask.sum() > 0:
                true_probs.append(y_true[mask].mean())
                pred_probs.append(y_prob[mask].mean())

        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        plt.plot(pred_probs, true_probs, "o-", label=model_name)
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title(f"Calibration Curve - {model_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save
        plot_path = self.validation_dir / f"calibration_{model_name}.png"
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()

        return plot_path

    def generate_validation_report(self, results: Dict):
        """Generate comprehensive validation report"""
        report = {
            "symbol": self.symbol,
            "timestamp": pd.Timestamp.now().isoformat(),
            "validation_results": results,
        }

        # Save JSON report
        report_path = self.validation_dir / "validation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Generate markdown report
        md_lines = [
            f"# Validation Report - {self.symbol}",
            f"\nGenerated: {report['timestamp']}",
            f"\n## Model Performance\n",
        ]

        if "models" in results:
            for model, metrics in results["models"].items():
                md_lines.append(f"\n### {model.upper()}")
                md_lines.append(f"- Accuracy: {metrics.get('accuracy', 0):.3f}")
                md_lines.append(f"- Precision: {metrics.get('precision', 0):.3f}")
                md_lines.append(f"- Recall: {metrics.get('recall', 0):.3f}")
                md_lines.append(f"- F1: {metrics.get('f1', 0):.3f}")
                if "auc" in metrics:
                    md_lines.append(f"- AUC: {metrics['auc']:.3f}")

        md_path = self.validation_dir / "validation_report.md"
        with open(md_path, "w") as f:
            f.write("\n".join(md_lines))

        return report_path, md_path


class ArtifactManager:
    """Manage training artifacts and model versioning"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.artifacts_dir = DATA_ROOT / "artifacts" / symbol
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def save_artifact(
        self,
        source_path: pathlib.Path,
        artifact_type: str,
        version: Optional[str] = None,
    ) -> pathlib.Path:
        """Save and version an artifact"""
        if not source_path.exists():
            raise FileNotFoundError(f"Source not found: {source_path}")

        # Generate version if not provided
        if version is None:
            version = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        # Create artifact directory
        artifact_dir = self.artifacts_dir / artifact_type / version
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Copy artifact
        import shutil

        dest_path = artifact_dir / source_path.name
        shutil.copy2(source_path, dest_path)

        # Save metadata
        metadata = {
            "symbol": self.symbol,
            "type": artifact_type,
            "version": version,
            "source": str(source_path),
            "timestamp": pd.Timestamp.now().isoformat(),
            "file_size_kb": dest_path.stat().st_size / 1024,
        }

        meta_path = artifact_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return dest_path

    def list_artifacts(self, artifact_type: Optional[str] = None) -> List[Dict]:
        """List all artifacts"""
        artifacts = []

        if artifact_type:
            type_dirs = [self.artifacts_dir / artifact_type]
        else:
            type_dirs = [d for d in self.artifacts_dir.iterdir() if d.is_dir()]

        for type_dir in type_dirs:
            if not type_dir.exists():
                continue

            for version_dir in type_dir.iterdir():
                if not version_dir.is_dir():
                    continue

                meta_path = version_dir / "metadata.json"
                if meta_path.exists():
                    with open(meta_path, "r") as f:
                        artifacts.append(json.load(f))

        # Sort by timestamp
        artifacts.sort(key=lambda x: x["timestamp"], reverse=True)

        return artifacts

    def get_latest_artifact(self, artifact_type: str) -> Optional[pathlib.Path]:
        """Get the latest artifact of a type"""
        artifacts = self.list_artifacts(artifact_type)

        if not artifacts:
            return None

        latest = artifacts[0]
        version = latest["version"]

        artifact_dir = self.artifacts_dir / artifact_type / version

        # Find the actual artifact file (not metadata)
        for file in artifact_dir.iterdir():
            if file.name != "metadata.json":
                return file

        return None

    def cleanup_old_artifacts(self, artifact_type: str, keep_n: int = 5):
        """Remove old artifacts, keeping the most recent N"""
        artifacts = self.list_artifacts(artifact_type)

        if len(artifacts) <= keep_n:
            return

        # Remove old artifacts
        to_remove = artifacts[keep_n:]

        for artifact in to_remove:
            version = artifact["version"]
            artifact_dir = self.artifacts_dir / artifact_type / version

            if artifact_dir.exists():
                import shutil

                shutil.rmtree(artifact_dir)
                print(f"Removed old artifact: {artifact_type}/{version}")


def validate_training_pipeline(symbol: str):
    """Validate complete training pipeline"""
    print(f"\n{'='*60}")
    print(f"Validating Training Pipeline - {symbol}")
    print(f"{'='*60}")

    validator = ModelValidator(symbol)
    artifact_mgr = ArtifactManager(symbol)

    results = {"symbol": symbol, "checks": {}}

    # Check ONNX models
    print("\nValidating ONNX models...")
    for model in ["lstm", "xgb", "cnn"]:
        onnx_path = DATA_ROOT / "models" / symbol / f"{model}.onnx"
        validation = validator.validate_onnx_model(onnx_path)
        results["checks"][f"{model}_onnx"] = validation

        if validation.get("status") == "valid":
            print(
                f"  [OK] {model}: {validation['file_size_kb']:.1f} KB, {validation['inference_ms']:.2f} ms"
            )
        else:
            print(f"  [FAIL] {model}: {validation.get('error', 'Not found')}")

    # Check scored NPZ
    print("\nValidating predictions...")
    scored_npz = DATA_ROOT / "features_cache" / symbol / "train_m15_scored.npz"
    if scored_npz.exists():
        pred_validation = validator.validate_predictions(scored_npz)
        results["predictions"] = pred_validation

        for model, metrics in pred_validation.get("models", {}).items():
            print(
                f"  {model}: AUC={metrics.get('auc', 0):.3f}, Acc={metrics['accuracy']:.3f}"
            )

    # Check calibration
    print("\nChecking calibration artifacts...")
    cal_dir = DATA_ROOT / "calibration" / symbol
    if cal_dir.exists():
        cal_files = list(cal_dir.glob("*.pkl"))
        results["calibration"] = {
            "n_calibrators": len([f for f in cal_files if "calibrator" in f.name]),
            "has_fusion": (cal_dir / "fusion_model.pkl").exists(),
        }
        print(f"  Calibrators: {results['calibration']['n_calibrators']}")
        print(
            f"  Fusion model: {'Yes' if results['calibration']['has_fusion'] else 'No'}"
        )

    # Generate report
    report_path, md_path = validator.generate_validation_report(results)
    print(f"\nReports saved:")
    print(f"  JSON: {report_path}")
    print(f"  Markdown: {md_path}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validation utilities")
    parser.add_argument("--symbol", required=True, help="Symbol")
    parser.add_argument("--validate", action="store_true", help="Run validation")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup old artifacts")

    args = parser.parse_args()

    if args.validate:
        validate_training_pipeline(args.symbol)

    if args.cleanup:
        mgr = ArtifactManager(args.symbol)
        for artifact_type in ["models", "calibration", "reports"]:
            mgr.cleanup_old_artifacts(artifact_type, keep_n=3)
            print(f"Cleaned up {artifact_type} artifacts")


if __name__ == "__main__":
    main()
