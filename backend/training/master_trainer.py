"""
Master Training Runner - Complete end-to-end training pipeline
Orchestrates data prep, model training, inference, calibration, and deployment
"""

import os
import pathlib
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional
import shutil

PROJECT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.getenv("ARIA_DATA_ROOT", PROJECT / "data"))
TRAINING_DIR = pathlib.Path(__file__).parent


class MasterTrainer:
    """Master orchestrator for complete training pipeline"""

    def __init__(self, symbol: str, config_path: Optional[pathlib.Path] = None):
        self.symbol = symbol
        self.start_time = time.time()

        # Load or create config
        if config_path and config_path.exists():
            with open(config_path, "r") as f:
                self.config = json.load(f)
        else:
            self.config = self._default_config()

        # Setup paths
        self.setup_directories()

        # Track progress
        self.results = {
            "symbol": symbol,
            "start_time": datetime.now().isoformat(),
            "config": self.config,
            "steps": {},
        }

    def _default_config(self) -> Dict:
        """Default training configuration"""
        return {
            "data_prep": {
                "start_date": "2019-01-01",
                "end_date": "2024-01-01",
                "horizon": 8,
                "features": [
                    "r",
                    "abs_r",
                    "ewma_sig",
                    "atr14",
                    "rsi",
                    "bb_pos",
                    "mom5",
                    "mom10",
                    "mom20",
                ],
            },
            "models": {
                "lstm": {
                    "enabled": True,
                    "epochs": 20,
                    "batch_size": 64,
                    "learning_rate": 0.001,
                    "seq_len": 16,
                    "hidden_size": 32,
                },
                "xgb": {
                    "enabled": True,
                    "n_estimators": 100,
                    "max_depth": 4,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                },
                "cnn": {
                    "enabled": True,
                    "epochs": 15,
                    "batch_size": 64,
                    "learning_rate": 0.001,
                    "seq_len": 32,
                    "model_type": "dilated",
                },
            },
            "walk_forward": {"enabled": True, "train_months": 4, "test_months": 2},
            "calibration": {"enabled": True, "method": "isotonic"},
            "deployment": {"copy_to_backend": True, "generate_metadata": True},
        }

    def setup_directories(self):
        """Create necessary directories"""
        self.dirs = {
            "data": DATA_ROOT / "parquet" / self.symbol,
            "features": DATA_ROOT / "features_cache" / self.symbol,
            "models": DATA_ROOT / "models" / self.symbol,
            "calibration": DATA_ROOT / "calibration" / self.symbol,
            "walk_forward": DATA_ROOT / "walk_forward" / self.symbol,
            "logs": DATA_ROOT / "training_logs" / self.symbol,
        }

        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)

    def run_command(self, cmd: List[str], step_name: str) -> bool:
        """Run a subprocess command and track results"""
        print(f"\n{'='*60}")
        print(f"Running: {step_name}")
        print(f"{'='*60}")

        log_file = self.dirs["logs"] / f"{step_name}_{datetime.now():%Y%m%d_%H%M%S}.log"

        start = time.time()

        with open(log_file, "w") as log:
            result = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT, text=True
            )

        elapsed = time.time() - start

        self.results["steps"][step_name] = {
            "command": " ".join(cmd),
            "return_code": result.returncode,
            "elapsed_time": elapsed,
            "log_file": str(log_file),
        }

        if result.returncode == 0:
            print(f"[OK] {step_name} completed in {elapsed:.1f}s")
            return True
        else:
            print(f"[FAIL] {step_name} failed (see {log_file})")
            return False

    def step_data_preparation(self) -> Optional[pathlib.Path]:
        """Step 1: Prepare training data"""
        config = self.config["data_prep"]

        output_npz = self.dirs["features"] / "train_m15.npz"

        # Skip if exists
        if output_npz.exists():
            print(f"Using existing NPZ: {output_npz}")
            return output_npz

        # Find M1 parquet file
        m1_file = DATA_ROOT / "parquet" / self.symbol / f"{self.symbol}_m1.parquet"
        if not m1_file.exists():
            print(f"M1 data not found at {m1_file}")
            print("Attempting to download real M1 data via MetaTrader 5...")
            dl_cmd = [
                sys.executable,
                str(TRAINING_DIR / "download_m1_mt5.py"),
                "--symbol",
                self.symbol,
                "--start",
                self.config["data_prep"]["start_date"],
                "--end",
                self.config["data_prep"]["end_date"],
                "--out",
                str(m1_file),
            ]
            if not self.run_command(dl_cmd, "download_m1"):
                print(
                    "Automatic M1 download failed. Please provide a real M1 parquet at the expected path and retry."
                )
                return None
            if not m1_file.exists():
                print("M1 file still missing after download attempt.")
                return None

        cmd = [
            sys.executable,
            str(TRAINING_DIR / "prepare_m15_npz.py"),
            "--symbol",
            self.symbol,
            "--m1",
            str(m1_file),
            "--horizon",
            str(config["horizon"]),
            "--out",
            str(output_npz),
        ]

        if self.run_command(cmd, "data_preparation"):
            return output_npz
        return None

    def step_train_models(self, npz_path: pathlib.Path) -> Dict:
        """Step 2: Train individual models"""
        results = {}

        for model_name, model_config in self.config["models"].items():
            if not model_config.get("enabled", True):
                continue

            if model_name == "lstm":
                cmd = [
                    sys.executable,
                    str(TRAINING_DIR / "train_lstm.py"),
                    "--symbol",
                    self.symbol,
                    "--npz",
                    str(npz_path),
                    "--epochs",
                    str(model_config["epochs"]),
                    "--batch_size",
                    str(model_config["batch_size"]),
                    "--lr",
                    str(model_config["learning_rate"]),
                ]
            elif model_name == "xgb":
                cmd = [
                    sys.executable,
                    str(TRAINING_DIR / "train_xgb.py"),
                    "--symbol",
                    self.symbol,
                    "--npz",
                    str(npz_path),
                    "--rounds",
                    str(model_config["n_estimators"]),
                    "--depth",
                    str(model_config["max_depth"]),
                    "--lr",
                    str(model_config["learning_rate"]),
                ]
            elif model_name == "cnn":
                cmd = [
                    sys.executable,
                    str(TRAINING_DIR / "train_cnn1d.py"),
                    "--symbol",
                    self.symbol,
                    "--npz",
                    str(npz_path),
                    "--epochs",
                    str(model_config["epochs"]),
                    "--batch_size",
                    str(model_config["batch_size"]),
                    "--lr",
                    str(model_config["learning_rate"]),
                    "--model",
                    model_config["model_type"],
                ]
            else:
                continue

            success = self.run_command(cmd, f"train_{model_name}")
            results[model_name] = success

            # Check ONNX output
            onnx_path = self.dirs["models"] / f"{model_name}.onnx"
            if onnx_path.exists():
                results[f"{model_name}_onnx_size"] = onnx_path.stat().st_size / 1024

        return results

    def step_inference(self, npz_path: pathlib.Path) -> Optional[pathlib.Path]:
        """Step 3: Generate model scores"""
        models = [m for m, c in self.config["models"].items() if c.get("enabled", True)]

        cmd = [
            sys.executable,
            str(TRAINING_DIR / "inference_to_npz.py"),
            "--symbol",
            self.symbol,
            "--npz",
            str(npz_path),
            "--models",
        ] + models

        if self.run_command(cmd, "inference"):
            scored_npz = npz_path.parent / f"{npz_path.stem}_scored.npz"
            if scored_npz.exists():
                return scored_npz

        return None

    def step_walk_forward(self) -> bool:
        """Step 4: Run walk-forward validation"""
        if not self.config["walk_forward"]["enabled"]:
            print("Walk-forward validation disabled")
            return True

        wf_config = self.config["walk_forward"]

        cmd = [
            sys.executable,
            str(TRAINING_DIR / "walk_forward_orchestrator.py"),
            "--symbol",
            self.symbol,
            "--start",
            self.config["data_prep"]["start_date"],
            "--end",
            self.config["data_prep"]["end_date"],
            "--train_months",
            str(wf_config["train_months"]),
            "--test_months",
            str(wf_config["test_months"]),
        ]

        return self.run_command(cmd, "walk_forward")

    def step_calibration(self, scored_npz: pathlib.Path) -> bool:
        """Step 5: Calibrate and fuse models"""
        if not self.config["calibration"]["enabled"]:
            print("Calibration disabled")
            return True

        models = [m for m, c in self.config["models"].items() if c.get("enabled", True)]

        cmd = (
            [
                sys.executable,
                str(TRAINING_DIR / "calibrate_and_fuse.py"),
                "--symbol",
                self.symbol,
                "--npz",
                str(scored_npz),
                "--models",
            ]
            + models
            + ["--method", self.config["calibration"]["method"]]
        )

        return self.run_command(cmd, "calibration")

    def step_deployment(self) -> bool:
        """Step 6: Deploy models to backend"""
        if not self.config["deployment"]["copy_to_backend"]:
            print("Deployment disabled")
            return True

        print(f"\n{'='*60}")
        print(f"Deploying Models")
        print(f"{'='*60}")

        # Target directory
        backend_models = PROJECT / "backend" / "models"
        backend_models.mkdir(parents=True, exist_ok=True)

        deployed = []

        # Copy ONNX models
        for model_name in ["lstm", "xgb", "cnn"]:
            if not self.config["models"].get(model_name, {}).get("enabled", True):
                continue

            src_onnx = self.dirs["models"] / f"{model_name}.onnx"
            if src_onnx.exists():
                dst_onnx = backend_models / f"{model_name}_{self.symbol.lower()}.onnx"
                shutil.copy2(src_onnx, dst_onnx)
                deployed.append(str(dst_onnx))
                print(f"  Deployed {model_name} -> {dst_onnx.name}")

        # Copy calibrators
        cal_dir = self.dirs["calibration"]
        if cal_dir.exists():
            for cal_file in cal_dir.glob("*.pkl"):
                dst_cal = backend_models / f"{cal_file.stem}_{self.symbol.lower()}.pkl"
                shutil.copy2(cal_file, dst_cal)
                deployed.append(str(dst_cal))
                print(f"  Deployed calibrator -> {dst_cal.name}")

        # Generate deployment metadata
        if self.config["deployment"]["generate_metadata"]:
            metadata = {
                "symbol": self.symbol,
                "models": list(self.config["models"].keys()),
                "deployed_files": deployed,
                "training_config": self.config,
                "deployment_time": datetime.now().isoformat(),
            }

            meta_path = backend_models / f"deployment_{self.symbol.lower()}.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"  Saved metadata -> {meta_path.name}")

        self.results["steps"]["deployment"] = {
            "deployed_files": deployed,
            "target_dir": str(backend_models),
        }

        return True

    def generate_report(self):
        """Generate final training report"""
        self.results["end_time"] = datetime.now().isoformat()
        self.results["total_time"] = time.time() - self.start_time

        # Save report
        report_path = (
            self.dirs["logs"] / f"training_report_{datetime.now():%Y%m%d_%H%M%S}.json"
        )
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Training Complete")
        print(f"{'='*60}")
        print(f"Symbol: {self.symbol}")
        print(f"Total time: {self.results['total_time']:.1f}s")
        print(f"Report saved: {report_path}")

        # Print summary
        print(f"\nSteps Summary:")
        for step, info in self.results["steps"].items():
            status = "[OK]" if info.get("return_code", 1) == 0 else "[FAIL]"
            elapsed = info.get("elapsed_time", 0)
            print(f"  {status} {step}: {elapsed:.1f}s")

    def run_pipeline(self):
        """Run complete training pipeline"""
        print(f"\n{'='*60}")
        print(f"ARIA Training Pipeline - {self.symbol}")
        print(f"{'='*60}")

        # Step 1: Data preparation
        npz_path = self.step_data_preparation()
        if npz_path is None:
            print("Pipeline failed at data preparation")
            return False

        # Step 2: Train models
        train_results = self.step_train_models(npz_path)

        # Step 3: Inference
        scored_npz = self.step_inference(npz_path)
        if scored_npz is None:
            print("Pipeline failed at inference")
            return False

        # Step 4: Walk-forward validation (optional)
        self.step_walk_forward()

        # Step 5: Calibration
        self.step_calibration(scored_npz)

        # Step 6: Deployment
        self.step_deployment()

        # Generate report
        self.generate_report()

        return True


def main():
    parser = argparse.ArgumentParser(description="Master Training Runner")
    parser.add_argument("--symbol", default="XAUUSD", help="Symbol to train")
    parser.add_argument("--config", help="Configuration JSON file")
    parser.add_argument(
        "--skip-walk-forward", action="store_true", help="Skip walk-forward validation"
    )
    parser.add_argument(
        "--skip-deployment", action="store_true", help="Skip deployment step"
    )

    args = parser.parse_args()

    # Load config if provided
    config_path = pathlib.Path(args.config) if args.config else None

    # Create trainer
    trainer = MasterTrainer(args.symbol, config_path)

    # Override config if needed
    if args.skip_walk_forward:
        trainer.config["walk_forward"]["enabled"] = False
    if args.skip_deployment:
        trainer.config["deployment"]["copy_to_backend"] = False

    # Run pipeline
    trainer.run_pipeline()


if __name__ == "__main__":
    main()
