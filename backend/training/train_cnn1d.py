"""
CNN 1D Trainer - Compact convolutional model for time series
CPU-optimized with minimal memory footprint
"""

import os
import pathlib
import argparse
import json
import numpy as np
import random
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, log_loss

PROJECT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.getenv("ARIA_DATA_ROOT", PROJECT / "data"))
DEVICE = torch.device("cpu")

# CPU optimization
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True


class CNN1DDataset(Dataset):
    """1D CNN dataset for time series classification"""

    def __init__(self, arr: Dict, seq_len: int = 32, feature_cols: List[str] = None):
        self.seq_len = seq_len

        # Default features for 1D convolution
        if feature_cols is None:
            feature_cols = ["r", "abs_r", "ewma_sig", "atr14", "rsi", "bb_pos"]

        # Stack features
        features = []
        for col in feature_cols:
            if col in arr:
                features.append(arr[col])

        # Shape: (n_samples, n_features)
        self.X = np.stack(features, axis=1).astype(np.float32)
        self.y = arr["label"].astype(np.int64)

        # Normalize
        self.mean = self.X.mean(axis=0, keepdims=True)
        self.std = self.X.std(axis=0, keepdims=True) + 1e-8
        self.X = (self.X - self.mean) / self.std

        self.n_features = self.X.shape[1]

    def __len__(self):
        return max(0, len(self.y) - self.seq_len)

    def __getitem__(self, idx):
        # Get sequence window
        x = self.X[idx : idx + self.seq_len]  # (seq_len, n_features)
        # Transpose for Conv1d: (n_features, seq_len)
        x = x.T
        y = self.y[idx + self.seq_len]
        return x, y


class TinyCNN1D(nn.Module):
    """Compact 1D CNN for CPU inference"""

    def __init__(self, in_channels: int = 6, seq_len: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len

        # Convolutional layers with increasing dilation
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 2)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Convolutional feature extraction
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # Flatten
        x = x.squeeze(-1)

        # Classification
        logits = self.classifier(x)
        return logits


class DilatedCNN1D(nn.Module):
    """Alternative: Dilated CNN for larger receptive field"""

    def __init__(self, in_channels: int = 6):
        super().__init__()
        self.in_channels = in_channels

        # Dilated convolutions
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, padding=4, dilation=4)
        self.conv4 = nn.Conv1d(32, 32, kernel_size=3, padding=8, dilation=8)

        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(32)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 2)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)


def set_seed(seed: int = 0):
    """Set all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Compute balanced class weights"""
    unique, counts = np.unique(y, return_counts=True)
    weights = len(y) / (len(unique) * counts)
    weight_dict = dict(zip(unique, weights))
    return torch.tensor(
        [weight_dict.get(i, 1.0) for i in range(2)], dtype=torch.float32
    )


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    n_samples = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        n_samples += X.size(0)

    return total_loss / n_samples


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            loss = criterion(logits, y)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * X.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    accuracy = (all_preds == all_labels).mean()

    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.5

    avg_loss = total_loss / len(all_labels)

    return avg_loss, accuracy, auc


def train(symbol: str, npz_path: pathlib.Path, config: Dict):
    """Main training function"""
    set_seed(config.get("seed", 0))

    # Load data
    print(f"Loading data from {npz_path}")
    arr = np.load(npz_path, allow_pickle=False)

    # Create dataset
    feature_cols = config.get(
        "features", ["r", "abs_r", "ewma_sig", "atr14", "rsi", "bb_pos"]
    )
    ds = CNN1DDataset(arr, seq_len=config["seq_len"], feature_cols=feature_cols)

    # Train/val split
    n_val = int(len(ds) * config.get("val_split", 0.2))
    n_train = len(ds) - n_val

    if n_train <= 0 or n_val <= 0:
        print(f"Insufficient data: train={n_train}, val={n_val}")
        return None

    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    # Dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )

    # Model selection
    model_type = config.get("model_type", "standard")
    if model_type == "dilated":
        model = DilatedCNN1D(in_channels=len(feature_cols)).to(DEVICE)
    else:
        model = TinyCNN1D(in_channels=len(feature_cols), seq_len=config["seq_len"]).to(
            DEVICE
        )

    # Optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 1e-4),
    )

    # Class-weighted loss
    class_weights = compute_class_weights(arr["label"])
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Training loop
    best_auc = 0.0
    patience_counter = 0
    max_patience = config.get("early_stop_patience", 5)

    ckpt_dir = DATA_ROOT / "models" / symbol
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    metrics_history = []

    for epoch in range(config["epochs"]):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)

        # Validate
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, DEVICE)

        # Update scheduler
        scheduler.step(val_loss)

        epoch_time = time.time() - start_time

        # Log metrics
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_auc": val_auc,
            "epoch_time": epoch_time,
        }
        metrics_history.append(metrics)

        print(
            f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f} | Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0

            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config,
                "metrics": metrics,
                "feature_cols": feature_cols,
                "normalization": {"mean": ds.mean.tolist(), "std": ds.std.tolist()},
            }
            torch.save(checkpoint, ckpt_dir / "cnn_best.pt")
            print(f"  -> Saved best model (AUC: {val_auc:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Export ONNX
    print("\nExporting to ONNX...")
    model.load_state_dict(
        torch.load(ckpt_dir / "cnn_best.pt", map_location=DEVICE)["model_state"]
    )
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, len(feature_cols), config["seq_len"], device=DEVICE)

    # Export
    onnx_path = ckpt_dir / "cnn.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
    )

    print(f"Exported ONNX to {onnx_path}")

    # Save metadata
    metadata = {
        "symbol": symbol,
        "best_auc": float(best_auc),
        "config": config,
        "feature_cols": feature_cols,
        "metrics_history": metrics_history,
        "model_type": model_type,
        "onnx_size_kb": onnx_path.stat().st_size / 1024,
    }

    with open(ckpt_dir / "cnn_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train CNN 1D model")
    parser.add_argument("--symbol", required=True, help="Symbol to train")
    parser.add_argument("--npz", default=None, help="Path to NPZ file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=32, help="Sequence length")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--model",
        choices=["standard", "dilated"],
        default="standard",
        help="Model type",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    args = parser.parse_args()

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
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "learning_rate": args.lr,
        "weight_decay": 1e-4,
        "dropout": 0.2,
        "val_split": 0.2,
        "early_stop_patience": 5,
        "model_type": args.model,
        "seed": args.seed,
    }

    # Train
    metadata = train(args.symbol, npz_path, config)

    if metadata:
        print(f"\nTraining complete!")
        print(f"Best AUC: {metadata['best_auc']:.4f}")
        print(f"ONNX size: {metadata['onnx_size_kb']:.1f} KB")


if __name__ == "__main__":
    main()
