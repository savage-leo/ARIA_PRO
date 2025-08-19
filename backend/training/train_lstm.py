"""
LSTM Trainer - CPU-optimized for T470
Produces calibrated P(Long)/P(Short) with compact ONNX export
"""

import os
import pathlib
import argparse
import json
import numpy as np
import random
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, log_loss

PROJECT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.getenv("ARIA_DATA_ROOT", PROJECT / "data"))
DEVICE = torch.device("cpu")

# CPU optimization for reproducibility
torch.set_num_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class SeqDataset(Dataset):
    """Sequence dataset for LSTM training"""

    def __init__(self, arrs: Dict, seq_len: int = 16, feature_cols: list = None):
        self.seq_len = seq_len

        # Default features optimized for forex
        if feature_cols is None:
            feature_cols = ["r", "abs_r", "ewma_sig", "atr14", "rsi", "bb_pos", "mom5"]

        # Stack features
        features = []
        for col in feature_cols:
            if col in arrs:
                features.append(arrs[col])

        self.X = np.stack(features, axis=1).astype(np.float32)
        self.y = arrs["label"].astype(np.int64)

        # Normalize features for stability
        self.mean = self.X.mean(axis=0, keepdims=True)
        self.std = self.X.std(axis=0, keepdims=True) + 1e-8
        self.X = (self.X - self.mean) / self.std

    def __len__(self):
        return max(0, len(self.y) - self.seq_len)

    def __getitem__(self, idx):
        x = self.X[idx : idx + self.seq_len]
        y = self.y[idx + self.seq_len]
        return x, y


class TinyLSTM(nn.Module):
    """Compact LSTM for CPU inference"""

    def __init__(
        self, in_dim: int = 7, hidden: int = 32, layers: int = 1, dropout: float = 0.1
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden

        # LSTM with dropout for regularization
        self.lstm = nn.LSTM(
            in_dim,
            hidden,
            layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0,
        )

        # Classification head with residual connection
        self.fc = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),  # 2 outputs for binary classification
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    def forward(self, x):
        # LSTM forward
        out, (h_n, c_n) = self.lstm(x)

        # Use last hidden state
        h = out[:, -1, :]

        # Classification
        logits = self.fc(h)
        return logits


def set_seed(seed: int = 0):
    """Set all random seeds for reproducibility"""
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

    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * X.size(0)
        n_samples += X.size(0)

    return total_loss / n_samples


def validate(model, dataloader, criterion, device):
    """Validate model and compute metrics"""
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

            # Get probabilities
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * X.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # P(Long)
            all_labels.extend(y.cpu().numpy())

    # Compute metrics
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    accuracy = (all_preds == all_labels).mean()

    # Handle edge cases for AUC
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.5

    avg_loss = total_loss / len(all_labels)

    return avg_loss, accuracy, auc, all_probs


def train(symbol: str, npz_path: pathlib.Path, config: Dict):
    """Main training function"""
    set_seed(config.get("seed", 0))

    # Load data
    print(f"Loading data from {npz_path}")
    arr = np.load(npz_path, allow_pickle=False)

    # Create dataset
    feature_cols = config.get(
        "features", ["r", "abs_r", "ewma_sig", "atr14", "rsi", "bb_pos", "mom5"]
    )
    ds = SeqDataset(arr, seq_len=config["seq_len"], feature_cols=feature_cols)

    # Train/val split
    n_val = int(len(ds) * config.get("val_split", 0.2))
    n_train = len(ds) - n_val

    if n_train <= 0 or n_val <= 0:
        print(f"Insufficient data: train={n_train}, val={n_val}")
        return None

    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    # Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,  # Single thread for CPU
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0
    )

    # Model
    model = TinyLSTM(
        in_dim=len(feature_cols),
        hidden=config["hidden_size"],
        layers=config["n_layers"],
        dropout=config.get("dropout", 0.1),
    ).to(DEVICE)

    # Optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 1e-4),
    )

    # Class-weighted loss for imbalanced data
    class_weights = compute_class_weights(arr["label"])
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float("inf")
    best_auc = 0.0
    best_acc = 0.0
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
        val_loss, val_acc, val_auc, val_probs = validate(
            model, val_loader, criterion, DEVICE
        )

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
            best_val_loss = val_loss
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
            torch.save(checkpoint, ckpt_dir / "lstm_best.pt")
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
        torch.load(ckpt_dir / "lstm_best.pt", map_location=DEVICE, weights_only=False)[
            "model_state"
        ]
    )
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, config["seq_len"], len(feature_cols), device=DEVICE)

    # Export with optimizations
    onnx_path = ckpt_dir / "lstm.onnx"
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

    # Save training metadata
    metadata = {
        "symbol": symbol,
        "best_auc": float(best_auc),
        "best_val_loss": float(best_val_loss),
        "config": config,
        "feature_cols": feature_cols,
        "metrics_history": metrics_history,
        "onnx_size_kb": onnx_path.stat().st_size / 1024,
    }

    with open(ckpt_dir / "lstm_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train LSTM model")
    parser.add_argument("--symbol", required=True, help="Symbol to train")
    parser.add_argument("--npz", default=None, help="Path to NPZ file")
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=16, help="Sequence length")
    parser.add_argument("--hidden", type=int, default=32, help="Hidden size")
    parser.add_argument("--layers", type=int, default=1, help="Number of LSTM layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
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
        "hidden_size": args.hidden,
        "n_layers": args.layers,
        "learning_rate": args.lr,
        "weight_decay": 1e-4,
        "dropout": 0.1,
        "val_split": 0.2,
        "early_stop_patience": 5,
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
