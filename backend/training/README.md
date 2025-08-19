# ARIA Institutional Forex AI Training Playbook

## Overview
Complete CPU-optimized training pipeline for institutional-grade Forex AI models (LSTM, CNN, XGBoost) with calibration, fusion, and walk-forward validation. Designed for efficient local training on Lenovo T470 hardware.

## Quick Start

### 1. Single Command Training
```bash
# Train all models for EURUSD
python master_trainer.py --symbol EURUSD

# Skip walk-forward validation for faster training
python master_trainer.py --symbol EURUSD --skip-walk-forward

# Use custom configuration
python master_trainer.py --symbol EURUSD --config custom_config.json
```

### 2. Individual Component Training
```bash
# Prepare data
python prepare_m15_npz.py --symbol EURUSD --start 2019-01-01 --end 2024-01-01

# Train individual models
python train_lstm.py --symbol EURUSD --npz data/features_cache/EURUSD/train_m15.npz
python train_xgb.py --symbol EURUSD --npz data/features_cache/EURUSD/train_m15.npz
python train_cnn1d.py --symbol EURUSD --npz data/features_cache/EURUSD/train_m15.npz

# Generate predictions
python inference_to_npz.py --symbol EURUSD --npz data/features_cache/EURUSD/train_m15.npz

# Calibrate and fuse
python calibrate_and_fuse.py --symbol EURUSD --npz data/features_cache/EURUSD/train_m15_scored.npz
```

### 3. Walk-Forward Validation
```bash
python walk_forward_orchestrator.py --symbol EURUSD --start 2019-01-01 --end 2024-01-01
```

## Pipeline Components

### Data Preparation (`prepare_m15_npz.py`)
- Aggregates M1 data to M15 bars
- Computes technical features (RSI, ATR, Bollinger Bands, momentum)
- Generates directional labels with configurable horizon
- Detects market regimes and trading sessions
- Exports compressed NPZ with metadata

### Model Training

#### LSTM (`train_lstm.py`)
- PyTorch CPU-optimized LSTM for sequence modeling
- Balanced class weights for imbalanced data
- Early stopping with learning rate scheduling
- ONNX export with opset 11
- Inference target: <50ms per batch

#### XGBoost (`train_xgb.py`)
- Gradient boosting with time series cross-validation
- Feature engineering with session/volatility dummies
- Sample weighting for class balance
- ONNX export via skl2onnx
- Compact model size (<500KB typical)

#### CNN 1D (`train_cnn1d.py`)
- 1D convolutions for time series patterns
- Standard and dilated CNN architectures
- Batch normalization and dropout
- CPU-optimized inference
- ONNX export with dynamic axes

### Inference Pipeline (`inference_to_npz.py`)
- Batch inference for all models
- CPU-optimized ONNX runtime
- Adds model scores to NPZ files
- Supports partial model sets
- Memory-efficient processing

### Walk-Forward Validation (`walk_forward_orchestrator.py`)
- Rolling window train/test splits
- Automated model retraining
- PSR and AUC metrics computation
- Optimal gating weight calculation
- Results aggregation and reporting

### Calibration & Fusion (`calibrate_and_fuse.py`)
- Platt scaling and isotonic regression
- Expected Calibration Error (ECE) metrics
- Logistic regression fusion head
- Cross-validated performance
- Calibrator persistence

### Master Orchestrator (`master_trainer.py`)
- End-to-end pipeline automation
- Configurable via JSON
- Progress tracking and logging
- Automatic deployment to backend
- Comprehensive reporting

### Validation Utilities (`validation_utils.py`)
- ONNX model validation
- Inference speed benchmarking
- Performance metrics computation
- Calibration curve plotting
- Artifact versioning

## Configuration

### Default Configuration Structure
```json
{
  "data_prep": {
    "start_date": "2019-01-01",
    "end_date": "2024-01-01",
    "horizon": 8,
    "features": ["r", "abs_r", "ewma_sig", "atr14", "rsi", "bb_pos", "mom5"]
  },
  "models": {
    "lstm": {
      "enabled": true,
      "epochs": 20,
      "batch_size": 64,
      "learning_rate": 0.001,
      "seq_len": 16,
      "hidden_size": 32
    },
    "xgb": {
      "enabled": true,
      "n_estimators": 100,
      "max_depth": 4,
      "learning_rate": 0.05
    },
    "cnn": {
      "enabled": true,
      "epochs": 15,
      "batch_size": 64,
      "model_type": "dilated"
    }
  },
  "walk_forward": {
    "enabled": true,
    "train_months": 4,
    "test_months": 2
  },
  "calibration": {
    "enabled": true,
    "method": "isotonic"
  }
}
```

## Directory Structure
```
data/
├── parquet/           # Raw M1 data
├── features_cache/    # Prepared NPZ files
├── models/           # Trained models and ONNX exports
├── calibration/      # Calibration models
├── walk_forward/     # Walk-forward results
├── validation/       # Validation reports
├── artifacts/        # Versioned artifacts
└── training_logs/    # Training logs
```

## Performance Targets

### T470 Hardware Constraints
- CPU: Intel i5-7300U (2 cores/4 threads)
- RAM: 8GB (16GB recommended)
- Storage: SSD recommended for data I/O

### Model Size Targets
- LSTM: <1MB ONNX
- XGBoost: <500KB ONNX
- CNN: <2MB ONNX
- Total deployment: <5MB

### Inference Latency
- Single model: <50ms
- All models: <200ms
- Calibration overhead: <10ms
- Fusion: <5ms

## Best Practices

### Data Preparation
1. Ensure sufficient data (>1 year recommended)
2. Handle session boundaries properly
3. Remove weekends and holidays
4. Normalize features appropriately
5. Balance classes via weighting

### Training
1. Set deterministic seeds for reproducibility
2. Use gradient accumulation for larger effective batches
3. Monitor validation loss for early stopping
4. Save checkpoints regularly
5. Export to ONNX immediately after training

### Validation
1. Use walk-forward for time series
2. Compute both AUC and PSR metrics
3. Check calibration quality (ECE)
4. Validate ONNX inference speed
5. Test on out-of-sample data

### Deployment
1. Version all artifacts
2. Keep calibrators with models
3. Document gating weights
4. Monitor inference latency
5. Implement fallback mechanisms

## Troubleshooting

### Common Issues

#### Out of Memory
- Reduce batch size
- Use gradient accumulation
- Process data in chunks
- Clear cache between models

#### Slow Training
- Ensure CPU thread settings (1 thread)
- Use smaller models
- Reduce sequence lengths
- Enable early stopping

#### Poor Performance
- Check class balance
- Increase training data
- Tune hyperparameters
- Verify feature quality
- Adjust label horizon

#### ONNX Export Failures
- Check PyTorch/ONNX versions
- Simplify model architecture
- Use supported operations only
- Set proper opset version

## Dependencies
```txt
numpy==1.26.4
pandas==2.2.0
scikit-learn==1.4.0
xgboost==2.0.3
torch==2.2.0
onnx==1.15.0
onnxruntime==1.18.1
skl2onnx==1.16.0
joblib==1.3.2
matplotlib==3.8.2
seaborn==0.13.2
```

## Example Training Session
```bash
# 1. Prepare environment
export ARIA_DATA_ROOT=/path/to/data
export PYTHONHASHSEED=0

# 2. Run complete pipeline
python master_trainer.py --symbol EURUSD

# 3. Validate results
python validation_utils.py --symbol EURUSD --validate

# 4. Check deployment
ls ../models/
# lstm_eurusd.onnx
# xgb_eurusd.onnx
# cnn_eurusd.onnx
# lstm_calibrator_eurusd.pkl
# fusion_model_eurusd.pkl
```

## Production Checklist
- [ ] Data quality verified
- [ ] Models trained successfully
- [ ] ONNX exports validated
- [ ] Inference speed acceptable (<200ms)
- [ ] Calibration applied
- [ ] Walk-forward metrics acceptable
- [ ] Gating weights computed
- [ ] Models deployed to backend
- [ ] Integration tests passed
- [ ] Monitoring enabled

## Support
For issues or questions about the training pipeline, check:
1. Training logs in `data/training_logs/`
2. Validation reports in `data/validation/`
3. Model metadata in `data/models/[symbol]/`
4. Walk-forward results in `data/walk_forward/`
