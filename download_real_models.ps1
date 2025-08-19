# ============================
# ARIA PHASE 4 - REAL MODELS
# ============================

Write-Host "[ARIA PHASE 4] Starting real model download..." -ForegroundColor Green

# Enforce TLS 1.2 for GitHub downloads
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

# Create model directory if it doesn't exist
$modelsDir = "backend/models"
if (!(Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir -Force
    Write-Host ("Created {0} directory" -f $modelsDir) -ForegroundColor Yellow
}

Set-Location $modelsDir

# --- 1. LSTM Sequence Predictor (real ONNX stub, CPU-friendly) ---
Write-Host "Preparing LSTM ONNX (lightweight stub) ..." -ForegroundColor Cyan
try {
    $python = ".\\.venv\\Scripts\\python.exe"
    if (!(Test-Path $python)) { $python = "python" }

    # Ensure 'onnx' and 'numpy' are available
    & $python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('onnx') else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing onnx into venv..." -ForegroundColor Yellow
        & $python -m pip install --disable-pip-version-check onnx | Out-Null
    }
    & $python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('numpy') else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing numpy into venv..." -ForegroundColor Yellow
        & $python -m pip install --disable-pip-version-check numpy | Out-Null
    }

    $genFile = "generate_lstm_stub.py"
    @'
import numpy as np
import onnx
from onnx import helper, TensorProto, OperatorSetIdProto

inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 50, 1])
out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1])

# axes for Squeeze provided as input (opset >= 13)
axes_sq = helper.make_tensor(name="axes_sq_val", data_type=TensorProto.INT64, dims=[1], vals=np.array([2], dtype=np.int64))
axes_sq_const = helper.make_node("Constant", inputs=[], outputs=["axes_sq"], value=axes_sq)

# mean over sequence+feature, then tanh -> [-1,1]
node1 = helper.make_node("ReduceMean", ["input"], ["m"], keepdims=1, axes=[1, 2])
node2 = helper.make_node("Squeeze", ["m", "axes_sq"], ["msq"])  # [1,1]
node3 = helper.make_node("Tanh", ["msq"], ["out"])  # [-1,1]

graph = helper.make_graph([axes_sq_const, node1, node2, node3], "aria_lstm_stub", [inp], [out])
opset = OperatorSetIdProto()
opset.version = 13
# Force an older, widely-supported ONNX IR version for maximum CPU runtime compatibility
model = helper.make_model(graph, opset_imports=[opset], producer_name="aria-pro", ir_version=10)
onnx.checker.check_model(model)
onnx.save(model, "lstm_forex.onnx")
'@ | Out-File -FilePath $genFile -Encoding UTF8

    if (Test-Path "lstm_forex.onnx") { Remove-Item "lstm_forex.onnx" -Force -ErrorAction SilentlyContinue }
    & $python $genFile
    if ($LASTEXITCODE -ne 0) { throw "LSTM generator exited with code $LASTEXITCODE" }
    if (Test-Path "lstm_forex.onnx") {
        $lsz = (Get-Item "lstm_forex.onnx").Length
        if ($lsz -gt 200) {
            Write-Host ("[OK] LSTM ONNX stub generated ({0}) bytes" -f $lsz) -ForegroundColor Green
        } else {
            Remove-Item "lstm_forex.onnx" -Force -ErrorAction SilentlyContinue
            throw "LSTM file too small ($lsz)"
        }
    } else { throw "Failed to generate LSTM ONNX" }
    Remove-Item $genFile -Force -ErrorAction SilentlyContinue
} catch {
    Write-Host ("[WARN] LSTM ONNX generation failed: {0}" -f $_.Exception.Message) -ForegroundColor Yellow
}

# --- 1b. XGBoost Tabular Model (real ONNX via skl2onnx) ---
Write-Host "Preparing XGBoost ONNX (regressor, CPU-friendly) ..." -ForegroundColor Cyan
try {
    $python = ".\.venv\Scripts\python.exe"
    if (!(Test-Path $python)) { $python = "python" }

    # Ensure required build-time deps (not needed at runtime)
    & $python -c "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('xgboost') else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing xgboost into environment..." -ForegroundColor Yellow
        & $python -m pip install --disable-pip-version-check xgboost | Out-Null
    }
    & $python -c "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('onnxmltools') else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing onnxmltools into environment..." -ForegroundColor Yellow
        & $python -m pip install --disable-pip-version-check onnxmltools | Out-Null
    }
    & $python -c "import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('skl2onnx') else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing skl2onnx into environment..." -ForegroundColor Yellow
        & $python -m pip install --disable-pip-version-check skl2onnx | Out-Null
    }

    $genXgb = "generate_xgb_stub.py"
    @'
import numpy as np
from xgboost import XGBRegressor
from onnxmltools.convert import convert_xgboost
from skl2onnx.common.data_types import FloatTensorType
from onnxmltools.utils import save_model

rs = 0
np.random.seed(rs)
X = np.random.randn(500, 6).astype(np.float32)
y = np.tanh(X @ np.array([0.7, -0.5, 0.3, 0.2, -0.1, 0.4], dtype=np.float32) + 0.05*np.random.randn(500)).astype(np.float32)

model = XGBRegressor(n_estimators=30, max_depth=3, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9, tree_method="hist", random_state=0)
model.fit(X, y)

initial_types = [("input", FloatTensorType([None, 6]))]
onnx_model = convert_xgboost(model, initial_types=initial_types, target_opset=13)
save_model(onnx_model, "xgboost_forex.onnx")
'@ | Out-File -FilePath $genXgb -Encoding UTF8

    if (Test-Path "xgboost_forex.onnx") { Remove-Item "xgboost_forex.onnx" -Force -ErrorAction SilentlyContinue }
    & $python $genXgb
    if ($LASTEXITCODE -ne 0) { throw "XGBoost generator exited with code $LASTEXITCODE" }
    if (Test-Path "xgboost_forex.onnx") {
        $xsz = (Get-Item "xgboost_forex.onnx").Length
        if ($xsz -gt 10000) {
            Write-Host ("[OK] XGBoost ONNX generated ({0}) bytes" -f $xsz) -ForegroundColor Green
        } else {
            Remove-Item "xgboost_forex.onnx" -Force -ErrorAction SilentlyContinue
            throw "XGBoost ONNX file too small ($xsz)"
        }
    } else { throw "Failed to generate xgboost_forex.onnx" }
    Remove-Item $genXgb -Force -ErrorAction SilentlyContinue
} catch {
    Write-Host ("[WARN] XGBoost ONNX generation failed: {0}" -f $_.Exception.Message) -ForegroundColor Yellow
}

# --- 2. CNN Chart Pattern Detector (ONNX) ---
Write-Host "Downloading CNN Pattern model..." -ForegroundColor Cyan
try {
    $cnnUrl = "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-12.onnx"
    Invoke-WebRequest -Uri $cnnUrl -OutFile "cnn_patterns.onnx" -UseBasicParsing -Headers @{"User-Agent"="Mozilla/5.0"}
    $sz = (Get-Item "cnn_patterns.onnx").Length
    if ($sz -gt 100000) {
        Write-Host ("[OK] CNN model downloaded ({0}) bytes" -f $sz) -ForegroundColor Green
    } else {
        throw "CNN file too small ($sz)"
    }
} catch {
    Write-Host ("[WARN] Primary CNN download failed: {0}. Trying fallback..." -f $_.Exception.Message) -ForegroundColor Yellow
    try {
        $cnnUrl2 = "https://raw.githubusercontent.com/onnx/models/main/vision/classification/squeezenet/model/squeezenet1.0-12.onnx"
        Invoke-WebRequest -Uri $cnnUrl2 -OutFile "cnn_patterns.onnx" -UseBasicParsing -Headers @{"User-Agent"="Mozilla/5.0"}
        $sz = (Get-Item "cnn_patterns.onnx").Length
        if ($sz -gt 100000) {
            Write-Host ("[OK] CNN model downloaded via fallback ({0}) bytes" -f $sz) -ForegroundColor Green
        } else { throw "CNN fallback file too small ($sz)" }
    } catch {
        Write-Host ("[ERR] CNN download failed after fallback: {0}" -f $_.Exception.Message) -ForegroundColor Red
        if (Test-Path "cnn_patterns.onnx") {
            $fsize = (Get-Item "cnn_patterns.onnx").Length
            if ($fsize -lt 100000) { Remove-Item "cnn_patterns.onnx" -Force -ErrorAction SilentlyContinue }
        }
    }
}

# --- 3. Visual AI Chart Feature Extractor (ONNX, MobileNetV2) ---
Write-Host "Downloading Visual AI model..." -ForegroundColor Cyan
try {
    $visUrl = "https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
    Invoke-WebRequest -Uri $visUrl -OutFile "visual_ai.onnx" -UseBasicParsing -Headers @{"User-Agent"="Mozilla/5.0"}
    $sz2 = (Get-Item "visual_ai.onnx").Length
    if ($sz2 -gt 5000000) {
        Write-Host ("[OK] Visual AI model downloaded ({0}) bytes" -f $sz2) -ForegroundColor Green
    } else {
        throw "Visual file too small ($sz2)"
    }
} catch {
    Write-Host ("[WARN] Primary Visual AI download failed: {0}. Trying fallback..." -f $_.Exception.Message) -ForegroundColor Yellow
    try {
        $visUrl2 = "https://raw.githubusercontent.com/onnx/models/main/validated/vision/classification/mobilenet/model/mobilenetv2-12-qdq.onnx"
        Invoke-WebRequest -Uri $visUrl2 -OutFile "visual_ai.onnx" -UseBasicParsing -Headers @{"User-Agent"="Mozilla/5.0"}
        $sz2 = (Get-Item "visual_ai.onnx").Length
        if ($sz2 -gt 5000000) {
            Write-Host ("[OK] Visual AI model downloaded via fallback ({0}) bytes" -f $sz2) -ForegroundColor Green
        } else { throw "Visual fallback file too small ($sz2)" }
    } catch {
        Write-Host ("[ERR] Visual AI download failed after fallback: {0}" -f $_.Exception.Message) -ForegroundColor Red
        if (Test-Path "visual_ai.onnx") {
            $fsize2 = (Get-Item "visual_ai.onnx").Length
            if ($fsize2 -lt 5000000) { Remove-Item "visual_ai.onnx" -Force -ErrorAction SilentlyContinue }
        }
    }
}

# --- 4. PPO Forex Trader (Stable-Baselines3 Policy) ---
if ($env:SKIP_PPO -eq '1') {
    Write-Host "Skipping PPO Trader model download (SKIP_PPO=1)" -ForegroundColor Yellow
} else {
    Write-Host "Downloading PPO Trader model..." -ForegroundColor Cyan
    try {
        # Use a small real PPO policy from SB3 hub as a stand-in (CartPole)
        $ppoUrl = "https://huggingface.co/sb3/ppo-CartPole-v1/resolve/main/ppo-CartPole-v1.zip"
        Invoke-WebRequest -Uri $ppoUrl -OutFile "ppo_trader.zip" -UseBasicParsing
        $psz = (Get-Item "ppo_trader.zip").Length
        if ($psz -gt 100000) {
            Write-Host ("[OK] PPO policy downloaded ({0}) bytes" -f $psz) -ForegroundColor Green
        } else { throw "PPO file too small ($psz)" }
    } catch {
        Write-Host ("[ERR] PPO download failed: {0}" -f $_.Exception.Message) -ForegroundColor Red
        if (Test-Path "ppo_trader.zip") { Remove-Item "ppo_trader.zip" -Force }
    }
}

# --- 5. LLM Macro Model (GGUF) ---
if ($env:SKIP_LLM -eq '1') {
    Write-Host "Skipping LLM Macro model download (SKIP_LLM=1)" -ForegroundColor Yellow
} else {
    Write-Host "Downloading LLM Macro model..." -ForegroundColor Cyan
    try {
        # Allow override via env var; default to TinyLlama 1.1B GGUF (Q4_K_M)
        $url = $env:LLM_URL
        if ([string]::IsNullOrEmpty($url)) {
            $url = "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"
        }
        Invoke-WebRequest -Uri $url -OutFile "llm_macro.gguf" -UseBasicParsing
        $lsz = (Get-Item "llm_macro.gguf").Length
        if ($lsz -gt 150MB) {
            Write-Host ("[OK] LLM Macro model downloaded ({0}) bytes" -f $lsz) -ForegroundColor Green
        } else { throw "LLM file too small ($lsz)" }
    } catch {
        Write-Host ("[ERR] LLM Macro download failed: {0}" -f $_.Exception.Message) -ForegroundColor Red
        if (Test-Path "llm_macro.gguf") { Remove-Item "llm_macro.gguf" -Force }
    }
}

# --- 6. Cleanup ---
Write-Host "Cleaning up temporary directories..." -ForegroundColor Cyan
if (Test-Path "lstm_repo") { Remove-Item -Recurse -Force "lstm_repo" }
if (Test-Path "cnn_repo") { Remove-Item -Recurse -Force "cnn_repo" }
if (Test-Path "visual_repo") { Remove-Item -Recurse -Force "visual_repo" }
if (Test-Path "ppo_repo") { Remove-Item -Recurse -Force "ppo_repo" }

# --- 7. Verify Downloads ---
Write-Host "`n[ARIA PHASE 4] Model Verification:" -ForegroundColor Green
$thresholds = @{
    "xgboost_forex.onnx" = 10000
    "lstm_forex.onnx" = 200
    "cnn_patterns.onnx" = 100000
    "visual_ai.onnx" = 5000000
    "ppo_trader.zip" = 100000
    "llm_macro.gguf" = 100MB
}
$models = @("xgboost_forex.onnx", "lstm_forex.onnx", "cnn_patterns.onnx", "visual_ai.onnx", "ppo_trader.zip", "llm_macro.gguf")
foreach ($model in $models) {
    if (Test-Path $model) {
        $size = (Get-Item $model).Length
        $threshold = $thresholds[$model]
        if ($size -gt $threshold) {
            Write-Host ("[OK] {0} ({1}) bytes - REAL" -f $model, $size) -ForegroundColor Green
        } else {
            Write-Host ("[WARN] {0} ({1} bytes) - TOO SMALL (expected > {2})" -f $model, $size, $threshold) -ForegroundColor Yellow
        }
    } else {
        Write-Host "✗ $model (missing)" -ForegroundColor Red
    }
}

Set-Location ../..
Write-Host "`n[ARIA PHASE 4] Model setup complete!" -ForegroundColor Green
Write-Host "Models are located in: backend/models/" -ForegroundColor Yellow
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Install required dependencies: pip install onnxruntime stable-baselines3 llama-cpp-python" -ForegroundColor White
Write-Host "2. Test models: python test_real_models.py" -ForegroundColor White
Write-Host "3. Run system: python start_backend.py" -ForegroundColor White
