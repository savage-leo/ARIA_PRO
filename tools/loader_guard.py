# tools/loader_guard.py
"""
Deterministic lightweight model loader and blocker for heavy-model downloads.
This script refuses to run if it detects heavy model downloader scripts
or attempts to fetch giant models. It enforces local-only, small artifacts.
"""
import sys
from pathlib import Path
import json

BASE = Path(__file__).resolve().parents[1]
BLOCKLIST = [
    "download_real_models.ps1",
    "download_models.sh",
    "weights_fetcher.py",
    "huggingface_download.py",
    "model_fetcher.py"
]

MAX_ARTIFACT_SIZE_MB = 150    # max allowed single model size (tuneable)

def scan_repo():
    found = []
    for name in BLOCKLIST:
        p = BASE / name
        if p.exists():
            found.append(str(p))
    return found

def check_models_dir():
    models_dir = BASE / "models"
    oversized = []
    if models_dir.exists():
        for f in models_dir.rglob("*"):
            if f.is_file():
                size_mb = f.stat().st_size / (1024*1024)
                if size_mb > MAX_ARTIFACT_SIZE_MB:
                    oversized.append((str(f), round(size_mb,2)))
    return oversized

def enforce():
    blocked = scan_repo()
    oversized = check_models_dir()
    if blocked:
        print("ABORT: Heavy downloader scripts detected. Remove them before running live.")
        for b in blocked:
            print(" -", b)
        sys.exit(2)
    if oversized:
        print("ABORT: Oversized model artifacts found (> %d MB):" % MAX_ARTIFACT_SIZE_MB)
        for f,s in oversized:
            print(" -", f, s, "MB")
        sys.exit(3)
    print("Loader guard OK: no heavy downloaders or oversized artifacts.")

if __name__ == "__main__":
    enforce()
