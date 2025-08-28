import os
import sys
from pathlib import Path

# Setup
os.chdir(Path(__file__).parent)
sys.path.insert(0, '.')
os.environ['PYTHONPATH'] = '.'

# Load env
from dotenv import load_dotenv
load_dotenv('production.env')

# Start server
import uvicorn
from backend.main import app

print("Starting ARIA PRO Backend on 127.0.0.1:8100...")
uvicorn.run(app, host="127.0.0.1", port=8100, log_level="info")
