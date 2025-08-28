@echo off
cd /d "c:\savage\ARIA_PRO"
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
echo Starting ARIA PRO Backend on port 8100...
python start_backend.py
pause
