#!/bin/bash

# Production startup script for ARIA Backend
export PYTHONPATH=/app
export PORT=${PORT:-8100}

# Create logs directory if it doesn't exist
mkdir -p logs

# Start the FastAPI application
exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
