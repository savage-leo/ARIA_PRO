#!/usr/bin/env bash
set -e
echo "Bootstrapping ARIA Phase1..."

# Backend venv
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi
. .venv/bin/activate
pip install -r backend/requirements.txt

# Frontend deps
cd frontend
if [ ! -d "node_modules" ]; then
  npm ci
fi
cd ..

echo "Bootstrap complete"
