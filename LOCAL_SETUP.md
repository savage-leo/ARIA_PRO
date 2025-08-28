# ARIA Institutional Pro — Local Setup (No Docker)

This project is intended to run locally without Docker. Follow the steps below to run both backend (FastAPI) and frontend (React + TypeScript) on your machine.

## Prerequisites
- Windows 10/11
- Python 3.11+
- Node.js 18+
- MetaTrader5 terminal installed and logged in (for live integrations)
- Visual Studio Build Tools (if you plan to build C++ components)

## 1) Environment variables
Use `production.env.template` as a template and create `production.env` in the project root:

- Template: `ARIA_PRO/production.env.template`
- Create:   `ARIA_PRO/production.env` (auto-loaded by `start_backend.py`)
- Frontend may read `ARIA_PRO/frontend/.env` (optional) for Vite variables

Important variables (minimum for local):
- MT5_LOGIN, MT5_PASSWORD, MT5_SERVER (only if connecting to MT5)
- JWT_SECRET_KEY (>= 32 chars), ADMIN_API_KEY (>= 16 chars)
- ARIA_WS_TOKEN (>= 16 chars) for WebSocket auth
- Optional: DATABASE_URL, REDIS_URL

## 2) Backend setup (FastAPI)
```
# From project root
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Run dev server (recommended)
python ..\start_backend.py
# or (from backend dir)
uvicorn main:app --host 0.0.0.0 --port 8100 --reload
```

Health check:
- Open http://localhost:8100/docs to view the API

## 3) Frontend setup (React + TypeScript)
```
# From project root
cd frontend
npm install
npm run dev
```

- Vite dev server: http://localhost:5173 (default)
- Configure backend URLs in `frontend/.env` if needed:
  - `VITE_BACKEND_BASE=http://localhost:8100`
  - `VITE_BACKEND_WS=ws://localhost:8100`

## 4) One-command development (optional)
From the project root you can use the root `package.json` scripts:
```
# Install everything
npm run install:all

# Run backend and frontend together
npm run dev
```

This uses `concurrently` to run both processes in one terminal.

## 5) Testing
```
# Backend tests
cd backend
python -m pytest

# Frontend tests
cd frontend
npm run test
```

## Notes
- Docker is not used in this setup. Ignore any Dockerfiles or compose files. Run everything locally.
- If you need HTTPS locally, use a dev proxy (e.g. `vite.config.ts` proxy) or run Nginx locally as an optional add-on.
- For MT5: ensure terminal is running and credentials are valid. Some endpoints may require a connected terminal.

See also: `PORTS_AND_CONNECTIONS.md` for a concise port and connection matrix.
