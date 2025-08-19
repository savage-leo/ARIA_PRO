# ARIA Institutional Pro — Local Setup (No Docker)

This project is intended to run locally without Docker. Follow the steps below to run both backend (FastAPI) and frontend (React + TypeScript) on your machine.

## Prerequisites
- Windows 10/11
- Python 3.11+
- Node.js 18+
- MetaTrader5 terminal installed and logged in (for live integrations)
- Visual Studio Build Tools (if you plan to build C++ components)

## 1) Environment variables
Use `env.example` as a template and create your `.env` in the project root:

- Root file: `ARIA_PRO/.env`
- Backend may also read `ARIA_PRO/backend/.env` (optional)
- Frontend may read `ARIA_PRO/frontend/.env` (optional) for Vite variables

Important variables:
- MT5_LOGIN, MT5_PASSWORD, MT5_SERVER
- JWT_SECRET
- Optional: DATABASE_URL, REDIS_URL (if you use external services locally)

## 2) Backend setup (FastAPI)
```
# From project root
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Run dev server
python ..\start_backend.py
# or
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:
- Open http://localhost:8000/docs to view the API

## 3) Frontend setup (React + TypeScript)
```
# From project root
cd frontend
npm install
npm run dev
```

- Vite dev server: http://localhost:5173 (default)
- Configure API URL in `frontend/.env` if needed, e.g. `VITE_API_URL=http://localhost:8000`

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
