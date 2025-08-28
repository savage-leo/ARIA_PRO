# Ports and Connections

This document summarizes the local development ports, base URLs, environment variables, and proxy behavior for ARIA PRO.

## Summary
- Backend (FastAPI): http://localhost:8100
- Backend WebSocket: ws://localhost:8100/ws
- Frontend (Vite): http://localhost:5173
- REST base path: `/api`
- Vite dev proxy: `/api` -> http://localhost:8100

## Frontend configuration
- `frontend/src/config.ts`
  - `API_BASE_URL` default: `http://localhost:8100`
  - `WS_BASE_URL` default: `ws://localhost:8100`
- Environment overrides (optional, in `frontend/.env`):
  - `VITE_BACKEND_BASE=http://localhost:8100`
  - `VITE_BACKEND_WS=ws://localhost:8100`

## Backend configuration
- Startup: `python start_backend.py` (binds 127.0.0.1:8100)
- API routes are prefixed with `/api` (see `backend/main.py`)
- WebSocket endpoint: `/ws` (see `backend/routes/websocket.py`)
- CORS: configured via `ARIA_CORS_ORIGINS` to include frontend dev ports (e.g. 5173)
- Auth and security (required for protected endpoints and WS):
  - `JWT_SECRET_KEY` (>= 32 chars)
  - `ADMIN_API_KEY` (>= 16 chars)
  - `ARIA_WS_TOKEN` (>= 16 chars)

## Vite dev proxy
`frontend/vite.config.ts` proxies API calls during dev:
```ts
proxy: {
  '/api': {
    target: 'http://localhost:8100',
    changeOrigin: true,
    secure: false,
  }
}
```
This allows the frontend to call `/api/...` without CORS issues.

## WebSocket authentication
The backend requires a token. Provide one of the following:
- Query: `ws://localhost:8100/ws?token=YOUR_TOKEN`
- Header: `Authorization: Bearer YOUR_TOKEN`
- Header: `X-ARIA-TOKEN: YOUR_TOKEN`

Valid tokens: `ARIA_WS_TOKEN`, `ADMIN_API_KEY`, or a valid JWT (refresh).

## Notes
- Prefer using `start_backend.py` to ensure env loading from `production.env` and consistent 8100 port.
- Keep frontend and backend on the above ports to match docs and examples.
