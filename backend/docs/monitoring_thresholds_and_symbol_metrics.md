# ARIA PRO Monitoring: Env-Configurable Thresholds and Per-Symbol Metrics

This document summarizes the new environment-configurable monitoring thresholds and the per-symbol performance metrics available via REST and WebSocket.

## Environment Variables

- ARIA_THRESH_CPU_WARN
  - Description: CPU usage warning threshold (%).
  - Default: 90
- ARIA_THRESH_MEM_AVAIL_CRIT
  - Description: Critical threshold for available memory (% of total). Alert when below this value.
  - Default: 10
- ARIA_THRESH_LATENCY_WARN_MS
  - Description: Warning threshold for model latency EMA (milliseconds) for both global model metrics and per-symbol metrics.
  - Default: 1000

These values are read when the global PerformanceMonitor is instantiated. Set them before the process starts for desired thresholds.

## Key Backend Components

- `backend/core/performance_monitor.py`
  - Class `PerformanceMonitor` maintains:
    - Global model metrics
    - Per-symbol metrics: `symbol -> model -> metrics`
  - Thresholds available at `PerformanceMonitor.thresholds`.
  - Methods:
    - `get_metrics(model_name?: str)`
    - `get_symbol_metrics(symbol?: str)`
    - `get_system_metrics()`

- `backend/routes/monitoring.py`
  - Provides REST + WebSocket endpoints described below.

## REST Endpoints

Base prefix: `/monitoring`

- GET `/performance/metrics`
  - Returns `{ system, models, symbols, thresholds }`.
- GET `/performance/models/{model_name}`
  - Returns detailed metrics for a specific model or 404 JSON if none.
- GET `/performance/symbols`
  - Returns all per-symbol metrics: `{ [symbol]: { [model]: metrics } }`.
- GET `/performance/symbols/{symbol}`
  - Returns per-symbol metrics for one symbol or 404 JSON if none.
- GET `/performance/thresholds`
  - Returns the current thresholds sourced from env with defaults.

## WebSocket

- WS `/monitoring/ws/performance`
  - Broadcasts messages of types:
    - `system_metrics`
    - `model_metrics`
    - `symbol_metrics` (new)
    - `alert` (driven by thresholds)

## Frontend Integration Notes

- The dashboard should connect to `/monitoring/ws/performance` for real-time updates.
- Use `/monitoring/performance/thresholds` to colorize UI based on env-config thresholds.
- Use `/monitoring/performance/symbols` to populate per-symbol views and charts.

## Testing

- Backend tests are located at `backend/tests/test_monitoring_endpoints.py`.
  - Validate thresholds endpoint reads env values.
  - Validate per-symbol endpoints return model metrics after updates.

## Operational Guidance

- Set the thresholds via environment before starting the backend service.
- Monitor for increased memory usage from deeper per-symbol histories; `PerformanceMonitor.max_metrics` limits history length.
