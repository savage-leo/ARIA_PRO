# SMC Edge Core Implementation

## Overview
SMC Edge Core combines Smart Money Concepts analysis with liquidity trap detection for institutional-grade trading signals.

## Components
- `trap_detector.py`: Liquidity trap detection using heuristics
- `smc_edge_core.py`: Main orchestration engine
- `trade_memory.py`: SQLite-based trade idea storage
- `order_executor.py`: Partial order planning

## Safety Features
- Dry-run default
- Admin API key required for execution
- Environment controls for live trading
- No mock data - expects real market data

## Environment Setup
```bash
SMC_EDGE_ENABLED=true
AUTO_EXEC_ENABLED=false
ALLOW_LIVE=0
ADMIN_API_KEY=changeme
```

## Usage
1. Start services: `./scripts/start_all.sh`
2. Test prepare: Use frontend AISecretPanel or API
3. View history: `GET /api/smc/history`
4. Execute: Admin only with proper environment flags

## Testing
```bash
python scripts/test_smc_edge.py
```

## Logs
- Trade Memory: `logs/trade_memory.sqlite`
- Trade Logs: `logs/trades.log`
