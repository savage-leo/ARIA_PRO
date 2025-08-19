# ARIA Pro MT5 Trading Guide

This guide explains how to run the ARIA Pro auto trader system with live MT5 data feeds and execution.

## Prerequisites

1. MT5 account credentials (login, password, server)
2. MT5 terminal installed and running
3. Valid API keys for any external services (Alpha Vantage, etc.)

## Configuration

The system is configured via environment variables. The main configuration file is `production.env`:

```env
# ===== MT5 CONNECTION =====
ARIA_ENABLE_MT5=1
MT5_LOGIN=your_login
MT5_PASSWORD=your_password
MT5_SERVER=your_server

# ===== AUTO TRADING =====
AUTO_TRADE_ENABLED=1
AUTO_TRADE_PROB_THRESHOLD=0.75
AUTO_TRADE_INTERVAL_SEC=60
AUTO_TRADE_PRIMARY_MODEL=xgb
AUTO_TRADE_DRY_RUN=0
AUTO_TRADE_ATR_PERIOD=14
AUTO_TRADE_ATR_SL_MULT=1.5
AUTO_TRADE_ATR_TP_MULT=2.0
AUTO_TRADE_COOLDOWN_SEC=300
AUTO_TRADE_SYMBOLS=EURUSD,GBPUSD,USDJPY,XAUUSD,BTCUSD

# ===== RISK MANAGEMENT =====
ARIA_RISK_PER_TRADE=0.005
ARIA_ATR_STOP_MULT=2.5
ARIA_ATR_TP_MULT=4.0

# ===== EXECUTION & ADMIN =====
ARIA_ENABLE_EXEC=1
AUTO_EXEC_ENABLED=true
ALLOW_LIVE=1
```

## Running the System

### 1. Test MT5 Connection

First, verify that the system can connect to your MT5 account:

```bash
cd ARIA_PRO
python test_mt5_connection.py
```

### 2. Run Backtest with MT5 Data

Run a backtest using historical data from MT5:

```bash
cd ARIA_PRO
python run_mt5_backtest.py
```

Or use the original backtest script with MT5 data:

```bash
cd ARIA_PRO
python backend/scripts/backtest_bars.py --symbol EURUSD --days 30
```

### 3. Run Live Trading

To run live trading with MT5 execution:

```bash
cd ARIA_PRO
python run_live_trading.py
```

### 4. Run Production System

For production deployment, use the production startup script:

```bash
cd ARIA_PRO
python start_production.py
```

## Monitoring

The system provides monitoring endpoints:

- `GET /monitoring/auto-trader/status` - Auto trader status
- `GET /monitoring/models/status` - AI model status
- `GET /monitoring/data-feed/status` - Data feed status

## Risk Management

The system includes several risk management features:

1. **Position Sizing**: Risk per trade is controlled by `ARIA_RISK_PER_TRADE`
2. **Stop Loss**: ATR-based stop loss with multiplier `ARIA_ATR_STOP_MULT`
3. **Take Profit**: ATR-based take profit with multiplier `ARIA_ATR_TP_MULT`
4. **Drawdown Control**: Maximum drawdown limit via `ARIA_MAX_DD`
5. **Cooldown Period**: Prevents over-trading with `AUTO_TRADE_COOLDOWN_SEC`

## Troubleshooting

### Common Issues

1. **MT5 Connection Failed**: Verify credentials in `production.env` and ensure MT5 terminal is running
2. **No Market Data**: Check MT5 symbol names and ensure they are available in your account
3. **Execution Errors**: Verify account has sufficient margin and check symbol specifications

### Logs

Check the following log files for detailed information:

- `logs/backend.log` - Main application logs
- `logs/production.log` - Production system logs
- `logs/enhanced_fusion.log` - Fusion engine logs

## Security

1. Keep your `production.env` file secure and never commit it to version control
2. Use strong passwords and secure storage for MT5 credentials
3. Monitor trading activity regularly
4. Use dry run mode (`AUTO_TRADE_DRY_RUN=1`) for testing
