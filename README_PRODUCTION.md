# ARIA PRO Production Setup Guide

## Overview
ARIA PRO is an institutional-grade forex trading system featuring:
- **Phase 3 Orchestrator**: Real-time market data processing and signal fusion
- **Enhanced SMC Fusion Core**: Advanced Smart Money Concepts with AI meta-weighting
- **Multi-Model AI Integration**: LSTM, CNN, PPO, Visual AI, and LLM Macro models
- **Risk Management**: ATR-based position sizing, Kelly criterion, anomaly detection
- **MT5 Integration**: Live market data and trade execution

## Quick Start

### 1. Environment Setup
```bash
# Copy and configure production environment
cp production.env .env
# Edit .env with your MT5 credentials and preferences
```

### 2. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Launch Production System
```bash
# Windows
launch_production.bat

# Linux/Mac
python start_production.py
```

## Configuration

### MT5 Connection
```env
ARIA_ENABLE_MT5=1
MT5_LOGIN=your_mt5_login
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_mt5_server
```

### Trading Symbols
```env
ARIA_SYMBOLS=EURUSD,GBPUSD,USDJPY,XAUUSD
ARIA_BAR_SECONDS=60
```

### Risk Management
```env
ARIA_RISK_PER_TRADE=0.005      # 0.5% risk per trade
ARIA_MAX_DD=0.03              # 3% max daily drawdown
ARIA_KELLY_CAP=0.25           # 25% Kelly criterion cap
ARIA_ATR_STOP_MULT=2.5        # ATR stop loss multiplier
ARIA_ATR_TP_MULT=4.0          # ATR take profit multiplier
```

### Execution Control
```env
ARIA_ENABLE_EXEC=1            # Enable live trading
ARIA_ALLOW_SHORT=1            # Allow short positions
ARIA_MAX_SLIPPAGE_PIPS=1.5    # Max slippage tolerance
```

## System Architecture

### Phase 3 Orchestrator
- **Bar Builder**: Aggregates MT5 ticks into timeframe bars
- **Signal Fusion**: Integrates AI model signals with SMC analysis
- **Meta-Weighting**: Online learning fusion of multiple AI models
- **Execution Router**: Risk-managed trade execution via MT5

### Enhanced SMC Fusion Core
- **Order Block Detection**: Identifies institutional order blocks
- **Fair Value Gap Analysis**: Detects market inefficiencies
- **Liquidity Zone Mapping**: Maps institutional liquidity levels
- **Regime Detection**: Identifies market conditions (trend/mean-reversion)

### AI Model Integration
- **LSTM**: Time series forecasting
- **CNN**: Pattern recognition
- **PPO**: Reinforcement learning agent
- **Visual AI**: Chart pattern analysis
- **LLM Macro**: Fundamental analysis

## Monitoring & Debugging

### API Endpoints
```bash
# Health check
GET http://localhost:8000/health

# Detailed health check
GET http://localhost:8000/debug/health/detailed

# Latest trading ideas
GET http://localhost:8000/debug/ideas
GET http://localhost:8000/debug/idea/EURUSD

# Orchestrator status
GET http://localhost:8000/debug/orchestrator/status

# Fusion core state
GET http://localhost:8000/debug/fusion/state/EURUSD
```

### Log Files
- `logs/production.log`: Main system logs
- `logs/enhanced_fusion.log`: SMC fusion core logs
- `logs/trade_memory.sqlite`: Trade history database

## Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -r backend/requirements.txt

EXPOSE 8000

CMD ["python", "start_production.py"]
```

### System Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ recommended
- **Storage**: 10GB+ for logs and models
- **Network**: Stable internet for MT5 connection
- **OS**: Windows 10/11 (for MT5), Linux/Mac (backend only)

### Security Considerations
- Use strong API keys: `ADMIN_API_KEY=your_secure_key`
- Restrict network access to trusted IPs
- Monitor system logs for anomalies
- Regular backup of trade history and state files

## Performance Optimization

### Model Loading
- Models are loaded once at startup
- GPU acceleration available for CNN/Visual models
- Memory usage scales with number of symbols

### Latency Optimization
- Bar timeframe: 60 seconds (configurable)
- Feed timeout: 30 seconds (configurable)
- Execution retry: 0.75 seconds (configurable)

### Scaling
- Multiple symbols: Each symbol runs independent fusion core
- Horizontal scaling: Deploy multiple instances per symbol group
- Load balancing: Use reverse proxy for API endpoints

## Troubleshooting

### Common Issues

1. **MT5 Connection Failed**
   - Verify credentials in production.env
   - Check MT5 terminal is running
   - Ensure network connectivity

2. **Module Import Errors**
   - Run: `pip install -r backend/requirements.txt`
   - Check Python path includes project root

3. **No Trading Ideas Generated**
   - Check market hours for symbols
   - Verify SMC patterns are being detected
   - Review anomaly threshold settings

4. **High Memory Usage**
   - Reduce number of symbols
   - Increase bar cleanup frequency
   - Monitor for memory leaks

### Debug Mode
```env
LOG_LEVEL=DEBUG
ENABLE_DEBUG_LOGGING=true
```

## Support

For issues and questions:
1. Check logs in `logs/` directory
2. Review API endpoints for system status
3. Verify environment configuration
4. Test with single symbol first

## License
ARIA PRO - Institutional Forex AI Trading System
Copyright (c) 2024

