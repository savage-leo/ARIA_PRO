# ARIA MT5 Production Engine

**Complete Institutional-Grade AI Trading System with Full MT5 Integration**

## 🚀 Overview

The ARIA MT5 Production Engine is a comprehensive, institutional-grade algorithmic trading system that integrates multiple AI models (LSTM, CNN, PPO, Image-based) with MetaTrader 5 for live trading. Built for production deployment with full risk management, persistent trade memory, and real-time monitoring.

## ✨ Key Features

### 🤖 AI Integration
- **LSTM Models**: Time series prediction and trend analysis
- **CNN Models**: Pattern recognition and chart analysis
- **PPO Models**: Reinforcement learning for optimal trading decisions
- **Image Models**: Chart image analysis for technical patterns
- **Ensemble Decision Making**: Weighted combination of all AI signals

### 🛡️ Risk Management
- **Dynamic Position Sizing**: Kelly criterion-based lot calculation
- **Drawdown Protection**: Real-time drawdown monitoring and limits
- **Correlation Risk**: Cross-asset correlation monitoring
- **Trade Cooldowns**: Prevents over-trading with time-based restrictions
- **Kill Switch**: Emergency stop functionality for critical conditions

### 📊 Production Features
- **Persistent Trade Memory**: Complete trade journaling and history
- **Real-Time Dashboard**: WebSocket and REST API for monitoring
- **Auto-Reconnect**: Automatic MT5 connection recovery
- **Comprehensive Logging**: Institutional-grade audit trails
- **Backup & Restore**: Data persistence and recovery systems

### 🔧 Technical Excellence
- **Multi-Symbol Support**: Trade multiple currency pairs simultaneously
- **Live Market Data**: Real-time tick processing from MT5
- **Execution Latency Monitoring**: Performance tracking for institutional requirements
- **Environment-Driven Configuration**: All settings via environment variables
- **Modular Architecture**: Clean separation of concerns

## 📋 Requirements

### System Requirements
- **Python**: 3.8+
- **MetaTrader 5**: Installed and configured
- **RAM**: 4GB+ recommended
- **Storage**: 1GB+ for logs and data

### Python Dependencies
```bash
pip install MetaTrader5 fastapi uvicorn websockets numpy pandas python-dotenv
```

### MT5 Requirements
- Valid MT5 account (demo or live)
- MT5 terminal installed and running
- API trading enabled

## 🚀 Quick Start

### 1. Configuration Setup

Create your configuration file:
```bash
cp aria_mt5_config.env.example aria_mt5_config.env
```

Edit `aria_mt5_config.env` with your MT5 credentials:
```env
# MT5 Connection Settings
MT5_ACCOUNT=your_account_number
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server

# Trading Parameters
TRADE_SYMBOLS=EURUSD,GBPUSD,USDJPY
LIVE_TRADING=false  # Set to true for live trading
```

### 2. Test Setup

Test your configuration:
```bash
python aria_mt5_launcher.py test
```

### 3. Start Engine

Start in production mode:
```bash
python aria_mt5_launcher.py start
```

Start with dashboard:
```bash
python aria_mt5_launcher.py start --mode dashboard
```

### 4. Access Dashboard

- **REST API**: http://127.0.0.1:8000
- **WebSocket**: ws://127.0.0.1:8765
- **API Documentation**: http://127.0.0.1:8000/docs

## 📖 Usage Guide

### Launcher Commands

```bash
# Start the engine
python aria_mt5_launcher.py start [--mode production|dashboard]

# Stop the engine
python aria_mt5_launcher.py stop

# Restart the engine
python aria_mt5_launcher.py restart

# Check status
python aria_mt5_launcher.py status

# View logs
python aria_mt5_launcher.py logs [--lines 50]

# Test setup
python aria_mt5_launcher.py test

# Backup data
python aria_mt5_launcher.py backup

# Restore data
python aria_mt5_launcher.py restore --backup-dir backups/backup_20250117_143022

# List backups
python aria_mt5_launcher.py list-backups
```

### API Endpoints

#### Engine Status
```bash
GET /status
```
Returns current engine status, trade counts, and account information.

#### Recent Trades
```bash
GET /trades
```
Returns the last 10 executed trades with full details.

#### AI Signals
```bash
GET /signals
```
Returns current AI signals for all configured symbols.

#### Risk Status
```bash
GET /risk
```
Returns risk management status and violations.

#### Control Engine
```bash
POST /control/start
POST /control/stop
```
Start or stop the trading engine.

### WebSocket Events

Connect to `ws://127.0.0.1:8765` for real-time updates:

```javascript
const ws = new WebSocket('ws://127.0.0.1:8765');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    switch(data.type) {
        case 'trade_executed':
            console.log('Trade executed:', data.data);
            break;
        case 'ai_signal':
            console.log('AI signal:', data.data);
            break;
        case 'risk_violation':
            console.log('Risk violation:', data.data);
            break;
        case 'heartbeat':
            console.log('System heartbeat');
            break;
    }
};
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MT5_ACCOUNT` | MT5 account number | Required |
| `MT5_PASSWORD` | MT5 password | Required |
| `MT5_SERVER` | MT5 server | Required |
| `TRADE_SYMBOLS` | Comma-separated symbols | EURUSD,GBPUSD,USDJPY |
| `TRADE_INTERVAL` | Trading interval (seconds) | 60 |
| `LIVE_TRADING` | Enable live trading | false |
| `MAX_OPEN_TRADES` | Maximum open positions | 5 |
| `DAILY_DRAWDOWN_LIMIT` | Daily drawdown limit | 100.0 |
| `BASE_LOT` | Base lot size | 0.01 |
| `MIN_CONFIDENCE` | Minimum AI confidence | 0.65 |

### AI Signal Weights

Configure the importance of each AI model:

```env
LSTM_WEIGHT=0.3      # LSTM time series analysis
CNN_WEIGHT=0.25      # CNN pattern recognition
PPO_WEIGHT=0.25      # PPO reinforcement learning
IMAGE_WEIGHT=0.2     # Image-based analysis
```

## 🏗️ Architecture

### Core Components

```
ARIA MT5 Production Engine
├── Configuration Manager
├── MT5 Connection Manager
├── AI Signal Generator
│   ├── LSTM Signal Generator
│   ├── CNN Signal Generator
│   ├── PPO Signal Generator
│   └── Image Signal Generator
├── Risk Manager
├── Trade Memory
├── Dashboard Manager
└── Main Trading Engine
```

### Data Flow

1. **Market Data** → MT5 Connection
2. **AI Analysis** → Signal Generation (LSTM, CNN, PPO, Image)
3. **Ensemble Decision** → Weighted combination of signals
4. **Risk Validation** → Position sizing and risk checks
5. **Trade Execution** → MT5 order placement
6. **Memory Storage** → Persistent trade journaling
7. **Dashboard Update** → Real-time monitoring

## 🔒 Security & Risk Management

### Production Safety Features

- **Demo Mode**: Safe testing with `LIVE_TRADING=false`
- **Account Validation**: Prevents demo account live trading
- **Position Limits**: Maximum open positions per symbol
- **Drawdown Protection**: Automatic stop on drawdown limits
- **Correlation Monitoring**: Prevents over-concentration
- **Execution Latency**: Monitors for performance issues

### Risk Parameters

```env
# Risk Management
MAX_OPEN_TRADES=5              # Max positions per symbol
DAILY_DRAWDOWN_LIMIT=100.0     # Daily loss limit
BASE_LOT=0.01                  # Minimum lot size
MAX_LOT=1.0                    # Maximum lot size
MIN_CONFIDENCE=0.65            # Minimum AI confidence
MAX_CORRELATION=0.7            # Maximum correlation threshold
```

## 📊 Monitoring & Analytics

### Dashboard Features

- **Real-Time Status**: Engine status and MT5 connection
- **Trade History**: Complete trade journal with AI attribution
- **AI Performance**: Individual model performance tracking
- **Risk Metrics**: Drawdown, correlation, and violation tracking
- **Account Overview**: Balance, equity, and P&L

### Logging

Comprehensive logging for institutional compliance:

- **Trade Logs**: All executed trades with full details
- **AI Logs**: Signal generation and confidence levels
- **Risk Logs**: Violations and risk management events
- **System Logs**: Connection status and errors
- **Performance Logs**: Execution latency and performance metrics

## 🔧 Development & Customization

### Adding New AI Models

1. Extend `AISignalGenerator` class
2. Implement signal generation method
3. Add weight configuration
4. Update ensemble calculation

### Custom Risk Rules

1. Extend `RiskManager` class
2. Implement custom validation method
3. Add configuration parameters
4. Update validation pipeline

### Dashboard Extensions

1. Add new API endpoints in `DashboardManager`
2. Implement data retrieval methods
3. Add WebSocket event types
4. Update frontend integration

## 🚨 Important Notes

### Live Trading Safety

⚠️ **CRITICAL**: Before enabling live trading:

1. **Test thoroughly** in demo mode
2. **Validate configuration** with `python aria_mt5_launcher.py test`
3. **Review risk parameters** for your account size
4. **Monitor initial trades** closely
5. **Have emergency stop** procedures ready

### Demo vs Live

- **Demo Mode**: Safe for testing and development
- **Live Mode**: Real money trading - use with extreme caution
- **Configuration**: Same settings work for both modes
- **Validation**: System prevents demo account live trading

### Performance Considerations

- **Execution Latency**: Monitor for <100ms institutional requirements
- **Memory Usage**: Trade memory grows over time
- **CPU Usage**: AI model inference can be resource-intensive
- **Network**: Stable internet required for MT5 connection

## 🆘 Troubleshooting

### Common Issues

1. **MT5 Connection Failed**
   - Check account credentials
   - Verify MT5 terminal is running
   - Check internet connection

2. **AI Models Not Loading**
   - Install required dependencies
   - Check model file paths
   - Verify Python environment

3. **Dashboard Not Accessible**
   - Check port availability
   - Verify firewall settings
   - Check FastAPI logs

4. **Performance Issues**
   - Monitor system resources
   - Check execution latency
   - Review AI model complexity

### Debug Mode

Enable detailed logging:
```env
LOG_LEVEL=DEBUG
PERFORMANCE_MONITORING=true
```

### Support

For issues and questions:
1. Check logs: `python aria_mt5_launcher.py logs`
2. Test setup: `python aria_mt5_launcher.py test`
3. Review configuration
4. Check system requirements

## 📈 Performance Benchmarks

### Institutional Requirements

- **Execution Latency**: <100ms
- **Data Quality**: 99.9% uptime
- **Risk Management**: Real-time monitoring
- **Audit Trail**: Complete logging
- **Recovery Time**: <30 seconds

### Expected Performance

- **Trade Execution**: 50-100ms typical
- **AI Signal Generation**: 10-50ms per model
- **Risk Validation**: <5ms
- **Dashboard Updates**: Real-time via WebSocket
- **Memory Usage**: 100-500MB typical

## 🔄 Updates & Maintenance

### Regular Maintenance

1. **Backup Data**: `python aria_mt5_launcher.py backup`
2. **Review Logs**: `python aria_mt5_launcher.py logs`
3. **Update Configuration**: Review and adjust parameters
4. **Monitor Performance**: Check execution metrics
5. **Clean Old Data**: Archive old trade memory

### Version Updates

1. **Backup Current Setup**: `python aria_mt5_launcher.py backup`
2. **Update Code**: Pull latest version
3. **Test Configuration**: `python aria_mt5_launcher.py test`
4. **Restart Engine**: `python aria_mt5_launcher.py restart`

## 📄 License

This software is provided for educational and research purposes. Use at your own risk. The authors are not responsible for any financial losses.

## 🤝 Contributing

Contributions are welcome! Please:

1. Test thoroughly in demo mode
2. Follow existing code style
3. Add comprehensive documentation
4. Include unit tests
5. Update this README

---

**ARIA MT5 Production Engine** - Institutional-Grade AI Trading System

*Built for production deployment with comprehensive risk management and real-time monitoring.*

