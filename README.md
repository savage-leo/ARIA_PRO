# ARIA PRO - Institutional Forex AI Trading Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.8+-blue.svg)](https://www.typescriptlang.org/)
[![React 19](https://img.shields.io/badge/React-19+-61DAFB.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)

**ARIA PRO** is a production-ready, institutional-grade AI-powered trading platform that combines multiple machine learning models, Enhanced Smart Money Concepts (SMC) analysis, and real-time market data processing for algorithmic trading in the Forex market. Built for professional traders and institutions requiring maximum precision, reliability, and performance.

## 🌟 Key Features

### 🧠 Enhanced AI Signal Generation
- **6-Model Ensemble**: XGBoost, LSTM, CNN, PPO, Vision AI, and LLM Macro models
- **Enhanced SMC Fusion Core**: Advanced Smart Money Concepts with bias engine integration
- **Real-time Processing**: Sub-50ms latency with WebSocket streaming
- **Market Regime Detection**: HMM-based Viterbi smoothing with persistence bias
- **Risk Budget Engine**: Kelly criterion optimization with dynamic position sizing
- **Microstructure Analysis**: Order flow imbalances and liquidity trap detection

### ⚡ Production Trading Engine
- **MT5 Integration**: Live MetaTrader 5 connectivity with fallback mechanisms
- **Enhanced AutoTrader**: Multi-symbol automated execution with kill-switch protection
- **Multi-Timeframe Analysis**: M1, M5, M15, M30, H1, H4, D1 with regime conditioning
- **Bias Engine**: Market sentiment analysis with confluence scoring
- **Performance Monitor**: Real-time system metrics and trade attribution
- **Emergency Controls**: Global kill switch with automatic drawdown protection

### 🔒 Institutional Risk Management
- **Dynamic Position Sizing**: 0.3-1% adaptive sizing based on market regime
- **Portfolio Optimization**: Correlation-aware multi-asset position management
- **Drawdown Protection**: Automatic risk reduction during high volatility periods
- **Live Guard Middleware**: Production safety enforcement with trading restrictions
- **Rate Limiting**: Per-IP and per-endpoint protection with burst allowance
- **Health Monitoring**: Comprehensive system health checks with liveness probes

### 📊 Professional Dashboard Features
- **Neon HUD Theme**: Futuristic blue/cyan interface inspired by institutional trading floors
- **Real-time WebSocket**: Live data streaming for ticks, signals, and order updates
- **Flow Monitor**: Data pipeline visualization with animated particle effects
- **Institutional AI Tab**: Advanced signal analysis with model performance metrics
- **Orders & Positions**: Live order management with WebSocket updates
- **Settings & Monitoring**: System configuration with real-time performance metrics

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- MetaTrader 5 (for live trading)
- Redis (for caching and pub/sub)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/aria-pro.git
   cd ARIA_PRO
   ```

2. **Set up Python environment**
   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac

   # Install backend dependencies
   cd backend
   pip install -r requirements.txt
   ```

3. **Set up frontend**
   ```bash
   cd ../frontend
   npm install
   ```

4. **Configure environment variables**
   Copy the production environment template:
   ```bash
   cp production.env.template production.env
   # Edit production.env with your MT5 credentials and API keys
   ```

5. **Download AI models**
   ```powershell
   # Windows PowerShell
   .\download_real_models.ps1
   ```

### Running the Application

#### Production Mode (Recommended)
```bash
# Start the complete production stack
python start_production.py
```

#### Development Mode
1. **Start the backend**
   ```bash
   cd backend
   uvicorn main:app --host 127.0.0.1 --port 8001 --reload
   ```

2. **Start the frontend**
   ```bash
   cd frontend
   npm run dev
   ```

3. **Access the dashboard**
   - Frontend: http://localhost:5175
   - Backend API: http://localhost:8001/docs

## 🛠️ Configuration

### Key Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ARIA_ENABLE_MT5` | Enable MetaTrader 5 integration | `1` |
| `MT5_LOGIN` | MT5 account number | `101984611` |
| `MT5_SERVER` | MT5 server address | `FBS-Demo` |
| `AUTO_TRADE_ENABLED` | Enable automated trading | `1` |
| `AUTO_TRADE_DRY_RUN` | Enable dry-run mode (no real trades) | `0` |
| `ARIA_INCLUDE_XGB` | Include XGBoost model | `1` |
| `ARIA_ENABLE_EXEC` | Enable trade execution | `1` |
| `AUTO_TRADE_SYMBOLS` | Trading symbols | `EURUSD,GBPUSD,USDJPY,XAUUSD,BTCUSD` |
| `ARIA_USE_HMM` | Enable HMM regime detection | `1` |
| `JWT_SECRET_KEY` | Secret key for JWT tokens (32+ chars) | - |
| `ADMIN_API_KEY` | API key for admin endpoints (16+ chars) | - |
| `ARIA_CORS_ORIGINS` | Allowed CORS origins | `http://localhost:5175` |

## 📚 Documentation

### System Architecture

```
ARIA PRO v1.2 - Institutional Architecture
├── Frontend (React 19 + TypeScript 5.8)
│   ├── Neon HUD Dashboard
│   ├── Institutional AI Tab
│   ├── Flow Monitor (Real-time)
│   ├── Orders & Positions
│   ├── Watchlist & Settings
│   └── Performance Analytics
│
├── Backend (FastAPI + Python 3.11)
│   ├── Enhanced SMC Fusion Core
│   ├── 6-Model AI Ensemble
│   ├── AutoTrader Engine
│   ├── Risk Budget Engine
│   ├── Bias Engine
│   ├── Performance Monitor
│   └── WebSocket Broadcaster
│
├── Data Layer
│   ├── MT5 Market Feed
│   ├── Dukascopy Connector
│   ├── Trade Memory Database
│   └── Performance Metrics
│
├── AI Models
│   ├── XGBoost (ONNX)
│   ├── LSTM (ONNX)
│   ├── CNN Patterns (ONNX)
│   ├── PPO Trader (ZIP)
│   ├── Vision AI
│   └── LLM Macro
│
└── Infrastructure
    ├── WebSocket Streaming
    ├── Rate Limiting
    ├── Live Guard Middleware
    ├── Health Monitoring
    └── Security Headers
```

### API Documentation

Run the backend and visit:
- API Docs: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

### Additional Documentation
- [Local Setup Guide](./LOCAL_SETUP.md) - Development environment setup
- [Production Deployment](./ARIA_DEPLOYMENT_INSTRUCTIONS.md) - Production deployment guide
- [Enhanced SMC Implementation](./ENHANCED_FUSION_IMPLEMENTATION.md) - SMC Fusion Core details
- [WebSocket API](./WEBSOCKET_README.md) - Real-time data streaming
- [Production Security Audit](./PRODUCTION_SECURITY_AUDIT.md) - Security implementation
- [Production Readiness Report](./PRODUCTION_READINESS_REPORT.md) - System validation
- [Ports and Connections](./PORTS_AND_CONNECTIONS.md) - Network configuration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact
\ARIA_PRO
For inquiries, please contact [your-email@example.com](mailto:your-email@example.com)

---

<div align="center">
  <p>Made with ❤️ by the ARIA PRO Team</p>
  <p>🚀 Trading the future, today.</p>
</div>
