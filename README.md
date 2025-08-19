# ARIA Institutional Pro v1.2

[![TypeScript](https://img.shields.io/badge/TypeScript-5.8-blue)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-19-61dafb)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688)](https://fastapi.tiangolo.com/)
[![Material-UI](https://img.shields.io/badge/MUI-7-007fff)](https://mui.com/)

**Real-time Institutional-Grade Forex AI Trading Platform**

> Note: This repository is configured for a local-only workflow. Docker is not used. See LOCAL_SETUP.md for step-by-step instructions.

## 🚀 Features

- **Real-time MT5 Integration** - Live market data and trade execution
- **Advanced AI Signal Generation** - LSTM, CNN, PPO, Visual AI, and LLM Macro signals
- **Enhanced SMC Analysis** - Smart Money Concepts with edge detection
- **Institutional Dashboard** - Professional trading interface with Material-UI
- **Risk Management** - Advanced position sizing and risk controls
- **WebSocket Real-time Updates** - Live market feeds and trade notifications
- **Multi-timeframe Analysis** - Comprehensive market structure analysis

## 🏗️ Architecture

```
ARIA_PRO/
├── backend/           # FastAPI + MT5 + AI Models
├── frontend/          # React 19 + TypeScript + MUI
├── cpp_core/          # High-performance C++ components
├── scripts/           # Automation and deployment scripts
├── logs/              # Application logs
└── docs/              # Documentation
```

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **MetaTrader5** - Trading platform integration
- **TensorFlow/PyTorch** - AI model inference
- **WebSockets** - Real-time communication
- **SQLAlchemy** - Database ORM

### Frontend
- **React 19** - Modern UI framework
- **TypeScript 5.8** - Type-safe development
- **Material-UI 7** - Professional component library
- **Redux Toolkit** - State management
- **React Router 7** - Navigation
- **Vite** - Fast build tool

### Infrastructure
- **C++ Core** - High-performance calculations
- **Nginx** (optional) - Reverse proxy for local HTTPS
- **PostgreSQL** (optional) - Local database if needed

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- MetaTrader5 Terminal
- Visual Studio Build Tools (for C++ components)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-org/aria-institutional-pro.git
cd aria-institutional-pro
```

2. **Backend Setup**
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

3. **Frontend Setup**
```bash
cd frontend
npm install
```

4. **Environment Configuration**
```bash
cp .env.example .env
# Configure your MT5 credentials and API keys
```

5. **Start Development**
```bash
# Terminal 1 - Backend
python start_backend.py

# Terminal 2 - Frontend
cd frontend && npm run dev
```

## 📊 Trading Features

### AI Signal Generation
- **LSTM Networks** - Time series prediction with trend analysis
- **CNN Models** - Pattern recognition and candlestick analysis
- **PPO Agents** - Reinforcement learning for optimal entry/exit
- **Visual AI** - Chart pattern recognition
- **LLM Macro** - Fundamental analysis with sentiment scoring

### Smart Money Concepts (SMC)
- **Order Block Detection** - Institutional order flow analysis
- **Fair Value Gaps** - Imbalance identification
- **Liquidity Analysis** - Volume and volatility scoring
- **Market Structure** - Higher highs/lows tracking
- **Support/Resistance** - Dynamic level calculation

### Risk Management
- **Position Sizing** - Kelly Criterion and fixed fractional
- **Stop Loss Management** - ATR-based and technical levels
- **Portfolio Correlation** - Cross-pair risk assessment
- **Drawdown Protection** - Dynamic position reduction

## 🔧 Development

### Code Quality
- **TypeScript Strict Mode** - Zero tolerance for `any` types
- **ESLint + Prettier** - Automated code formatting
- **Pre-commit Hooks** - Quality gates before commits
- **Unit Testing** - Comprehensive test coverage

### Build Commands
```bash
# Frontend
npm run build          # Production build
npm run type-check     # TypeScript validation
npm run lint           # Code linting
npm run test           # Run tests

# Backend
python -m pytest      # Run all tests
python -m mypy .       # Type checking
python -m black .      # Code formatting
```

## 📈 Performance

- **Sub-millisecond Latency** - C++ core for critical calculations
- **Real-time Updates** - WebSocket streaming at 60fps
- **Memory Optimized** - Efficient data structures
- **Scalable Architecture** - Microservices ready

## 🔐 Security

- **API Authentication** - JWT tokens with refresh
- **Rate Limiting** - DDoS protection
- **Input Validation** - Pydantic models
- **Secure Communications** - TLS/SSL encryption

## 📚 Documentation

- [API Documentation](docs/api.md)
- [Frontend Components](docs/frontend.md)
- [Trading Strategies](docs/strategies.md)
- [Deployment Guide](docs/deployment.md)
 - [Local Setup](LOCAL_SETUP.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is proprietary software. All rights reserved.

## 🆘 Support

For technical support and questions:
- Email: support@aria-institutional.com
- Documentation: [docs.aria-institutional.com](https://docs.aria-institutional.com)
- Issues: [GitHub Issues](https://github.com/your-org/aria-institutional-pro/issues)

---

**Built with ❤️ for institutional traders**
