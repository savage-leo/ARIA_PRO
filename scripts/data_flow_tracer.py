#!/usr/bin/env python3
"""
Data Flow Tracer: Maps how data enters ARIA PRO system from API/MT5 through signal generation
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def trace_data_flow():
    print("=== ARIA PRO Data Flow Analysis ===\n")

    print("1. DATA ENTRY POINTS:")
    print("   a) MT5 Connection (MetaTrader5 package)")
    print("      - backend/services/market_data_feed.py")
    print("      - Fetches OHLCV data via MT5.copy_rates_from()")
    print("      - Controlled by ARIA_ENABLE_MT5 env var")
    print("   b) REST API Endpoints")
    print("      - /market/data - Manual market data injection")
    print("      - /signals/generate - Direct signal generation")
    print("      - /trading/execute - Manual trade execution")
    print("   c) WebSocket Feed")
    print("      - /ws - Real-time data streaming")
    print("      - backend/routes/websocket.py")

    print("\n2. DATA PROCESSING PIPELINE:")
    print("   Raw Data → Data Source Manager → Signal Generators → Trade Memory")
    print("   ")
    print("   a) DataSourceManager (backend/services/data_source_manager.py)")
    print("      - Aggregates data from multiple sources")
    print("      - Manages lifecycle of data feeds")
    print("      - Registered sources: market_data_feed, cpp_service")
    print("   ")
    print("   b) Signal Generation Chain:")
    print("      - RealAISignalGenerator (backend/services/signal_generator.py)")
    print("      - Uses XGBoost + LSTM models (backend/models/)")
    print("      - Enhanced SMC Fusion Core (backend/core/enhanced_smc_fusion.py)")
    print("      - BiasEngine (backend/core/bias_engine.py)")
    print("   ")
    print("   c) Trade Execution Path:")
    print("      - AutoTrader (backend/services/auto_trader.py)")
    print("      - Processes signals when |score| >= threshold")
    print("      - ATR-based SL/TP calculation")
    print("      - MT5 trade execution via backend/services/mt5_service.py")

    print("\n3. DATA FLOW SEQUENCE:")
    print("   Step 1: Market Data Ingestion")
    print("   ├─ MT5: copy_rates_from() → OHLCV bars")
    print("   ├─ API: POST /market/data → manual injection")
    print("   └─ WebSocket: real-time streaming")
    print("   ")
    print("   Step 2: Feature Engineering")
    print("   ├─ RealAISignalGenerator._extract_features()")
    print("   ├─ XGBoost: 6D features (momentum x4, volatility, z-score)")
    print("   └─ LSTM: sequence processing for temporal patterns")
    print("   ")
    print("   Step 3: Signal Generation")
    print("   ├─ AI Models → raw predictions")
    print("   ├─ Enhanced SMC Fusion → EnhancedTradeIdea")
    print("   ├─ BiasEngine → bias_factor adjustment")
    print("   └─ Confidence scaling: idea.confidence * bias_factor")
    print("   ")
    print("   Step 4: Trade Decision")
    print("   ├─ AutoTrader._process_symbol()")
    print("   ├─ Threshold gating: |score| >= AUTO_TRADE_THRESHOLD")
    print("   ├─ Risk checks: position size, drawdown limits")
    print("   └─ MT5 execution or dry-run logging")
    print("   ")
    print("   Step 5: Memory & Tracking")
    print("   ├─ TradeMemory.insert_trade_idea() → SQLite storage")
    print("   ├─ Outcome tracking via /trade-memory/{id}/outcome")
    print("   └─ Performance analytics")

    print("\n4. KEY INTEGRATION POINTS:")
    print("   - backend/main.py: FastAPI app with all routers")
    print("   - backend/services/data_source_manager: Central data orchestration")
    print("   - backend/core/trade_memory.py: Persistent idea/outcome storage")
    print("   - backend/services/auto_trader.py: Automated execution engine")
    print("   - frontend/src/: React UI consuming REST/WebSocket APIs")

    print("\n5. ENVIRONMENT CONTROLS:")
    print("   - ARIA_ENABLE_MT5: Enable real MT5 connection")
    print("   - AUTO_TRADE_ENABLED: Enable automated trading")
    print("   - AUTO_TRADE_THRESHOLD: Signal confidence threshold")
    print("   - ARIA_SYMBOLS: Comma-separated trading pairs")
    print("   - TRADE_DB: SQLite path for trade memory")


if __name__ == "__main__":
    trace_data_flow()
