#!/usr/bin/env python3
"""
ARIA MT5 Production Dashboard
Real-time WebSocket and REST API for monitoring and controlling the trading engine
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import asdict
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
import threading
import queue

# Import the production engine
from aria_mt5_production_engine import ARIAMT5ProductionEngine, TradeDecision, AISignal

# ============================================================================
# DASHBOARD MODELS
# ============================================================================

class DashboardConfig:
    """Dashboard configuration"""
    def __init__(self):
        self.websocket_port = 8765
        self.rest_port = 8000
        self.host = "127.0.0.1"
        self.enable_cors = True

class TradeStatus(BaseModel):
    """Trade status for API responses"""
    symbol: str
    action: str
    confidence: float
    lot_size: float
    status: str
    timestamp: str
    ai_signals: List[Dict]

class EngineStatus(BaseModel):
    """Engine status for API responses"""
    running: bool
    connected: bool
    total_trades: int
    successful_trades: int
    daily_pnl: float
    risk_violations: int
    last_update: str

class AISignalStatus(BaseModel):
    """AI signal status"""
    source: str
    confidence: float
    direction: str
    timestamp: str

# ============================================================================
# DASHBOARD MANAGER
# ============================================================================

class DashboardManager:
    """Manages real-time dashboard updates and connections"""
    
    def __init__(self, engine: ARIAMT5ProductionEngine):
        self.engine = engine
        self.config = DashboardConfig()
        self.websocket_connections: List[WebSocket] = []
        self.message_queue = queue.Queue()
        self.running = False
        
        # Setup logging
        self.logger = logging.getLogger("ARIA_DASHBOARD")
        
        # Initialize FastAPI
        self.app = FastAPI(title="ARIA MT5 Dashboard", version="1.0.0")
        self.setup_routes()
        self.setup_cors()
    
    def setup_cors(self):
        """Setup CORS middleware"""
        if self.config.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
    
    def setup_routes(self):
        """Setup REST API routes"""
        
        @self.app.get("/")
        async def root():
            return {"message": "ARIA MT5 Production Dashboard", "status": "running"}
        
        @self.app.get("/status")
        async def get_engine_status():
            """Get current engine status"""
            return self.get_status_data()
        
        @self.app.get("/trades")
        async def get_recent_trades():
            """Get recent trades"""
            return self.get_recent_trades_data()
        
        @self.app.get("/signals")
        async def get_ai_signals():
            """Get current AI signals"""
            return self.get_ai_signals_data()
        
        @self.app.get("/risk")
        async def get_risk_status():
            """Get risk management status"""
            return self.get_risk_status_data()
        
        @self.app.post("/control/start")
        async def start_engine():
            """Start the trading engine"""
            try:
                if not self.engine.running:
                    threading.Thread(target=self.engine.start, daemon=True).start()
                    return {"status": "success", "message": "Engine starting"}
                else:
                    return {"status": "error", "message": "Engine already running"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/control/stop")
        async def stop_engine():
            """Stop the trading engine"""
            try:
                self.engine.stop()
                return {"status": "success", "message": "Engine stopped"}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(self, websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                # Send initial status
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "data": self.get_status_data()
                }))
                
                # Keep connection alive and send updates
                while True:
                    try:
                        # Wait for messages from queue
                        message = self.message_queue.get(timeout=1)
                        await websocket.send_text(json.dumps(message))
                    except queue.Empty:
                        # Send heartbeat
                        await websocket.send_text(json.dumps({
                            "type": "heartbeat",
                            "timestamp": datetime.now().isoformat()
                        }))
                        
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)
    
    def get_status_data(self) -> Dict:
        """Get current engine status data"""
        try:
            account_info = self.engine.mt5_connection.get_market_data("EURUSD")
            balance = account_info.get("balance", 0) if account_info else 0
            
            return {
                "running": self.engine.running,
                "connected": self.engine.mt5_connection.connected,
                "total_trades": len(self.engine.trade_memory.memory.get("trades", [])),
                "successful_trades": sum(
                    1 for trade in self.engine.trade_memory.memory.get("trades", [])
                    if trade.get("mt5_result", {}).get("retcode") == 10009
                ),
                "daily_pnl": 0.0,  # Calculate from trade memory
                "risk_violations": len(self.engine.trade_memory.memory.get("risk_violations", [])),
                "account_balance": balance,
                "last_update": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting status: {e}")
            return {"error": str(e)}
    
    def get_recent_trades_data(self) -> List[Dict]:
        """Get recent trades data"""
        try:
            trades = self.engine.trade_memory.memory.get("trades", [])
            return trades[-10:]  # Last 10 trades
        except Exception as e:
            self.logger.error(f"Error getting trades: {e}")
            return []
    
    def get_ai_signals_data(self) -> Dict:
        """Get current AI signals data"""
        try:
            # Get latest signals for each symbol
            signals_data = {}
            for symbol in self.engine.config.trade_symbols:
                market_data = self.engine.mt5_connection.get_market_data(symbol)
                if market_data:
                    ensemble_signal = self.engine.ai_generator.generate_ensemble_signal(symbol, market_data)
                    signals_data[symbol] = {
                        "ensemble": {
                            "confidence": ensemble_signal.confidence,
                            "direction": ensemble_signal.direction.value,
                            "timestamp": datetime.fromtimestamp(ensemble_signal.timestamp).isoformat()
                        },
                        "individual": [
                            {
                                "source": signal.source.value,
                                "confidence": signal.confidence,
                                "direction": signal.direction.value
                            }
                            for signal in ensemble_signal.metadata.get("individual_signals", [])
                        ]
                    }
            return signals_data
        except Exception as e:
            self.logger.error(f"Error getting signals: {e}")
            return {}
    
    def get_risk_status_data(self) -> Dict:
        """Get risk management status"""
        try:
            return {
                "daily_drawdown": 0.0,  # Calculate from account info
                "open_positions": len(self.engine.mt5_connection.get_market_data("EURUSD") or []),
                "risk_violations": self.engine.trade_memory.memory.get("risk_violations", []),
                "symbol_stats": self.engine.trade_memory.memory.get("symbols", {}),
                "last_update": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error getting risk status: {e}")
            return {"error": str(e)}
    
    def broadcast_update(self, update_type: str, data: Dict):
        """Broadcast update to all WebSocket connections"""
        message = {
            "type": update_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to queue for WebSocket processing
        self.message_queue.put(message)
        
        # Also log important updates
        if update_type in ["trade_executed", "risk_violation", "ai_signal"]:
            self.logger.info(f"Dashboard update: {update_type} - {data}")
    
    def start(self):
        """Start the dashboard"""
        self.running = True
        self.logger.info(f"Starting ARIA MT5 Dashboard on {self.config.host}:{self.config.rest_port}")
        
        # Start FastAPI server
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.rest_port,
            log_level="info"
        )
    
    def stop(self):
        """Stop the dashboard"""
        self.running = False
        self.logger.info("ARIA MT5 Dashboard stopped")

# ============================================================================
# INTEGRATION WITH PRODUCTION ENGINE
# ============================================================================

class DashboardAwareEngine(ARIAMT5ProductionEngine):
    """Production engine with dashboard integration"""
    
    def __init__(self):
        super().__init__()
        self.dashboard = DashboardManager(self)
        self.dashboard_thread = None
    
    def start(self):
        """Start engine with dashboard"""
        # Start dashboard in separate thread
        self.dashboard_thread = threading.Thread(target=self.dashboard.start, daemon=True)
        self.dashboard_thread.start()
        
        # Start main engine
        super().start()
    
    def stop(self):
        """Stop engine and dashboard"""
        self.dashboard.stop()
        super().stop()
    
    def _process_symbol(self, symbol: str):
        """Process symbol with dashboard updates"""
        try:
            # Get market data
            market_data = self.mt5_connection.get_market_data(symbol)
            if not market_data:
                return
            
            # Generate AI signals
            ensemble_signal = self.ai_generator.generate_ensemble_signal(symbol, market_data)
            
            # Broadcast AI signal to dashboard
            self.dashboard.broadcast_update("ai_signal", {
                "symbol": symbol,
                "confidence": ensemble_signal.confidence,
                "direction": ensemble_signal.direction.value,
                "source": ensemble_signal.source.value
            })
            
            # Log AI signal
            self.logger.log_ai_signal(ensemble_signal, symbol)
            
            # Create trade decision
            trade_decision = self._create_trade_decision(symbol, ensemble_signal, market_data)
            
            # Validate trade
            is_valid, reason = self.risk_manager.validate_trade(symbol, trade_decision)
            
            if not is_valid:
                self.logger.log_risk_violation(symbol, reason)
                # Broadcast risk violation
                self.dashboard.broadcast_update("risk_violation", {
                    "symbol": symbol,
                    "reason": reason,
                    "timestamp": datetime.now().isoformat()
                })
                return
            
            # Execute trade
            if self.config.live_trading:
                result = self.mt5_connection.execute_trade(trade_decision)
                
                # Record trade
                self.trade_memory.record_trade(trade_decision, result)
                self.logger.log_trade(trade_decision, result)
                
                # Broadcast trade execution
                self.dashboard.broadcast_update("trade_executed", {
                    "symbol": symbol,
                    "action": trade_decision.action.value,
                    "lot_size": trade_decision.lot_size,
                    "confidence": trade_decision.confidence,
                    "result": result.get("retcode", "UNKNOWN"),
                    "timestamp": datetime.now().isoformat()
                })
            else:
                self.logger.logger.info(f"DRY_RUN: Would execute {trade_decision.action.value} {symbol}")
                
                # Broadcast dry run
                self.dashboard.broadcast_update("dry_run", {
                    "symbol": symbol,
                    "action": trade_decision.action.value,
                    "lot_size": trade_decision.lot_size,
                    "confidence": trade_decision.confidence,
                    "timestamp": datetime.now().isoformat()
                })
            
        except Exception as e:
            self.logger.logger.error(f"Error processing {symbol}: {e}")
            
            # Broadcast error
            self.dashboard.broadcast_update("error", {
                "symbol": symbol,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            })

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for dashboard-enabled engine"""
    try:
        # Create dashboard-aware engine
        engine = DashboardAwareEngine()
        
        # Start the engine (includes dashboard)
        engine.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

