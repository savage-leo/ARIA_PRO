#!/usr/bin/env python3
"""
Simple WebSocket test script to verify broadcasting functionality
"""

import asyncio
import json
import random
import time
from datetime import datetime
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from backend.services.ws_broadcaster import (
    broadcaster,
    broadcast_tick,
    broadcast_signal,
    broadcast_order_update,
    broadcast_idea,
    broadcast_prepared_payload,
)


async def simulate_tick_data():
    """Simulate live tick data for major forex pairs"""
    symbols = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]

    while True:
        for symbol in symbols:
            # Generate realistic price movements
            base_price = {
                "EURUSD": 1.0850,
                "GBPUSD": 1.2650,
                "USDJPY": 148.50,
                "AUDUSD": 0.6650,
                "USDCAD": 1.3550,
            }[symbol]

            # Add some random movement
            movement = random.uniform(-0.0010, 0.0010)
            bid = base_price + movement
            ask = bid + random.uniform(0.0001, 0.0003)  # Spread

            await broadcast_tick(symbol, bid, ask)
            print(f"📊 Tick: {symbol} {bid:.5f}/{ask:.5f}")

        await asyncio.sleep(2)  # Update every 2 seconds


async def simulate_trading_signals():
    """Simulate trading signals from AI models"""
    symbols = ["EURUSD", "GBPUSD", "USDJPY"]

    while True:
        for symbol in symbols:
            if random.random() < 0.3:  # 30% chance of signal
                side = random.choice(["buy", "sell"])
                strength = random.uniform(0.6, 0.95)
                confidence = random.uniform(70, 95)

                signal_data = {
                    "symbol": symbol,
                    "side": side,
                    "strength": round(strength, 3),
                    "confidence": round(confidence, 1),
                    "model": random.choice(["LSTM", "SMC_Fusion", "Trap_Detector"]),
                    "timestamp": datetime.now().isoformat(),
                }

                await broadcast_signal(signal_data)
                print(
                    f"🎯 Signal: {side.upper()} {symbol} (Strength: {strength:.3f}, Confidence: {confidence:.1f}%)"
                )

        await asyncio.sleep(5)  # Check for signals every 5 seconds


async def simulate_order_updates():
    """Simulate order status updates"""
    order_statuses = ["pending", "filled", "cancelled", "partial_fill"]

    while True:
        if random.random() < 0.2:  # 20% chance of order update
            symbol = random.choice(["EURUSD", "GBPUSD", "USDJPY"])
            side = random.choice(["buy", "sell"])
            status = random.choice(order_statuses)
            volume = random.uniform(0.01, 1.0)
            price = (
                random.uniform(1.0800, 1.0900)
                if symbol == "EURUSD"
                else random.uniform(1.2600, 1.2700)
            )

            order_data = {
                "order_id": f"ORD_{int(time.time())}_{random.randint(1000, 9999)}",
                "symbol": symbol,
                "side": side,
                "status": status,
                "volume": round(volume, 2),
                "price": round(price, 5),
                "timestamp": datetime.now().isoformat(),
            }

            await broadcast_order_update(order_data)
            print(
                f"📋 Order: {status.upper()} {side.upper()} {symbol} {volume} lots @ {price}"
            )

        await asyncio.sleep(8)  # Order updates every 8 seconds


async def simulate_trading_ideas():
    """Simulate trading ideas from SMC analysis"""
    while True:
        if random.random() < 0.15:  # 15% chance of idea
            symbol = random.choice(["EURUSD", "GBPUSD", "USDJPY"])

            idea_data = {
                "symbol": symbol,
                "type": random.choice(["breakout", "reversal", "continuation"]),
                "timeframe": random.choice(["H1", "H4", "D1"]),
                "confidence": random.uniform(60, 90),
                "entry_price": random.uniform(1.0800, 1.0900),
                "stop_loss": random.uniform(1.0750, 1.0850),
                "take_profit": random.uniform(1.0900, 1.1000),
                "reasoning": f"SMC analysis suggests {random.choice(['bullish', 'bearish'])} momentum",
                "timestamp": datetime.now().isoformat(),
            }

            await broadcast_idea(idea_data)
            print(
                f"💡 Idea: {idea_data['type'].upper()} {symbol} on {idea_data['timeframe']}"
            )

        await asyncio.sleep(10)  # Ideas every 10 seconds


async def main():
    """Main function to run all simulations"""
    print("🚀 Starting WebSocket test broadcaster...")
    print("📡 Broadcasting to connected clients...")
    print("✅ WebSocket broadcaster ready")
    print("🎭 Starting data simulations...")
    print("📊 Press Ctrl+C to stop")

    try:
        # Run all simulations concurrently
        await asyncio.gather(
            simulate_tick_data(),
            simulate_trading_signals(),
            simulate_order_updates(),
            simulate_trading_ideas(),
        )
    except KeyboardInterrupt:
        print("\n🛑 Stopping WebSocket broadcaster...")
        print("✅ WebSocket broadcaster stopped")


if __name__ == "__main__":
    asyncio.run(main())
