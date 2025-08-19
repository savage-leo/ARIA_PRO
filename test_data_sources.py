#!/usr/bin/env python3
"""
Test script to verify data sources are working
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from backend.services.data_source_manager import data_source_manager
from backend.services.ws_broadcaster import broadcaster


async def test_data_sources():
    """Test all data sources"""
    print("🚀 Testing ARIA PRO Data Sources...")
    print("=" * 50)

    try:
        # Start data sources
        print("📡 Starting data sources...")
        await data_source_manager.start_all()

        # Wait a moment for sources to start
        await asyncio.sleep(2)

        # Check status
        status = data_source_manager.get_status()
        print("\n📊 Data Source Status:")
        for source_name, source_status in status["data_sources"].items():
            status_icon = "✅" if source_status["running"] else "❌"
            print(
                f"  {status_icon} {source_name}: {'Running' if source_status['running'] else 'Stopped'}"
            )

        # Monitor WebSocket clients
        print(f"\n🔌 WebSocket Clients: {len(broadcaster.clients)}")

        print("\n🎭 Data sources are running! Check your frontend to see live data.")
        print("📊 Press Ctrl+C to stop...")

        # Keep running
        while True:
            await asyncio.sleep(5)
            client_count = len(broadcaster.clients)
            print(f"🔌 Active WebSocket clients: {client_count}")

    except KeyboardInterrupt:
        print("\n🛑 Stopping data sources...")
        await data_source_manager.stop_all()
        print("✅ Data sources stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        await data_source_manager.stop_all()


if __name__ == "__main__":
    asyncio.run(test_data_sources())
