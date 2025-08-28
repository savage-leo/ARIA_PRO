"""Test script for MT5 AI Integration"""
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / 'backend'))

from services.mt5_ai_integration import mt5_ai_integration

async def test_ai_integration():
    print("Starting MT5 AI Integration test...")
    
    try:
        # Start the AI integration service
        await mt5_ai_integration.start()
        print("MT5 AI Integration started successfully")
        
        # Keep running for a while to process some ticks
        print("Processing market data for 30 seconds...")
        await asyncio.sleep(30)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        await mt5_ai_integration.stop()
        print("Test completed")

if __name__ == "__main__":
    asyncio.run(test_ai_integration())
