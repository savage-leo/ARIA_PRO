"""Test comprehensive MT5 AI Integration with all models"""
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / 'backend'))

from services.mt5_ai_integration import mt5_ai_integration

async def test_comprehensive_ai():
    print("Testing comprehensive MT5 AI Integration...")
    
    try:
        # Start the AI integration service
        await mt5_ai_integration.start()
        print("MT5 AI Integration started - processing with all AI models")
        
        # Let it run for 60 seconds to see AI signals
        print("Running AI analysis for 60 seconds...")
        await asyncio.sleep(60)
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        await mt5_ai_integration.stop()
        print("AI Integration test completed")

if __name__ == "__main__":
    asyncio.run(test_comprehensive_ai())
