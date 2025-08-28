"""Script to integrate AI service into ARIA Pro"""
import asyncio
from backend.services.mt5_ai_integration import mt5_ai_integration

async def main():
    print("Starting AI Integration...")
    await mt5_ai_integration.start()
    print("AI Integration started")

if __name__ == "__main__":
    asyncio.run(main())
