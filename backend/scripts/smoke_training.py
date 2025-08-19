import asyncio
import os
import sys

# Ensure project root on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from backend.core.training_connector import TrainingConnector


async def main():
    # Ensure demo-safe
    os.environ.setdefault("ARIA_ASSUME_DEMO", "1")
    os.environ.setdefault("ARIA_ALLOW_LIVE_TRADING", "0")

    connector = TrainingConnector()
    # Quick dry-run style training using placeholder trainers
    results = await connector.train_model(
        model_type="lstm",
        symbol="XAUUSD",
        timeframe="M5",
        days_back=1,
        batch_size=16,
        epochs=1,
    )
    print("Smoke training results:", results)


if __name__ == "__main__":
    asyncio.run(main())
