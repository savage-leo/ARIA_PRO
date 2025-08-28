# Replace backend/services/data_source_manager.py or overwrite relevant functions
# Ensure no demo/mock fallback when MT5 enabled
import os
import asyncio
import logging
from typing import Any, Dict

logger = logging.getLogger("DATA.SRC")


class FeedUnavailableError(RuntimeError):
    pass


class DataSourceManager:
    def __init__(self, mt5_feed=None, ai_signal_generator=None):
        self.mt5_feed = mt5_feed
        self.ai_signal_generator = ai_signal_generator
        self.running = False
        self.data_sources = []

    def require_live_mode(self):
        # when env enables MT5, disallow mock fallbacks
        if os.environ.get("ARIA_ENABLE_MT5", "0") == "1" and (
            self.mt5_feed is None or not getattr(self.mt5_feed, "_connected", False)
        ):
            raise FeedUnavailableError("MT5 enabled but MT5 feed not connected")

    def get_last_bar(self, symbol: str) -> Dict:
        # prefer real MT5 feed
        if self.mt5_feed and getattr(self.mt5_feed, "_connected", False):
            # expect mt5_feed to provide get_last_bar or build a bar externally via bar_builder
            if hasattr(self.mt5_feed, "get_last_bar"):
                return self.mt5_feed.get_last_bar(symbol)
            # otherwise raise to force correct integration
            raise FeedUnavailableError(
                "MT5 feed does not implement get_last_bar; implement bar_builder"
            )
        # Strict live-only policy: no fallbacks allowed
        raise FeedUnavailableError("MT5 feed not connected; live-only mode enforced")

    def get_ai_signals(self, symbol: str, features: Dict) -> Dict:
        # require ai_signal_generator (real models) when MT5 enabled
        if os.environ.get("ARIA_ENABLE_MT5", "0") == "1":
            if self.ai_signal_generator is None:
                raise FeedUnavailableError(
                    "AI signal generator required when MT5 enabled"
                )
            return self.ai_signal_generator.get_signals(symbol, features)
        # dev mode: if generator is available, use it; otherwise empty
        if self.ai_signal_generator is not None:
            return self.ai_signal_generator.get_signals(symbol, features)
        return {}

    async def start_all(self):
        """Start all data sources"""
        if self.running:
            logger.warning("Data sources are already running")
            return

        self.running = True
        logger.info("Starting all data sources...")

        try:
            # Start all data sources concurrently
            tasks = []
            for source in self.data_sources:
                task = asyncio.create_task(source.start())
                tasks.append(task)

            # Wait for all to start
            await asyncio.gather(*tasks, return_exceptions=True)

            logger.info("All data sources started successfully")

        except Exception as e:
            logger.error(f"Error starting data sources: {e}")
            self.running = False

    async def stop_all(self):
        """Stop all data sources"""
        if not self.running:
            logger.warning("Data sources are not running")
            return

        self.running = False
        logger.info("Stopping all data sources...")

        try:
            # Stop all data sources
            for source in self.data_sources:
                await source.stop()

            logger.info("All data sources stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping data sources: {e}")

    def get_status(self) -> dict:
        """Get status of all data sources"""
        status = {"manager_running": self.running, "data_sources": {}}

        for source in self.data_sources:
            source_name = source.__class__.__name__
            status["data_sources"][source_name] = {
                "running": getattr(source, "running", False)
            }

        return status


# Global instance
data_source_manager = DataSourceManager()

def get_data_source_manager() -> DataSourceManager:
    """Get the global data source manager instance"""
    return data_source_manager
