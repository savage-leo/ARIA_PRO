"""
System Monitor for ARIA Pro Backend
Monitors system resources and performance metrics
"""

import psutil
import logging
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SystemMonitor:
    """System resource monitoring"""
    
    def __init__(self):
        self.is_running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start system monitoring"""
        if self.is_running:
            return
            
        self.is_running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("System monitor started")
    
    async def start_monitoring(self):
        """Compatibility shim - call the start method"""
        return await self.start()
        
    async def stop(self):
        """Stop system monitoring"""
        self.is_running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("System monitor stopped")
        
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                metrics = self.get_system_metrics()
                # Log critical metrics
                if metrics['memory_percent'] > 90:
                    logger.warning(f"High memory usage: {metrics['memory_percent']:.1f}%")
                if metrics['cpu_percent'] > 90:
                    logger.warning(f"High CPU usage: {metrics['cpu_percent']:.1f}%")
                    
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Error in system monitor loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error
                
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        try:
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_gb': psutil.virtual_memory().used / (1024**3),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'disk_percent': psutil.disk_usage('/').percent if hasattr(psutil.disk_usage('/'), 'percent') else 0,
                'network_sent_mb': psutil.net_io_counters().bytes_sent / (1024**2),
                'network_recv_mb': psutil.net_io_counters().bytes_recv / (1024**2),
                'process_count': len(psutil.pids()),
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'cpu_percent': 0,
                'memory_percent': 0,
                'memory_used_gb': 0,
                'memory_total_gb': 0,
                'disk_percent': 0,
                'network_sent_mb': 0,
                'network_recv_mb': 0,
                'process_count': 0,
                'load_average': [0, 0, 0],
                'error': str(e)
            }

# Global instance
system_monitor = SystemMonitor()

def get_system_monitor():
    """Get the global system monitor instance"""
    return system_monitor
