"""Performance monitoring for ARIA's ML models and system components."""

import time
import psutil
import numpy as np
import threading
import functools
import os
import inspect
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import json
import logging
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class CPUMetrics:
    """Track CPU-specific metrics for model inference."""
    thread_id: int
    model_name: str
    start_time: float
    cpu_percent: List[float] = field(default_factory=list)
    cpu_times: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    last_measurement: float = field(default_factory=time.time)

@dataclass
class ModelMetrics:
    """Track performance metrics for a single model."""
    model_name: str
    call_count: int = 0
    total_latency: float = 0.0
    min_latency: float = float('inf')
    max_latency: float = 0.0
    ema_latency: Optional[float] = None
    ema_alpha: float = 0.1  # Smoothing factor for EMA
    memory_usage: List[float] = field(default_factory=list)
    cpu_percent: List[float] = field(default_factory=list)
    errors: int = 0
    last_call_time: float = 0.0
    cpu_metrics_history: List[Dict] = field(default_factory=list)
    
    def update(self, latency_ms: float, cpu_metrics: Optional[Dict] = None):
        """Update metrics with new measurement."""
        self.call_count += 1
        self.total_latency += latency_ms
        self.min_latency = min(self.min_latency, latency_ms)
        self.max_latency = max(self.max_latency, latency_ms)
        
        # Update EMA
        if self.ema_latency is None:
            self.ema_latency = latency_ms
        else:
            self.ema_latency = (1 - self.ema_alpha) * self.ema_latency + self.ema_alpha * latency_ms
        
        # Track CPU metrics if provided
        if cpu_metrics:
            self.cpu_percent.append(cpu_metrics.get('cpu_percent_avg', 0.0))
            self.memory_usage.append(cpu_metrics.get('memory_mb_avg', 0.0))
            self.cpu_metrics_history.append(cpu_metrics)
        else:
            # Fallback to current system metrics
            process = psutil.Process()
            self.cpu_percent.append(psutil.cpu_percent())
            self.memory_usage.append(process.memory_info().rss / 1024 / 1024)
        
        self.last_call_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            "model": self.model_name,
            "calls": self.call_count,
            "latency_ms": {
                "min": round(self.min_latency, 2),
                "max": round(self.max_latency, 2),
                "avg": round(self.total_latency / max(1, self.call_count), 2),
                "ema": round(self.ema_latency, 2) if self.ema_latency is not None else 0.0,
            },
            "memory_mb": {
                "current": round(self.memory_usage[-1], 2) if self.memory_usage else 0.0,
                "avg": round(np.mean(self.memory_usage), 2) if self.memory_usage else 0.0,
                "p95": round(np.percentile(self.memory_usage, 95), 2) if self.memory_usage else 0.0,
            },
            "cpu_percent": {
                "current": round(self.cpu_percent[-1], 2) if self.cpu_percent else 0.0,
                "avg": round(np.mean(self.cpu_percent), 2) if self.cpu_percent else 0.0,
                "p95": round(np.percentile(self.cpu_percent, 95), 2) if self.cpu_percent else 0.0,
            },
            "errors": self.errors,
            "last_call": datetime.fromtimestamp(self.last_call_time).isoformat(),
        }

class CPUMonitor:
    """Monitor CPU and thread-level metrics."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.thread_metrics: Dict[int, CPUMetrics] = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def start_tracking(self, model_name: str) -> int:
        """Start tracking a model's CPU usage in a separate thread."""
        thread_id = threading.get_ident()
        with self.lock:
            self.thread_metrics[thread_id] = CPUMetrics(
                thread_id=thread_id,
                model_name=model_name,
                start_time=time.time()
            )
        # Start measuring in the available context (async loop or thread fallback)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.measure(thread_id))
        except RuntimeError:
            self.executor.submit(self._measure_sync, thread_id)
        return thread_id
        
    async def measure(self, thread_id: int):
        """Measure CPU usage in a loop for the given thread."""
        metrics = self.thread_metrics.get(thread_id)
        if not metrics:
            return
            
        while thread_id in self.thread_metrics:
            try:
                with self.process.oneshot():
                    # Non-blocking CPU measurement
                    cpu_percent = self.process.cpu_percent(interval=None)
                    cpu_times = self.process.cpu_times()
                    memory_mb = self.process.memory_info().rss / 1024 / 1024
                    
                with self.lock:
                    if thread_id in self.thread_metrics:
                        # Only append non-zero CPU readings
                        if cpu_percent > 0:
                            metrics.cpu_percent.append(cpu_percent)
                        metrics.cpu_times.append(sum(cpu_times))
                        metrics.memory_usage.append(memory_mb)
                        metrics.last_measurement = time.time()
                        
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"CPU monitoring error: {e}")
                break
                
    def _measure_sync(self, thread_id: int):
        """Synchronous fallback for CPU measurement when no event loop is running."""
        metrics = self.thread_metrics.get(thread_id)
        if not metrics:
            return
        # Initialize for delta calculation
        last_cpu_times = None
        last_time = time.time()
        
        while thread_id in self.thread_metrics:
            try:
                with self.process.oneshot():
                    # Non-blocking CPU measurement
                    cpu_percent = self.process.cpu_percent(interval=None)
                    cpu_times = self.process.cpu_times()
                    memory_mb = self.process.memory_info().rss / 1024 / 1024
                    
                # Calculate CPU percentage from delta if possible
                if last_cpu_times is not None:
                    current_time = time.time()
                    time_delta = current_time - last_time
                    if time_delta > 0:
                        cpu_total_delta = sum(cpu_times) - sum(last_cpu_times)
                        cpu_cores = psutil.cpu_count() or 1
                        cpu_percent_calc = (cpu_total_delta / time_delta) * 100 / cpu_cores
                        cpu_percent = max(cpu_percent, cpu_percent_calc)
                    last_time = current_time
                last_cpu_times = cpu_times
                    
                with self.lock:
                    if thread_id in self.thread_metrics:
                        # Only append non-zero CPU readings
                        if cpu_percent > 0:
                            metrics.cpu_percent.append(cpu_percent)
                        metrics.cpu_times.append(sum(cpu_times))
                        metrics.memory_usage.append(memory_mb)
                        metrics.last_measurement = time.time()
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"CPU monitoring error (sync): {e}")
                break
                
    def stop_tracking(self, thread_id: int) -> Optional[Dict]:
        """Stop tracking and return final metrics."""
        with self.lock:
            metrics = self.thread_metrics.pop(thread_id, None)
            
        if metrics:
            # Filter out zero CPU readings for more accurate stats
            valid_cpu = [c for c in metrics.cpu_percent if c > 0]
            return {
                "cpu_percent_avg": np.mean(valid_cpu) if valid_cpu else 0.0,
                "cpu_percent_max": max(valid_cpu) if valid_cpu else 0.0,
                "memory_mb_avg": np.mean(metrics.memory_usage) if metrics.memory_usage else 0.0,
                "memory_mb_max": max(metrics.memory_usage) if metrics.memory_usage else 0.0,
                "duration_sec": time.time() - metrics.start_time
            }
        return None

class PerformanceMonitor:
    """Monitor performance of ML models and system components."""
    
    def __init__(self, max_metrics: int = 1000):
        self.models: Dict[str, ModelMetrics] = {}
        self.cpu_monitor = CPUMonitor()
        self.max_metrics = max_metrics
        # Thread-safe lock for synchronous operations
        self._thread_lock = threading.Lock()
        # Lazily created within the event loop context to avoid cross-loop usage
        self._lock: Optional[asyncio.Lock] = None
        self._lock_loop: Optional[asyncio.AbstractEventLoop] = None
        self.start_time = time.time()
        self.active_connections: List[Any] = []
        self._monitoring_task = None
        # Event loop used for scheduling monitor tasks; ensures consistency
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        # Per-symbol metrics: symbol -> model -> ModelMetrics
        self.symbol_metrics: Dict[str, Dict[str, ModelMetrics]] = {}
        # Environment-configurable thresholds with sensible defaults
        self.thresholds = {
            "cpu_warn_percent": float(os.environ.get("ARIA_THRESH_CPU_WARN", "90")),
            "mem_available_percent_crit": float(os.environ.get("ARIA_THRESH_MEM_AVAIL_CRIT", "10")),
            "latency_warn_ms": float(os.environ.get("ARIA_THRESH_LATENCY_WARN_MS", "1000")),
        }
        
    def _in_pm_loop(self) -> bool:
        """Return True if currently executing inside the monitor's event loop."""
        try:
            return asyncio.get_running_loop() is self._loop
        except RuntimeError:
            return False

    async def _run_in_pm_loop(self, coro):
        """Run the given coroutine on the monitor's loop, awaiting completion.
        If called from the same loop, executes directly. If no loop is bound,
        executes in the current loop.
        """
        if self._loop and self._loop.is_running() and not self._in_pm_loop():
            cf = asyncio.run_coroutine_threadsafe(coro, self._loop)
            try:
                # Bridge to the current loop safely
                return await asyncio.wrap_future(cf)
            except RuntimeError:
                # No running loop in this thread; block for result
                return cf.result()
        # Fallback: run in the current loop
        return await coro
        
    async def start_monitoring(self):
        """Start background monitoring tasks."""
        if self._monitoring_task is None:
            # Bind to the currently running loop
            self._loop = asyncio.get_running_loop()
            self._monitoring_task = asyncio.create_task(self._monitor_system())

    def _loop_worker(self, loop: asyncio.AbstractEventLoop) -> None:
        """Run an event loop forever in a background thread."""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def _ensure_background_loop(self) -> None:
        """Ensure there is a dedicated background loop available and running."""
        if self._loop and self._loop.is_running():
            return
        loop = asyncio.new_event_loop()
        t = threading.Thread(target=self._loop_worker, args=(loop,), daemon=True)
        t.start()
        self._loop = loop
        
    def _schedule_update(self, coro):
        """Schedule an async update safely, even if no event loop is running."""
        # Prefer the internal loop if present
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self._loop)
            return
        # Otherwise, try to use any current running loop (e.g., FastAPI/uvicorn loop)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(coro)
            self._loop = loop
        except RuntimeError:
            # No running loop in this thread; spin up a background loop
            self._ensure_background_loop()
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        
    async def _monitor_system(self):
        """Background task to monitor system metrics."""
        while True:
            try:
                system_metrics = self.get_system_metrics()
                await self._broadcast_metrics({
                    "type": "system_metrics",
                    "metrics": system_metrics
                })
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(5)
    
    @contextmanager
    def track_model(self, model_name: str, symbol: Optional[str] = None):
        """Context manager for tracking model inference with CPU metrics."""
        thread_id = self.cpu_monitor.start_tracking(model_name)
        start_time = time.time()
        
        try:
            yield
        finally:
            # Get CPU metrics
            cpu_metrics = self.cpu_monitor.stop_tracking(thread_id) or {}
            latency_ms = (time.time() - start_time) * 1000
            
            # Update model metrics
            self._schedule_update(
                self._update_metrics(model_name, latency_ms, cpu_metrics, symbol)
            )
    
    async def _update_metrics_impl(self, model_name: str, latency_ms: float, cpu_metrics: Dict, symbol: Optional[str] = None):
        """Implementation: update metrics for a model on the monitor's loop."""
        current_loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not current_loop:
            # Create/recreate lock in the active loop context
            self._lock = asyncio.Lock()
            self._lock_loop = current_loop
        async with self._lock:
            # Use thread lock for models dict access
            with self._thread_lock:
                if model_name not in self.models:
                    self.models[model_name] = ModelMetrics(model_name=model_name)
            try:
                self.models[model_name].update(latency_ms, cpu_metrics)

                # Truncate old metrics to prevent memory leaks
                metrics = self.models[model_name]
                for metric_list in [metrics.cpu_percent, metrics.memory_usage]:
                    if len(metric_list) > self.max_metrics:
                        metric_list[:] = metric_list[-self.max_metrics:]

                # Broadcast update
                await self._broadcast_metrics({
                    "type": "model_metrics",
                    "model": model_name,
                    "metrics": metrics.to_dict(),
                })

                # Per-symbol metrics update if symbol provided
                if symbol:
                    with self._thread_lock:
                        if symbol not in self.symbol_metrics:
                            self.symbol_metrics[symbol] = {}
                        if model_name not in self.symbol_metrics[symbol]:
                            self.symbol_metrics[symbol][model_name] = ModelMetrics(model_name=model_name)
                    sym_metrics = self.symbol_metrics[symbol][model_name]
                    sym_metrics.update(latency_ms, cpu_metrics)
                    # Truncate symbol metric lists as well
                    for metric_list in [sym_metrics.cpu_percent, sym_metrics.memory_usage]:
                        if len(metric_list) > self.max_metrics:
                            metric_list[:] = metric_list[-self.max_metrics:]
                    # Broadcast symbol-level update
                    await self._broadcast_metrics({
                        "type": "symbol_metrics",
                        "symbol": symbol,
                        "model": model_name,
                        "metrics": sym_metrics.to_dict(),
                    })
            except Exception as e:
                logger.error(f"Error updating metrics for {model_name}: {e}")
                with self._thread_lock:
                    if model_name in self.models:
                        self.models[model_name].errors += 1

    async def _update_metrics(self, model_name: str, latency_ms: float, cpu_metrics: Dict, symbol: Optional[str] = None):
        """Public API used by tests and track_model: route to monitor loop if needed."""
        return await self._run_in_pm_loop(
            self._update_metrics_impl(model_name, latency_ms, cpu_metrics, symbol)
        )
    
    async def track_model_async(
        self, 
        model_name: str, 
        latency_ms: float, 
        cpu_metrics: Optional[Dict] = None,
        symbol: Optional[str] = None,
    ) -> None:
        """Track model inference performance asynchronously."""
        await self._update_metrics(model_name, latency_ms, cpu_metrics or {}, symbol)
    
    def get_metrics(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics for all models or a specific model."""
        with self._thread_lock:
            if model_name:
                mm = self.models.get(model_name)
                return mm.to_dict() if isinstance(mm, ModelMetrics) else {}
            return {name: metrics.to_dict() for name, metrics in self.models.items()}

    def get_symbol_metrics(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Get per-symbol performance metrics.
        - If symbol is provided: returns {model_name: metrics_dict}
        - Else: returns {symbol: {model_name: metrics_dict}}
        """
        with self._thread_lock:
            if symbol:
                models = self.symbol_metrics.get(symbol, {})
                return {m: mm.to_dict() for m, mm in models.items()}
            return {s: {m: mm.to_dict() for m, mm in models.items()} for s, models in self.symbol_metrics.items()}
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide performance metrics."""
        process = psutil.Process()
        mem_info = process.memory_info()
        
        # Non-blocking CPU measurement
        cpu_pct = psutil.cpu_percent(interval=None)
        
        with self._thread_lock:
            models_count = len(self.models)
            total_inferences = sum(m.call_count for m in self.models.values())
            symbols_count = len(self.symbol_metrics)
            total_symbol_inferences = sum(
                m.call_count for sym in self.symbol_metrics.values() for m in sym.values()
            )
        
        return {
            "system": {
                "uptime_seconds": time.time() - self.start_time,
                "cpu_percent": cpu_pct,
                "memory": {
                    "rss_mb": mem_info.rss / 1024 / 1024,
                    "vms_mb": mem_info.vms / 1024 / 1024,
                    "available_percent": psutil.virtual_memory().available * 100 / psutil.virtual_memory().total,
                },
                "threads": process.num_threads(),
                "open_files": len(process.open_files()),
            },
            "models_tracked": models_count,
            "total_inferences": total_inferences,
            "symbols_tracked": symbols_count,
            "total_symbol_inferences": total_symbol_inferences,
        }
    
    async def _broadcast_metrics(self, message: Dict[str, Any]):
        """Broadcast metrics to connected WebSocket clients."""
        if not self.active_connections:
            return
            
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.active_connections.remove(conn)
    
    def add_connection(self, websocket):
        """Add a WebSocket connection for real-time updates."""
        self.active_connections.append(websocket)
    
    def remove_connection(self, websocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def _check_thresholds_impl(self):
        """Implementation: run threshold checks and broadcast on the monitor's loop."""
        system = self.get_system_metrics()
        cpu_warn = self.thresholds.get("cpu_warn_percent", 90.0)
        mem_crit = self.thresholds.get("mem_available_percent_crit", 10.0)
        lat_warn = self.thresholds.get("latency_warn_ms", 1000.0)

        # CPU threshold check
        if system['system']['cpu_percent'] > cpu_warn:
            await self._broadcast_metrics({
                "type": "alert",
                "message": f"High CPU Usage: {system['system']['cpu_percent']:.1f}% (>{cpu_warn}%)",
                "timestamp": datetime.now().isoformat(),
                "severity": "warning",
            })

        # Memory threshold check
        if system['system']['memory']['available_percent'] < mem_crit:
            await self._broadcast_metrics({
                "type": "alert",
                "message": f"Low available memory (< {mem_crit}%)",
                "timestamp": datetime.now().isoformat(),
                "severity": "critical",
            })

        # Model latency threshold checks
        with self._thread_lock:
            model_items = list(self.models.items())
        for model_name, metrics in model_items:
            if metrics.ema_latency and metrics.ema_latency > lat_warn:
                await self._broadcast_metrics({
                    "type": "alert",
                    "message": f"High latency in {model_name}: {metrics.ema_latency:.1f}ms (>{lat_warn}ms)",
                    "timestamp": datetime.now().isoformat(),
                    "severity": "warning",
                })

        # Per-symbol latency threshold checks
        with self._thread_lock:
            symbol_items = list(self.symbol_metrics.items())
        for symbol, models in symbol_items:
            for model_name, metrics in models.items():
                if metrics.ema_latency and metrics.ema_latency > lat_warn:
                    await self._broadcast_metrics({
                        "type": "alert",
                        "message": f"High latency {symbol}/{model_name}: {metrics.ema_latency:.1f}ms (>{lat_warn}ms)",
                        "timestamp": datetime.now().isoformat(),
                        "severity": "warning",
                    })

    async def check_thresholds(self):
        """Check metrics against thresholds and generate alerts.
        Ensures execution on the monitor's event loop to safely broadcast.
        """
        return await self._run_in_pm_loop(self._check_thresholds_impl())

    async def log_metrics(self) -> None:
        """Log performance metrics periodically."""
        while True:
            try:
                # Check thresholds and send alerts
                await self.check_thresholds()
                
                system_metrics = self.get_system_metrics()
                logger.info(
                    "Performance Metrics - System: %s",
                    json.dumps(system_metrics, indent=2)
                )
                
                for model_name in list(self.models.keys()):
                    metrics = self.get_metrics(model_name)
                    logger.info(
                        "Performance Metrics - %s: %s",
                        model_name,
                        json.dumps(metrics, indent=2)
                    )
                
            except Exception as e:
                logger.error(f"Error logging metrics: {e}")
                
            await asyncio.sleep(60)  # Log every 60 seconds

# Global instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get or create the global performance monitor."""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor

# Decorator for easy function timing
def track_performance(model_name: Optional[str] = None):
    """Decorator to track function performance with CPU metrics."""
    def decorator(func):
        sig = inspect.signature(func)
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            nonlocal model_name
            name = model_name or f"{func.__module__}.{func.__name__}"
            monitor = get_performance_monitor()
            # Attempt to detect a 'symbol' argument for per-symbol metrics
            symbol_val: Optional[str] = None
            try:
                if 'symbol' in sig.parameters:
                    bound = sig.bind_partial(*args, **kwargs)
                    sv = bound.arguments.get('symbol')
                    symbol_val = str(sv) if isinstance(sv, str) else None
            except Exception:
                symbol_val = None

            with monitor.track_model(name, symbol=symbol_val):
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)
                
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            nonlocal model_name
            name = model_name or f"{func.__module__}.{func.__name__}"
            monitor = get_performance_monitor()
            # Attempt to detect a 'symbol' argument for per-symbol metrics
            symbol_val: Optional[str] = None
            try:
                if 'symbol' in sig.parameters:
                    bound = sig.bind_partial(*args, **kwargs)
                    sv = bound.arguments.get('symbol')
                    symbol_val = str(sv) if isinstance(sv, str) else None
            except Exception:
                symbol_val = None

            with monitor.track_model(name, symbol=symbol_val):
                return func(*args, **kwargs)
                
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# Legacy decorator for backward compatibility
class timed:
    """Decorator to time function execution and track performance."""
    
    def __init__(self, name: Optional[str] = None, track_memory: bool = True):
        self.name = name
        self.track_memory = track_memory
        self.monitor = get_performance_monitor()
    
    def __call__(self, func):
        @functools.wraps(func)
        async def async_wrapped(*args, **kwargs):
            with self.monitor.track_model(self.name or f"{func.__module__}.{func.__name__}"):
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                return func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapped(*args, **kwargs):
            with self.monitor.track_model(self.name or f"{func.__module__}.{func.__name__}"):
                return func(*args, **kwargs)
        
        return async_wrapped if asyncio.iscoroutinefunction(func) else sync_wrapped
