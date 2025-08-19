import asyncio
import logging
from typing import Optional, List, Dict, Any

import httpx

from backend.core.config import get_settings
from backend.services.auto_trader import auto_trader


class QueueingLogHandler(logging.Handler):
    """Thread-safe log handler that enqueues messages into an asyncio Queue."""

    def __init__(self, level: int = logging.WARNING) -> None:
        super().__init__(level=level)
        self.queue: Optional[asyncio.Queue[str]] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))

    def set_async_ctx(
        self, queue: asyncio.Queue[str], loop: asyncio.AbstractEventLoop
    ) -> None:
        self.queue = queue
        self.loop = loop

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if self.queue is None or self.loop is None:
                return
            msg = self.format(record)
            # Ensure thread-safe enqueue from logging threads
            self.loop.call_soon_threadsafe(self.queue.put_nowait, msg)
        except Exception:
            # Never break logging
            pass


class LLMMonitorService:
    """Background service that batches logs and consults the DAN proxy for analysis.

    Optionally applies bounded tuning suggestions to AutoTrader when enabled.
    """

    def __init__(self) -> None:
        self.running: bool = False
        self._client: Optional[httpx.AsyncClient] = None
        self._task: Optional[asyncio.Task[None]] = None
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1000)
        self._handler = QueueingLogHandler(level=logging.WARNING)
        self._last_tune_ts: float = 0.0

        # Config snapshot (loaded on start)
        self._proxy_url_base: str = ""
        self._interval: int = 60
        self._tuning_enabled: bool = False
        self._max_rel: float = 0.2
        self._tune_cooldown: int = 300

    async def start(self) -> None:
        if self.running:
            logging.getLogger(__name__).warning("LLMMonitorService already running")
            return

        settings = get_settings()
        self._proxy_url_base = settings.LLM_MONITOR_DAN_URL.rstrip("/")
        self._interval = int(settings.LLM_MONITOR_INTERVAL_SEC)
        self._tuning_enabled = bool(settings.LLM_TUNING_ENABLED)
        self._max_rel = float(settings.LLM_TUNING_MAX_REL_DELTA)
        self._tune_cooldown = int(settings.LLM_TUNING_COOLDOWN_SEC)

        self._client = httpx.AsyncClient(timeout=15.0)

        loop = asyncio.get_running_loop()
        self._handler.set_async_ctx(self._queue, loop)
        logging.getLogger().addHandler(self._handler)

        self.running = True
        self._task = asyncio.create_task(self._run())
        logging.getLogger(__name__).info("LLMMonitorService started")

    async def stop(self) -> None:
        self.running = False
        try:
            logging.getLogger().removeHandler(self._handler)
        except Exception:
            pass
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except Exception:
                pass
            self._task = None
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None
        logging.getLogger(__name__).info("LLMMonitorService stopped")

    async def _run(self) -> None:
        buffer: List[str] = []
        while self.running:
            try:
                try:
                    while True:
                        msg = await asyncio.wait_for(
                            self._queue.get(), timeout=self._interval
                        )
                        buffer.append(msg)
                        # Cap buffer to avoid unbounded growth
                        if len(buffer) > 200:
                            buffer = buffer[-200:]
                except asyncio.TimeoutError:
                    pass

                if not buffer:
                    continue

                logs_to_send = buffer[-100:]
                buffer.clear()
                await self._analyze_and_maybe_tune(logs_to_send)
            except asyncio.CancelledError:
                break
            except Exception:
                logging.getLogger(__name__).exception("LLMMonitor loop error")

    def _build_prompt(self, logs: List[str]) -> str:
        import json as _json

        at = auto_trader
        params = {
            "interval_sec": getattr(at, "interval_sec", None),
            "threshold": getattr(at, "threshold", None),
            "atr_period": getattr(at, "atr_period", None),
            "atr_sl_mult": getattr(at, "atr_sl_mult", None),
            "atr_tp_mult": getattr(at, "atr_tp_mult", None),
            "cooldown_sec": getattr(at, "cooldown_sec", None),
            "dry_run": getattr(at, "dry_run", None),
        }

        instruction = (
            "You are ARIA LLM Monitor. Analyze backend errors/warnings and output strict JSON only: "
            "{\"summary\":\"...\", \"risk_level\":\"low|medium|high\", "
            "\"suggested_tuning\":{\"interval_sec\":int?,\"threshold\":float?,\"atr_period\":int?,"
            "\"atr_sl_mult\":float?,\"atr_tp_mult\":float?,\"cooldown_sec\":int?}}. "
            "Use null for suggested_tuning if no change is warranted."
        )
        context = f"Current AutoTrader params: {_json.dumps(params)}"
        logs_text = "\n".join(logs)
        prompt = f"{instruction}\n{context}\nRecent Logs:\n{logs_text}"
        return prompt

    async def _analyze_and_maybe_tune(self, logs: List[str]) -> None:
        if self._client is None:
            return
        prompt = self._build_prompt(logs)
        try:
            resp = await self._client.post(
                f"{self._proxy_url_base}/v1/generate",
                json={
                    "prompt": prompt,
                    "temperature": 0.0,
                    "max_tokens": 400,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            text = data.get("text")
            if not text:
                # OpenAI-compatible safety
                text = (
                    (data.get("choices") or [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
            if not text:
                return
            parsed = self._extract_json(text)
            if not parsed:
                return
            tuning = parsed.get("suggested_tuning") if isinstance(parsed, dict) else None
            if self._tuning_enabled and isinstance(tuning, dict) and tuning:
                await self._apply_bounded_tuning(tuning)
        except Exception:
            logging.getLogger(__name__).exception("LLM analyze request failed")

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        import re
        import json as _json

        # Try fenced code block first
        try:
            m = re.search(r"`{3,}.*?\n(.*?)\n`{3,}", text, flags=re.S)
            if m:
                return _json.loads(m.group(1).strip())
        except Exception:
            pass
        # Fallback: locate first {...}
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                return _json.loads(text[start : end + 1])
        except Exception:
            return None
        return None

    async def _apply_bounded_tuning(self, proposal: Dict[str, Any]) -> None:
        import time

        now = time.time()
        if now - self._last_tune_ts < self._tune_cooldown:
            return

        at = auto_trader
        mapping = [
            ("interval_sec", int),
            ("threshold", float),
            ("atr_period", int),
            ("atr_sl_mult", float),
            ("atr_tp_mult", float),
            ("cooldown_sec", int),
        ]
        updates: Dict[str, Any] = {}
        for key, caster in mapping:
            if key in proposal and getattr(at, key, None) is not None:
                current_val = getattr(at, key)
                try:
                    new_val = caster(proposal[key])
                except Exception:
                    continue
                # Relative bound clamp
                if isinstance(current_val, (int, float)) and current_val != 0:
                    rel = abs((float(new_val) - float(current_val)) / float(current_val))
                    if rel > self._max_rel:
                        delta = self._max_rel * abs(float(current_val))
                        if float(new_val) > float(current_val):
                            new_val = caster(float(current_val) + delta)
                        else:
                            new_val = caster(float(current_val) - delta)
                updates[key] = new_val

        if updates:
            await at.apply_tuning(**updates)
            self._last_tune_ts = now
            logging.getLogger(__name__).info("Applied LLM tuning: %s", updates)


# Global instance
llm_monitor_service = LLMMonitorService()
