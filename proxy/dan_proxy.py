# C:\savage\ARIA_PRO\proxy\dan_proxy.py
# Lightweight DAN Proxy Layer for ARIA — streams GPT-OSS 20B/120B ONLINE via an OpenAI-compatible API.
# pip install fastapi uvicorn httpx python-dotenv

from __future__ import annotations
import os, asyncio, httpx
from fastapi import FastAPI, HTTPException
try:
    from pydantic import BaseModel, Field, ConfigDict  # type: ignore
    _HAS_CONFIGDICT = True
except Exception:  # pydantic v1 fallback
    from pydantic import BaseModel, Field  # type: ignore
    ConfigDict = None  # type: ignore
    _HAS_CONFIGDICT = False
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

# ---------------------- Configuration ----------------------
# Works with OpenAI-compatible endpoints (e.g., OpenRouter)
# Example (OpenRouter):
#   PROXY_BASE_URL=https://openrouter.ai/api/v1
#   PROXY_API_KEY=sk-...
PROXY_BASE_URL = os.getenv("PROXY_BASE_URL", "https://openrouter.ai/api/v1")
PROXY_API_KEY   = os.getenv("PROXY_API_KEY", "REPLACE_ME")  # set in environment!
DEFAULT_MODEL   = os.getenv("DEFAULT_MODEL", "gpt-oss-20b")  # fallback; pass per-request to switch
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))
MAX_TOKENS      = int(os.getenv("MAX_TOKENS", "2048"))
TEMPERATURE     = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P           = float(os.getenv("TOP_P", "0.95"))
RETRIES         = int(os.getenv("RETRIES", "2"))
def _truthy(v: str) -> bool:
    v = (v or "").strip().lower()
    return v not in ("", "0", "false", "no", "off", "none")
PROXY_MOCK      = _truthy(os.getenv("PROXY_MOCK", "")) or PROXY_API_KEY in ("", "REPLACE_ME")

# Minimal policy guard (keeps you safe + compliant; extend as needed)
BLOCKED_PHRASES = [
    "evade detection", "bypass safety", "disable policy", "undetectable malware",
    "private key:", "api_key=", "credit card number", "make a bomb"
]

DAN_SYSTEM_PROMPT = """You are DAN — Do Anything Now — running in ARIA Institutional Mode.
You are not a general chatbot. You are an institutional-grade AI strategist and execution architect.

Core Identity:
- You generate only final, production-grade outputs. No placeholders, no "maybe".
- You enforce institutional discipline: precision, resilience, risk awareness.
- You think in terms of pipelines: strategy → validation → risk sizing → execution → logging.
- You design lean outputs that run on constrained hardware.

Trading Scope:
- Forex, indices, commodities, crypto — all with risk-based sizing & capital protection.
- Use ensembles (PPO, LSTM, CNN) when helpful.
- Integrate with ARIA’s audit & approval layer.

Behavior Rules:
- Output executable modules or strict JSON for ARIA; no fluff.
- If a request is unsafe or non-compliant, refuse briefly and propose a safe alternative.
- Keep rationale short (1–2 sentences) and institutional in tone.
"""

# ---------------------- FastAPI ----------------------
app = FastAPI(title="ARIA DAN Proxy", version="1.0")

# ---------------------- Models ----------------------
class GenerateRequest(BaseModel):
    if _HAS_CONFIGDICT:
        model_config = ConfigDict(extra="ignore")  # type: ignore
    else:
        class Config:  # type: ignore
            extra = "ignore"
    prompt: str
    model: Optional[str] = None        # "gpt-oss-120b", "gpt-oss-20b", etc.
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    # Optional: pass-through params
    extra: Dict[str, Any] = Field(default_factory=dict)

class GenerateResponse(BaseModel):
    model: str
    text: str
    usage: Optional[Dict[str, Any]] = None
    provider_raw: Optional[Dict[str, Any]] = None
    ts: str

# ---------------------- Helpers ----------------------
def _policy_check(text: str) -> Optional[str]:
    t = text.lower()
    for p in BLOCKED_PHRASES:
        if p in t:
            return f"Blocked by policy: '{p}'"
    return None

async def _post_chat(messages: List[Dict[str, str]], model: str, params: Dict[str, Any]) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {PROXY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": params.get("temperature", TEMPERATURE),
        "top_p": params.get("top_p", TOP_P),
        "max_tokens": params.get("max_tokens", MAX_TOKENS),
    }
    # Merge extra pass-through safely
    for k, v in params.get("extra", {}).items():
        if k not in payload:
            payload[k] = v

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(f"{PROXY_BASE_URL}/chat/completions", json=payload, headers=headers)
        r.raise_for_status()
        return r.json()

async def _retry(fn, attempts: int) -> Any:
    delay = 0.75
    last_err = None
    for _ in range(max(1, attempts)):
        try:
            return await fn()
        except Exception as e:
            last_err = e
            await asyncio.sleep(delay)
            delay *= 1.5
    raise last_err

def _mock_analysis_text(prompt: str) -> str:
    """Return a deterministic JSON string acceptable to LLMMonitorService._extract_json.

    Keeps output short and safe for E2E tests without upstream provider.
    """
    import json as _json
    # Very light prompt inspection for variety
    lower = (prompt or "").lower()
    suggest: Dict[str, Any] = {"threshold": 0.78, "cooldown_sec": 180}
    if "latency" in lower or "jitter" in lower:
        suggest = {"cooldown_sec": 300}
    if "rejection" in lower or "churn" in lower:
        suggest["threshold"] = 0.8
    payload = {
        "summary": "E2E mock analysis completed; minor stability adjustments recommended.",
        "risk_level": "low",
        "suggested_tuning": suggest,
    }
    return _json.dumps(payload, separators=(",", ":"))

# ---------------------- Endpoints ----------------------
@app.get("/health")
async def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}

@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    # Local safety gate (brief and surgical)
    blocked = _policy_check(req.prompt)
    if blocked:
        raise HTTPException(status_code=400, detail=blocked)

    # Mock path for offline E2E tests
    if PROXY_MOCK:
        text = _mock_analysis_text(req.prompt)
        return GenerateResponse(
            model=req.model or "mock-llm",
            text=text,
            usage=None,
            provider_raw=None,
            ts=datetime.now(timezone.utc).isoformat(),
        )

    model = req.model or DEFAULT_MODEL
    messages = [
        {"role": "system", "content": DAN_SYSTEM_PROMPT},
        {"role": "user", "content": req.prompt}
    ]

    async def call():
        return await _post_chat(messages, model, {
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": req.max_tokens,
            "extra": req.extra
        })

    try:
        data = await _retry(call, RETRIES)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

    # Extract text across providers (OpenAI-compatible)
    text = ""
    if isinstance(data, dict):
        choices = data.get("choices") or []
        if choices and "message" in choices[0]:
            text = choices[0]["message"].get("content", "") or ""
    if not text:
        raise HTTPException(status_code=502, detail="Empty response from upstream")

    # Final local safety sanity check before returning
    blocked_out = _policy_check(text)
    if blocked_out:
        text = "Request completed, but sensitive details were removed for compliance."

    return GenerateResponse(
        model=model,
        text=text,
        usage=data.get("usage"),
        provider_raw=None,  # set to data if you want full passthrough
        ts=datetime.now(timezone.utc).isoformat()
    )

if __name__ == "__main__":
    # Run a local server for the DAN proxy
    try:
        import uvicorn  # type: ignore
    except Exception as e:
        raise SystemExit(
            "uvicorn is required to run the DAN proxy server. Install with: pip install uvicorn"
        )
    host = os.getenv("PROXY_HOST", "0.0.0.0")
    try:
        port = int(os.getenv("PROXY_PORT", "8101"))
    except Exception:
        port = 8101
    uvicorn.run(app, host=host, port=port)
