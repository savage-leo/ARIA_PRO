#!/usr/bin/env python3
# ----------------------------------------------------------------------
# institutional_proxy.py – ARIA ↔ Ollama + GPT‑OS full bridge
# ----------------------------------------------------------------------

import os
import json
import uuid
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import AsyncGenerator, Dict, Any, Optional

import httpx
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI(
    title="ARIA ↔ Ollama + GPT‑OS Institutional Bridge",
    description="Full proxy for ARIA PRO, supporting multiple local and remote models."
)

# ----------------------------------------------------------------------
# 0️⃣ CONFIG – edit these
# ----------------------------------------------------------------------
LOCAL_OLLAMA_URL = os.getenv("LOCAL_OLLAMA_URL", "http://127.0.0.1:11434")

# Remote models with API keys: name → {url, api_key, model_id, task_specialty}
REMOTE_MODELS = {
    # Reasoning & Strategy Models
    "llama-3.1-405b": {
        "url": "https://api.together.ai/v1",
        "api_key": "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",
        "model_id": "meta-llama/llama-3.1-405b-instruct",
        "specialty": ["strategy", "reasoning", "analysis", "default"]
    },
    "llama-3.3-70b": {
        "url": "https://api.together.ai/v1",
        "api_key": "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",
        "model_id": "meta-llama/llama-3.3-70b-instruct",
        "specialty": ["strategy", "reasoning", "analysis", "default"]
    },
    
    # Coding & Development Models
    "qwen-coder-32b": {
        "url": "https://api.together.ai/v1",
        "api_key": "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",
        "model_id": "qwen/qwen-2.5-coder-32b-instruct",
        "specialty": ["code", "development", "debugging", "audit"]
    },
    "deepseek-r1-14b": {
        "url": "https://api.together.ai/v1",
        "api_key": "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",
        "model_id": "deepseek/deepseek-r1-distill-qwen-14b",
        "specialty": ["code", "development", "debugging", "math"]
    },
    
    # Vision & Multimodal Models
    "qwen-vl-72b": {
        "url": "https://api.together.ai/v1",
        "api_key": "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",
        "model_id": "qwen/qwen2.5-vl-72b-instruct",
        "specialty": ["vision", "multimodal", "analysis"]
    },
    
    # Fast & Efficient Models
    "reka-flash-3": {
        "url": "https://api.together.ai/v1",
        "api_key": "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",
        "model_id": "rekaai/reka-flash-3:free",
        "specialty": ["fast", "efficient", "default"]
    },
    "qwq-32b": {
        "url": "https://api.together.ai/v1",
        "api_key": "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",
        "model_id": "qwen/qwq-32b:free",
        "specialty": ["fast", "efficient", "default"]
    },
    "llama-3.2-3b": {
        "url": "https://api.together.ai/v1",
        "api_key": "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",
        "model_id": "meta-llama/llama-3.2-3b-instruct",
        "specialty": ["fast", "efficient", "default"]
    }
}

# Local models – set of names
LOCAL_MODELS = set(os.getenv("LOCAL_MODELS", "mistral:latest,qwen2.5-coder:1.5b-base,gemma3:4b,nomic-embed-text:latest").split(","))

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

# Optional caching
USE_CACHE = True
CACHE_SIZE = 128

# ----------------------------------------------------------------------
# 1️⃣ Helpers
# ----------------------------------------------------------------------
def is_local_model(model_name: str) -> bool:
    return model_name in LOCAL_MODELS

def is_remote_model(model_name: str) -> bool:
    return model_name in REMOTE_MODELS

def get_best_model_for_task(task: str, available_models: list) -> str:
    """Intelligent model selection based on task type"""
    task = task.lower()
    
    # Task to model priority mapping
    task_priorities = {
        "strategy": ["llama-3.1-405b", "llama-3.3-70b", "qwen-vl-72b"],
        "reasoning": ["llama-3.1-405b", "llama-3.3-70b", "qwen-vl-72b"],
        "analysis": ["llama-3.1-405b", "llama-3.3-70b", "qwen-vl-72b"],
        "code": ["qwen-coder-32b", "deepseek-r1-14b", "llama-3.3-70b"],
        "development": ["qwen-coder-32b", "deepseek-r1-14b", "llama-3.3-70b"],
        "debugging": ["qwen-coder-32b", "deepseek-r1-14b", "llama-3.3-70b"],
        "audit": ["qwen-coder-32b", "deepseek-r1-14b", "llama-3.1-405b"],
        "math": ["deepseek-r1-14b", "llama-3.1-405b", "llama-3.3-70b"],
        "vision": ["qwen-vl-72b", "llama-3.3-70b", "llama-3.1-405b"],
        "multimodal": ["qwen-vl-72b", "llama-3.3-70b"],
        "fast": ["reka-flash-3", "qwq-32b", "llama-3.2-3b"],
        "efficient": ["reka-flash-3", "qwq-32b", "llama-3.2-3b"],
        "default": ["llama-3.3-70b", "reka-flash-3", "qwq-32b"]
    }
    
    # Get priority list for this task
    priorities = task_priorities.get(task, task_priorities["default"])
    
    # Check remote models first
    for model in priorities:
        if model in REMOTE_MODELS:
            return model
    
    # Check local models as fallback
    for model in available_models:
        if any(priority.lower() in model.lower() for priority in priorities):
            return model
    
    # Final fallback
    return available_models[0] if available_models else "reka-flash-3"

# ----------------------------------------------------------------------
# 2️⃣ Forward to local Ollama
# ----------------------------------------------------------------------
async def forward_to_local_ollama(endpoint: str, payload: Dict[str, Any], stream: bool) -> httpx.Response:
    url = f"{LOCAL_OLLAMA_URL.rstrip('/')}{endpoint}"
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if endpoint == "/api/tags":
                # Tags endpoint uses GET, not POST
                return await client.get(url, timeout=30.0)
            elif stream:
                return await client.post(url, json=payload, timeout=None, stream=True)
            else:
                return await client.post(url, json=payload, timeout=120.0)
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail=f"Cannot connect to Ollama at {LOCAL_OLLAMA_URL}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Ollama request timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error forwarding to Ollama: {str(e)}")

# ----------------------------------------------------------------------
# 3️⃣ Call remote GPT‑OS
# ----------------------------------------------------------------------
async def call_remote_together(model_name: str, payload: Dict[str, Any], stream: bool) -> httpx.Response:
    model_info = REMOTE_MODELS[model_name]
    url = f"{model_info['url'].rstrip('/')}/chat/completions" if "messages" in payload else f"{model_info['url'].rstrip('/')}/completions"
    headers = {"Authorization": f"Bearer {model_info['api_key']}", "Content-Type": "application/json"}
    
    # Convert Ollama format to OpenAI format
    openai_payload = {
        "model": model_info["model_id"],
        "stream": stream,
        "temperature": payload.get("options", {}).get("temperature", 0.7),
        "max_tokens": payload.get("options", {}).get("max_tokens", 512)
    }
    
    if "messages" in payload:
        openai_payload["messages"] = payload["messages"]
    else:
        openai_payload["prompt"] = payload.get("prompt", "")
    
    for attempt in range(MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                if stream:
                    return await client.post(url, headers=headers, json=openai_payload, timeout=None, stream=True)
                else:
                    return await client.post(url, headers=headers, json=openai_payload, timeout=120.0)
        except httpx.HTTPError as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise HTTPException(status_code=502, detail=str(e))

# ----------------------------------------------------------------------
# 4️⃣ Translate OpenAI → Ollama JSON
# ----------------------------------------------------------------------
def openai_chunk_to_ollama(chunk: dict, model_name: str, final: bool=False) -> dict:
    delta = chunk.get("choices", [{}])[0].get("delta", {})
    content = delta.get("content", "") if not final else ""
    done = final or chunk.get("choices", [{}])[0].get("finish_reason") is not None
    return {
        "model": model_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "response": content,
        "done": done
    }

# ----------------------------------------------------------------------
# 5️⃣ /api/generate
# ----------------------------------------------------------------------
@app.post("/api/generate")
async def generate(request: Request):
    body = await request.json()
    model_name = body.get("model", "")
    prompt = body.get("prompt", "")
    stream = bool(body.get("stream", False))
    options = body.get("options", {})
    task = body.get("task", "default")  # New task parameter for intelligent routing

    # CASE: Auto-select best model if none specified
    if not model_name:
        # Get available models
        try:
            local_resp = await forward_to_local_ollama("/api/tags", {}, stream=False)
            local_models = [m.get("name", "") for m in local_resp.json().get("models", [])]
        except:
            local_models = []
        
        # Add remote models
        all_models = local_models + list(REMOTE_MODELS.keys())
        model_name = get_best_model_for_task(task, all_models)
        print(f"[ARIA] Auto-selected model '{model_name}' for task '{task}'")

    # CASE: local model
    print(f"[DEBUG] Checking if '{model_name}' is local model...")
    print(f"[DEBUG] LOCAL_MODELS: {LOCAL_MODELS}")
    print(f"[DEBUG] is_local_model result: {is_local_model(model_name)}")
    if is_local_model(model_name):
        print(f"[DEBUG] Using local model: {model_name}")
        resp = await forward_to_local_ollama("/api/generate", body, stream)
        if stream:
            async def fwd_stream() -> AsyncGenerator[bytes, None]:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            return StreamingResponse(fwd_stream(), media_type="application/json")
        else:
            # Translate Ollama response to expected format
            data = resp.json()
            response_text = data.get("response", "")
            return JSONResponse({
                "model": model_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "response": response_text,
                "done": True
            })

    # CASE: remote model
    if not is_remote_model(model_name):
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    openai_payload = {
        "model": REMOTE_MODELS[model_name]["model_id"],
        "prompt": prompt,
        "temperature": options.get("temperature", 0.7),
        "max_tokens": options.get("max_tokens", 512),
        "stream": stream
    }

    remote_resp = await call_remote_together(model_name, body, stream)

    if remote_resp.status_code != 200:
        err = remote_resp.json().get("error", {})
        raise HTTPException(status_code=remote_resp.status_code, detail=err.get("message", "Remote error"))

    if stream:
        async def ollama_stream() -> AsyncGenerator[bytes, None]:
            async for line in remote_resp.aiter_lines():
                if not line.strip():
                    continue
                json_str = line.lstrip("data: ").strip()
                try:
                    chunk = json.loads(json_str)
                except json.JSONDecodeError:
                    continue
                ollama_obj = openai_chunk_to_ollama(chunk, model_name)
                yield (json.dumps(ollama_obj) + "\n").encode()
        return StreamingResponse(ollama_stream(), media_type="application/json")
    else:
        data = remote_resp.json()
        full_text = data.get("choices", [{}])[0].get("text", "")
        return JSONResponse({
            "model": model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "response": full_text,
            "done": True
        })

# ----------------------------------------------------------------------
# 6️⃣ /api/chat
# ----------------------------------------------------------------------
@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    model_name = body.get("model", "")
    messages = body.get("messages", [])
    stream = bool(body.get("stream", False))
    options = body.get("options", {})
    task = body.get("task", "default")  # New task parameter for intelligent routing

    # CASE: Auto-select best model if none specified
    if not model_name:
        # Get available models
        try:
            local_resp = await forward_to_local_ollama("/api/tags", {}, stream=False)
            local_models = [m.get("name", "") for m in local_resp.json().get("models", [])]
        except:
            local_models = []
        
        # Add remote models
        all_models = local_models + list(REMOTE_MODELS.keys())
        model_name = get_best_model_for_task(task, all_models)
        print(f"[ARIA] Auto-selected model '{model_name}' for task '{task}'")

    # local
    if is_local_model(model_name):
        resp = await forward_to_local_ollama("/api/chat", body, stream)
        if stream:
            async def fwd() -> AsyncGenerator[bytes, None]:
                async for chunk in resp.aiter_bytes():
                    yield chunk
            return StreamingResponse(fwd(), media_type="application/json")
        else:
            # Translate Ollama response to expected format
            data = resp.json()
            message_content = data.get("message", {}).get("content", "")
            return JSONResponse({
                "model": model_name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "response": message_content,
                "done": True
            })

    # remote
    if not is_remote_model(model_name):
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

    openai_payload = {
        "model": REMOTE_MODELS[model_name]["model_id"],
        "messages": messages,
        "temperature": options.get("temperature", 0.7),
        "max_tokens": options.get("max_tokens", 512),
        "stream": stream
    }

    remote_resp = await call_remote_together(model_name, body, stream)
    if remote_resp.status_code != 200:
        err = remote_resp.json().get("error", {})
        raise HTTPException(status_code=remote_resp.status_code, detail=err.get("message", "Remote chat error"))

    if stream:
        async def chat_stream() -> AsyncGenerator[bytes, None]:
            async for line in remote_resp.aiter_lines():
                if not line.strip():
                    continue
                json_str = line.lstrip("data: ").strip()
                try:
                    chunk = json.loads(json_str)
                except json.JSONDecodeError:
                    continue
                yield (json.dumps(openai_chunk_to_ollama(chunk, model_name)) + "\n").encode()
        return StreamingResponse(chat_stream(), media_type="application/json")
    else:
        data = remote_resp.json()
        full_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return JSONResponse({
            "model": model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "response": full_text,
            "done": True
        })

# ----------------------------------------------------------------------
# 7️⃣ /api/tags – merge local + remote
# ----------------------------------------------------------------------
@app.get("/api/tags")
async def tags():
    try:
        print("Fetching local models from Ollama...")
        local_resp = await forward_to_local_ollama("/api/tags", None, stream=False)
        print(f"Local response status: {local_resp.status_code}")
        if local_resp.status_code == 200:
            local_data = local_resp.json()
            print(f"Local models found: {len(local_data.get('models', []))}")
        else:
            print(f"Local response error: {local_resp.text}")
            local_data = {"models": []}
    except Exception as e:
        print(f"Warning: Could not fetch local models: {e}")
        local_data = {"models": []}
    
    # Create remote model entries with specialties
    remote_entries = []
    for model_name, model_info in REMOTE_MODELS.items():
        specialties = ", ".join(model_info.get("specialty", ["general"]))
        remote_entries.append({
            "name": model_name,
            "model": model_name,
            "modified_at": datetime.now(timezone.utc).isoformat(),
            "size": 0,
            "digest": f"remote-{model_name}",
            "specialty": specialties,
            "model_id": model_info["model_id"]
        })
    merged = {"models": local_data.get("models", []) + remote_entries}
    print(f"Total models in response: {len(merged['models'])}")
    return JSONResponse(merged)

# ----------------------------------------------------------------------
# 8️⃣ /healthz
# ----------------------------------------------------------------------
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

# ----------------------------------------------------------------------
# 9️⃣ Run standalone
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("institutional_proxy:app", host="0.0.0.0", port=11435, log_level="info")

