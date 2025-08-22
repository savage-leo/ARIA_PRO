#!/usr/bin/env python3
import asyncio
import httpx
import json

async def test_chat():
    base_url = "http://localhost:11435"
    
    payload = {
        "model": "qwen2.5-coder:1.5b-base",
        "messages": [
            {"role": "user", "content": "Say hello!"}
        ],
        "stream": False,
        "options": {
            "temperature": 0.7,
            "max_tokens": 20
        }
    }
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{base_url}/api/chat", json=payload)
            print(f"Status: {resp.status_code}")
            if resp.status_code == 200:
                result = resp.json()
                print(f"✅ Success: {result.get('response', '')}")
            else:
                print(f"❌ Error: {resp.text}")
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_chat())
