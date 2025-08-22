#!/usr/bin/env python3
import httpx
import asyncio

async def debug_test():
    async with httpx.AsyncClient() as client:
        payload = {
            "model": "qwen2.5-coder:1.5b-base",
            "prompt": "hello",
            "stream": False
        }
        
        try:
            resp = await client.post("http://localhost:11435/api/generate", json=payload)
            print(f"Status: {resp.status_code}")
            print(f"Response: {resp.text}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(debug_test())
