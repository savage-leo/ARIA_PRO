#!/usr/bin/env python3
import httpx
import asyncio

async def test_ollama_direct():
    print("Testing Ollama directly...")
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        payload = {
            "model": "qwen2.5-coder:1.5b-base",
            "prompt": "hello",
            "stream": False
        }
        
        try:
            print("Sending request to Ollama directly...")
            resp = await client.post("http://localhost:11434/api/generate", json=payload)
            print(f"Ollama response: {resp.status_code}")
            if resp.status_code == 200:
                result = resp.json()
                print(f"Response: {result.get('response', '')[:100]}")
            else:
                print(f"Error: {resp.text}")
        except Exception as e:
            print(f"Ollama test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_ollama_direct())
