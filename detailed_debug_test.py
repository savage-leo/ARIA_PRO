#!/usr/bin/env python3
import httpx
import asyncio
import json

async def detailed_debug_test():
    print("Testing proxy endpoints...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test health first
        try:
            resp = await client.get("http://localhost:11435/healthz")
            print(f"Health check: {resp.status_code} - {resp.json()}")
        except Exception as e:
            print(f"Health check failed: {e}")
            return
        
        # Test model listing
        try:
            resp = await client.get("http://localhost:11435/api/tags")
            print(f"Model listing: {resp.status_code}")
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                print(f"Found {len(models)} models")
                local_models = [m for m in models if "specialty" not in m]
                print(f"Local models: {[m['name'] for m in local_models]}")
        except Exception as e:
            print(f"Model listing failed: {e}")
        
        # Test generate with local model
        print("\nTesting generate with local model...")
        payload = {
            "model": "qwen2.5-coder:1.5b-base",
            "prompt": "hello",
            "stream": False
        }
        
        try:
            print(f"Sending payload: {json.dumps(payload, indent=2)}")
            resp = await client.post("http://localhost:11435/api/generate", json=payload)
            print(f"Generate response: {resp.status_code}")
            print(f"Response text: {resp.text[:500]}")
        except Exception as e:
            print(f"Generate failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(detailed_debug_test())
