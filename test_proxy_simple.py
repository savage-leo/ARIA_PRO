#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Simple Proxy Test with Smaller Model
# ----------------------------------------------------------------------

import asyncio
import httpx
import json

async def test_simple():
    """Test proxy with smaller model"""
    
    base_url = "http://localhost:11435"
    
    print("🧪 Simple Proxy Test with Smaller Model")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1️⃣ Health check...")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/healthz")
            print(f"✅ Health: {resp.status_code}")
    except Exception as e:
        print(f"❌ Health failed: {e}")
        return
    
    # Test 2: List models
    print("\n2️⃣ List models...")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/api/tags")
            models = resp.json()
            print(f"✅ Models: {len(models.get('models', []))}")
            for model in models.get('models', []):
                print(f"   - {model.get('name', 'Unknown')}")
    except Exception as e:
        print(f"❌ Models failed: {e}")
    
    # Test 3: Generate with smaller model
    print("\n3️⃣ Test generation with qwen2.5-coder:1.5b-base...")
    try:
        payload = {
            "model": "qwen2.5-coder:1.5b-base",
            "prompt": "Hello!",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "max_tokens": 20
            }
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{base_url}/api/generate", json=payload)
            if resp.status_code == 200:
                result = resp.json()
                print(f"✅ Generation successful!")
                print(f"   Response: {result.get('response', '')}")
            else:
                print(f"❌ Generation failed: {resp.status_code}")
                print(f"   Error: {resp.text}")
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
    
    # Test 4: Test remote model routing
    print("\n4️⃣ Test remote model routing...")
    try:
        payload = {
            "model": "gptos-120b",
            "prompt": "Hello!",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "max_tokens": 20
            }
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{base_url}/api/generate", json=payload)
            if resp.status_code == 502:
                print("✅ Remote model routing working (expected connection error)")
            else:
                print(f"❌ Unexpected response: {resp.status_code}")
    except Exception as e:
        print(f"❌ Remote test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Simple test complete!")

if __name__ == "__main__":
    asyncio.run(test_simple())
