#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Test Institutional Proxy
# ----------------------------------------------------------------------

import asyncio
import httpx
import json

async def test_proxy():
    """Test the institutional proxy endpoints"""
    
    base_url = "http://localhost:11435"
    
    print("🧪 Testing ARIA PRO Institutional Proxy...")
    
    # Test 1: Health check
    print("\n1️⃣ Testing /healthz...")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/healthz")
            print(f"✅ Health check: {resp.status_code} - {resp.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test 2: List models
    print("\n2️⃣ Testing /api/tags...")
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base_url}/api/tags")
            models = resp.json()
            print(f"✅ Models available: {len(models.get('models', []))}")
            for model in models.get('models', [])[:5]:  # Show first 5
                print(f"   - {model.get('name', 'Unknown')}")
    except Exception as e:
        print(f"❌ Model listing failed: {e}")
    
    # Test 3: Generate (non-streaming)
    print("\n3️⃣ Testing /api/generate (non-streaming)...")
    try:
        payload = {
            "model": "mistral:latest",  # Use a local model
            "prompt": "Hello, this is a test!",
            "stream": False,
            "options": {
                "temperature": 0.7,
                "max_tokens": 50
            }
        }
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{base_url}/api/generate", json=payload)
            if resp.status_code == 200:
                result = resp.json()
                print(f"✅ Generation successful: {result.get('response', '')[:100]}...")
            else:
                print(f"❌ Generation failed: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
    
    # Test 4: Chat (non-streaming)
    print("\n4️⃣ Testing /api/chat (non-streaming)...")
    try:
        payload = {
            "model": "mistral:latest",
            "messages": [
                {"role": "user", "content": "Say hello in a friendly way!"}
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "max_tokens": 50
            }
        }
        
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{base_url}/api/chat", json=payload)
            if resp.status_code == 200:
                result = resp.json()
                print(f"✅ Chat successful: {result.get('response', '')[:100]}...")
            else:
                print(f"❌ Chat failed: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"❌ Chat test failed: {e}")
    
    print("\n🎉 Proxy testing complete!")

if __name__ == "__main__":
    asyncio.run(test_proxy())
