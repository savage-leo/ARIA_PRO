#!/usr/bin/env python3
"""
Test Remote Models with Updated API Keys
"""
import asyncio
import httpx
import json

async def test_remote_models():
    print("🔍 Testing Remote Models with Updated API Keys")
    print("=" * 60)
    
    base_url = "http://localhost:11435"
    
    # Test models to try
    test_models = [
        "reka-flash-3",
        "qwq-32b", 
        "llama-3.3-70b",
        "qwen-coder-32b"
    ]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for model in test_models:
            print(f"\n🎯 Testing model: {model}")
            try:
                payload = {
                    "model": model,
                    "prompt": "Write a simple hello world message",
                    "stream": False
                }
                
                resp = await client.post(f"{base_url}/api/generate", json=payload)
                print(f"   Status: {resp.status_code}")
                
                if resp.status_code == 200:
                    result = resp.json()
                    print(f"   ✅ SUCCESS!")
                    print(f"   Response: {result.get('response', '')[:100]}...")
                else:
                    print(f"   ❌ FAILED: {resp.text}")
                    
            except Exception as e:
                print(f"   ❌ ERROR: {str(e)}")
            
            await asyncio.sleep(2)  # Wait between requests
    
    print("\n" + "=" * 60)
    print("🎉 Remote model testing complete!")

if __name__ == "__main__":
    asyncio.run(test_remote_models())

