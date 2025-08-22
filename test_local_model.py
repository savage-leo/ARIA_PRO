#!/usr/bin/env python3
"""
Test Local Model Functionality
"""
import asyncio
import httpx
import json

async def test_local_model():
    print("🔍 Testing Local Model Functionality")
    print("=" * 50)
    
    base_url = "http://localhost:11435"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            payload = {
                "model": "qwen2.5-coder:1.5b-base",
                "prompt": "Write a simple Python function",
                "stream": False
            }
            
            print(f"🎯 Testing model: qwen2.5-coder:1.5b-base")
            print(f"📤 Sending request...")
            
            resp = await client.post(f"{base_url}/api/generate", json=payload)
            print(f"📥 Status: {resp.status_code}")
            
            if resp.status_code == 200:
                result = resp.json()
                print(f"✅ SUCCESS!")
                print(f"📝 Response: {result.get('response', '')[:200]}...")
                print(f"📊 Model used: {result.get('model', 'Unknown')}")
                print(f"📏 Response length: {len(result.get('response', ''))}")
            else:
                print(f"❌ FAILED: {resp.text}")
                
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🎉 Local model test complete!")

if __name__ == "__main__":
    asyncio.run(test_local_model())
