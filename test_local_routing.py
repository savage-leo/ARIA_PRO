#!/usr/bin/env python3
"""
Test Local Model Routing - Verify intelligent routing works with local models
"""

import asyncio
import httpx
import json

async def test_local_routing():
    base_url = "http://localhost:11435"
    print("🧪 Testing Local Model Routing Logic")
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Test with local models explicitly
            print("1️⃣ Testing explicit local model selection...")
            
            local_models = ["qwen2.5-coder:1.5b-base", "gemma3:4b"]
            
            for model_name in local_models:
                print(f"\n   🎯 Testing local model: {model_name}")
                
                payload = {
                    "model": model_name,
                    "prompt": "Write a simple hello world function",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 50
                    }
                }
                
                try:
                    resp = await client.post(f"{base_url}/api/generate", json=payload)
                    if resp.status_code == 200:
                        result = resp.json()
                        response_text = result.get("response", "")[:100] + "..." if len(result.get("response", "")) > 100 else result.get("response", "")
                        print(f"      ✅ Success! Response: {response_text}")
                    else:
                        print(f"      ❌ Failed: {resp.status_code} - {resp.text[:200]}")
                except Exception as e:
                    print(f"      ❌ Error: {str(e)[:100]}")
            
            # Test task-based routing (should fall back to local models)
            print("\n2️⃣ Testing task-based routing with local fallback...")
            
            tasks = [
                ("code", "Write a Python function"),
                ("strategy", "Give me a simple plan"),
                ("fast", "Quick answer please")
            ]
            
            for task, prompt in tasks:
                print(f"\n   🎯 Testing '{task}' task (should use local fallback)...")
                
                payload = {
                    "prompt": prompt,
                    "task": task,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 50
                    }
                }
                
                try:
                    resp = await client.post(f"{base_url}/api/generate", json=payload)
                    if resp.status_code == 200:
                        result = resp.json()
                        model_used = result.get("model", "unknown")
                        response_text = result.get("response", "")[:100] + "..." if len(result.get("response", "")) > 100 else result.get("response", "")
                        print(f"      ✅ Success! Model: {model_used}")
                        print(f"      📝 Response: {response_text}")
                    else:
                        print(f"      ❌ Failed: {resp.status_code} - {resp.text[:200]}")
                except Exception as e:
                    print(f"      ❌ Error: {str(e)[:100]}")
                
                await asyncio.sleep(1)
            
            print("\n" + "=" * 50)
            print("✅ Local routing test completed!")
            print("📝 Note: Remote models failing due to API key issues")
            print("🔧 Next: Update API keys or test with valid credentials")
            
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_local_routing())
