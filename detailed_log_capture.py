#!/usr/bin/env python3
"""
Detailed Log Capture for LLM Review
Captures comprehensive error details and response information
"""

import asyncio
import httpx
import json
from datetime import datetime

async def detailed_log_capture():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"detailed_llm_logs_{timestamp}.json"
    
    logs = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "test_type": "detailed_proxy_review",
            "proxy_url": "http://localhost:11435"
        },
        "test_results": [],
        "summary": {}
    }
    
    def log_test(test_name, status, details=None, error=None, response_text=None):
        test_result = {
            "timestamp": datetime.now().isoformat(),
            "test_name": test_name,
            "status": status,
            "details": details or {},
            "error": error,
            "response_text": response_text
        }
        logs["test_results"].append(test_result)
        print(f"[{test_result['timestamp']}] {test_name}: {status}")
        if error:
            print(f"   Error: {error}")
        if response_text:
            print(f"   Response: {response_text[:200]}...")
    
    base_url = "http://localhost:11435"
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            
            # 1. Health Check
            print("🔍 Testing Health...")
            try:
                resp = await client.get(f"{base_url}/healthz")
                log_test("health_check", "PASS" if resp.status_code == 200 else "FAIL", 
                        {"status_code": resp.status_code}, 
                        response_text=resp.text)
            except Exception as e:
                log_test("health_check", "ERROR", error=str(e))
            
            # 2. Model Inventory
            print("📋 Testing Model Inventory...")
            try:
                resp = await client.get(f"{base_url}/api/tags")
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    local_models = [m for m in models if "specialty" not in m]
                    remote_models = [m for m in models if "specialty" in m]
                    
                    log_test("model_inventory", "PASS", {
                        "total_models": len(models),
                        "local_models": len(local_models),
                        "remote_models": len(remote_models),
                        "local_model_names": [m['name'] for m in local_models],
                        "remote_model_names": [m['name'] for m in remote_models]
                    }, response_text=resp.text)
                else:
                    log_test("model_inventory", "FAIL", {"status_code": resp.status_code}, response_text=resp.text)
            except Exception as e:
                log_test("model_inventory", "ERROR", error=str(e))
            
            # 3. Direct Local Model Test
            print("🏠 Testing Direct Local Model...")
            try:
                payload = {
                    "model": "qwen2.5-coder:1.5b-base",
                    "prompt": "Write a simple Python function",
                    "stream": False
                }
                
                resp = await client.post(f"{base_url}/api/generate", json=payload)
                log_test("direct_local_model", "PASS" if resp.status_code == 200 else "FAIL", 
                        {"status_code": resp.status_code}, 
                        response_text=resp.text)
            except Exception as e:
                log_test("direct_local_model", "ERROR", error=str(e))
            
            # 4. Task Routing Tests with Details
            print("🎯 Testing Task Routing with Details...")
            tasks = [
                ("code", "Write a function to sort a list"),
                ("strategy", "Give me a simple plan"),
                ("fast", "Quick answer please")
            ]
            
            for task, prompt in tasks:
                try:
                    payload = {
                        "prompt": prompt,
                        "task": task,
                        "stream": False
                    }
                    
                    resp = await client.post(f"{base_url}/api/generate", json=payload)
                    log_test(f"task_routing_{task}", "PASS" if resp.status_code == 200 else "FAIL", 
                            {"status_code": resp.status_code, "task": task, "prompt": prompt}, 
                            response_text=resp.text)
                except Exception as e:
                    log_test(f"task_routing_{task}", "ERROR", {"task": task}, error=str(e))
                
                await asyncio.sleep(1)
            
            # 5. Chat Interface Test
            print("💬 Testing Chat Interface...")
            try:
                chat_payload = {
                    "messages": [
                        {"role": "user", "content": "Hello, how are you?"}
                    ],
                    "task": "default",
                    "stream": False
                }
                
                resp = await client.post(f"{base_url}/api/chat", json=chat_payload)
                log_test("chat_interface", "PASS" if resp.status_code == 200 else "FAIL", 
                        {"status_code": resp.status_code}, 
                        response_text=resp.text)
            except Exception as e:
                log_test("chat_interface", "ERROR", error=str(e))
            
            # 6. Ollama Direct Test (for comparison)
            print("🔗 Testing Ollama Direct...")
            try:
                payload = {
                    "model": "qwen2.5-coder:1.5b-base",
                    "prompt": "Hello",
                    "stream": False
                }
                
                resp = await client.post("http://localhost:11434/api/generate", json=payload)
                log_test("ollama_direct", "PASS" if resp.status_code == 200 else "FAIL", 
                        {"status_code": resp.status_code}, 
                        response_text=resp.text)
            except Exception as e:
                log_test("ollama_direct", "ERROR", error=str(e))
            
            # Generate Summary
            total_tests = len(logs["test_results"])
            passed_tests = len([t for t in logs["test_results"] if t["status"] == "PASS"])
            failed_tests = len([t for t in logs["test_results"] if t["status"] == "FAIL"])
            error_tests = len([t for t in logs["test_results"] if t["status"] == "ERROR"])
            
            logs["summary"] = {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "errors": error_tests,
                "success_rate": f"{(passed_tests / total_tests * 100):.1f}%" if total_tests > 0 else "0%",
                "deployment_status": "READY" if passed_tests >= 3 else "NEEDS_ATTENTION"
            }
            
            # Save logs
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
            
            print(f"\n📄 Detailed logs saved to: {log_file}")
            print(f"📊 Success Rate: {logs['summary']['success_rate']}")
            print(f"🚀 Deployment Status: {logs['summary']['deployment_status']}")
            
            return logs
            
    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        return logs

if __name__ == "__main__":
    asyncio.run(detailed_log_capture())
