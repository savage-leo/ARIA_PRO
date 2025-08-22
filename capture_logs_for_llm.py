#!/usr/bin/env python3
"""
Capture Comprehensive Logs for LLM Review
"""

import asyncio
import httpx
import json
from datetime import datetime

async def capture_logs():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"llm_review_logs_{timestamp}.json"
    
    logs = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "test_type": "comprehensive_proxy_review"
        },
        "test_results": [],
        "summary": {}
    }
    
    def log_test(test_name, status, details=None, error=None):
        test_result = {
            "timestamp": datetime.now().isoformat(),
            "test_name": test_name,
            "status": status,
            "details": details or {},
            "error": error
        }
        logs["test_results"].append(test_result)
        print(f"[{test_result['timestamp']}] {test_name}: {status}")
    
    base_url = "http://localhost:11435"
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            
            # 1. Health Check
            print("🔍 Testing Health...")
            try:
                resp = await client.get(f"{base_url}/healthz")
                if resp.status_code == 200:
                    log_test("health_check", "PASS", {"status_code": resp.status_code})
                else:
                    log_test("health_check", "FAIL", {"status_code": resp.status_code})
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
                        "remote_models": len(remote_models)
                    })
                else:
                    log_test("model_inventory", "FAIL", {"status_code": resp.status_code})
            except Exception as e:
                log_test("model_inventory", "ERROR", error=str(e))
            
            # 3. Local Model Test
            print("🏠 Testing Local Models...")
            try:
                payload = {
                    "model": "qwen2.5-coder:1.5b-base",
                    "prompt": "Write a simple Python function",
                    "stream": False
                }
                
                resp = await client.post(f"{base_url}/api/generate", json=payload)
                if resp.status_code == 200:
                    result = resp.json()
                    log_test("local_model", "PASS", {
                        "model": result.get("model"),
                        "response_length": len(result.get("response", ""))
                    })
                else:
                    log_test("local_model", "FAIL", {"status_code": resp.status_code})
            except Exception as e:
                log_test("local_model", "ERROR", error=str(e))
            
            # 4. Task Routing Test
            print("🎯 Testing Task Routing...")
            tasks = ["code", "strategy", "fast"]
            for task in tasks:
                try:
                    payload = {
                        "prompt": "Test prompt",
                        "task": task,
                        "stream": False
                    }
                    
                    resp = await client.post(f"{base_url}/api/generate", json=payload)
                    if resp.status_code == 200:
                        result = resp.json()
                        log_test(f"task_routing_{task}", "PASS", {
                            "model_selected": result.get("model")
                        })
                    else:
                        log_test(f"task_routing_{task}", "FAIL", {"status_code": resp.status_code})
                except Exception as e:
                    log_test(f"task_routing_{task}", "ERROR", error=str(e))
                
                await asyncio.sleep(1)
            
            # Generate Summary
            total_tests = len(logs["test_results"])
            passed_tests = len([t for t in logs["test_results"] if t["status"] == "PASS"])
            
            logs["summary"] = {
                "total_tests": total_tests,
                "passed": passed_tests,
                "success_rate": f"{(passed_tests / total_tests * 100):.1f}%" if total_tests > 0 else "0%"
            }
            
            # Save logs
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
            
            print(f"\n📄 Logs saved to: {log_file}")
            print(f"📊 Success Rate: {logs['summary']['success_rate']}")
            
            return logs
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return logs

if __name__ == "__main__":
    asyncio.run(capture_logs())
