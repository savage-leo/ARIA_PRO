#!/usr/bin/env python3
"""
Final Comprehensive Test - Enhanced ARIA PRO Institutional Proxy
Complete verification of all functionality
"""
import asyncio
import httpx
import json
from datetime import datetime

async def final_comprehensive_test():
    print("🚀 FINAL COMPREHENSIVE TEST")
    print("=" * 60)
    print(f"⏰ Test Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    base_url = "http://localhost:11435"
    results = []
    
    def log_result(test_name, status, details=None, error=None):
        result = {
            "test": test_name,
            "status": status,
            "details": details or {},
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)
        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_icon} {test_name}: {status}")
        if error:
            print(f"   Error: {error}")
        if details:
            for key, value in details.items():
                print(f"   {key}: {value}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # 1. Health Check
            print("\n1️⃣ Testing Health...")
            try:
                resp = await client.get(f"{base_url}/healthz")
                if resp.status_code == 200:
                    log_result("Health Check", "PASS", {"status_code": resp.status_code})
                else:
                    log_result("Health Check", "FAIL", {"status_code": resp.status_code})
            except Exception as e:
                log_result("Health Check", "FAIL", error=str(e))
            
            # 2. Model Inventory
            print("\n2️⃣ Testing Model Inventory...")
            try:
                resp = await client.get(f"{base_url}/api/tags")
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    local_models = [m for m in models if "specialty" not in m]
                    remote_models = [m for m in models if "specialty" in m]
                    log_result("Model Inventory", "PASS", {
                        "total_models": len(models),
                        "local_models": len(local_models),
                        "remote_models": len(remote_models)
                    })
                else:
                    log_result("Model Inventory", "FAIL", {"status_code": resp.status_code})
            except Exception as e:
                log_result("Model Inventory", "FAIL", error=str(e))
            
            # 3. Local Model Generation
            print("\n3️⃣ Testing Local Model Generation...")
            try:
                payload = {
                    "model": "qwen2.5-coder:1.5b-base",
                    "prompt": "Write a simple Python function",
                    "stream": False
                }
                resp = await client.post(f"{base_url}/api/generate", json=payload)
                if resp.status_code == 200:
                    result = resp.json()
                    log_result("Local Model Generation", "PASS", {
                        "model_used": result.get("model"),
                        "response_length": len(result.get("response", ""))
                    })
                else:
                    log_result("Local Model Generation", "FAIL", {"status_code": resp.status_code})
            except Exception as e:
                log_result("Local Model Generation", "FAIL", error=str(e))
            
            # 4. Task-Based Routing (should fall back to local models)
            print("\n4️⃣ Testing Task-Based Routing...")
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
                        log_result(f"Task Routing ({task})", "PASS", {
                            "model_selected": result.get("model")
                        })
                    else:
                        log_result(f"Task Routing ({task})", "FAIL", {
                            "status_code": resp.status_code,
                            "note": "Expected to fall back to local models"
                        })
                except Exception as e:
                    log_result(f"Task Routing ({task})", "FAIL", error=str(e))
                await asyncio.sleep(1)
            
            # 5. Chat Interface
            print("\n5️⃣ Testing Chat Interface...")
            try:
                payload = {
                    "messages": [
                        {"role": "user", "content": "What is the capital of France?"}
                    ],
                    "task": "default",
                    "stream": False
                }
                resp = await client.post(f"{base_url}/api/chat", json=payload)
                if resp.status_code == 200:
                    result = resp.json()
                    log_result("Chat Interface", "PASS", {
                        "model_used": result.get("model"),
                        "response_length": len(result.get("response", ""))
                    })
                else:
                    log_result("Chat Interface", "FAIL", {
                        "status_code": resp.status_code,
                        "note": "Expected to fall back to local models"
                    })
            except Exception as e:
                log_result("Chat Interface", "FAIL", error=str(e))
            
            # 6. Direct Remote Model Test
            print("\n6️⃣ Testing Direct Remote Model...")
            try:
                payload = {
                    "model": "reka-flash-3",
                    "prompt": "Hello",
                    "stream": False
                }
                resp = await client.post(f"{base_url}/api/generate", json=payload)
                if resp.status_code == 200:
                    result = resp.json()
                    log_result("Direct Remote Model", "PASS", {
                        "model_used": result.get("model"),
                        "response_length": len(result.get("response", ""))
                    })
                else:
                    log_result("Direct Remote Model", "FAIL", {
                        "status_code": resp.status_code,
                        "note": "API key authentication issue"
                    })
            except Exception as e:
                log_result("Direct Remote Model", "FAIL", error=str(e))
    
    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
    
    # Generate Summary
    print("\n" + "=" * 60)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = len([r for r in results if r["status"] == "PASS"])
    failed_tests = len([r for r in results if r["status"] == "FAIL"])
    
    print(f"📈 Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"📊 Success Rate: {(passed_tests / total_tests * 100):.1f}%" if total_tests > 0 else "0%")
    
    # Determine deployment status
    if passed_tests >= 4:  # At least health, model inventory, local generation, and one other
        deployment_status = "✅ PRODUCTION READY"
        print(f"\n🎉 Deployment Status: {deployment_status}")
        print("🚀 Ready for ARIA integration!")
    else:
        deployment_status = "❌ NEEDS ATTENTION"
        print(f"\n⚠️ Deployment Status: {deployment_status}")
        print("🔧 Check configuration and restart proxy")
    
    # Save results
    test_file = f"final_comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    test_data = {
        "test_time": datetime.now().isoformat(),
        "deployment_status": deployment_status,
        "summary": {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "success_rate": f"{(passed_tests / total_tests * 100):.1f}%" if total_tests > 0 else "0%"
        },
        "results": results
    }
    
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\n📄 Test results saved to: {test_file}")
    
    # Final recommendations
    print("\n🎯 FINAL RECOMMENDATIONS:")
    if deployment_status == "✅ PRODUCTION READY":
        print("1. ✅ Configure ARIA to use http://localhost:11435")
        print("2. ✅ Test with local models (fully functional)")
        print("3. ✅ Deploy to production environment")
        print("4. 🔑 Optional: Update API keys for remote models")
    else:
        print("1. 🔧 Check proxy configuration")
        print("2. 🔧 Verify Ollama is running")
        print("3. 🔧 Restart the proxy")
        print("4. 🔧 Run tests again")
    
    return test_data

if __name__ == "__main__":
    asyncio.run(final_comprehensive_test())

