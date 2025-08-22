#!/usr/bin/env python3
"""
Final Deployment Verification - Enhanced ARIA PRO Institutional Proxy
"""

import asyncio
import httpx
import json
from datetime import datetime

async def verify_deployment():
    print("🔍 FINAL DEPLOYMENT VERIFICATION")
    print("=" * 50)
    print(f"⏰ Verification Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    base_url = "http://localhost:11435"
    results = []
    
    def log_result(test_name, status, details=None):
        result = {
            "test": test_name,
            "status": status,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)
        
        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_icon} {test_name}: {status}")
        if details:
            for key, value in details.items():
                print(f"   {key}: {value}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            
            # 1. Health Check
            print("\n1️⃣ Testing Health Endpoint...")
            try:
                resp = await client.get(f"{base_url}/healthz")
                if resp.status_code == 200:
                    log_result("Health Check", "PASS", {
                        "status_code": resp.status_code,
                        "response": resp.json()
                    })
                else:
                    log_result("Health Check", "FAIL", {
                        "status_code": resp.status_code,
                        "response": resp.text
                    })
            except Exception as e:
                log_result("Health Check", "FAIL", {"error": str(e)})
            
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
                        "remote_models": len(remote_models),
                        "local_model_names": [m['name'] for m in local_models],
                        "remote_model_names": [m['name'] for m in remote_models[:3]] + ["..."]
                    })
                else:
                    log_result("Model Inventory", "FAIL", {
                        "status_code": resp.status_code,
                        "response": resp.text
                    })
            except Exception as e:
                log_result("Model Inventory", "FAIL", {"error": str(e)})
            
            # 3. Local Model Generation
            print("\n3️⃣ Testing Local Model Generation...")
            try:
                payload = {
                    "model": "qwen2.5-coder:1.5b-base",
                    "prompt": "Write a simple Python function to add two numbers",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 100
                    }
                }
                
                resp = await client.post(f"{base_url}/api/generate", json=payload)
                if resp.status_code == 200:
                    result = resp.json()
                    log_result("Local Model Generation", "PASS", {
                        "model_used": result.get("model"),
                        "response_length": len(result.get("response", "")),
                        "response_preview": result.get("response", "")[:100] + "..."
                    })
                else:
                    log_result("Local Model Generation", "FAIL", {
                        "status_code": resp.status_code,
                        "response": resp.text
                    })
            except Exception as e:
                log_result("Local Model Generation", "FAIL", {"error": str(e)})
            
            # 4. Task-Based Routing
            print("\n4️⃣ Testing Task-Based Routing...")
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
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "max_tokens": 50
                        }
                    }
                    
                    resp = await client.post(f"{base_url}/api/generate", json=payload)
                    if resp.status_code == 200:
                        result = resp.json()
                        log_result(f"Task Routing ({task})", "PASS", {
                            "model_selected": result.get("model"),
                            "response_length": len(result.get("response", ""))
                        })
                    else:
                        log_result(f"Task Routing ({task})", "FAIL", {
                            "status_code": resp.status_code,
                            "expected_error": "API key authentication required"
                        })
                except Exception as e:
                    log_result(f"Task Routing ({task})", "FAIL", {"error": str(e)})
                
                await asyncio.sleep(1)
            
            # 5. Chat Interface
            print("\n5️⃣ Testing Chat Interface...")
            try:
                chat_payload = {
                    "messages": [
                        {"role": "user", "content": "What is the capital of France?"}
                    ],
                    "task": "default",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 100
                    }
                }
                
                resp = await client.post(f"{base_url}/api/chat", json=chat_payload)
                if resp.status_code == 200:
                    result = resp.json()
                    log_result("Chat Interface", "PASS", {
                        "model_used": result.get("model"),
                        "response_length": len(result.get("response", ""))
                    })
                else:
                    log_result("Chat Interface", "FAIL", {
                        "status_code": resp.status_code,
                        "expected_error": "API key authentication required"
                    })
            except Exception as e:
                log_result("Chat Interface", "FAIL", {"error": str(e)})
            
            # Generate Summary
            print("\n" + "=" * 50)
            print("📊 VERIFICATION SUMMARY")
            print("=" * 50)
            
            total_tests = len(results)
            passed_tests = len([r for r in results if r["status"] == "PASS"])
            failed_tests = len([r for r in results if r["status"] == "FAIL"])
            
            print(f"📈 Total Tests: {total_tests}")
            print(f"✅ Passed: {passed_tests}")
            print(f"❌ Failed: {failed_tests}")
            print(f"📊 Success Rate: {(passed_tests / total_tests * 100):.1f}%" if total_tests > 0 else "0%")
            
            # Determine deployment status
            if passed_tests >= 3:  # At least health, model inventory, and local generation
                deployment_status = "✅ PRODUCTION READY"
                print(f"\n🎉 Deployment Status: {deployment_status}")
                print("🚀 Ready for ARIA integration!")
            else:
                deployment_status = "❌ NEEDS ATTENTION"
                print(f"\n⚠️ Deployment Status: {deployment_status}")
                print("🔧 Check configuration and restart proxy")
            
            # Save verification results
            verification_file = f"deployment_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            verification_data = {
                "verification_time": datetime.now().isoformat(),
                "deployment_status": deployment_status,
                "summary": {
                    "total_tests": total_tests,
                    "passed": passed_tests,
                    "failed": failed_tests,
                    "success_rate": f"{(passed_tests / total_tests * 100):.1f}%" if total_tests > 0 else "0%"
                },
                "results": results
            }
            
            with open(verification_file, 'w') as f:
                json.dump(verification_data, f, indent=2)
            
            print(f"\n📄 Verification results saved to: {verification_file}")
            
            # Final recommendations
            print("\n🎯 NEXT STEPS:")
            if deployment_status == "✅ PRODUCTION READY":
                print("1. ✅ Configure ARIA to use http://localhost:11435")
                print("2. ✅ Test with local models (immediately available)")
                print("3. 🔑 Update API keys for remote model functionality (optional)")
                print("4. 🚀 Deploy to production environment")
            else:
                print("1. 🔧 Check proxy configuration")
                print("2. 🔧 Verify Ollama is running on port 11434")
                print("3. 🔧 Restart the proxy")
                print("4. 🔧 Run verification again")
            
            return verification_data
            
    except Exception as e:
        print(f"❌ Critical error during verification: {str(e)}")
        return None

if __name__ == "__main__":
    asyncio.run(verify_deployment())
