#!/usr/bin/env python3
"""
Comprehensive System Audit - Enhanced ARIA PRO Institutional Proxy
Line-by-line analysis of all components and endpoints
"""

import asyncio
import httpx
import json
import os
import sys
from datetime import datetime

class SystemAudit:
    def __init__(self):
        self.proxy_url = "http://localhost:11435"
        self.ollama_url = "http://localhost:11434"
        self.audit_results = []
        
    def log_audit(self, component, test_name, status, details=None, error=None):
        """Log audit result with timestamp"""
        result = {
            "timestamp": datetime.now().isoformat(),
            "component": component,
            "test": test_name,
            "status": status,
            "details": details or {},
            "error": error
        }
        self.audit_results.append(result)
        
        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_icon} [{component}] {test_name}: {status}")
        if error:
            print(f"   Error: {error}")
        if details:
            for key, value in details.items():
                print(f"   {key}: {value}")
    
    async def audit_proxy_health(self):
        """Audit proxy health endpoint"""
        print("\n🔍 AUDITING PROXY HEALTH")
        print("=" * 50)
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.proxy_url}/healthz")
                if resp.status_code == 200:
                    self.log_audit("PROXY", "Health Check", "PASS", {
                        "status_code": resp.status_code,
                        "response": resp.json()
                    })
                else:
                    self.log_audit("PROXY", "Health Check", "FAIL", {
                        "status_code": resp.status_code,
                        "response": resp.text
                    })
        except Exception as e:
            self.log_audit("PROXY", "Health Check", "FAIL", error=str(e))
    
    async def audit_ollama_health(self):
        """Audit Ollama health"""
        print("\n🔍 AUDITING OLLAMA HEALTH")
        print("=" * 50)
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.ollama_url}/api/tags")
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    self.log_audit("OLLAMA", "Health Check", "PASS", {
                        "status_code": resp.status_code,
                        "models_count": len(models),
                        "model_names": [m['name'] for m in models]
                    })
                else:
                    self.log_audit("OLLAMA", "Health Check", "FAIL", {
                        "status_code": resp.status_code,
                        "response": resp.text
                    })
        except Exception as e:
            self.log_audit("OLLAMA", "Health Check", "FAIL", error=str(e))
    
    async def audit_proxy_model_inventory(self):
        """Audit proxy model inventory"""
        print("\n🔍 AUDITING PROXY MODEL INVENTORY")
        print("=" * 50)
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.proxy_url}/api/tags")
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    local_models = [m for m in models if "specialty" not in m]
                    remote_models = [m for m in models if "specialty" in m]
                    
                    self.log_audit("PROXY", "Model Inventory", "PASS", {
                        "total_models": len(models),
                        "local_models": len(local_models),
                        "remote_models": len(remote_models),
                        "local_model_names": [m['name'] for m in local_models],
                        "remote_model_names": [m['name'] for m in remote_models]
                    })
                else:
                    self.log_audit("PROXY", "Model Inventory", "FAIL", {
                        "status_code": resp.status_code,
                        "response": resp.text
                    })
        except Exception as e:
            self.log_audit("PROXY", "Model Inventory", "FAIL", error=str(e))
    
    async def audit_ollama_model_generation(self):
        """Audit Ollama direct model generation"""
        print("\n🔍 AUDITING OLLAMA MODEL GENERATION")
        print("=" * 50)
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "model": "qwen2.5-coder:1.5b-base",
                    "prompt": "Write a simple Python function",
                    "stream": False
                }
                
                resp = await client.post(f"{self.ollama_url}/api/generate", json=payload)
                if resp.status_code == 200:
                    result = resp.json()
                    self.log_audit("OLLAMA", "Model Generation", "PASS", {
                        "status_code": resp.status_code,
                        "model_used": result.get("model"),
                        "response_length": len(result.get("response", "")),
                        "response_preview": result.get("response", "")[:100] + "..."
                    })
                else:
                    self.log_audit("OLLAMA", "Model Generation", "FAIL", {
                        "status_code": resp.status_code,
                        "response": resp.text
                    })
        except Exception as e:
            self.log_audit("OLLAMA", "Model Generation", "FAIL", error=str(e))
    
    async def audit_proxy_model_generation(self):
        """Audit proxy model generation"""
        print("\n🔍 AUDITING PROXY MODEL GENERATION")
        print("=" * 50)
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                payload = {
                    "model": "qwen2.5-coder:1.5b-base",
                    "prompt": "Write a simple Python function",
                    "stream": False
                }
                
                resp = await client.post(f"{self.proxy_url}/api/generate", json=payload)
                if resp.status_code == 200:
                    result = resp.json()
                    self.log_audit("PROXY", "Model Generation", "PASS", {
                        "status_code": resp.status_code,
                        "model_used": result.get("model"),
                        "response_length": len(result.get("response", "")),
                        "response_preview": result.get("response", "")[:100] + "..."
                    })
                else:
                    self.log_audit("PROXY", "Model Generation", "FAIL", {
                        "status_code": resp.status_code,
                        "response": resp.text
                    })
        except Exception as e:
            self.log_audit("PROXY", "Model Generation", "FAIL", error=str(e))
    
    async def audit_task_routing(self):
        """Audit task-based routing"""
        print("\n🔍 AUDITING TASK-BASED ROUTING")
        print("=" * 50)
        
        tasks = [
            ("code", "Write a function to sort a list"),
            ("strategy", "Give me a simple plan"),
            ("fast", "Quick answer please")
        ]
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for task, prompt in tasks:
                    payload = {
                        "prompt": prompt,
                        "task": task,
                        "stream": False
                    }
                    
                    try:
                        resp = await client.post(f"{self.proxy_url}/api/generate", json=payload)
                        if resp.status_code == 200:
                            result = resp.json()
                            self.log_audit("PROXY", f"Task Routing ({task})", "PASS", {
                                "model_selected": result.get("model"),
                                "response_length": len(result.get("response", ""))
                            })
                        else:
                            self.log_audit("PROXY", f"Task Routing ({task})", "FAIL", {
                                "status_code": resp.status_code,
                                "expected_error": "API key authentication required"
                            })
                    except Exception as e:
                        self.log_audit("PROXY", f"Task Routing ({task})", "FAIL", error=str(e))
                    
                    await asyncio.sleep(1)
        except Exception as e:
            self.log_audit("PROXY", "Task Routing", "FAIL", error=str(e))
    
    async def audit_chat_interface(self):
        """Audit chat interface"""
        print("\n🔍 AUDITING CHAT INTERFACE")
        print("=" * 50)
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {
                    "messages": [
                        {"role": "user", "content": "What is the capital of France?"}
                    ],
                    "task": "default",
                    "stream": False
                }
                
                resp = await client.post(f"{self.proxy_url}/api/chat", json=payload)
                if resp.status_code == 200:
                    result = resp.json()
                    self.log_audit("PROXY", "Chat Interface", "PASS", {
                        "model_used": result.get("model"),
                        "response_length": len(result.get("response", ""))
                    })
                else:
                    self.log_audit("PROXY", "Chat Interface", "FAIL", {
                        "status_code": resp.status_code,
                        "expected_error": "API key authentication required"
                    })
        except Exception as e:
            self.log_audit("PROXY", "Chat Interface", "FAIL", error=str(e))
    
    async def audit_proxy_configuration(self):
        """Audit proxy configuration files"""
        print("\n🔍 AUDITING PROXY CONFIGURATION")
        print("=" * 50)
        
        # Check main proxy file
        proxy_file = "backend/services/institutional_proxy.py"
        if os.path.exists(proxy_file):
            self.log_audit("CONFIG", "Proxy File Exists", "PASS", {
                "file_path": proxy_file,
                "file_size": os.path.getsize(proxy_file)
            })
        else:
            self.log_audit("CONFIG", "Proxy File Exists", "FAIL", {
                "file_path": proxy_file,
                "error": "File not found"
            })
        
        # Check startup scripts
        startup_files = [
            "start_proxy_for_aria.py",
            "scripts/start_institutional_proxy.ps1"
        ]
        
        for file_path in startup_files:
            if os.path.exists(file_path):
                self.log_audit("CONFIG", f"Startup Script: {file_path}", "PASS", {
                    "file_path": file_path,
                    "file_size": os.path.getsize(file_path)
                })
            else:
                self.log_audit("CONFIG", f"Startup Script: {file_path}", "FAIL", {
                    "file_path": file_path,
                    "error": "File not found"
                })
    
    def generate_audit_summary(self):
        """Generate comprehensive audit summary"""
        print("\n" + "=" * 60)
        print("📊 SYSTEM AUDIT SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.audit_results)
        passed_tests = len([r for r in self.audit_results if r["status"] == "PASS"])
        failed_tests = len([r for r in self.audit_results if r["status"] == "FAIL"])
        
        print(f"📈 Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"📊 Success Rate: {(passed_tests / total_tests * 100):.1f}%" if total_tests > 0 else "0%")
        
        # Component breakdown
        components = {}
        for result in self.audit_results:
            comp = result["component"]
            if comp not in components:
                components[comp] = {"total": 0, "passed": 0, "failed": 0}
            components[comp]["total"] += 1
            if result["status"] == "PASS":
                components[comp]["passed"] += 1
            else:
                components[comp]["failed"] += 1
        
        print("\n📋 Component Breakdown:")
        for comp, stats in components.items():
            success_rate = (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            print(f"   {comp}: {stats['passed']}/{stats['total']} ({success_rate:.1f}%)")
        
        # Critical issues
        critical_issues = [r for r in self.audit_results if r["status"] == "FAIL" and "Generation" in r["test"]]
        if critical_issues:
            print(f"\n🚨 Critical Issues Found: {len(critical_issues)}")
            for issue in critical_issues:
                print(f"   - {issue['component']}: {issue['test']}")
                if issue.get("error"):
                    print(f"     Error: {issue['error']}")
        
        # Save audit results
        audit_file = f"system_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        audit_data = {
            "audit_time": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": f"{(passed_tests / total_tests * 100):.1f}%" if total_tests > 0 else "0%"
            },
            "components": components,
            "critical_issues": len(critical_issues),
            "results": self.audit_results
        }
        
        with open(audit_file, 'w') as f:
            json.dump(audit_data, f, indent=2)
        
        print(f"\n📄 Audit results saved to: {audit_file}")
        
        # Final recommendations
        print("\n🎯 RECOMMENDATIONS:")
        if passed_tests >= total_tests * 0.7:
            print("✅ System is mostly operational")
            if critical_issues:
                print("🔧 Fix critical generation issues for full functionality")
        else:
            print("❌ System needs significant attention")
            print("🔧 Review all failed components")
        
        return audit_data
    
    async def run_complete_audit(self):
        """Run complete system audit"""
        print("🔍 COMPREHENSIVE SYSTEM AUDIT")
        print("=" * 60)
        print(f"⏰ Audit Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Run all audit components
        await self.audit_proxy_health()
        await self.audit_ollama_health()
        await self.audit_proxy_model_inventory()
        await self.audit_ollama_model_generation()
        await self.audit_proxy_model_generation()
        await self.audit_task_routing()
        await self.audit_chat_interface()
        await self.audit_proxy_configuration()
        
        # Generate summary
        return self.generate_audit_summary()

async def main():
    """Main audit function"""
    audit = SystemAudit()
    results = await audit.run_complete_audit()
    
    print("\n🎉 System audit complete!")
    return results

if __name__ == "__main__":
    asyncio.run(main())
