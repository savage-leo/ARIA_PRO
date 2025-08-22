#!/usr/bin/env python3
"""
Enhanced ARIA PRO Institutional Proxy - Deployment Script
Captures comprehensive logs for LLM review and monitoring
"""

import asyncio
import subprocess
import time
import json
import os
import sys
from datetime import datetime
import httpx
import threading
import queue
import signal

class ProxyDeployment:
    def __init__(self):
        self.proxy_process = None
        self.log_queue = queue.Queue()
        self.is_running = False
        self.log_file = f"proxy_deployment_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.logs = []
        
    def log_event(self, event_type, message, data=None):
        """Log an event with timestamp and metadata"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message,
            "data": data or {}
        }
        self.logs.append(log_entry)
        self.log_queue.put(log_entry)
        print(f"[{log_entry['timestamp']}] {event_type.upper()}: {message}")
        
    def save_logs(self):
        """Save all logs to JSON file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
        print(f"📄 Logs saved to: {self.log_file}")
        
    def start_proxy(self):
        """Start the institutional proxy"""
        try:
            self.log_event("deployment", "Starting Enhanced ARIA PRO Institutional Proxy")
            
            # Change to the correct directory
            proxy_path = os.path.join(os.getcwd(), "backend", "services", "institutional_proxy.py")
            if not os.path.exists(proxy_path):
                self.log_event("error", f"Proxy file not found: {proxy_path}")
                return False
                
            # Start the proxy process
            self.proxy_process = subprocess.Popen(
                [sys.executable, proxy_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.log_event("deployment", f"Proxy process started with PID: {self.proxy_process.pid}")
            return True
            
        except Exception as e:
            self.log_event("error", f"Failed to start proxy: {str(e)}")
            return False
    
    def monitor_proxy_output(self):
        """Monitor proxy stdout and stderr"""
        if not self.proxy_process:
            return
            
        def read_output(stream, stream_type):
            try:
                for line in stream:
                    if line.strip():
                        self.log_event("proxy_output", line.strip(), {"stream": stream_type})
            except Exception as e:
                self.log_event("error", f"Error reading {stream_type}: {str(e)}")
        
        # Start monitoring threads
        stdout_thread = threading.Thread(target=read_output, args=(self.proxy_process.stdout, "stdout"))
        stderr_thread = threading.Thread(target=read_output, args=(self.proxy_process.stderr, "stderr"))
        
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        
        stdout_thread.start()
        stderr_thread.start()
        
        self.log_event("monitoring", "Started monitoring proxy output streams")
    
    async def health_check(self):
        """Perform comprehensive health checks"""
        self.log_event("health_check", "Starting comprehensive health checks")
        
        checks = []
        
        # Check if proxy is responding
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Health endpoint
                resp = await client.get("http://localhost:11435/healthz")
                checks.append({
                    "endpoint": "/healthz",
                    "status": resp.status_code,
                    "response": resp.json() if resp.status_code == 200 else None
                })
                
                # Model listing
                resp = await client.get("http://localhost:11435/api/tags")
                if resp.status_code == 200:
                    models = resp.json().get("models", [])
                    local_models = [m for m in models if "specialty" not in m]
                    remote_models = [m for m in models if "specialty" in m]
                    checks.append({
                        "endpoint": "/api/tags",
                        "status": resp.status_code,
                        "total_models": len(models),
                        "local_models": len(local_models),
                        "remote_models": len(remote_models),
                        "local_model_names": [m['name'] for m in local_models],
                        "remote_model_names": [m['name'] for m in remote_models]
                    })
                else:
                    checks.append({
                        "endpoint": "/api/tags",
                        "status": resp.status_code,
                        "error": resp.text
                    })
                
        except Exception as e:
            checks.append({
                "error": f"Health check failed: {str(e)}"
            })
        
        self.log_event("health_check", "Health checks completed", {"checks": checks})
        return checks
    
    async def test_local_model(self):
        """Test local model functionality"""
        self.log_event("testing", "Testing local model generation")
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {
                    "model": "qwen2.5-coder:1.5b-base",
                    "prompt": "Write a simple Python function to add two numbers",
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 100
                    }
                }
                
                resp = await client.post("http://localhost:11435/api/generate", json=payload)
                
                if resp.status_code == 200:
                    result = resp.json()
                    self.log_event("testing", "Local model test successful", {
                        "model": result.get("model"),
                        "response_length": len(result.get("response", "")),
                        "response_preview": result.get("response", "")[:200]
                    })
                    return True
                else:
                    self.log_event("testing", "Local model test failed", {
                        "status_code": resp.status_code,
                        "error": resp.text
                    })
                    return False
                    
        except Exception as e:
            self.log_event("testing", f"Local model test error: {str(e)}")
            return False
    
    async def test_task_routing(self):
        """Test task-based routing functionality"""
        self.log_event("testing", "Testing task-based routing")
        
        tasks = [
            ("code", "Write a function to sort a list"),
            ("strategy", "Give me a simple plan"),
            ("fast", "Quick answer please")
        ]
        
        results = []
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for task, prompt in tasks:
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
                        resp = await client.post("http://localhost:11435/api/generate", json=payload)
                        results.append({
                            "task": task,
                            "status_code": resp.status_code,
                            "success": resp.status_code == 200,
                            "error": resp.text if resp.status_code != 200 else None
                        })
                    except Exception as e:
                        results.append({
                            "task": task,
                            "error": str(e)
                        })
                    
                    await asyncio.sleep(1)
                    
        except Exception as e:
            self.log_event("testing", f"Task routing test error: {str(e)}")
        
        self.log_event("testing", "Task routing tests completed", {"results": results})
        return results
    
    async def deployment_summary(self):
        """Generate comprehensive deployment summary"""
        self.log_event("summary", "Generating deployment summary")
        
        summary = {
            "deployment_time": datetime.now().isoformat(),
            "proxy_pid": self.proxy_process.pid if self.proxy_process else None,
            "proxy_running": self.proxy_process and self.proxy_process.poll() is None,
            "total_log_entries": len(self.logs),
            "log_file": self.log_file
        }
        
        # Add health check results
        health_checks = await self.health_check()
        summary["health_checks"] = health_checks
        
        # Add test results
        local_test = await self.test_local_model()
        task_tests = await self.test_task_routing()
        
        summary["tests"] = {
            "local_model": local_test,
            "task_routing": task_tests
        }
        
        self.log_event("summary", "Deployment summary generated", summary)
        return summary
    
    def stop_proxy(self):
        """Stop the proxy process"""
        if self.proxy_process:
            self.log_event("deployment", "Stopping proxy process")
            self.proxy_process.terminate()
            try:
                self.proxy_process.wait(timeout=10)
                self.log_event("deployment", "Proxy process stopped gracefully")
            except subprocess.TimeoutExpired:
                self.proxy_process.kill()
                self.log_event("deployment", "Proxy process force killed")
    
    async def run_deployment(self):
        """Run the complete deployment process"""
        try:
            self.is_running = True
            
            # Start proxy
            if not self.start_proxy():
                return False
            
            # Start monitoring
            self.monitor_proxy_output()
            
            # Wait for proxy to start
            self.log_event("deployment", "Waiting for proxy to start...")
            await asyncio.sleep(5)
            
            # Run comprehensive tests
            summary = await self.deployment_summary()
            
            # Save logs
            self.save_logs()
            
            # Print summary
            print("\n" + "="*60)
            print("🚀 ENHANCED ARIA PRO INSTITUTIONAL PROXY DEPLOYMENT")
            print("="*60)
            print(f"📄 Logs saved to: {self.log_file}")
            print(f"🔄 Proxy PID: {summary['proxy_pid']}")
            print(f"✅ Proxy Running: {summary['proxy_running']}")
            print(f"📊 Total Log Entries: {summary['total_log_entries']}")
            
            if summary.get('health_checks'):
                health = summary['health_checks']
                print(f"🏥 Health Status: {len([h for h in health if 'error' not in h])}/{len(health)} endpoints healthy")
            
            if summary.get('tests'):
                tests = summary['tests']
                print(f"🧪 Local Model Test: {'✅' if tests['local_model'] else '❌'}")
                task_success = len([t for t in tests['task_routing'] if t.get('success')])
                print(f"🎯 Task Routing Tests: {task_success}/{len(tests['task_routing'])} successful")
            
            print("\n📋 Next Steps:")
            print("1. Point ARIA to http://localhost:11435")
            print("2. Update API keys for remote model functionality")
            print("3. Monitor logs in the generated JSON file")
            print("\n🎉 Deployment Complete!")
            
            return True
            
        except Exception as e:
            self.log_event("error", f"Deployment failed: {str(e)}")
            return False
        finally:
            self.is_running = False

async def main():
    """Main deployment function"""
    deployment = ProxyDeployment()
    
    # Handle graceful shutdown
    def signal_handler(signum, frame):
        print("\n🛑 Shutdown signal received...")
        deployment.stop_proxy()
        deployment.save_logs()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run deployment
    success = await deployment.run_deployment()
    
    if success:
        print("\n✅ Deployment successful! Proxy is running and ready for ARIA integration.")
        print("📄 Review the logs in the generated JSON file for detailed information.")
    else:
        print("\n❌ Deployment failed. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
