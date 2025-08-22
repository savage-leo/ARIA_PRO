#!/usr/bin/env python3
# ----------------------------------------------------------------------
# ARIA Ollama Router — Aria ⇄ Ollama (remote/local) Brain, Logging, Optional Gui Mirroring
# ----------------------------------------------------------------------

import argparse
import json
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Generator
import requests
import os

# Optional GUI mirroring
try:
    import pyautogui
    import pygetwindow as gw
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False

class OllamaClient:
    """Client for Ollama API with institutional proxy support"""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint.rstrip('/')
        
    def _make_request(self, method: str, path: str, **kwargs) -> requests.Response:
        """Make HTTP request to Ollama endpoint"""
        url = f"{self.endpoint}{path}"
        return requests.request(method, url, **kwargs)
    
    def discover_models(self) -> List[Dict[str, Any]]:
        """Discover available models"""
        resp = self._make_request("GET", "/api/tags")
        resp.raise_for_status()
        data = resp.json()
        return data.get("models", [])
    
    def generate(self, model: str, prompt: str, stream: bool = False, options: Dict[str, Any] = None) -> Generator[str, None, None]:
        """Generate text using Ollama"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": options or {}
        }
        
        resp = self._make_request("POST", "/api/generate", json=payload, stream=stream)
        resp.raise_for_status()
        
        if stream:
            for line in resp.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode())
                        if chunk.get("done", False):
                            break
                        yield chunk.get("response", "")
                    except json.JSONDecodeError:
                        continue
        else:
            data = resp.json()
            yield data.get("response", "")
    
    def chat(self, model: str, messages: List[Dict[str, str]], stream: bool = False, options: Dict[str, Any] = None) -> Generator[str, None, None]:
        """Chat using Ollama"""
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": options or {}
        }
        
        resp = self._make_request("POST", "/api/chat", json=payload, stream=stream)
        resp.raise_for_status()
        
        if stream:
            for line in resp.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode())
                        if chunk.get("done", False):
                            break
                        yield chunk.get("response", "")
                    except json.JSONDecodeError:
                        continue
        else:
            data = resp.json()
            yield data.get("response", "")

class AriaBrainRouter:
    """Intelligent model routing based on task type"""
    
    def __init__(self, available_models: List[str]):
        self.available_models = [m.lower() for m in available_models]
        
        # Task to model mapping with GPT-OS 120B as default
        self.task_routing = {
            "strategy": ["gptos-120b", "120b", "llama-3.1-405b", "mistral:latest"],
            "memory": ["gptos-120b", "qwen2.5:72b-instruct", "120b", "mistral:latest"],
            "code": ["gptos-120b", "qwen2.5-coder:32b-instruct", "qwen2.5-coder:1.5b-base", "120b"],
            "execute": ["gptos-120b", "mistral-nemo:12b", "mistral:latest", "120b"],
            "vision": ["gptos-120b", "llama-3.2-11b-vision", "120b"],
            "default": ["gptos-120b", "120b", "mistral:latest"]
        }
    
    def pick(self, task: str) -> str:
        """Pick the best available model for the task"""
        task_models = self.task_routing.get(task.lower(), self.task_routing["default"])
        
        for model in task_models:
            if any(model.lower() in available for available in self.available_models):
                return model
        
        # Fallback to first available model
        return self.available_models[0] if self.available_models else "gptos-120b"

class AriaLogger:
    """Logging for ARIA interactions"""
    
    def __init__(self, log_file: str = "aria_llm_log.jsonl"):
        self.log_file = log_file
    
    def log(self, record: Dict[str, Any]):
        """Log a record to JSONL file"""
        record["timestamp"] = datetime.now().isoformat()
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

class GuiMirror:
    """Optional GUI mirroring for Ollama desktop app"""
    
    def __init__(self):
        if not GUI_AVAILABLE:
            raise ImportError("GUI mirroring requires pyautogui and pygetwindow")
        
        self.ollama_window = None
        self._find_ollama_window()
    
    def _find_ollama_window(self):
        """Find Ollama desktop app window"""
        try:
            windows = gw.getAllTitles()
            for title in windows:
                if "ollama" in title.lower():
                    self.ollama_window = gw.getWindowsWithTitle(title)[0]
                    break
        except Exception as e:
            print(f"[!] Could not find Ollama window: {e}")
    
    def type_prompt(self, prompt: str):
        """Type prompt into Ollama GUI"""
        if self.ollama_window:
            try:
                self.ollama_window.activate()
                time.sleep(0.5)
                pyautogui.write(prompt)
            except Exception as e:
                print(f"[!] GUI mirror error: {e}")
    
    def show_response(self, response: str):
        """Show response in Ollama GUI"""
        if self.ollama_window:
            try:
                self.ollama_window.activate()
                time.sleep(0.5)
                # This would need to be adapted based on Ollama GUI structure
                print(f"[GUI] Response displayed: {response[:100]}...")
            except Exception as e:
                print(f"[!] GUI mirror error: {e}")

def discover_models(endpoint: str) -> List[str]:
    """Discover models on endpoint"""
    try:
        resp = requests.get(f"{endpoint.rstrip('/')}/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [model.get("name", "") for model in data.get("models", [])]
    except Exception as e:
        print(f"[!] Model discovery failed: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="ARIA Ollama Router")
    parser.add_argument("--endpoint", default="http://localhost:11435", 
                       help="Ollama endpoint (default: institutional proxy)")
    parser.add_argument("--model", help="Force specific model")
    parser.add_argument("--task", choices=["strategy", "memory", "code", "execute", "vision"],
                       help="Task type for intelligent routing")
    parser.add_argument("--prompt", required=True, help="Input prompt")
    parser.add_argument("--chat", action="store_true", help="Use chat mode")
    parser.add_argument("--stream", action="store_true", help="Enable streaming")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--log", default="aria_llm_log.jsonl", help="Log file")
    parser.add_argument("--mirror-gui", action="store_true", help="Mirror to Ollama GUI")
    
    args = parser.parse_args()
    
    # Initialize client
    client = OllamaClient(args.endpoint)
    
    # Discover models on this host to drive routing
    try:
        models = discover_models(args.endpoint)
    except Exception as e:
        print(f"[!] Failed to discover models on {args.endpoint}: {e}", file=sys.stderr)
        models = []
    
    # Choose model
    chosen_model = args.model
    if not chosen_model:
        router = AriaBrainRouter(models or ["gptos-120b"])  # fallback to GPT-OS 120B
        chosen_model = router.pick(args.task or "default")
    
    # Initialize logger
    logger = AriaLogger(args.log)
    
    # Optional GUI mirror
    gui = None
    if args.mirror_gui:
        if GUI_AVAILABLE:
            try:
                gui = GuiMirror()
            except ImportError:
                print("[!] GUI mirroring not available - install pyautogui pygetwindow")
        else:
            print("[!] GUI mirroring not available - install pyautogui pygetwindow")
    
    # Prepare options
    options = {"temperature": args.temperature}
    
    # Send prompt
    if gui:
        gui.type_prompt(args.prompt)
    
    print(f"[ARIA] endpoint={args.endpoint} model={chosen_model} task={args.task or 'default'}")
    
    start = time.time()
    response_text = ""
    
    try:
        if args.chat:
            messages = [{"role": "user", "content": args.prompt}]
            stream_iter = client.chat(chosen_model, messages, stream=args.stream, options=options)
        else:
            stream_iter = client.generate(chosen_model, args.prompt, stream=args.stream, options=options)
        
        if args.stream:
            for chunk in stream_iter:
                response_text += chunk
                print(chunk, end="", flush=True)
            print()
        else:
            # non-stream: generator will produce a single full string
            response_text = "".join(stream_iter)
            print(response_text)
    except requests.HTTPError as e:
        print(f"\n[!] HTTP error from Ollama: {e}")
        sys.exit(2)
    except requests.RequestException as e:
        print(f"\n[!] Network error: {e}")
        sys.exit(3)
    
    dur = time.time() - start
    
    # Mirror response into GUI if requested
    if gui and response_text:
        gui.show_response(response_text)
    
    # Log record
    logger.log({
        "endpoint": args.endpoint,
        "task": args.task or "default",
        "model": chosen_model,
        "prompt": args.prompt,
        "response": response_text,
        "duration_sec": round(dur, 3),
        "options": options,
    })
    
    # Exit status 0 = success

if __name__ == "__main__":
    main()
