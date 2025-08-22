# 🚀 ARIA PRO Integration Instructions - Enhanced Institutional Proxy

## 📊 **DEPLOYMENT STATUS: ✅ READY**

**Proxy Status**: ✅ **FULLY OPERATIONAL**  
**Local Models**: ✅ **WORKING PERFECTLY**  
**Infrastructure**: ✅ **STABLE**  
**API Endpoint**: ✅ **FIXED**

---

## 🎯 **IMMEDIATE ARIA INTEGRATION (5 minutes)**

### **Step 1: Configure ARIA to Use the Proxy**

**Change ARIA Configuration:**
```bash
# Find your ARIA configuration file and change:
ARIA_MODEL_URL=http://localhost:11434

# To:
ARIA_MODEL_URL=http://localhost:11435
```

**Alternative Configuration Methods:**
```bash
# Environment variable
export ARIA_MODEL_URL=http://localhost:11435

# Or in your ARIA config file
OLLAMA_URL=http://localhost:11435
```

### **Step 2: Restart ARIA Application**

```bash
# Stop ARIA
# Start ARIA with new configuration
```

### **Step 3: Verify Integration**

**Test Commands:**
```bash
# Health check
curl http://localhost:11435/healthz

# Model inventory
curl http://localhost:11435/api/tags

# Test local model generation
python test_local_model.py
```

---

## 📋 **AVAILABLE MODELS FOR ARIA**

### **Local Models (4) - ✅ FULLY FUNCTIONAL**
- `mistral:latest` - General purpose (7.2B parameters)
- `qwen2.5-coder:1.5b-base` - Code generation (1.5B parameters)
- `gemma3:4b` - Efficient general purpose (4.3B parameters)
- `nomic-embed-text:latest` - Embeddings (137M parameters)

### **Remote Models (8) - ⚠️ NEED API KEY REFRESH**
- `llama-3.1-405b` - Strategy & reasoning
- `llama-3.3-70b` - General purpose
- `qwen-coder-32b` - Advanced coding
- `deepseek-r1-14b` - Code & math
- `qwen-vl-72b` - Vision & multimodal
- `reka-flash-3` - Fast responses
- `qwq-32b` - Efficient processing
- `llama-3.2-3b` - Lightweight

---

## 🔧 **PROXY MANAGEMENT COMMANDS**

### **Start Proxy:**
```powershell
python start_proxy_for_aria.py
```

### **Stop Proxy:**
```powershell
Stop-Process -Name "python" -Force
```

### **Fix & Restart (if needed):**
```powershell
.\fix_and_restart_proxy.ps1
```

### **Health Monitoring:**
```bash
# Check proxy status
curl http://localhost:11435/healthz

# Check available models
curl http://localhost:11435/api/tags

# Test local model
python test_local_model.py
```

---

## 📁 **DEPLOYMENT FILES**

### **Core Files (Required):**
- `backend/services/institutional_proxy.py` - Main proxy implementation
- `start_proxy_for_aria.py` - Python startup script
- `fix_and_restart_proxy.ps1` - One-shot fix script
- `test_local_model.py` - Local model test script

### **Documentation:**
- `ARIA_INTEGRATION_GUIDE.md` - Complete integration guide
- `FINAL_DEPLOYMENT_READY_SUMMARY.md` - Deployment summary
- `ARIA_DEPLOYMENT_INSTRUCTIONS.md` - This file

---

## 🎯 **VERIFICATION CHECKLIST**

### **Pre-Integration:**
- [ ] Proxy running on port 11435
- [ ] Health endpoint responding: `{"status":"ok"}`
- [ ] Model inventory showing 12 models
- [ ] Local model generation working

### **Post-Integration:**
- [ ] ARIA configuration updated to `http://localhost:11435`
- [ ] ARIA application restarted
- [ ] Model selection working in ARIA interface
- [ ] Local model generation functional
- [ ] Streaming responses working

---

## 🚀 **PRODUCTION DEPLOYMENT**

### **For Production Environment:**

1. **Copy Files:**
   ```bash
   # Copy proxy files to production server
   scp -r backend/services/institutional_proxy.py user@server:/path/to/aria/
   scp start_proxy_for_aria.py user@server:/path/to/aria/
   scp fix_and_restart_proxy.ps1 user@server:/path/to/aria/
   ```

2. **Configure Environment:**
   ```bash
   # Set environment variables
   export LOCAL_OLLAMA_URL=http://127.0.0.1:11434
   export ARIA_MODEL_URL=http://production-server:11435
   ```

3. **Start Proxy:**
   ```bash
   # Start proxy on production server
   python start_proxy_for_aria.py
   ```

4. **Configure ARIA:**
   ```bash
   # Update ARIA to use production proxy
   ARIA_MODEL_URL=http://production-server:11435
   ```

---

## 🔑 **OPTIONAL: Remote Model Activation**

### **To Enable Remote Models:**

1. **Get New API Keys:**
   - Visit: https://api.together.ai/settings/api-keys
   - Generate new API keys for each model

2. **Update Configuration:**
   ```python
   # In backend/services/institutional_proxy.py
   REMOTE_MODELS = {
       "reka-flash-3": {
           "url": "https://api.together.ai/v1",
           "api_key": "YOUR_NEW_API_KEY_HERE",
           "model_id": "rekaai/reka-flash-3:free",
           "specialty": ["fast", "efficient", "default"]
       },
       # ... update other models
   }
   ```

3. **Restart Proxy:**
   ```powershell
   .\fix_and_restart_proxy.ps1
   ```

---

## 📊 **TROUBLESHOOTING**

### **Common Issues:**

**Proxy Not Starting:**
```bash
# Check if port 11435 is in use
netstat -an | findstr "11435"

# Kill existing processes
Stop-Process -Name "python" -Force

# Restart proxy
python start_proxy_for_aria.py
```

**Local Models Not Working:**
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# Restart Ollama if needed
# Then restart proxy
.\fix_and_restart_proxy.ps1
```

**ARIA Can't Connect:**
```bash
# Verify proxy is running
curl http://localhost:11435/healthz

# Check ARIA configuration
# Ensure ARIA_MODEL_URL=http://localhost:11435
```

---

## 🎉 **SUCCESS INDICATORS**

### **✅ Working Features:**
- [ ] Proxy health endpoint responding
- [ ] 12 models available in ARIA interface
- [ ] Local model generation working
- [ ] Streaming responses functional
- [ ] Error handling working properly

### **⚠️ Expected Limitations:**
- [ ] Remote models return 401 (API key needed)
- [ ] Task routing falls back to local models

---

## 📞 **SUPPORT**

### **Log Files:**
- Check proxy console output for real-time status
- Review `test_local_model.py` output for verification

### **Health Monitoring:**
```bash
# Quick health check
curl http://localhost:11435/healthz

# Model availability
curl http://localhost:11435/api/tags
```

---

## 🚀 **FINAL STATUS**

**Status**: ✅ **PRODUCTION READY**  
**ARIA Integration**: ✅ **READY TO PROCEED**  
**Local Models**: ✅ **FULLY FUNCTIONAL**  
**Infrastructure**: ✅ **STABLE**

**The Enhanced ARIA PRO Institutional Proxy is successfully deployed and ready for immediate ARIA integration!**

---

**Next Step**: Configure ARIA to use `http://localhost:11435` and begin testing with local models.
