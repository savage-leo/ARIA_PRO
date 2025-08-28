# ARIA PRO v1.2 Production Deployment Guide

## 🚀 Pre-Deployment Checklist

### System Requirements
- **OS**: Windows 10/11 Pro or Windows Server 2019+
- **Python**: 3.11+ with virtual environment
- **Redis**: 6.0+ for caching and distributed locking
- **Memory**: 16GB+ RAM recommended
- **Storage**: 100GB+ SSD for optimal performance
- **Network**: Stable internet for MT5 connectivity

### Dependencies Verification
```powershell
# Verify Python version
python --version  # Should be 3.11+

# Verify Redis availability
redis-cli ping  # Should return PONG

# Check MT5 installation
# Ensure MetaTrader 5 is installed and configured
```

## 📋 Environment Configuration

### 1. Production Environment File
Create `ARIA_PRO/production.env` with institutional-grade settings:

```env
# === CORE SYSTEM ===
ARIA_ENVIRONMENT=production
ARIA_DEBUG=0
ARIA_LOG_LEVEL=INFO

# === MT5 CONFIGURATION ===
ARIA_ENABLE_MT5=1
MT5_LOGIN=your_live_account
MT5_PASSWORD=your_secure_password
MT5_SERVER=your_broker_server
MT5_PATH=C:\Program Files\MetaTrader 5\terminal64.exe

# === TRADING CONFIGURATION ===
AUTO_TRADE_ENABLED=1
AUTO_TRADE_DRY_RUN=0
AUTO_TRADE_SYMBOLS=EURUSD,GBPUSD,USDJPY,XAUUSD,BTCUSD
AUTO_TRADE_CB_COOLDOWN=3600
AUTO_TRADE_POS_LIMITS=EURUSD:2,GBPUSD:2,USDJPY:1,XAUUSD:1,BTCUSD:1
AUTO_TRADE_BASE_TIMEOUT=30
AUTO_TRADE_MAX_TIMEOUT=120

# === RISK MANAGEMENT ===
RISK_MAX_POSITION_SIZE=0.01
RISK_MAX_DAILY_LOSS=1000
RISK_VAR_CONFIDENCE=0.95
RISK_CVAR_CONFIDENCE=0.99
RISK_KELLY_MAX_SIZE=0.25
RISK_LOOKBACK_DAYS=252

# === AI MODELS ===
ARIA_INCLUDE_XGB=1
ARIA_INCLUDE_LSTM=1
ARIA_INCLUDE_CNN=1
AI_CIRCUIT_BREAKER_THRESHOLD=0.3
AI_PERFORMANCE_WINDOW=100
AI_BASE_ANALYSIS_INTERVAL=60
AI_MODEL_TIMEOUT=30

# === SECURITY ===
JWT_SECRET_KEY=your_32_char_production_secret_key
ADMIN_API_KEY=your_16_char_admin_key
JWT_ROTATION_HOURS=24
JWT_SECRET_OVERLAP_HOURS=2
JWT_MAX_SECRETS=5

# === REDIS CONFIGURATION ===
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your_redis_password
REDIS_MAX_CONNECTIONS=20

# === CORS & SECURITY ===
ARIA_CORS_ORIGINS=https://your-domain.com,https://app.your-domain.com
ARIA_ALLOWED_HOSTS=your-domain.com,app.your-domain.com
ARIA_ENABLE_HTTPS=1

# === MONITORING & ALERTING ===
MONITORING_ENABLED=1
ALERT_EMAIL_ENABLED=1
ALERT_EMAIL_SMTP_HOST=smtp.your-provider.com
ALERT_EMAIL_SMTP_PORT=587
ALERT_EMAIL_USERNAME=alerts@your-domain.com
ALERT_EMAIL_PASSWORD=your_email_password
ALERT_EMAIL_TO=admin@your-domain.com
ALERT_WEBHOOK_URL=https://your-webhook-endpoint.com

# === BACKUP CONFIGURATION ===
BACKUP_RETENTION_DAYS=30
BACKUP_MAX_COUNT=100
BACKUP_COMPRESSION=6
BACKUP_VERIFY=1

# === PERFORMANCE TUNING ===
UVICORN_WORKERS=4
UVICORN_MAX_REQUESTS=1000
UVICORN_TIMEOUT_KEEP_ALIVE=5
```

### 2. Security Hardening
```powershell
# Set restrictive file permissions
icacls "ARIA_PRO\production.env" /grant:r "%USERNAME%:R" /inheritance:r

# Secure log directories
icacls "ARIA_PRO\logs" /grant:r "%USERNAME%:F" /inheritance:r

# Protect model files
icacls "ARIA_PRO\backend\models" /grant:r "%USERNAME%:R" /inheritance:r
```

## 🔧 Installation Steps

### 1. System Preparation
```powershell
# Clone/update ARIA PRO
cd C:\
git clone https://github.com/your-org/ARIA_PRO.git
cd ARIA_PRO

# Create production virtual environment
python -m venv .venv_prod
.venv_prod\Scripts\activate

# Install production dependencies
pip install -r backend\requirements.txt
pip install gunicorn uvicorn[standard] psutil
```

### 2. Database Initialization
```powershell
# Initialize SQLite databases
python -c "
import sqlite3
import os

# Create data directory
os.makedirs('data', exist_ok=True)

# Initialize trade memory database
conn = sqlite3.connect('data/trade_memory.db')
conn.execute('''
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY,
        symbol TEXT,
        action TEXT,
        volume REAL,
        price REAL,
        timestamp TEXT,
        profit REAL
    )
''')
conn.close()
print('Database initialized')
"
```

### 3. Redis Setup
```powershell
# Install Redis (if not already installed)
# Download from https://github.com/microsoftarchive/redis/releases

# Start Redis service
redis-server --service-install
redis-server --service-start

# Verify Redis connectivity
redis-cli ping
```

### 4. SSL/TLS Configuration (Production)
```powershell
# Generate self-signed certificate for development
# For production, use proper SSL certificates from CA

openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes
```

## 🚀 Deployment Process

### 1. Pre-Deployment Validation
```powershell
# Run comprehensive test suite
python backend\tests\test_runner.py all

# Validate configuration
python -c "
from backend.core.config import get_settings
settings = get_settings()
print('✅ Configuration loaded successfully')
print(f'Environment: {settings.environment}')
print(f'MT5 Enabled: {settings.enable_mt5}')
print(f'Auto Trading: {settings.auto_trade_enabled}')
"

# Check component imports
python -c "
import sys
sys.path.append('.')

from backend.services.auto_trader import AutoTrader
from backend.services.risk_engine import RiskEngine  
from backend.core.comprehensive_monitoring import MonitoringSystem
from backend.core.distributed_lock import LockManager
from backend.core.jwt_rotation import JWTRotationManager

print('✅ All enhanced components imported successfully')
"
```

### 2. Production Startup
```powershell
# Start ARIA PRO backend
.venv_prod\Scripts\uvicorn.exe backend.main:app ^
    --host 0.0.0.0 ^
    --port 8001 ^
    --workers 4 ^
    --access-log ^
    --log-level info ^
    --ssl-keyfile key.pem ^
    --ssl-certfile cert.pem
```

### 3. Health Check Verification
```powershell
# Test health endpoints
curl -k https://localhost:8001/health/live
curl -k https://localhost:8001/health/ready
curl -k https://localhost:8001/monitoring/metrics

# Verify WebSocket connectivity
# Use WebSocket client to test wss://localhost:8001/ws
```

## 📊 Monitoring Setup

### 1. System Monitoring
```powershell
# Create monitoring script
echo '@echo off
:loop
curl -s https://localhost:8001/monitoring/metrics > temp_metrics.json
python -c "
import json
with open('temp_metrics.json') as f:
    metrics = json.load(f)
    print(f'CPU: {metrics.get(\"cpu_percent\", 0):.1f}%%')
    print(f'Memory: {metrics.get(\"memory_percent\", 0):.1f}%%')
    print(f'Active Trades: {metrics.get(\"active_positions\", 0)}')
"
timeout /t 30 > nul
goto loop' > monitor.bat

# Run monitoring
monitor.bat
```

### 2. Log Monitoring
```powershell
# Set up log rotation
# Create logs directory structure
mkdir logs\archive

# Monitor critical logs
Get-Content logs\backend.log -Wait -Tail 10
```

### 3. Alert Configuration
Ensure email/webhook alerts are configured in `production.env` for:
- Circuit breaker activations
- High CPU/memory usage (>80%)
- Trading errors
- MT5 connection issues
- Security violations

## 🔒 Security Checklist

### 1. Access Control
- [ ] Strong passwords for all accounts
- [ ] JWT secrets are 32+ characters
- [ ] Admin API keys are secure
- [ ] File permissions are restrictive
- [ ] Network access is limited

### 2. Data Protection
- [ ] Database files are backed up
- [ ] Sensitive data is encrypted
- [ ] Logs don't contain secrets
- [ ] API keys are not hardcoded

### 3. Network Security
- [ ] HTTPS/WSS enabled
- [ ] CORS origins restricted
- [ ] Rate limiting active
- [ ] Firewall configured

## 📦 Backup Procedures

### 1. Automated Backups
```powershell
# Create backup schedule
schtasks /create /tn "ARIA Backup" /tr "python C:\ARIA_PRO\scripts\backup_system.py backup --type full" /sc daily /st 02:00

# Test backup system
python scripts\backup_system.py backup --type full
python scripts\backup_system.py list
```

### 2. Recovery Testing
```powershell
# Test restore procedure (use test environment)
python scripts\backup_system.py restore --backup-file backups\aria_full_backup_20241225_020000.zip --target-dir test_restore
```

## 🔄 Maintenance Procedures

### Daily Tasks
- Monitor system health via `/monitoring/metrics`
- Check log files for errors
- Verify trading performance
- Review alert notifications

### Weekly Tasks
- Run full backup
- Update dependencies if needed
- Review security logs
- Performance analysis

### Monthly Tasks
- Security audit
- Backup retention cleanup
- System performance review
- Documentation updates

## 🚨 Troubleshooting

### Common Issues

#### MT5 Connection Failed
```powershell
# Check MT5 service
Get-Process -Name "terminal64" -ErrorAction SilentlyContinue

# Verify credentials
python -c "
import MetaTrader5 as mt5
if mt5.initialize():
    print('✅ MT5 connection successful')
    mt5.shutdown()
else:
    print('❌ MT5 connection failed')
"
```

#### Redis Connection Issues
```powershell
# Check Redis service
redis-cli ping

# Restart Redis if needed
redis-server --service-stop
redis-server --service-start
```

#### High Memory Usage
```powershell
# Check process memory
Get-Process -Name "python" | Select-Object ProcessName, WorkingSet

# Review memory metrics
curl -s https://localhost:8001/monitoring/metrics | python -c "
import json, sys
data = json.load(sys.stdin)
print(f'Memory: {data.get(\"memory_percent\", 0):.1f}%')
"
```

#### Circuit Breaker Activated
```powershell
# Check circuit breaker status
curl -s https://localhost:8001/monitoring/system/status | python -c "
import json, sys
data = json.load(sys.stdin)
print(f'Circuit Breaker: {data.get(\"circuit_breaker_active\", False)}')
"

# Reset if needed (admin access required)
curl -X POST https://localhost:8001/admin/reset-circuit-breaker \
  -H "X-Admin-Key: your_admin_key"
```

## 📈 Performance Optimization

### System Tuning
```env
# Add to production.env for high-performance deployment
UVICORN_WORKERS=8                    # 2x CPU cores
UVICORN_MAX_REQUESTS=2000           # Higher request limit
REDIS_MAX_CONNECTIONS=50            # More Redis connections
AI_MODEL_TIMEOUT=60                 # Longer model timeout
MONITORING_COLLECTION_INTERVAL=5    # More frequent monitoring
```

### Resource Monitoring
- CPU usage should stay below 70%
- Memory usage should stay below 80%
- Disk I/O should be minimal
- Network latency to MT5 should be <50ms

## 🎯 Success Metrics

### Trading Performance
- Signal generation: >50 signals/hour
- Order execution: <100ms average latency
- Risk calculations: <10ms per calculation
- Circuit breaker activations: <1 per day

### System Performance
- Uptime: >99.9%
- Response time: <200ms for API calls
- Memory usage: <80% of available
- Error rate: <0.1% of requests

### Security Metrics
- Failed authentication attempts: <10/hour
- Rate limit violations: <50/hour
- Security alerts: <5/day
- Backup success rate: 100%

This deployment guide ensures your ARIA PRO v1.2 system is production-ready with institutional-grade reliability, security, and performance.
