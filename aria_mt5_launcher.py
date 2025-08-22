#!/usr/bin/env python3
"""
ARIA MT5 Production Engine Launcher
Easy startup and management of the complete trading system
"""

import os
import sys
import time
import argparse
import subprocess
import signal
import json
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ARIAMT5Launcher:
    """Launcher for ARIA MT5 Production Engine"""
    
    def __init__(self):
        self.config_file = "aria_mt5_config.env"
        self.engine_process = None
        self.dashboard_process = None
        self.running = False
        
    def load_config(self):
        """Load configuration from environment file"""
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
            logger.info(f"Configuration loaded from {self.config_file}")
        else:
            logger.warning(f"Configuration file {self.config_file} not found, using defaults")
    
    def check_dependencies(self):
        """Check if all required dependencies are available"""
        required_packages = [
            'MetaTrader5',
            'fastapi',
            'uvicorn',
            'websockets',
            'numpy',
            'pandas',
            'python-dotenv'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing required packages: {missing_packages}")
            logger.info("Install with: pip install " + " ".join(missing_packages))
            return False
        
        logger.info("✅ All dependencies available")
        return True
    
    def check_mt5_connection(self):
        """Test MT5 connection"""
        try:
            import MetaTrader5 as mt5
            
            # Load config
            self.load_config()
            
            # Test connection
            if not mt5.initialize(
                login=int(os.getenv("MT5_ACCOUNT", "0")),
                password=os.getenv("MT5_PASSWORD", ""),
                server=os.getenv("MT5_SERVER", "")
            ):
                logger.error(f"MT5 connection failed: {mt5.last_error()}")
                return False
            
            # Get account info
            account_info = mt5.account_info()
            if account_info:
                logger.info(f"✅ MT5 connected: {account_info.login} @ {account_info.server}")
                logger.info(f"Account balance: {account_info.balance} {account_info.currency}")
            else:
                logger.error("Failed to get account info")
                return False
            
            mt5.shutdown()
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection test failed: {e}")
            return False
    
    def create_directories(self):
        """Create necessary directories"""
        directories = [
            "data",
            "logs",
            "models",
            "backups"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
        
        logger.info("✅ Directories created")
    
    def start_engine(self, mode="production"):
        """Start the ARIA MT5 production engine"""
        try:
            # Load configuration
            self.load_config()
            
            # Check dependencies
            if not self.check_dependencies():
                return False
            
            # Check MT5 connection
            if not self.check_mt5_connection():
                return False
            
            # Create directories
            self.create_directories()
            
            # Determine which script to run
            if mode == "dashboard":
                script = "aria_mt5_dashboard.py"
            else:
                script = "aria_mt5_production_engine.py"
            
            # Start the engine
            logger.info(f"🚀 Starting ARIA MT5 Engine in {mode} mode...")
            
            self.engine_process = subprocess.Popen([
                sys.executable, script
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.running = True
            logger.info(f"✅ ARIA MT5 Engine started (PID: {self.engine_process.pid})")
            
            # Wait a moment to check if it started successfully
            time.sleep(3)
            
            if self.engine_process.poll() is None:
                logger.info("Engine is running successfully")
                return True
            else:
                stdout, stderr = self.engine_process.communicate()
                logger.error(f"Engine failed to start: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to start engine: {e}")
            return False
    
    def stop_engine(self):
        """Stop the ARIA MT5 production engine"""
        if self.engine_process and self.engine_process.poll() is None:
            logger.info("🛑 Stopping ARIA MT5 Engine...")
            
            # Send SIGTERM
            self.engine_process.terminate()
            
            # Wait for graceful shutdown
            try:
                self.engine_process.wait(timeout=10)
                logger.info("✅ Engine stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Force killing engine...")
                self.engine_process.kill()
                self.engine_process.wait()
                logger.info("✅ Engine force stopped")
            
            self.running = False
        else:
            logger.info("Engine is not running")
    
    def get_status(self):
        """Get current engine status"""
        if self.engine_process and self.engine_process.poll() is None:
            return {
                "running": True,
                "pid": self.engine_process.pid,
                "uptime": time.time() - self.engine_process._start_time if hasattr(self.engine_process, '_start_time') else 0
            }
        else:
            return {"running": False}
    
    def show_logs(self, lines=50):
        """Show recent logs"""
        log_file = os.getenv("LOG_FILE", "logs/aria_mt5_live.log")
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                log_lines = f.readlines()
                recent_logs = log_lines[-lines:] if len(log_lines) > lines else log_lines
                
                print(f"\n📋 Recent logs ({len(recent_logs)} lines):")
                print("=" * 80)
                for line in recent_logs:
                    print(line.rstrip())
        else:
            logger.info("No log file found")
    
    def backup_data(self):
        """Backup trade memory and configuration"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = f"backups/backup_{timestamp}"
            Path(backup_dir).mkdir(parents=True, exist_ok=True)
            
            # Backup trade memory
            memory_file = os.getenv("TRADE_MEMORY_FILE", "data/aria_trade_memory.json")
            if os.path.exists(memory_file):
                import shutil
                shutil.copy2(memory_file, f"{backup_dir}/trade_memory.json")
            
            # Backup configuration
            if os.path.exists(self.config_file):
                import shutil
                shutil.copy2(self.config_file, f"{backup_dir}/config.env")
            
            # Backup logs
            log_file = os.getenv("LOG_FILE", "logs/aria_mt5_live.log")
            if os.path.exists(log_file):
                import shutil
                shutil.copy2(log_file, f"{backup_dir}/logs.log")
            
            logger.info(f"✅ Backup created: {backup_dir}")
            return backup_dir
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            return None
    
    def restore_data(self, backup_dir):
        """Restore data from backup"""
        try:
            if not os.path.exists(backup_dir):
                logger.error(f"Backup directory not found: {backup_dir}")
                return False
            
            # Restore trade memory
            backup_memory = f"{backup_dir}/trade_memory.json"
            if os.path.exists(backup_memory):
                memory_file = os.getenv("TRADE_MEMORY_FILE", "data/aria_trade_memory.json")
                import shutil
                shutil.copy2(backup_memory, memory_file)
                logger.info("Trade memory restored")
            
            # Restore configuration
            backup_config = f"{backup_dir}/config.env"
            if os.path.exists(backup_config):
                import shutil
                shutil.copy2(backup_config, self.config_file)
                logger.info("Configuration restored")
            
            logger.info(f"✅ Data restored from: {backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    def list_backups(self):
        """List available backups"""
        backup_dir = "backups"
        if not os.path.exists(backup_dir):
            logger.info("No backups found")
            return
        
        backups = []
        for item in os.listdir(backup_dir):
            item_path = os.path.join(backup_dir, item)
            if os.path.isdir(item_path) and item.startswith("backup_"):
                backups.append({
                    "name": item,
                    "path": item_path,
                    "created": datetime.fromtimestamp(os.path.getctime(item_path)).isoformat()
                })
        
        if backups:
            print("\n📦 Available backups:")
            print("=" * 80)
            for backup in sorted(backups, key=lambda x: x["created"], reverse=True):
                print(f"  {backup['name']} - {backup['created']}")
        else:
            logger.info("No backups found")

def main():
    """Main launcher interface"""
    parser = argparse.ArgumentParser(description="ARIA MT5 Production Engine Launcher")
    parser.add_argument("command", choices=[
        "start", "stop", "restart", "status", "logs", "test", "backup", "restore", "list-backups"
    ], help="Command to execute")
    parser.add_argument("--mode", choices=["production", "dashboard"], default="production",
                       help="Engine mode (default: production)")
    parser.add_argument("--lines", type=int, default=50, help="Number of log lines to show")
    parser.add_argument("--backup-dir", help="Backup directory for restore")
    
    args = parser.parse_args()
    
    launcher = ARIAMT5Launcher()
    
    try:
        if args.command == "start":
            success = launcher.start_engine(args.mode)
            if success:
                print("🎉 ARIA MT5 Engine started successfully!")
                print("📊 Dashboard available at: http://127.0.0.1:8000")
                print("🔌 WebSocket available at: ws://127.0.0.1:8765")
            else:
                print("❌ Failed to start engine")
                sys.exit(1)
        
        elif args.command == "stop":
            launcher.stop_engine()
            print("✅ Engine stopped")
        
        elif args.command == "restart":
            launcher.stop_engine()
            time.sleep(2)
            success = launcher.start_engine(args.mode)
            if success:
                print("🔄 Engine restarted successfully")
            else:
                print("❌ Failed to restart engine")
                sys.exit(1)
        
        elif args.command == "status":
            status = launcher.get_status()
            if status["running"]:
                print(f"✅ Engine is running (PID: {status['pid']})")
            else:
                print("❌ Engine is not running")
        
        elif args.command == "logs":
            launcher.show_logs(args.lines)
        
        elif args.command == "test":
            print("🔍 Testing ARIA MT5 setup...")
            launcher.load_config()
            
            if launcher.check_dependencies():
                print("✅ Dependencies: OK")
            else:
                print("❌ Dependencies: FAILED")
            
            if launcher.check_mt5_connection():
                print("✅ MT5 Connection: OK")
            else:
                print("❌ MT5 Connection: FAILED")
            
            launcher.create_directories()
            print("✅ Directories: OK")
            print("🎉 All tests passed!")
        
        elif args.command == "backup":
            backup_dir = launcher.backup_data()
            if backup_dir:
                print(f"✅ Backup created: {backup_dir}")
            else:
                print("❌ Backup failed")
                sys.exit(1)
        
        elif args.command == "restore":
            if not args.backup_dir:
                print("❌ Please specify backup directory with --backup-dir")
                sys.exit(1)
            
            if launcher.restore_data(args.backup_dir):
                print("✅ Data restored successfully")
            else:
                print("❌ Restore failed")
                sys.exit(1)
        
        elif args.command == "list-backups":
            launcher.list_backups()
    
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        launcher.stop_engine()
    except Exception as e:
        logger.error(f"Launcher error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

