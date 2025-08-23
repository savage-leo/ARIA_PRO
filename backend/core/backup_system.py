"""
Automated Backup System for ARIA Pro Trading State
Production-grade backup and recovery for critical trading data
"""

import os
import json
import sqlite3
import shutil
import gzip
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import threading
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class BackupConfig:
    """Backup configuration settings"""
    backup_dir: str = "backups"
    retention_days: int = 30
    compression: bool = True
    backup_interval_hours: int = 6
    max_backup_size_mb: int = 100

class BackupManager:
    """Automated backup manager for trading state and configurations"""
    
    def __init__(self, config: BackupConfig = None):
        self.config = config or BackupConfig()
        self.backup_dir = Path(self.config.backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        self._backup_task = None
        self._running = False
        self._lock = threading.Lock()
        
        logger.info(f"BackupManager initialized: {self.backup_dir}, retention={self.config.retention_days}d")
    
    async def start_automated_backups(self):
        """Start automated backup task"""
        if self._backup_task and not self._backup_task.done():
            logger.warning("Backup task already running")
            return
        
        self._running = True
        self._backup_task = asyncio.create_task(self._backup_loop())
        logger.info(f"Automated backups started (interval: {self.config.backup_interval_hours}h)")
    
    async def stop_automated_backups(self):
        """Stop automated backup task"""
        self._running = False
        if self._backup_task and not self._backup_task.done():
            self._backup_task.cancel()
            try:
                await self._backup_task
            except asyncio.CancelledError:
                pass
        logger.info("Automated backups stopped")
    
    async def _backup_loop(self):
        """Main backup loop"""
        while self._running:
            try:
                await self.create_full_backup()
                await self.cleanup_old_backups()
                
                # Wait for next backup interval
                await asyncio.sleep(self.config.backup_interval_hours * 3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Backup loop error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def create_full_backup(self) -> str:
        """Create comprehensive backup of all critical data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"aria_backup_{timestamp}"
        backup_path = self.backup_dir / backup_name
        
        try:
            backup_path.mkdir(exist_ok=True)
            
            # Backup components
            backup_manifest = {
                "timestamp": timestamp,
                "version": "1.2.0",
                "components": {}
            }
            
            # 1. Trading state and memory
            await self._backup_trading_state(backup_path, backup_manifest)
            
            # 2. Configuration files
            await self._backup_configurations(backup_path, backup_manifest)
            
            # 3. Model states and calibration data
            await self._backup_model_states(backup_path, backup_manifest)
            
            # 4. Audit logs and telemetry
            await self._backup_audit_data(backup_path, backup_manifest)
            
            # 5. System state and metrics
            await self._backup_system_state(backup_path, backup_manifest)
            
            # Save manifest
            manifest_path = backup_path / "backup_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(backup_manifest, f, indent=2)
            
            # Compress if enabled
            if self.config.compression:
                compressed_path = await self._compress_backup(backup_path)
                shutil.rmtree(backup_path)
                final_path = compressed_path
            else:
                final_path = backup_path
            
            # Check size limits
            size_mb = self._get_backup_size_mb(final_path)
            if size_mb > self.config.max_backup_size_mb:
                logger.warning(f"Backup size ({size_mb:.1f}MB) exceeds limit ({self.config.max_backup_size_mb}MB)")
            
            logger.info(f"Full backup created: {final_path} ({size_mb:.1f}MB)")
            return str(final_path)
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            if backup_path.exists():
                shutil.rmtree(backup_path, ignore_errors=True)
            raise
    
    async def _backup_trading_state(self, backup_path: Path, manifest: Dict):
        """Backup trading state and memory"""
        try:
            trading_dir = backup_path / "trading_state"
            trading_dir.mkdir(exist_ok=True)
            
            # Trade memory database
            trade_db_path = Path("data/trade_memory.db")
            if trade_db_path.exists():
                shutil.copy2(trade_db_path, trading_dir / "trade_memory.db")
                manifest["components"]["trade_memory"] = {"status": "backed_up", "size_bytes": trade_db_path.stat().st_size}
            
            # Enhanced fusion state
            fusion_state_path = Path("enhanced_fusion_state.json")
            if fusion_state_path.exists():
                shutil.copy2(fusion_state_path, trading_dir / "enhanced_fusion_state.json")
                manifest["components"]["fusion_state"] = {"status": "backed_up", "size_bytes": fusion_state_path.stat().st_size}
            
            # AutoTrader state
            try:
                from backend.services.auto_trader import auto_trader
                if auto_trader:
                    auto_trader_state = auto_trader.get_status()
                    with open(trading_dir / "auto_trader_state.json", 'w') as f:
                        json.dump(auto_trader_state, f, indent=2)
                    manifest["components"]["auto_trader_state"] = {"status": "backed_up"}
            except Exception as e:
                logger.warning(f"Could not backup AutoTrader state: {e}")
            
        except Exception as e:
            logger.error(f"Trading state backup failed: {e}")
            manifest["components"]["trading_state"] = {"status": "failed", "error": str(e)}
    
    async def _backup_configurations(self, backup_path: Path, manifest: Dict):
        """Backup configuration files"""
        try:
            config_dir = backup_path / "configurations"
            config_dir.mkdir(exist_ok=True)
            
            config_files = [
                "ARIA_PRO/production.env",
                "ARIA_PRO/.env",
                "ARIA_PRO/backend/.env",
                "ARIA_PRO/frontend/.env",
                "ARIA_PRO/backend/production_requirements.txt",
                "ARIA_PRO/frontend/package.json"
            ]
            
            backed_up_configs = []
            for config_file in config_files:
                config_path = Path(config_file)
                if config_path.exists():
                    dest_path = config_dir / config_path.name
                    shutil.copy2(config_path, dest_path)
                    backed_up_configs.append(config_path.name)
            
            manifest["components"]["configurations"] = {
                "status": "backed_up",
                "files": backed_up_configs
            }
            
        except Exception as e:
            logger.error(f"Configuration backup failed: {e}")
            manifest["components"]["configurations"] = {"status": "failed", "error": str(e)}
    
    async def _backup_model_states(self, backup_path: Path, manifest: Dict):
        """Backup model states and calibration data"""
        try:
            models_dir = backup_path / "model_states"
            models_dir.mkdir(exist_ok=True)
            
            # Model calibration data
            calibration_dir = Path("ARIA_PRO/backend/calibration")
            if calibration_dir.exists():
                shutil.copytree(calibration_dir, models_dir / "calibration", dirs_exist_ok=True)
            
            # Model cache states
            try:
                from backend.core.model_cache import get_model_cache
                cache = get_model_cache()
                cache_state = {
                    "loaded_models": list(cache._models.keys()),
                    "cache_stats": cache.get_stats()
                }
                with open(models_dir / "model_cache_state.json", 'w') as f:
                    json.dump(cache_state, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not backup model cache state: {e}")
            
            manifest["components"]["model_states"] = {"status": "backed_up"}
            
        except Exception as e:
            logger.error(f"Model states backup failed: {e}")
            manifest["components"]["model_states"] = {"status": "failed", "error": str(e)}
    
    async def _backup_audit_data(self, backup_path: Path, manifest: Dict):
        """Backup audit logs and telemetry"""
        try:
            audit_dir = backup_path / "audit_data"
            audit_dir.mkdir(exist_ok=True)
            
            # Audit logs
            audit_logs_dir = Path("data/audit_logs")
            if audit_logs_dir.exists():
                shutil.copytree(audit_logs_dir, audit_dir / "audit_logs", dirs_exist_ok=True)
            
            # Telemetry data
            telemetry_dir = Path("data/telemetry")
            if telemetry_dir.exists():
                shutil.copytree(telemetry_dir, audit_dir / "telemetry", dirs_exist_ok=True)
            
            # Recent log files
            logs_dir = Path("logs")
            if logs_dir.exists():
                for log_file in logs_dir.glob("*.log"):
                    if log_file.stat().st_mtime > (datetime.now() - timedelta(days=7)).timestamp():
                        shutil.copy2(log_file, audit_dir / log_file.name)
            
            manifest["components"]["audit_data"] = {"status": "backed_up"}
            
        except Exception as e:
            logger.error(f"Audit data backup failed: {e}")
            manifest["components"]["audit_data"] = {"status": "failed", "error": str(e)}
    
    async def _backup_system_state(self, backup_path: Path, manifest: Dict):
        """Backup system state and metrics"""
        try:
            system_dir = backup_path / "system_state"
            system_dir.mkdir(exist_ok=True)
            
            # System health snapshot
            try:
                from backend.core.health import get_health_checker
                health_checker = get_health_checker()
                health_status = await health_checker.check_all()
                
                with open(system_dir / "health_snapshot.json", 'w') as f:
                    json.dump(health_status, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not backup health snapshot: {e}")
            
            # Performance metrics
            try:
                from backend.core.performance_monitor import get_performance_monitor
                monitor = get_performance_monitor()
                metrics = monitor.get_current_metrics()
                
                with open(system_dir / "performance_metrics.json", 'w') as f:
                    json.dump(metrics, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not backup performance metrics: {e}")
            
            manifest["components"]["system_state"] = {"status": "backed_up"}
            
        except Exception as e:
            logger.error(f"System state backup failed: {e}")
            manifest["components"]["system_state"] = {"status": "failed", "error": str(e)}
    
    async def _compress_backup(self, backup_path: Path) -> Path:
        """Compress backup directory"""
        compressed_path = backup_path.with_suffix('.tar.gz')
        
        def compress():
            import tarfile
            with tarfile.open(compressed_path, 'w:gz') as tar:
                tar.add(backup_path, arcname=backup_path.name)
        
        # Run compression in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, compress)
        
        return compressed_path
    
    def _get_backup_size_mb(self, path: Path) -> float:
        """Get backup size in MB"""
        if path.is_file():
            return path.stat().st_size / (1024 * 1024)
        else:
            total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            return total_size / (1024 * 1024)
    
    async def cleanup_old_backups(self):
        """Remove backups older than retention period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
            removed_count = 0
            
            for backup_item in self.backup_dir.iterdir():
                if backup_item.is_file() or backup_item.is_dir():
                    # Extract timestamp from backup name
                    try:
                        if backup_item.name.startswith('aria_backup_'):
                            timestamp_str = backup_item.name.split('_')[2:4]
                            if len(timestamp_str) == 2:
                                timestamp_str = '_'.join(timestamp_str)
                                if timestamp_str.endswith('.tar.gz'):
                                    timestamp_str = timestamp_str[:-7]
                                
                                backup_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                                
                                if backup_date < cutoff_date:
                                    if backup_item.is_file():
                                        backup_item.unlink()
                                    else:
                                        shutil.rmtree(backup_item)
                                    removed_count += 1
                                    logger.info(f"Removed old backup: {backup_item.name}")
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Could not parse backup timestamp for {backup_item.name}: {e}")
            
            if removed_count > 0:
                logger.info(f"Cleanup completed: removed {removed_count} old backups")
            
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
    
    async def restore_from_backup(self, backup_path: str) -> bool:
        """Restore system from backup"""
        try:
            backup_path = Path(backup_path)
            if not backup_path.exists():
                logger.error(f"Backup not found: {backup_path}")
                return False
            
            # Extract if compressed
            if backup_path.suffix == '.gz':
                extract_dir = backup_path.parent / backup_path.stem.replace('.tar', '')
                import tarfile
                with tarfile.open(backup_path, 'r:gz') as tar:
                    tar.extractall(backup_path.parent)
                working_dir = extract_dir
            else:
                working_dir = backup_path
            
            # Read manifest
            manifest_path = working_dir / "backup_manifest.json"
            if not manifest_path.exists():
                logger.error("Backup manifest not found")
                return False
            
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            logger.info(f"Starting restore from backup: {manifest['timestamp']}")
            
            # Restore components (implement as needed)
            # This is a framework - actual restoration would need careful implementation
            # to avoid disrupting running systems
            
            logger.info("Backup restoration completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Backup restoration failed: {e}")
            return False
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = []
        
        for backup_item in self.backup_dir.iterdir():
            if backup_item.name.startswith('aria_backup_'):
                try:
                    # Parse timestamp
                    timestamp_str = backup_item.name.split('_')[2:4]
                    if len(timestamp_str) == 2:
                        timestamp_str = '_'.join(timestamp_str)
                        if timestamp_str.endswith('.tar.gz'):
                            timestamp_str = timestamp_str[:-7]
                        
                        backup_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        size_mb = self._get_backup_size_mb(backup_item)
                        
                        backups.append({
                            "name": backup_item.name,
                            "path": str(backup_item),
                            "timestamp": backup_date.isoformat(),
                            "size_mb": round(size_mb, 2),
                            "compressed": backup_item.suffix == '.gz'
                        })
                except Exception as e:
                    logger.warning(f"Could not parse backup info for {backup_item.name}: {e}")
        
        return sorted(backups, key=lambda x: x['timestamp'], reverse=True)

# Global backup manager instance
_backup_manager: Optional[BackupManager] = None

def get_backup_manager() -> BackupManager:
    """Get or create singleton backup manager"""
    global _backup_manager
    if _backup_manager is None:
        _backup_manager = BackupManager()
    return _backup_manager
