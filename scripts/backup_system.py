"""
ARIA PRO Backup and Recovery System
Institutional-grade backup procedures for critical system components
"""

import os
import json
import shutil
import sqlite3
import zipfile
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import asyncio
import aiofiles


class BackupManager:
    """Comprehensive backup management for ARIA PRO system"""
    
    def __init__(self, backup_root: str = "./backups"):
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Backup configuration
        self.config = {
            'retention_days': int(os.getenv('BACKUP_RETENTION_DAYS', '30')),
            'max_backups': int(os.getenv('BACKUP_MAX_COUNT', '100')),
            'compression_level': int(os.getenv('BACKUP_COMPRESSION', '6')),
            'verify_backups': os.getenv('BACKUP_VERIFY', '1') == '1'
        }
    
    def create_full_backup(self) -> Dict[str, str]:
        """Create comprehensive system backup"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"aria_full_backup_{timestamp}"
        backup_path = self.backup_root / backup_name
        backup_path.mkdir(exist_ok=True)
        
        self.logger.info(f"Starting full backup: {backup_name}")
        
        backup_manifest = {
            'backup_name': backup_name,
            'timestamp': timestamp,
            'type': 'full',
            'components': {}
        }
        
        try:
            # Backup configuration files
            config_backup = self._backup_configuration(backup_path)
            backup_manifest['components']['configuration'] = config_backup
            
            # Backup database files
            db_backup = self._backup_databases(backup_path)
            backup_manifest['components']['databases'] = db_backup
            
            # Backup AI models
            models_backup = self._backup_ai_models(backup_path)
            backup_manifest['components']['models'] = models_backup
            
            # Backup logs
            logs_backup = self._backup_logs(backup_path)
            backup_manifest['components']['logs'] = logs_backup
            
            # Backup state files
            state_backup = self._backup_state_files(backup_path)
            backup_manifest['components']['state'] = state_backup
            
            # Create compressed archive
            archive_path = self._create_compressed_archive(backup_path, backup_name)
            backup_manifest['archive_path'] = str(archive_path)
            backup_manifest['archive_size'] = archive_path.stat().st_size
            
            # Save manifest
            manifest_path = backup_path / 'backup_manifest.json'
            with open(manifest_path, 'w') as f:
                json.dump(backup_manifest, f, indent=2)
            
            # Verify backup if enabled
            if self.config['verify_backups']:
                self._verify_backup(archive_path, backup_manifest)
            
            self.logger.info(f"Full backup completed: {archive_path}")
            return backup_manifest
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            # Cleanup partial backup
            if backup_path.exists():
                shutil.rmtree(backup_path)
            raise
    
    def create_incremental_backup(self, last_backup_time: Optional[datetime] = None) -> Dict[str, str]:
        """Create incremental backup of changed files"""
        if last_backup_time is None:
            last_backup_time = datetime.now() - timedelta(hours=24)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"aria_incremental_backup_{timestamp}"
        backup_path = self.backup_root / backup_name
        backup_path.mkdir(exist_ok=True)
        
        self.logger.info(f"Starting incremental backup since: {last_backup_time}")
        
        backup_manifest = {
            'backup_name': backup_name,
            'timestamp': timestamp,
            'type': 'incremental',
            'since': last_backup_time.isoformat(),
            'components': {}
        }
        
        try:
            # Backup changed files
            changed_files = self._find_changed_files(last_backup_time)
            if changed_files:
                files_backup = self._backup_changed_files(backup_path, changed_files)
                backup_manifest['components']['changed_files'] = files_backup
            
            # Backup recent logs
            logs_backup = self._backup_recent_logs(backup_path, last_backup_time)
            backup_manifest['components']['logs'] = logs_backup
            
            # Create compressed archive
            archive_path = self._create_compressed_archive(backup_path, backup_name)
            backup_manifest['archive_path'] = str(archive_path)
            backup_manifest['archive_size'] = archive_path.stat().st_size
            
            # Save manifest
            manifest_path = backup_path / 'backup_manifest.json'
            with open(manifest_path, 'w') as f:
                json.dump(backup_manifest, f, indent=2)
            
            self.logger.info(f"Incremental backup completed: {archive_path}")
            return backup_manifest
            
        except Exception as e:
            self.logger.error(f"Incremental backup failed: {e}")
            if backup_path.exists():
                shutil.rmtree(backup_path)
            raise
    
    def _backup_configuration(self, backup_path: Path) -> Dict[str, str]:
        """Backup configuration files"""
        config_path = backup_path / 'configuration'
        config_path.mkdir(exist_ok=True)
        
        config_files = [
            'production.env',
            '.env',
            'backend/requirements.txt',
            'backend/core/config.py',
            'pytest.ini'
        ]
        
        backed_up = []
        for config_file in config_files:
            source = Path(config_file)
            if source.exists():
                dest = config_path / source.name
                shutil.copy2(source, dest)
                backed_up.append(str(source))
        
        return {
            'path': str(config_path),
            'files': backed_up,
            'count': len(backed_up)
        }
    
    def _backup_databases(self, backup_path: Path) -> Dict[str, str]:
        """Backup database files"""
        db_path = backup_path / 'databases'
        db_path.mkdir(exist_ok=True)
        
        # Find database files
        db_files = []
        for pattern in ['*.db', '*.sqlite', '*.sqlite3']:
            db_files.extend(Path('.').rglob(pattern))
        
        backed_up = []
        for db_file in db_files:
            if db_file.exists() and db_file.stat().st_size > 0:
                dest = db_path / db_file.name
                
                # For SQLite databases, create a backup using SQL dump
                if db_file.suffix in ['.db', '.sqlite', '.sqlite3']:
                    self._backup_sqlite_database(db_file, dest)
                else:
                    shutil.copy2(db_file, dest)
                
                backed_up.append(str(db_file))
        
        return {
            'path': str(db_path),
            'files': backed_up,
            'count': len(backed_up)
        }
    
    def _backup_sqlite_database(self, source: Path, dest: Path):
        """Create SQL dump backup of SQLite database"""
        try:
            conn = sqlite3.connect(str(source))
            with open(f"{dest}.sql", 'w') as f:
                for line in conn.iterdump():
                    f.write(f"{line}\n")
            conn.close()
            
            # Also copy the original file
            shutil.copy2(source, dest)
            
        except Exception as e:
            self.logger.warning(f"Failed to create SQL dump for {source}: {e}")
            # Fallback to file copy
            shutil.copy2(source, dest)
    
    def _backup_ai_models(self, backup_path: Path) -> Dict[str, str]:
        """Backup AI model files"""
        models_path = backup_path / 'models'
        models_path.mkdir(exist_ok=True)
        
        model_dirs = [
            'backend/models',
            'models'
        ]
        
        backed_up = []
        for model_dir in model_dirs:
            source_dir = Path(model_dir)
            if source_dir.exists():
                dest_dir = models_path / source_dir.name
                shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
                backed_up.append(str(source_dir))
        
        return {
            'path': str(models_path),
            'directories': backed_up,
            'count': len(backed_up)
        }
    
    def _backup_logs(self, backup_path: Path) -> Dict[str, str]:
        """Backup log files"""
        logs_path = backup_path / 'logs'
        logs_path.mkdir(exist_ok=True)
        
        log_dirs = ['logs', 'backend/logs']
        log_patterns = ['*.log', '*.err', '*.out']
        
        backed_up = []
        for log_dir in log_dirs:
            source_dir = Path(log_dir)
            if source_dir.exists():
                for pattern in log_patterns:
                    for log_file in source_dir.glob(pattern):
                        dest = logs_path / log_file.name
                        shutil.copy2(log_file, dest)
                        backed_up.append(str(log_file))
        
        return {
            'path': str(logs_path),
            'files': backed_up,
            'count': len(backed_up)
        }
    
    def _backup_state_files(self, backup_path: Path) -> Dict[str, str]:
        """Backup system state files"""
        state_path = backup_path / 'state'
        state_path.mkdir(exist_ok=True)
        
        state_files = [
            'enhanced_fusion_state.json',
            'data/trade_memory.db',
            'backend/state.json'
        ]
        
        backed_up = []
        for state_file in state_files:
            source = Path(state_file)
            if source.exists():
                dest = state_path / source.name
                shutil.copy2(source, dest)
                backed_up.append(str(source))
        
        return {
            'path': str(state_path),
            'files': backed_up,
            'count': len(backed_up)
        }
    
    def _find_changed_files(self, since: datetime) -> List[Path]:
        """Find files modified since given timestamp"""
        changed_files = []
        
        # Check important directories
        check_dirs = [
            'backend/services',
            'backend/core',
            'backend/middleware',
            'data',
            'logs'
        ]
        
        for check_dir in check_dirs:
            dir_path = Path(check_dir)
            if dir_path.exists():
                for file_path in dir_path.rglob('*'):
                    if file_path.is_file():
                        mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if mod_time > since:
                            changed_files.append(file_path)
        
        return changed_files
    
    def _backup_changed_files(self, backup_path: Path, changed_files: List[Path]) -> Dict[str, str]:
        """Backup list of changed files"""
        files_path = backup_path / 'changed_files'
        files_path.mkdir(exist_ok=True)
        
        backed_up = []
        for file_path in changed_files:
            # Preserve directory structure
            rel_path = file_path.relative_to('.')
            dest_path = files_path / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(file_path, dest_path)
            backed_up.append(str(file_path))
        
        return {
            'path': str(files_path),
            'files': backed_up,
            'count': len(backed_up)
        }
    
    def _backup_recent_logs(self, backup_path: Path, since: datetime) -> Dict[str, str]:
        """Backup recent log entries"""
        logs_path = backup_path / 'recent_logs'
        logs_path.mkdir(exist_ok=True)
        
        # This is a simplified version - in production, you'd want to
        # parse log files and extract entries since the timestamp
        return self._backup_logs(backup_path)
    
    def _create_compressed_archive(self, backup_path: Path, backup_name: str) -> Path:
        """Create compressed archive of backup"""
        archive_path = self.backup_root / f"{backup_name}.zip"
        
        with zipfile.ZipFile(
            archive_path, 
            'w', 
            zipfile.ZIP_DEFLATED, 
            compresslevel=self.config['compression_level']
        ) as zipf:
            for file_path in backup_path.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(backup_path)
                    zipf.write(file_path, arcname)
        
        # Remove uncompressed backup directory
        shutil.rmtree(backup_path)
        
        return archive_path
    
    def _verify_backup(self, archive_path: Path, manifest: Dict) -> bool:
        """Verify backup integrity"""
        try:
            with zipfile.ZipFile(archive_path, 'r') as zipf:
                # Test archive integrity
                bad_files = zipf.testzip()
                if bad_files:
                    raise Exception(f"Corrupt files in archive: {bad_files}")
                
                # Verify manifest exists
                if 'backup_manifest.json' not in zipf.namelist():
                    raise Exception("Backup manifest missing")
                
                self.logger.info(f"Backup verification successful: {archive_path}")
                return True
                
        except Exception as e:
            self.logger.error(f"Backup verification failed: {e}")
            return False
    
    def restore_backup(self, backup_path: str, target_dir: str = ".") -> bool:
        """Restore system from backup"""
        backup_file = Path(backup_path)
        if not backup_file.exists():
            self.logger.error(f"Backup file not found: {backup_path}")
            return False
        
        target_path = Path(target_dir)
        self.logger.info(f"Starting restore from: {backup_file}")
        
        try:
            # Extract backup
            with zipfile.ZipFile(backup_file, 'r') as zipf:
                zipf.extractall(target_path / 'restore_temp')
            
            # Read manifest
            manifest_path = target_path / 'restore_temp' / 'backup_manifest.json'
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            self.logger.info(f"Restoring backup: {manifest['backup_name']}")
            
            # Restore components based on manifest
            self._restore_components(target_path / 'restore_temp', target_path, manifest)
            
            # Cleanup
            shutil.rmtree(target_path / 'restore_temp')
            
            self.logger.info("Restore completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return False
    
    def _restore_components(self, restore_path: Path, target_path: Path, manifest: Dict):
        """Restore individual components from backup"""
        components = manifest.get('components', {})
        
        for component_name, component_info in components.items():
            self.logger.info(f"Restoring component: {component_name}")
            
            component_path = restore_path / component_name
            if component_path.exists():
                if component_name == 'configuration':
                    self._restore_configuration(component_path, target_path)
                elif component_name == 'databases':
                    self._restore_databases(component_path, target_path)
                elif component_name == 'models':
                    self._restore_models(component_path, target_path)
                elif component_name == 'logs':
                    self._restore_logs(component_path, target_path)
                elif component_name == 'state':
                    self._restore_state(component_path, target_path)
    
    def _restore_configuration(self, source_path: Path, target_path: Path):
        """Restore configuration files"""
        for config_file in source_path.iterdir():
            if config_file.is_file():
                dest = target_path / config_file.name
                shutil.copy2(config_file, dest)
                self.logger.info(f"Restored config: {config_file.name}")
    
    def _restore_databases(self, source_path: Path, target_path: Path):
        """Restore database files"""
        for db_file in source_path.iterdir():
            if db_file.is_file() and not db_file.name.endswith('.sql'):
                dest = target_path / 'data' / db_file.name
                dest.parent.mkdir(exist_ok=True)
                shutil.copy2(db_file, dest)
                self.logger.info(f"Restored database: {db_file.name}")
    
    def _restore_models(self, source_path: Path, target_path: Path):
        """Restore AI model files"""
        models_dest = target_path / 'backend' / 'models'
        models_dest.mkdir(parents=True, exist_ok=True)
        
        for item in source_path.iterdir():
            if item.is_dir():
                dest = models_dest.parent / item.name
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
                self.logger.info(f"Restored models: {item.name}")
    
    def _restore_logs(self, source_path: Path, target_path: Path):
        """Restore log files"""
        logs_dest = target_path / 'logs'
        logs_dest.mkdir(exist_ok=True)
        
        for log_file in source_path.iterdir():
            if log_file.is_file():
                dest = logs_dest / log_file.name
                shutil.copy2(log_file, dest)
                self.logger.info(f"Restored log: {log_file.name}")
    
    def _restore_state(self, source_path: Path, target_path: Path):
        """Restore state files"""
        for state_file in source_path.iterdir():
            if state_file.is_file():
                if 'trade_memory.db' in state_file.name:
                    dest = target_path / 'data' / state_file.name
                    dest.parent.mkdir(exist_ok=True)
                else:
                    dest = target_path / state_file.name
                shutil.copy2(state_file, dest)
                self.logger.info(f"Restored state: {state_file.name}")
    
    def cleanup_old_backups(self):
        """Remove old backups based on retention policy"""
        cutoff_date = datetime.now() - timedelta(days=self.config['retention_days'])
        
        backup_files = list(self.backup_root.glob("aria_*_backup_*.zip"))
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Keep max_backups most recent
        if len(backup_files) > self.config['max_backups']:
            for backup_file in backup_files[self.config['max_backups']:]:
                backup_file.unlink()
                self.logger.info(f"Removed old backup: {backup_file.name}")
        
        # Remove backups older than retention period
        for backup_file in backup_files:
            mod_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
            if mod_time < cutoff_date:
                backup_file.unlink()
                self.logger.info(f"Removed expired backup: {backup_file.name}")
    
    def list_backups(self) -> List[Dict]:
        """List available backups"""
        backups = []
        
        for backup_file in self.backup_root.glob("aria_*_backup_*.zip"):
            stat = backup_file.stat()
            backups.append({
                'name': backup_file.name,
                'path': str(backup_file),
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                'type': 'full' if 'full_backup' in backup_file.name else 'incremental'
            })
        
        return sorted(backups, key=lambda x: x['created'], reverse=True)


def main():
    """Main backup script entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ARIA PRO Backup System")
    parser.add_argument('action', choices=['backup', 'restore', 'list', 'cleanup'])
    parser.add_argument('--type', choices=['full', 'incremental'], default='full')
    parser.add_argument('--backup-file', help='Backup file for restore')
    parser.add_argument('--target-dir', default='.', help='Target directory for restore')
    
    args = parser.parse_args()
    
    backup_manager = BackupManager()
    
    if args.action == 'backup':
        if args.type == 'full':
            manifest = backup_manager.create_full_backup()
            print(f"Full backup created: {manifest['archive_path']}")
        else:
            manifest = backup_manager.create_incremental_backup()
            print(f"Incremental backup created: {manifest['archive_path']}")
    
    elif args.action == 'restore':
        if not args.backup_file:
            print("Error: --backup-file required for restore")
            return
        
        success = backup_manager.restore_backup(args.backup_file, args.target_dir)
        if success:
            print("Restore completed successfully")
        else:
            print("Restore failed")
    
    elif args.action == 'list':
        backups = backup_manager.list_backups()
        print(f"Available backups ({len(backups)}):")
        for backup in backups:
            size_mb = backup['size'] / (1024 * 1024)
            print(f"  {backup['name']} ({backup['type']}) - {size_mb:.1f}MB - {backup['created']}")
    
    elif args.action == 'cleanup':
        backup_manager.cleanup_old_backups()
        print("Backup cleanup completed")


if __name__ == "__main__":
    main()
