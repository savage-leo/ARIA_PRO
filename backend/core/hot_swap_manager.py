"""
Zero-Downtime Model Hot-Swapping Manager
Enables seamless model updates without service interruption
"""

import os
import time
import threading
import logging
import shutil
import hashlib
from typing import Dict, Optional, Callable, Any
from pathlib import Path
from dataclasses import dataclass
WATCHDOG_AVAILABLE = False
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    Observer = None
    FileSystemEventHandler = None
    WATCHDOG_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class SwapOperation:
    """Represents a model swap operation"""
    model_key: str
    old_path: str
    new_path: str
    backup_path: str
    timestamp: float
    status: str = "pending"  # pending, in_progress, completed, failed, rolled_back


if WATCHDOG_AVAILABLE:
    class ModelFileWatcher(FileSystemEventHandler):
        """Watches model files for changes"""
        
        def __init__(self, hot_swap_manager):
            self.hot_swap_manager = hot_swap_manager
            
        def on_modified(self, event):
            if event.is_directory:
                return
                
            file_path = event.src_path
            if self._is_model_file(file_path):
                logger.info(f"Model file modified: {file_path}")
                self.hot_swap_manager.trigger_auto_swap(file_path)
        
        def _is_model_file(self, file_path: str) -> bool:
            """Check if file is a model file"""
            extensions = ['.onnx', '.zip', '.gguf', '.pkl', '.joblib']
            return any(file_path.endswith(ext) for ext in extensions)
else:
    class ModelFileWatcher:
        """Dummy file watcher when watchdog unavailable"""
        def __init__(self, hot_swap_manager):
            pass


class HotSwapManager:
    """Manages zero-downtime model hot-swapping"""
    
    def __init__(self, model_cache, models_dir: str):
        self.model_cache = model_cache
        self.models_dir = Path(models_dir)
        self.backup_dir = self.models_dir / "backups"
        self.staging_dir = self.models_dir / "staging"
        
        # Create directories
        self.backup_dir.mkdir(exist_ok=True)
        self.staging_dir.mkdir(exist_ok=True)
        
        # Swap tracking
        self.active_swaps: Dict[str, SwapOperation] = {}
        self.swap_history: list = []
        self.lock = threading.RLock()
        
        # File watcher
        if WATCHDOG_AVAILABLE and Observer is not None:
            self.observer = Observer()
            self.file_watcher = ModelFileWatcher(self)
        else:
            self.observer = None
            self.file_watcher = None
        self.auto_swap_enabled = False
        
        # Validation callbacks
        self.validators: Dict[str, Callable] = {}
        
        # Performance tracking
        self.swap_count = 0
        self.total_swap_time = 0.0
        self.failed_swaps = 0
    
    def register_validator(self, model_type: str, validator_func: Callable):
        """Register a model validation function"""
        with self.lock:
            self.validators[model_type] = validator_func
    
    def enable_auto_swap(self):
        """Enable automatic hot-swapping on file changes"""
        if not self.auto_swap_enabled and WATCHDOG_AVAILABLE and self.observer and self.file_watcher:
            try:
                self.observer.schedule(self.file_watcher, str(self.models_dir), recursive=False)
                self.observer.start()
                self.auto_swap_enabled = True
                logger.info("Auto hot-swap enabled")
            except Exception as e:
                logger.warning(f"Failed to enable auto hot-swap: {e}")
                self.auto_swap_enabled = False
        elif not WATCHDOG_AVAILABLE:
            logger.warning("Auto hot-swap unavailable - watchdog not installed")
    
    def disable_auto_swap(self):
        """Disable automatic hot-swapping"""
        if self.auto_swap_enabled and self.observer:
            self.observer.stop()
            self.observer.join()
            self.auto_swap_enabled = False
            logger.info("Auto hot-swap disabled")
    
    def trigger_auto_swap(self, file_path: str):
        """Trigger automatic swap for modified file"""
        try:
            # Wait a bit for file to be fully written
            time.sleep(1.0)
            
            # Determine model key from file path
            file_name = Path(file_path).name
            model_key = self._get_model_key_from_filename(file_name)
            
            if model_key:
                logger.info(f"Auto-swapping model: {model_key}")
                self.hot_swap_model(model_key, file_path)
            
        except Exception as e:
            logger.error(f"Auto-swap failed for {file_path}: {e}")
    
    def _get_model_key_from_filename(self, filename: str) -> Optional[str]:
        """Map filename to model key"""
        mapping = {
            "lstm_forex.onnx": "lstm_forex",
            "cnn_patterns.onnx": "cnn_patterns", 
            "xgboost_forex.onnx": "xgb_forex",
            "ppo_trader.zip": "ppo_trader",
            "visual_ai.onnx": "visual_ai",
            "llm_macro.gguf": "llm_macro"
        }
        return mapping.get(filename)
    
    def hot_swap_model(self, model_key: str, new_model_path: str, validate: bool = True) -> bool:
        """Perform zero-downtime model hot-swap"""
        with self.lock:
            if model_key in self.active_swaps:
                logger.warning(f"Swap already in progress for {model_key}")
                return False
            
            start_time = time.time()
            
            # Create swap operation
            old_path = self._get_current_model_path(model_key)
            backup_path = self._create_backup_path(model_key)
            
            swap_op = SwapOperation(
                model_key=model_key,
                old_path=old_path,
                new_path=new_model_path,
                backup_path=backup_path,
                timestamp=start_time,
                status="in_progress"
            )
            
            self.active_swaps[model_key] = swap_op
            
            try:
                # Step 1: Validate new model
                if validate and not self._validate_model(model_key, new_model_path):
                    raise Exception("Model validation failed")
                
                # Step 2: Create backup of current model
                if old_path and os.path.exists(old_path):
                    shutil.copy2(old_path, backup_path)
                    logger.info(f"Backed up {model_key} to {backup_path}")
                
                # Step 3: Stage new model
                staged_path = self._stage_model(model_key, new_model_path)
                
                # Step 4: Atomic swap - invalidate cache first
                self.model_cache.invalidate(model_key)
                
                # Step 5: Move staged model to production
                if old_path:
                    shutil.move(staged_path, old_path)
                else:
                    # New model, determine target path
                    target_path = self._get_target_path(model_key)
                    shutil.move(staged_path, target_path)
                
                # Step 6: Preload new model into cache
                self.model_cache.get_model(model_key, force_reload=True)
                
                # Complete swap
                swap_op.status = "completed"
                swap_time = time.time() - start_time
                self.total_swap_time += swap_time
                self.swap_count += 1
                
                logger.info(f"Hot-swap completed for {model_key} in {swap_time:.3f}s")
                
                # Move to history
                self.swap_history.append(swap_op)
                del self.active_swaps[model_key]
                
                return True
                
            except Exception as e:
                logger.error(f"Hot-swap failed for {model_key}: {e}")
                
                # Rollback
                self._rollback_swap(swap_op)
                swap_op.status = "failed"
                self.failed_swaps += 1
                
                self.swap_history.append(swap_op)
                del self.active_swaps[model_key]
                
                return False
    
    def _validate_model(self, model_key: str, model_path: str) -> bool:
        """Validate model before swapping"""
        try:
            # Basic file checks
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
            
            if os.path.getsize(model_path) == 0:
                logger.error(f"Model file is empty: {model_path}")
                return False
            
            # Model type specific validation
            model_type = model_key.split('_')[0]
            if model_type in self.validators:
                validator = self.validators[model_type]
                return validator(model_path)
            
            # Default validation - try to load
            return self._basic_load_test(model_key, model_path)
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
    
    def _basic_load_test(self, model_key: str, model_path: str) -> bool:
        """Basic load test for model validation"""
        try:
            model_type = model_key.split('_')[0]
            
            if model_type in ['lstm', 'cnn', 'xgb', 'visual']:
                import onnxruntime as ort
                session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                return session is not None
                
            elif model_type == 'ppo':
                from stable_baselines3 import PPO
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = PPO.load(model_path)
                return model is not None
                
            elif model_type == 'llm':
                from llama_cpp import Llama
                model = Llama(model_path=model_path, n_threads=1, n_ctx=512, n_gpu_layers=0)
                return model is not None
            
            return True
            
        except Exception as e:
            logger.error(f"Basic load test failed: {e}")
            return False
    
    def _create_backup_path(self, model_key: str) -> str:
        """Create backup file path with timestamp"""
        timestamp = int(time.time())
        filename = f"{model_key}_backup_{timestamp}"
        
        # Determine extension
        current_path = self._get_current_model_path(model_key)
        if current_path:
            ext = Path(current_path).suffix
            filename += ext
        
        return str(self.backup_dir / filename)
    
    def _stage_model(self, model_key: str, source_path: str) -> str:
        """Stage model in staging directory"""
        staged_path = self.staging_dir / f"{model_key}_staged"
        
        # Add appropriate extension
        source_ext = Path(source_path).suffix
        staged_path = staged_path.with_suffix(source_ext)
        
        shutil.copy2(source_path, staged_path)
        return str(staged_path)
    
    def _get_current_model_path(self, model_key: str) -> Optional[str]:
        """Get current model file path"""
        mapping = {
            "lstm_forex": "lstm_forex.onnx",
            "cnn_patterns": "cnn_patterns.onnx",
            "xgb_forex": "xgboost_forex.onnx", 
            "ppo_trader": "ppo_trader.zip",
            "visual_ai": "visual_ai.onnx",
            "llm_macro": "llm_macro.gguf"
        }
        
        filename = mapping.get(model_key)
        if filename:
            path = self.models_dir / filename
            return str(path) if path.exists() else None
        return None
    
    def _get_target_path(self, model_key: str) -> str:
        """Get target path for new model"""
        mapping = {
            "lstm_forex": "lstm_forex.onnx",
            "cnn_patterns": "cnn_patterns.onnx", 
            "xgb_forex": "xgboost_forex.onnx",
            "ppo_trader": "ppo_trader.zip",
            "visual_ai": "visual_ai.onnx",
            "llm_macro": "llm_macro.gguf"
        }
        
        filename = mapping.get(model_key, f"{model_key}.model")
        return str(self.models_dir / filename)
    
    def _rollback_swap(self, swap_op: SwapOperation):
        """Rollback failed swap operation"""
        try:
            logger.info(f"Rolling back swap for {swap_op.model_key}")
            
            # Restore from backup if available
            if os.path.exists(swap_op.backup_path):
                if swap_op.old_path:
                    shutil.copy2(swap_op.backup_path, swap_op.old_path)
                
                # Invalidate cache and reload old model
                self.model_cache.invalidate(swap_op.model_key)
                self.model_cache.get_model(swap_op.model_key, force_reload=True)
                
                logger.info(f"Rollback completed for {swap_op.model_key}")
                swap_op.status = "rolled_back"
            
        except Exception as e:
            logger.error(f"Rollback failed for {swap_op.model_key}: {e}")
    
    def rollback_to_backup(self, model_key: str, backup_timestamp: int = None) -> bool:
        """Manually rollback to a specific backup"""
        with self.lock:
            try:
                # Find backup file
                if backup_timestamp:
                    backup_pattern = f"{model_key}_backup_{backup_timestamp}*"
                else:
                    # Find latest backup
                    backup_pattern = f"{model_key}_backup_*"
                
                import glob
                backup_files = glob.glob(str(self.backup_dir / backup_pattern))
                
                if not backup_files:
                    logger.error(f"No backup found for {model_key}")
                    return False
                
                # Use latest backup if multiple found
                backup_file = max(backup_files, key=os.path.getctime)
                
                # Perform rollback
                current_path = self._get_current_model_path(model_key)
                if current_path:
                    shutil.copy2(backup_file, current_path)
                    
                    # Invalidate cache and reload
                    self.model_cache.invalidate(model_key)
                    self.model_cache.get_model(model_key, force_reload=True)
                    
                    logger.info(f"Rolled back {model_key} to {backup_file}")
                    return True
                
                return False
                
            except Exception as e:
                logger.error(f"Manual rollback failed: {e}")
                return False
    
    def get_swap_status(self) -> Dict[str, Any]:
        """Get hot-swap status and statistics"""
        with self.lock:
            avg_swap_time = self.total_swap_time / self.swap_count if self.swap_count > 0 else 0.0
            
            return {
                "auto_swap_enabled": self.auto_swap_enabled,
                "active_swaps": len(self.active_swaps),
                "total_swaps": self.swap_count,
                "failed_swaps": self.failed_swaps,
                "success_rate": (self.swap_count - self.failed_swaps) / self.swap_count if self.swap_count > 0 else 0.0,
                "avg_swap_time": avg_swap_time,
                "recent_swaps": [
                    {
                        "model_key": swap.model_key,
                        "timestamp": swap.timestamp,
                        "status": swap.status
                    }
                    for swap in self.swap_history[-10:]  # Last 10 swaps
                ]
            }
    
    def cleanup_old_backups(self, max_age_days: int = 7):
        """Clean up old backup files"""
        try:
            cutoff_time = time.time() - (max_age_days * 24 * 3600)
            
            for backup_file in self.backup_dir.glob("*_backup_*"):
                if backup_file.stat().st_mtime < cutoff_time:
                    backup_file.unlink()
                    logger.info(f"Cleaned up old backup: {backup_file}")
                    
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
    
    def shutdown(self):
        """Shutdown hot-swap manager"""
        self.disable_auto_swap()
        logger.info("Hot-swap manager shutdown")


# Validation functions for different model types
def validate_onnx_model(model_path: str) -> bool:
    """Validate ONNX model"""
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Check inputs/outputs
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        if not inputs or not outputs:
            return False
        
        # Try a dummy inference if possible
        input_shape = inputs[0].shape
        if all(isinstance(dim, int) and dim > 0 for dim in input_shape):
            import numpy as np
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            session.run(None, {inputs[0].name: dummy_input})
        
        return True
        
    except Exception as e:
        logger.error(f"ONNX validation failed: {e}")
        return False


def validate_ppo_model(model_path: str) -> bool:
    """Validate PPO model"""
    try:
        from stable_baselines3 import PPO
        import warnings
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = PPO.load(model_path)
            
            # Basic checks
            if not hasattr(model, 'policy') or not hasattr(model, 'predict'):
                return False
            
            # Try a dummy prediction
            import numpy as np
            dummy_obs = np.random.randn(1, 10).astype(np.float32)
            action, _ = model.predict(dummy_obs, deterministic=True)
            
            return action is not None
            
    except Exception as e:
        logger.error(f"PPO validation failed: {e}")
        return False


def validate_llm_model(model_path: str) -> bool:
    """Validate LLM model"""
    try:
        from llama_cpp import Llama
        
        # Basic file checks
        if not model_path.endswith('.gguf'):
            return False
        
        # Try to load with minimal resources
        model = Llama(
            model_path=model_path,
            n_threads=1,
            n_ctx=256,
            n_gpu_layers=0
        )
        
        # Try a simple completion
        response = model.create_completion("Test", max_tokens=1, temperature=0.0)
        
        return response is not None and "choices" in response
        
    except Exception as e:
        logger.error(f"LLM validation failed: {e}")
        return False
