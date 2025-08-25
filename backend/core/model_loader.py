# ============================================
# ARIA Phase-4 Model Loader (Real Models)
# ============================================

import os
import logging
import warnings
import numpy as np
from typing import Optional, Dict, Any
from backend.services.models_interface import score_and_calibrate

# Setup logging
logger = logging.getLogger("ARIA.ModelLoader")

# Model directory path
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class XGBoostONNXAdapter:
    """Lightweight ONNX runtime adapter for XGBoost exports.

    Optional component. If the ONNX model is missing or onnxruntime is
    unavailable, predict() returns a neutral 0.0 score.
    """

    def __init__(self):
        self.session = None
        self.input_name = None
        self.output_name = None
        self.model_path = os.path.join(MODELS_DIR, "xgb_forex.onnx")

    def load(self) -> None:
        try:
            if os.path.exists(self.model_path):
                import onnxruntime as ort  # type: ignore

                self.session = ort.InferenceSession(self.model_path)
                self.input_name = self.session.get_inputs()[0].name
                self.output_name = self.session.get_outputs()[0].name
            else:
                # Model not present; adapter remains inactive
                self.session = None
        except Exception as e:
            logger.warning(f"XGBoostONNXAdapter load failed: {e}")
            self.session = None

    def predict(self, features: Dict[str, Any]) -> float:
        # Neutral fallback when adapter inactive
        if self.session is None:
            return 0.0

        x = None
        try:
            if isinstance(features, dict):
                if "tabular" in features:
                    x = np.asarray(features["tabular"], dtype=np.float32).reshape(1, -1)
                elif "series" in features:
                    x = np.asarray(features["series"], dtype=np.float32).reshape(1, -1)
                elif "ohlcv" in features:
                    arr = np.asarray(features["ohlcv"], dtype=np.float32)
                    x = arr.reshape(1, -1)
        except Exception:
            x = None

        if x is None:
            return 0.0

        try:
            result = self.session.run([self.output_name], {self.input_name: x})
            out = result[0]
            if isinstance(out, np.ndarray):
                val = float(out.flatten()[0])
            else:
                val = float(out)
            # squash to [-1, 1]
            return float(np.tanh(val))
        except Exception as e:
            logger.warning(f"XGBoostONNXAdapter inference failed: {e}")
            return 0.0


class ARIAModels:
    def __init__(self):
        self.lstm = None
        self.cnn = None
        self.visual_ai = None
        self.ppo = None
        self.llm_macro = None
        self.xgb = None
        self.models_loaded = False

    def load_all(self):
        """Load all available Phase-4 real models"""
        logger.info("[ARIA] Loading Phase-4 real models...")

        try:
            # --- 1. LSTM Forex Predictor ---
            lstm_path = os.path.join(MODELS_DIR, "lstm_forex.onnx")
            if os.path.exists(lstm_path):
                try:
                    import onnxruntime as ort

                    self.lstm = ort.InferenceSession(lstm_path)
                    logger.info("[ARIA] LSTM model loaded.")
                except Exception as e:
                    logger.warning(f"Failed to load LSTM model: {e}")

            # --- 2. CNN Pattern Detector ---
            cnn_path = os.path.join(MODELS_DIR, "cnn_patterns.onnx")
            if os.path.exists(cnn_path):
                try:
                    import onnxruntime as ort

                    self.cnn = ort.InferenceSession(cnn_path)
                    logger.info("[ARIA] CNN pattern model loaded.")
                except Exception as e:
                    logger.warning(f"Failed to load CNN model: {e}")

            # --- 3. Visual AI Extractor ---
            visual_path = os.path.join(MODELS_DIR, "visual_ai.onnx")
            if os.path.exists(visual_path):
                try:
                    import onnxruntime as ort

                    self.visual_ai = ort.InferenceSession(visual_path)
                    logger.info("[ARIA] Visual AI model loaded.")
                except Exception as e:
                    logger.warning(f"Failed to load Visual AI model: {e}")

            # --- 4. PPO Forex Trader ---
            ppo_path = os.path.join(MODELS_DIR, "ppo_trader.zip")
            if os.path.exists(ppo_path):
                try:
                    from stable_baselines3 import PPO

                    # Suppress benign deserialization warnings from stable-baselines3 during load
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", category=UserWarning, module="stable_baselines3.*"
                        )
                        warnings.filterwarnings(
                            "ignore",
                            category=FutureWarning,
                            module="stable_baselines3.*",
                        )
                        self.ppo = PPO.load(ppo_path)
                    logger.info("[ARIA] PPO trading agent loaded.")
                except Exception as e:
                    logger.warning(f"Failed to load PPO model: {e}")

            # --- XGBoost Forex Model ---
            try:
                self.xgb = XGBoostONNXAdapter()
                self.xgb.load()
                logger.info("[ARIA] XGBoost ONNX model ready.")
            except Exception as e:
                logger.warning(f"Failed to initialize XGBoost adapter: {e}")

            # --- 5. LLM Macro Model ---
            llm_path = os.path.join(MODELS_DIR, "llm_macro.gguf")
            if os.path.exists(llm_path):
                try:
                    from llama_cpp import Llama

                    self.llm_macro = Llama(
                        model_path=llm_path,
                        n_threads=4,
                        n_ctx=2048,
                        n_gpu_layers=0,  # CPU only for now
                    )
                    logger.info("[ARIA] LLM Macro model loaded.")
                except Exception as e:
                    logger.warning(f"Failed to load LLM model: {e}")

            self.models_loaded = True
            logger.info("[ARIA] All available Phase-4 models initialized.")

        except Exception as e:
            logger.error(f"Error during model loading: {e}")
            self.models_loaded = False

    def predict_lstm(self, seq: np.ndarray) -> Optional[float]:
        """Predict using LSTM model for forex sequence"""
        if self.lstm is None:
            return None

        try:
            # Prepare input for ONNX model
            if len(seq.shape) == 1:
                seq = seq.reshape(1, -1, 1)  # (batch, sequence, features)

            # Normalize sequence
            mean_val = np.mean(seq)
            std_val = np.std(seq) + 1e-9
            normalized_seq = (seq - mean_val) / std_val

            # Run inference
            input_name = self.lstm.get_inputs()[0].name
            output_name = self.lstm.get_outputs()[0].name
            result = self.lstm.run(
                [output_name], {input_name: normalized_seq.astype(np.float32)}
            )

            # Extract prediction
            prediction = float(result[0][0][0])
            return float(np.tanh(prediction))  # Ensure [-1, 1] range

        except Exception as e:
            logger.error(f"LSTM inference failed: {e}")
            return None

    def predict_cnn(self, image_tensor: np.ndarray) -> Optional[float]:
        """Predict using CNN model for chart patterns"""
        if self.cnn is None:
            return None

        try:
            # Ensure image_tensor is a numpy array
            if not isinstance(image_tensor, np.ndarray):
                image_tensor = np.array(image_tensor)

            # Prepare image for CNN input
            if len(image_tensor.shape) == 2:  # Grayscale
                image_tensor = np.stack([image_tensor] * 3, axis=-1)
            elif (
                len(image_tensor.shape) == 3 and image_tensor.shape[2] == 1
            ):  # Single channel
                image_tensor = np.repeat(image_tensor, 3, axis=2)

            # Resize to 224x224 if needed
            if image_tensor.shape[:2] != (224, 224):
                from PIL import Image

                # Convert to uint8 for PIL if needed
                if image_tensor.dtype != np.uint8:
                    # Scale to 0-255 range if in 0-1 range
                    if image_tensor.max() <= 1.0:
                        image_tensor = (image_tensor * 255).astype(np.uint8)
                    else:
                        image_tensor = image_tensor.astype(np.uint8)

                img_pil = Image.fromarray(image_tensor)
                img_pil = img_pil.resize((224, 224))
                image_tensor = np.array(img_pil)

            # Normalize to [0, 1] and convert to float32
            if image_tensor.dtype != np.float32:
                if image_tensor.max() > 1.0:
                    image_tensor = image_tensor.astype(np.float32) / 255.0
                else:
                    image_tensor = image_tensor.astype(np.float32)

            # Build both layouts
            nhwc = image_tensor.reshape(1, 224, 224, 3)
            nchw = image_tensor.transpose(2, 0, 1).reshape(1, 3, 224, 224)

            input_obj = self.cnn.get_inputs()[0]
            input_name = input_obj.name
            input_shape = getattr(input_obj, "shape", None)
            output_name = self.cnn.get_outputs()[0].name

            # Prefer layout based on input shape metadata when available
            prefer_nchw = True
            if isinstance(input_shape, (list, tuple)) and len(input_shape) == 4:
                cdim = input_shape[1]
                ddim = input_shape[3]
                if isinstance(cdim, int) and cdim == 3:
                    prefer_nchw = True
                elif isinstance(ddim, int) and ddim == 3:
                    prefer_nchw = False
                else:
                    prefer_nchw = True  # default to NCHW for ONNX vision models

            # Try preferred layout, then fallback to the other if it fails
            try:
                first = nchw if prefer_nchw else nhwc
                result = self.cnn.run([output_name], {input_name: first})
            except Exception:
                alt = nhwc if prefer_nchw else nchw
                result = self.cnn.run([output_name], {input_name: alt})

            # Extract pattern prediction
            # Handle different possible result formats
            result_array = result[0]
            if isinstance(result_array, np.ndarray):
                # Handle classification output (1, 1000, 1, 1) by taking the max class probability
                if len(result_array.shape) > 1 and result_array.shape[1] > 1:
                    # This is a classification model, extract a meaningful signal
                    # Take the max probability as a confidence measure
                    max_prob = float(np.max(result_array))
                    # Convert to a signal in [-1, 1] range
                    prediction = max_prob * 2 - 1
                elif result_array.size == 1:
                    prediction = float(result_array.item())
                else:
                    # For other cases, take the first element
                    prediction = float(result_array.flat[0])
            else:
                prediction = float(result_array)
            return float(np.tanh(prediction))  # Convert to [-1, 1]

        except Exception as e:
            logger.error(f"CNN inference failed: {e}")
            return None

    def predict_visual(self, image_tensor: np.ndarray) -> Optional[np.ndarray]:
        """Extract visual features using Visual AI model"""
        if self.visual_ai is None:
            return None

        try:
            # Prepare image for MobileNetV3 input
            if len(image_tensor.shape) == 2:  # Grayscale
                image_tensor = np.stack([image_tensor] * 3, axis=-1)
            elif image_tensor.shape[2] == 1:  # Single channel
                image_tensor = np.repeat(image_tensor, 3, axis=2)

            # Resize to 224x224 if needed
            if image_tensor.shape[:2] != (224, 224):
                from PIL import Image

                img_pil = Image.fromarray(image_tensor.astype(np.uint8))
                img_pil = img_pil.resize((224, 224))
                image_tensor = np.array(img_pil)

            # Normalize to [0, 1] and convert to float32
            image_tensor = image_tensor.astype(np.float32) / 255.0

            # Build both layouts
            nhwc = image_tensor.reshape(1, 224, 224, 3)
            nchw = image_tensor.transpose(2, 0, 1).reshape(1, 3, 224, 224)

            input_obj = self.visual_ai.get_inputs()[0]
            input_name = input_obj.name
            input_shape = getattr(input_obj, "shape", None)
            output_name = self.visual_ai.get_outputs()[0].name

            prefer_nchw = True
            if isinstance(input_shape, (list, tuple)) and len(input_shape) == 4:
                cdim = input_shape[1]
                ddim = input_shape[3]
                if isinstance(cdim, int) and cdim == 3:
                    prefer_nchw = True
                elif isinstance(ddim, int) and ddim == 3:
                    prefer_nchw = False
                else:
                    prefer_nchw = True

            try:
                first = nchw if prefer_nchw else nhwc
                result = self.visual_ai.run([output_name], {input_name: first})
            except Exception:
                alt = nhwc if prefer_nchw else nchw
                result = self.visual_ai.run([output_name], {input_name: alt})

            # Extract latent features
            latent = result[0][0]  # Remove batch dimension
            return latent

        except Exception as e:
            logger.error(f"Visual AI inference failed: {e}")
            return None

    def trade_with_ppo(self, obs: np.ndarray) -> Optional[float]:
        """Get trading action from PPO agent"""
        if self.ppo is None:
            return None

        try:
            # Ensure obs is a numpy array
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs)

            # Prepare observation for PPO agent
            if len(obs.shape) == 1:
                obs = obs.reshape(1, -1)
            elif len(obs.shape) == 0:
                # Handle scalar input
                obs = np.array([[float(obs)]])

            # Get action from PPO agent
            action, _ = self.ppo.predict(obs, deterministic=True)

            # Convert action to [-1, 1] range
            if isinstance(action, np.ndarray):
                if len(action) > 0:
                    action = float(action[0])
                else:
                    action = 0.0
            else:
                action = float(action)

            return float(np.tanh(action))

        except Exception as e:
            logger.error(f"PPO inference failed: {e}")
            return None

    def predict_xgb(self, features: Dict[str, Any]) -> Optional[float]:
        """Predict using XGBoost ONNX adapter.
        features can include one of: {"tabular": [...]} or {"series": [...]} or {"ohlcv": [[o,h,l,c,v], ...]}
        Returns float in [-1, 1] or None when unavailable.
        """
        if self.xgb is None:
            return None
        try:
            score = self.xgb.predict(features)
            return float(score)
        except Exception as e:
            logger.error(f"XGBoost inference failed: {e}")
            return None

    def query_llm(self, prompt: str) -> Optional[str]:
        """Query LLM for macro analysis"""
        if self.llm_macro is None:
            return None

        try:
            # Create completion from LLM
            response = self.llm_macro.create_completion(
                prompt, max_tokens=256, temperature=0.1, stop=["\n\n", "###", "---"]
            )

            # Extract response text
            result_text = response["choices"][0]["text"].strip()
            return result_text

        except Exception as e:
            logger.error(f"LLM inference failed: {e}")
            return None

    def get_model_status(self) -> Dict[str, bool]:
        """Get status of all models"""
        return {
            "lstm": self.lstm is not None,
            "cnn": self.cnn is not None,
            "visual_ai": self.visual_ai is not None,
            "ppo": self.ppo is not None,
            "llm_macro": self.llm_macro is not None,
            "xgb": self.xgb is not None,
            "models_loaded": self.models_loaded,
        }

    def is_ready(self) -> bool:
        """Check if at least one model is loaded"""
        return any(
            [
                self.lstm is not None,
                self.cnn is not None,
                self.visual_ai is not None,
                self.ppo is not None,
                self.llm_macro is not None,
                self.xgb is not None,
            ]
        )


"""
The block below should be at the end of the class definitions. It was corrupted by
an accidental paste causing indentation errors. We restore the intended order: define
the global `aria_models` and load models on import. The stray duplicated logic is removed.
"""

# Import cached models and hot-swap manager
from backend.core.model_cache import cached_aria_models
from backend.core.hot_swap_manager import (
    HotSwapManager, 
    validate_onnx_model, 
    validate_ppo_model, 
    validate_llm_model
)

# Global ARIA model instance (legacy)
aria_models = ARIAModels()

# Global cached model instance (new)
cached_models = cached_aria_models

# Initialize hot-swap manager
hot_swap_manager = HotSwapManager(
    model_cache=cached_models.cache,
    models_dir=MODELS_DIR
)

# Register validators
hot_swap_manager.register_validator("lstm", validate_onnx_model)
hot_swap_manager.register_validator("cnn", validate_onnx_model)
hot_swap_manager.register_validator("xgb", validate_onnx_model)
hot_swap_manager.register_validator("visual", validate_onnx_model)
hot_swap_manager.register_validator("ppo", validate_ppo_model)
hot_swap_manager.register_validator("llm", validate_llm_model)

# Auto hot-swapping disabled by default - enable on demand if needed
# hot_swap_manager.enable_auto_swap()


# Enhanced compatibility wrapper
class ModelLoader:
    """Enhanced wrapper with caching and hot-swapping capabilities."""

    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        
        if use_cache:
            self.models = cached_models
        else:
            # Fallback to legacy models
            try:
                if not getattr(aria_models, "models_loaded", False):
                    aria_models.load_all()
            except Exception:
                pass
            self.models = aria_models

    def get_model_status(self):
        if self.use_cache:
            return {
                "lstm": self.models.cache.get_model("lstm_forex") is not None,
                "cnn": self.models.cache.get_model("cnn_patterns") is not None,
                "visual_ai": self.models.cache.get_model("visual_ai") is not None,
                "ppo": self.models.cache.get_model("ppo_trader") is not None,
                "llm_macro": self.models.cache.get_model("llm_macro") is not None,
                "xgb": self.models.cache.get_model("xgb_forex") is not None,
                "models_loaded": True,
                "cache_enabled": True
            }
        else:
            return self.models.get_model_status()

    def is_ready(self) -> bool:
        if self.use_cache:
            return any([
                self.models.cache.get_model("lstm_forex") is not None,
                self.models.cache.get_model("cnn_patterns") is not None,
                self.models.cache.get_model("visual_ai") is not None,
                self.models.cache.get_model("ppo_trader") is not None,
                self.models.cache.get_model("llm_macro") is not None,
                self.models.cache.get_model("xgb_forex") is not None,
            ])
        else:
            return self.models.is_ready()
    
    def get_cache_stats(self):
        """Get cache performance statistics"""
        if self.use_cache:
            return self.models.get_cache_stats()
        return {}
    
    def hot_swap_model(self, model_key: str, new_model_path: str) -> bool:
        """Perform hot-swap of model"""
        return hot_swap_manager.hot_swap_model(model_key, new_model_path)
    
    def get_swap_status(self):
        """Get hot-swap status"""
        return hot_swap_manager.get_swap_status()
    
    def cleanup_cache(self):
        """Clean up expired cache entries"""
        if self.use_cache:
            self.models.cleanup_expired()
        hot_swap_manager.cleanup_old_backups()


# Auto-load models on import
if __name__ == "__main__":
    # Test both systems
    logger.info("Testing legacy models...")
    aria_models.load_all()
    logger.info(f"Legacy model status: {aria_models.get_model_status()}")
    
    logger.info("Testing cached models...")
    logger.info(f"Cache stats: {cached_models.get_cache_stats()}")
    
    logger.info("Testing hot-swap manager...")
    logger.info(f"Swap status: {hot_swap_manager.get_swap_status()}")
else:
    # Production: use cached models by default
    pass
