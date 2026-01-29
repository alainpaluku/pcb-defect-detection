"""
Model loading and inference functionality.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
import logging

# Add parent directory to path to import project modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.detector import PCBInspector
    from src.config import Config
except ImportError as e:
    logging.error(f"Failed to import project modules: {e}")
    PCBInspector = None
    Config = None

from .config import AUTO_MODEL_PATHS

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model loading and inference operations."""
    
    def __init__(self):
        self.inspector: Optional[PCBInspector] = None
        self.model_path: Optional[str] = None
        self.is_loaded = False
        self.model_format = None
        
    def get_best_model_for_desktop(self) -> Optional[str]:
        """Find the PyTorch model for desktop application.
        
        Returns the path to the PyTorch .pt model.
        """
        for model_path in AUTO_MODEL_PATHS:
            path = Path(model_path)
            if path.exists() and path.suffix.lower() == '.pt':
                logger.info(f"Found PyTorch model: {model_path}")
                return str(path)
        
        return None
        
    def auto_load_model(self) -> bool:
        """Try to automatically load the best model for desktop GUI."""
        if PCBInspector is None:
            logger.error("PCBInspector not available - check project imports")
            return False
            
        # First, try to find the best model for desktop performance
        best_model = self.get_best_model_for_desktop()
        if best_model:
            try:
                self.load_model(best_model)
                logger.info(f"Auto-loaded optimal model for desktop: {best_model}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load optimal model {best_model}: {e}")
        
        logger.info("No suitable models found for auto-loading")
        return False
    
    def load_model(self, model_path: Union[str, Path]) -> bool:
        """Load a model from the specified path."""
        if PCBInspector is None:
            raise ImportError("PCBInspector not available - check project setup")
            
        try:
            model_path = str(model_path)
            path = Path(model_path)
            
            # Determine model format
            self.model_format = path.suffix.lower()
            
            logger.info(f"Loading PyTorch model: {model_path}")
            
            self.inspector = PCBInspector(model_path=model_path)
            self.model_path = model_path
            self.is_loaded = True
            
            logger.info(f"Model loaded successfully: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            self.inspector = None
            self.model_path = None
            self.is_loaded = False
            self.model_format = None
            raise
    
    def run_inference(self, image_path: Union[str, Path], 
                     confidence: float = 0.25) -> List[Dict[str, Any]]:
        """Run inference on a single image."""
        if not self.is_loaded or self.inspector is None:
            raise RuntimeError("No model loaded")
            
        try:
            detections = self.inspector.inspect(
                image_path=image_path,
                conf=confidence
            )
            return detections
            
        except Exception as e:
            logger.error(f"Inference failed for {image_path}: {e}")
            raise
    
    def run_batch_inference(self, image_dir: Union[str, Path],
                           confidence: float = 0.25) -> Dict[str, List[Dict[str, Any]]]:
        """Run inference on a directory of images."""
        if not self.is_loaded or self.inspector is None:
            raise RuntimeError("No model loaded")
            
        try:
            results = self.inspector.inspect_batch(
                image_dir=image_dir,
                conf=confidence
            )
            return results
            
        except Exception as e:
            logger.error(f"Batch inference failed for {image_dir}: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {"status": "No model loaded"}
            
        model_name = Path(self.model_path).name if self.model_path else "Unknown"
        model_format = self.model_format or "Unknown"
        
        return {
            "status": "Loaded",
            "name": model_name,
            "format": model_format,
            "path": self.model_path or "Unknown",
            "performance": "PyTorch model ready for inference",
            "optimized_for": "Desktop GUI application"
        }
    
    def unload_model(self):
        """Unload the current model."""
        self.inspector = None
        self.model_path = None
        self.is_loaded = False
        self.model_format = None
        logger.info("Model unloaded")


def validate_model_path(model_path: Union[str, Path]) -> bool:
    """Validate if the model path exists and is a PyTorch model."""
    path = Path(model_path)
    
    if not path.exists():
        return False
        
    # Only PyTorch format supported
    return path.suffix.lower() == '.pt'


def get_model_performance_recommendation(model_path: Union[str, Path]) -> str:
    """Get performance recommendation for a model format."""
    path = Path(model_path)
    
    if path.suffix.lower() == '.pt':
        return 'PyTorch model - Full compatibility and features'
    else:
        return 'UNSUPPORTED: Use .pt format'