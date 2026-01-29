"""YOLO11 Model for PCB Defect Detection."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.config import Config, ModelConfig, InferenceConfig
from src.utils import get_logger

logger = get_logger(__name__)


class ModelLoadError(Exception):
    """Error during model loading."""
    pass


class YOLOWrapper:
    """Wrapper for YOLO model with error handling."""
    
    def __init__(self, model_path: Union[str, Path]):
        self.model_path = str(model_path)
        self.model = self._load_model()
    
    def _load_model(self):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            
            # Check if file exists (except for pre-trained models)
            model_file = Path(self.model_path)
            if not model_file.suffix == '.pt' or (model_file.exists() or self.model_path in [
                'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
                'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'
            ]):
                model = YOLO(self.model_path)
                logger.info(f"Model loaded: {self.model_path}")
                return model
            else:
                raise ModelLoadError(f"Model file not found: {self.model_path}")
                
        except ImportError as e:
            raise ModelLoadError(
                "ultralytics not installed. Run: pip install ultralytics"
            ) from e
        except Exception as e:
            raise ModelLoadError(f"Model loading error: {e}") from e
    
    def __getattr__(self, name: str):
        """Delegate calls to underlying model."""
        return getattr(self.model, name)


class PCBDetector:
    """YOLO11 detector for PCB defects."""
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        config: Optional[Config] = None
    ):
        """Initialize detector.
        
        Args:
            model_path: Path to trained model or pre-trained model name
            config: Custom configuration
        """
        self.config = config or Config()
        self.model_path = model_path or self.config.model.name
        self._model: Optional[YOLOWrapper] = None
    
    @property
    def model(self) -> YOLOWrapper:
        """Lazy loading of model."""
        if self._model is None:
            self._model = YOLOWrapper(self.model_path)
        return self._model
    
    def train(
        self,
        data_yaml: Union[str, Path],
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        img_size: Optional[int] = None,
        project: Optional[str] = None,
        name: str = "pcb_yolo"
    ) -> Any:
        """Train the model.
        
        Args:
            data_yaml: Path to dataset YAML config
            epochs: Number of training epochs
            batch_size: Batch size
            img_size: Image size
            project: Output directory
            name: Experiment name
        
        Returns:
            Training results
        """
        model_cfg = self.config.model
        epochs = epochs or model_cfg.epochs
        batch_size = batch_size or model_cfg.batch_size
        img_size = img_size or model_cfg.img_size
        project = project or str(Config.get_output_path())
        
        logger.info(f"Training YOLO11 for {epochs} epochs...")
        logger.info(f"Dataset: {data_yaml}")
        logger.info(f"Batch: {batch_size}, Image: {img_size}")
        logger.info(f"Model: {self.model_path}, Optimizer: {model_cfg.optimizer}")
        logger.info(f"LR: {model_cfg.learning_rate}, Patience: {model_cfg.patience}")
        
        return self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            patience=model_cfg.patience,
            save=True,
            project=project,
            name=name,
            exist_ok=True,
            pretrained=True,
            optimizer=model_cfg.optimizer,
            lr0=model_cfg.learning_rate,
            lrf=0.01,
            # YOLO11 augmentations
            augment=model_cfg.augment,
            mosaic=model_cfg.mosaic,
            mixup=model_cfg.mixup,
            copy_paste=model_cfg.copy_paste,
            auto_augment=model_cfg.auto_augment,
            erasing=model_cfg.erasing,
            crop_fraction=model_cfg.crop_fraction,
            # Geometric transformations
            degrees=model_cfg.degrees,
            translate=model_cfg.translate,
            scale=model_cfg.scale,
            shear=model_cfg.shear,
            perspective=model_cfg.perspective,
            flipud=model_cfg.flipud,
            fliplr=model_cfg.fliplr,
            # Color transformations
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            # Convergence parameters
            warmup_epochs=model_cfg.warmup_epochs,
            warmup_momentum=model_cfg.warmup_momentum,
            warmup_bias_lr=model_cfg.warmup_bias_lr,
            weight_decay=model_cfg.weight_decay,
            dropout=model_cfg.dropout,
            close_mosaic=model_cfg.close_mosaic,
            # Performance
            workers=model_cfg.workers,
            cache=model_cfg.cache,
            amp=True,  # Mixed precision
            # OPTIMIZED: Improve bounding box precision without extra time
            box=7.5,  # Box loss gain (increased for better bbox precision)
            cls=0.5,  # Class loss gain
            dfl=1.5,  # Distribution focal loss (helps with bbox accuracy)
            # Label smoothing for better generalization
            label_smoothing=0.1,
            # New YOLO11 features
            nbs=64,  # Nominal batch size
            overlap_mask=True,  # Overlapping masks
            mask_ratio=4,  # Mask ratio
            verbose=True,
        )
    
    def validate(self, data_yaml: Optional[Union[str, Path]] = None) -> Any:
        """Validate the model."""
        return self.model.val(data=str(data_yaml) if data_yaml else None)
    
    def predict(
        self,
        source: Union[str, Path],
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        save: bool = True,
        show: bool = False
    ) -> List[Any]:
        """Run inference on images."""
        inf_cfg = self.config.inference
        return self.model.predict(
            source=str(source),
            conf=conf or inf_cfg.conf_threshold,
            iou=iou or inf_cfg.iou_threshold,
            save=save,
            show=show,
        )
    
    def export(self, format: str = "onnx") -> Path:
        """Export model to different formats."""
        logger.info(f"Exporting model to {format}...")
        path = self.model.export(format=format)
        logger.info(f"Exported: {path}")
        return Path(path)
    
    def export_multiple_formats(self, formats: List[str] = None) -> Dict[str, Path]:
        """Export model to PyTorch format only.
        
        Args:
            formats: Not used, kept for compatibility
        
        Returns:
            Dictionary mapping format names to exported file paths
        """
        exported_paths = {}
        
        # Save PyTorch .pt format (primary format)
        try:
            pt_path = self.model.save()
            exported_paths["pt"] = Path(pt_path)
            logger.info(f"✅ PyTorch (.pt) saved: {pt_path}")
        except Exception as e:
            logger.warning(f"❌ Failed to save PyTorch model: {e}")
        
        return exported_paths
    
    @staticmethod
    def extract_metrics(results: Any) -> Dict[str, float]:
        """Extract metrics from validation results."""
        return {
            "detection_precision": float(results.box.map50),    # Mean AP at IoU 0.5 (mAP@0.5)
            "strict_precision": float(results.box.map),         # Mean AP at IoU 0.5:0.95 (mAP@0.5:0.95)
            "reliability": float(results.box.mp),               # Mean precision
            "detection_rate": float(results.box.mr),            # Mean recall
        }
    
    @classmethod
    def load_trained(cls, model_path: Union[str, Path]) -> "PCBDetector":
        """Load trained model."""
        return cls(model_path=model_path)
