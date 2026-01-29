"""Configuration for PCB Defect Detection System."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# Global constants
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


@dataclass
class ModelConfig:
    """YOLO11 model configuration."""
    name: str = "yolo11m.pt"  # YOLO11 Medium - best balance precision/speed
    img_size: int = 640
    batch_size: int = 16
    epochs: int = 50  # OPTIMIZED: Reduced from 100 (converges at epoch 20)
    patience: int = 15  # OPTIMIZED: Early stopping at 15 epochs without improvement
    learning_rate: float = 0.001  # Optimal LR for YOLO11
    optimizer: str = "auto"  # YOLO11 automatically chooses
    augment: bool = True
    
    # Optimized augmentations for YOLO11
    mosaic: float = 1.0
    mixup: float = 0.15  # OPTIMIZED: Reduced from 0.2 (faster, still effective)
    copy_paste: float = 0.2  # OPTIMIZED: Reduced from 0.3 (faster preprocessing)
    degrees: float = 10.0  # Rotation
    translate: float = 0.2  # Translation
    scale: float = 0.9  # Scale
    shear: float = 2.0  # Shear
    perspective: float = 0.0001  # Perspective
    flipud: float = 0.5  # Vertical flip (useful for PCB)
    fliplr: float = 0.5  # Horizontal flip
    
    # Convergence parameters
    warmup_epochs: float = 3.0  # OPTIMIZED: Reduced from 5.0 (faster warmup)
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.1
    weight_decay: float = 0.0005
    dropout: float = 0.0  # YOLO11 handles better without dropout
    close_mosaic: int = 10  # OPTIMIZED: Close mosaic earlier (epoch 10 vs 20)
    
    # Performance
    workers: int = 8  # More workers
    cache: str = "ram"  # Keep in RAM for speed
    
    # New YOLO11 features
    auto_augment: str = "randaugment"  # Automatic augmentation
    erasing: float = 0.3  # OPTIMIZED: Reduced from 0.4 (faster)
    crop_fraction: float = 1.0  # Crop fraction


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    conf_threshold: float = 0.3  # Higher threshold for more precision
    iou_threshold: float = 0.5  # Stricter IoU
    max_det: int = 300  # Maximum detections
    agnostic_nms: bool = False  # NMS per class


@dataclass
class DataConfig:
    """Data configuration."""
    kaggle_dataset: str = "akhatova/pcb-defects"
    val_split: float = 0.2
    random_seed: int = 42


class Config:
    """Central configuration for PCB defect detection."""
    
    CLASS_NAMES: List[str] = [
        "missing_hole",
        "mouse_bite",
        "open_circuit",
        "short",
        "spur",
        "spurious_copper"
    ]
    
    NUM_CLASSES: int = len(CLASS_NAMES)
    
    # Class name mapping (handles different case conventions)
    CLASS_MAP: Dict[str, int] = {}
    
    @classmethod
    def _init_class_map(cls) -> None:
        """Initialize class mapping with different case variants."""
        for idx, name in enumerate(cls.CLASS_NAMES):
            variants = {
                name,                                      # missing_hole
                name.upper(),                              # MISSING_HOLE
                name.title(),                              # Missing_Hole
                name.replace("_", " ").title().replace(" ", "_"),  # Missing_Hole
            }
            for variant in variants:
                cls.CLASS_MAP[variant] = idx
    
    @classmethod
    def get_class_id(cls, class_name: str) -> int:
        """Returns class ID, case insensitive."""
        # Direct search
        if class_name in cls.CLASS_MAP:
            return cls.CLASS_MAP[class_name]
        # Case insensitive search
        lower_name = class_name.lower().replace(" ", "_")
        for name, idx in cls.CLASS_MAP.items():
            if name.lower() == lower_name:
                return idx
        raise ValueError(f"Unknown class: {class_name}")
    
    # Default configurations
    model: ModelConfig = ModelConfig()
    inference: InferenceConfig = InferenceConfig()
    data: DataConfig = DataConfig()
    
    @staticmethod
    def is_kaggle() -> bool:
        """Check if we're in Kaggle environment."""
        return os.path.exists("/kaggle/input")
    
    @staticmethod
    def get_data_path() -> Path:
        """Returns dataset path."""
        if Config.is_kaggle():
            for p in ("/kaggle/input/pcb-defects", "/kaggle/input/pcbdefects"):
                if Path(p).exists():
                    return Path(p)
        return Path("data/pcb-defects")
    
    @staticmethod
    def get_output_path() -> Path:
        """Returns output directory."""
        if Config.is_kaggle():
            return Path("/kaggle/working")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        return output_dir
    
    @staticmethod
    def get_yolo_dataset_path() -> Path:
        """Returns YOLO format dataset path."""
        return Config.get_output_path() / "yolo_dataset"
    
    @classmethod
    def create(
        cls,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        img_size: Optional[int] = None,
        conf_threshold: Optional[float] = None,
    ) -> "Config":
        """Creates custom configuration."""
        config = cls()
        if epochs is not None:
            config.model.epochs = epochs
        if batch_size is not None:
            config.model.batch_size = batch_size
        if img_size is not None:
            config.model.img_size = img_size
        if conf_threshold is not None:
            config.inference.conf_threshold = conf_threshold
        return config


# Initialize class mapping on module load
Config._init_class_map()
