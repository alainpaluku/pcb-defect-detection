"""EN: Validated, independent settings. FR: Paramètres validés et indépendants."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})
CLASS_NAMES = ("missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper")


def normalize_class_name(name: str) -> str:
    """EN: Accept case, spaces and hyphens. FR: Accepter casse, espaces et tirets."""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def validate_probability(value: float, name: str) -> float:
    """EN: Reject invalid thresholds, including NaN. FR: Rejeter les seuils invalides."""
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1 / doit être compris entre 0 et 1")
    return value


@dataclass
class ModelConfig:
    """EN: Detection training parameters. FR: Paramètres d'entraînement de détection."""

    name: str = "yolo11m.pt"
    img_size: int = 640
    batch_size: int = 16
    epochs: int = 50
    patience: int = 15
    learning_rate: float = 0.001
    optimizer: str = "auto"
    workers: int = 2
    cache: bool | str = False
    device: str | None = None
    mosaic: float = 1.0
    mixup: float = 0.0
    degrees: float = 10.0
    translate: float = 0.1
    scale: float = 0.5
    flipud: float = 0.5
    fliplr: float = 0.5
    close_mosaic: int = 10

    def __post_init__(self) -> None:
        for name in ("img_size", "batch_size", "epochs"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive / doit être positif")
        if self.img_size % 32:
            raise ValueError("img_size must be a multiple of 32 / doit être un multiple de 32")
        if self.workers < 0 or self.patience < 0 or self.close_mosaic < 0:
            raise ValueError("workers, patience and close_mosaic must be nonnegative")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if self.cache not in (False, True, "ram", "disk"):
            raise ValueError("cache must be False, True, ram or disk")
        for name in ("mosaic", "mixup", "translate", "scale", "flipud", "fliplr"):
            validate_probability(getattr(self, name), name)


@dataclass
class InferenceConfig:
    """EN: Bounded CPU defaults for Spaces. FR: Valeurs CPU bornées pour Spaces."""

    conf_threshold: float = 0.3
    iou_threshold: float = 0.5
    max_det: int = 300
    img_size: int = 640
    agnostic_nms: bool = False
    device: str = "cpu"

    def __post_init__(self) -> None:
        validate_probability(self.conf_threshold, "confidence")
        validate_probability(self.iou_threshold, "iou")
        if self.max_det <= 0 or self.img_size <= 0 or self.img_size % 32:
            raise ValueError("max_det must be positive; img_size must be a positive multiple of 32")


@dataclass
class DataConfig:
    """EN: Reproducible splitting. FR: Séparation reproductible des données."""

    val_split: float = 0.2
    random_seed: int = 42
    group_by_board: bool = True

    def __post_init__(self) -> None:
        if not 0 < self.val_split < 1:
            raise ValueError("val_split must be strictly between 0 and 1")


@dataclass
class Config:
    """EN: Never share mutable defaults. FR: Ne jamais partager les valeurs mutables."""

    CLASS_NAMES: ClassVar[tuple[str, ...]] = CLASS_NAMES
    NUM_CLASSES: ClassVar[int] = len(CLASS_NAMES)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_path: Path = field(default_factory=lambda: Config.get_output_path())

    @staticmethod
    def get_class_id(class_name: str) -> int:
        try:
            return CLASS_NAMES.index(normalize_class_name(class_name))
        except ValueError as error:
            raise ValueError(f"Unknown PCB class / Classe PCB inconnue: {class_name}") from error

    @staticmethod
    def is_kaggle() -> bool:
        return Path("/kaggle/input").is_dir()

    @staticmethod
    def get_data_path() -> Path:
        if Config.is_kaggle():
            for name in ("pcb-defects", "pcbdefects"):
                candidate = Path("/kaggle/input") / name
                if candidate.is_dir():
                    return candidate
        return PROJECT_ROOT / "data" / "pcb-defects"

    @staticmethod
    def get_output_path() -> Path:
        # EN: Reading configuration does not create files. FR: La lecture ne crée aucun fichier.
        return Path("/kaggle/working") if Config.is_kaggle() else PROJECT_ROOT / "output"

    @staticmethod
    def get_yolo_dataset_path() -> Path:
        return Config.get_output_path() / "yolo_dataset"

    @classmethod
    def create(cls, epochs=None, batch_size=None, img_size=None, conf_threshold=None) -> "Config":
        model = {
            key: value
            for key, value in {
                "epochs": epochs,
                "batch_size": batch_size,
                "img_size": img_size,
            }.items()
            if value is not None
        }
        inference = {} if conf_threshold is None else {"conf_threshold": conf_threshold}
        return cls(model=ModelConfig(**model), inference=InferenceConfig(**inference))
