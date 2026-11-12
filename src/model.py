"""EN: One YOLO adapter for training and inference. FR: Adaptateur YOLO commun."""

from pathlib import Path
from threading import RLock
from typing import Any

from src.config import Config, normalize_class_name, validate_probability

PRETRAINED_MODELS = frozenset(f"yolo11{size}.pt" for size in "nsmlx")
EXPORT_FORMATS = ("onnx", "torchscript", "tflite", "coreml", "openvino")


class ModelLoadError(RuntimeError):
    """EN: Missing or incompatible weights. FR: Poids absents ou incompatibles."""


class YOLOWrapper:
    """EN: Import the ML stack only when needed. FR: Charger le moteur ML à la demande."""

    def __init__(self, model_path: str | Path):
        self.model_path = str(model_path)
        if not Path(model_path).is_file() and self.model_path not in PRETRAINED_MODELS:
            raise ModelLoadError(f"Model not found / Modèle introuvable: {model_path}")
        try:
            from ultralytics import YOLO

            self.model = YOLO(self.model_path, task="detect")
        except Exception as error:
            raise ModelLoadError(f"Cannot load model / Chargement impossible: {error}") from error

    def __getattr__(self, name: str) -> Any:
        return getattr(self.model, name)


def validate_model_classes(names: dict | list) -> None:
    """EN: Reject generic COCO models. FR: Refuser les modèles COCO génériques."""
    values = names.values() if isinstance(names, dict) else names
    normalized = [normalize_class_name(str(name)) for name in values]
    if len(normalized) != Config.NUM_CLASSES or set(normalized) != set(Config.CLASS_NAMES):
        raise ModelLoadError(
            "Expected a model trained on the six PCB classes / "
            "Un modèle entraîné sur les six classes PCB est requis."
        )


class PCBDetector:
    """EN: Serialize access to the mutable predictor. FR: Sérialiser l'accès au prédicteur."""

    def __init__(self, model_path: str | Path | None = None, config: Config | None = None):
        self.config = config or Config()
        self.model_path = model_path if model_path is not None else self.config.model.name
        self._model: YOLOWrapper | None = None
        self._lock = RLock()

    @property
    def model(self) -> YOLOWrapper:
        with self._lock:
            if self._model is None:
                self._model = YOLOWrapper(self.model_path)
            return self._model

    def ensure_pcb_model(self) -> None:
        with self._lock:
            if self.model.task != "detect":
                raise ModelLoadError("A detection model is required / Modèle de détection requis")
            validate_model_classes(self.model.names)

    def train(
        self, data_yaml, epochs=None, batch_size=None, img_size=None, project=None, name="pcb_yolo"
    ) -> Any:
        """EN: Use supported detection options only. FR: Options de détection prises en charge."""
        cfg = self.config.model
        cfg.__post_init__()
        values = {
            "epochs": cfg.epochs if epochs is None else epochs,
            "batch": cfg.batch_size if batch_size is None else batch_size,
            "imgsz": cfg.img_size if img_size is None else img_size,
        }
        if any(value <= 0 for value in values.values()) or values["imgsz"] % 32:
            raise ValueError("Invalid training dimensions / Dimensions d’entraînement invalides")
        options = {
            key: getattr(cfg, key)
            for key in (
                "patience",
                "optimizer",
                "workers",
                "cache",
                "mosaic",
                "mixup",
                "degrees",
                "translate",
                "scale",
                "flipud",
                "fliplr",
                "close_mosaic",
            )
        }
        if cfg.device is not None:
            options["device"] = cfg.device
        with self._lock:
            return self.model.train(
                data=str(Path(data_yaml).resolve()),
                **values,
                **options,
                lr0=cfg.learning_rate,
                seed=self.config.data.random_seed,
                project=str(project or self.config.output_path),
                name=name,
                exist_ok=False,
                save=True,
                plots=True,
            )

    def validate(self, data_yaml=None) -> Any:
        with self._lock:
            return self.model.val(
                data=str(data_yaml) if data_yaml is not None else None,
                imgsz=self.config.model.img_size,
                device=self.config.model.device,
                workers=self.config.model.workers,
                project=str(self.config.output_path),
                name="pcb_validation",
                exist_ok=False,
            )

    def predict(self, source, conf=None, iou=None, save=False, show=False) -> list[Any]:
        cfg = self.config.inference
        confidence = cfg.conf_threshold if conf is None else conf
        overlap = cfg.iou_threshold if iou is None else iou
        validate_probability(confidence, "confidence")
        validate_probability(overlap, "iou")
        with self._lock:
            self.ensure_pcb_model()
            return self.model.predict(
                source=str(source) if isinstance(source, Path) else source,
                conf=confidence,
                iou=overlap,
                max_det=cfg.max_det,
                imgsz=cfg.img_size,
                agnostic_nms=cfg.agnostic_nms,
                device=cfg.device,
                save=save,
                show=show,
                verbose=False,
            )

    def export(self, format="onnx", output: str | Path | None = None) -> Path:
        """EN: Return the actual artifact path. FR: Retourner le chemin du fichier créé."""
        import shutil

        if format not in EXPORT_FORMATS:
            raise ValueError(f"Unsupported export format: {format}")
        with self._lock:
            path = Path(self.model.export(format=format, imgsz=self.config.model.img_size))
        if output is not None:
            directory = Path(output)
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / path.name
            if target.resolve() != path.resolve():
                if target.exists():
                    raise FileExistsError(target)
                if path.is_dir():
                    shutil.copytree(path, target)
                else:
                    shutil.copy2(path, target)
                path = target
        return path

    def export_multiple_formats(self, formats=None, output=None) -> dict[str, Path]:
        formats = ("onnx", "torchscript") if formats is None else formats
        return {fmt: self.export(fmt, output=output) for fmt in formats}

    @staticmethod
    def extract_metrics(results: Any) -> dict[str, float]:
        return {
            "map50": float(results.box.map50),
            "map50_95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        }
