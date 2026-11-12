"""EN: Resource-bounded inference service for Spaces. FR: Service d'inférence borné pour Spaces."""

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter

from PIL import Image, ImageOps, UnidentifiedImageError

from src.config import PROJECT_ROOT, Config, InferenceConfig, validate_probability
from src.detector import parse_results
from src.model import ModelLoadError, PCBDetector


@dataclass(frozen=True)
class SpaceSettings:
    """EN: Operator-controlled settings only. FR: Paramètres réservés à l'administrateur."""

    model_path: str | None = None
    model_repo: str | None = None
    model_filename: str = "pcb_model.pt"
    model_revision: str = "main"
    device: str = "cpu"
    image_size: int = 640
    max_detections: int = 300
    max_pixels: int = 20_000_000
    max_file_mb: int = 20
    cpu_threads: int = 2

    def __post_init__(self) -> None:
        for name in ("max_pixels", "max_file_mb", "cpu_threads"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive / doit être positif")
        InferenceConfig(img_size=self.image_size, max_det=self.max_detections, device=self.device)
        if Path(self.model_filename).suffix.lower() != ".pt":
            raise ValueError("Spaces requires .pt weights / Spaces nécessite des poids .pt")

    @classmethod
    def from_env(cls) -> "SpaceSettings":
        return cls(
            model_path=os.getenv("PCB_MODEL_PATH") or None,
            model_repo=os.getenv("PCB_MODEL_REPO") or None,
            model_filename=os.getenv("PCB_MODEL_FILENAME", "pcb_model.pt"),
            model_revision=os.getenv("PCB_MODEL_REVISION", "main"),
            device=os.getenv("PCB_DEVICE", "cpu"),
            image_size=int(os.getenv("PCB_IMAGE_SIZE", "640")),
            max_detections=int(os.getenv("PCB_MAX_DETECTIONS", "300")),
            max_pixels=int(os.getenv("PCB_MAX_PIXELS", "20000000")),
            max_file_mb=int(os.getenv("PCB_MAX_FILE_MB", "20")),
            cpu_threads=int(os.getenv("PCB_CPU_THREADS", "2")),
        )


def resolve_model_path(settings: SpaceSettings) -> Path:
    """EN: Explicit path, Hub cache, then bundled weights. FR: Chemin, Hub, puis poids locaux."""
    if settings.model_path:
        path = Path(settings.model_path).expanduser()
        path = path if path.is_absolute() else PROJECT_ROOT / path
    elif settings.model_repo:
        from huggingface_hub import hf_hub_download

        # EN: HF_TOKEN stays in the SDK environment. FR: HF_TOKEN reste dans l'environnement du SDK.
        path = Path(
            hf_hub_download(
                repo_id=settings.model_repo,
                filename=settings.model_filename,
                revision=settings.model_revision,
            )
        )
    else:
        path = PROJECT_ROOT / "models" / "pcb_model.pt"
    if not path.is_file():
        raise ModelLoadError(
            "Trained weights are missing. Set PCB_MODEL_REPO or PCB_MODEL_PATH. / "
            "Poids entraînés absents. Définissez PCB_MODEL_REPO ou PCB_MODEL_PATH."
        )
    if path.suffix.lower() != ".pt":
        raise ModelLoadError("Spaces requires .pt weights / Spaces nécessite des poids .pt")
    return path


def load_image(path: str | Path, settings: SpaceSettings) -> Image.Image:
    """EN: Check size before decoding and strip metadata. FR: Vérifier avant décodage et retirer les métadonnées."""
    path = Path(path)
    if path.stat().st_size > settings.max_file_mb * 1024 * 1024:
        raise ValueError(f"Image exceeds / Image supérieure à {settings.max_file_mb} MB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                if image.width * image.height > settings.max_pixels:
                    raise ValueError(
                        f"Image exceeds / Image supérieure à {settings.max_pixels} pixels"
                    )
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                rgb.info.clear()
                return rgb
    except (
        UnidentifiedImageError,
        ImportError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise ValueError("Invalid or oversized image / Image invalide ou trop grande") from error


class InferenceService:
    """EN: One lazy model per application. FR: Un modèle chargé à la demande par application."""

    def __init__(self, settings: SpaceSettings | None = None):
        self.settings = settings or SpaceSettings.from_env()
        self._detector: PCBDetector | None = None
        self._lock = Lock()

    def _get_detector(self) -> PCBDetector:
        if self._detector is None:
            import torch

            torch.set_num_threads(self.settings.cpu_threads)
            config = Config(
                inference=InferenceConfig(
                    device=self.settings.device,
                    img_size=self.settings.image_size,
                    max_det=self.settings.max_detections,
                )
            )
            detector = PCBDetector(resolve_model_path(self.settings), config)
            detector.ensure_pcb_model()
            self._detector = detector
        return self._detector

    def predict(self, image_path, confidence=0.3, iou=0.5):
        if image_path is None:
            raise ValueError("Select an image / Sélectionnez une image")
        validate_probability(confidence, "confidence")
        validate_probability(iou, "iou")
        # EN: Bound memory even outside the Gradio queue. FR: Borner la mémoire hors de la file Gradio.
        with self._lock:
            image = load_image(image_path, self.settings)
            detector = self._get_detector()
            started = perf_counter()
            result = detector.predict(image, conf=confidence, iou=iou, save=False)[0]
            elapsed_ms = (perf_counter() - started) * 1000
            detections = parse_results([result])
            annotated = Image.fromarray(result.plot()[..., ::-1])
        rows = [
            [
                item["class_name"],
                round(item["confidence"], 4),
                *[round(value, 1) for value in item["bbox"].values()],
            ]
            for item in detections
        ]
        summary = (
            f"{len(detections)} défaut(s) détecté(s) / defect(s) detected · {elapsed_ms:.0f} ms"
            if detections
            else "Aucun défaut détecté à ce seuil / No defects detected at this threshold"
        )
        payload = {
            "image_size": list(image.size),
            "confidence_threshold": confidence,
            "iou_threshold": iou,
            "inference_ms": round(elapsed_ms, 2),
            "detections": detections,
        }
        return annotated, rows, summary, payload
