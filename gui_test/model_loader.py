"""EN: Desktop access to the shared inference core. FR: Accès bureau au moteur partagé."""

import logging
from pathlib import Path

from src.detector import PCBInspector

from .config import AUTO_MODEL_PATHS

logger = logging.getLogger(__name__)


class ModelManager:
    """EN: Declare a model loaded only after validation. FR: Valider avant d'annoncer le chargement."""

    def __init__(self):
        self.inspector = None
        self.model_path = None
        self.is_loaded = False
        self.model_format = None

    def get_best_model_for_desktop(self):
        return next((str(path) for path in AUTO_MODEL_PATHS if validate_model_path(path)), None)

    def auto_load_model(self):
        path = self.get_best_model_for_desktop()
        if path is None:
            return False
        return self.load_model(path)

    def load_model(self, model_path):
        if not validate_model_path(model_path):
            raise ValueError("Choose an existing .pt model / Choisir un modèle .pt existant")
        inspector = PCBInspector(model_path=model_path)
        inspector.detector.ensure_pcb_model()
        self.inspector = inspector
        self.model_path = str(model_path)
        self.model_format = Path(model_path).suffix.lower()
        self.is_loaded = True
        return True

    def run_inference(self, image_path, confidence=0.3, iou=0.5):
        if self.inspector is None:
            raise RuntimeError("No model loaded / Aucun modèle chargé")
        return self.inspector.inspect(image_path, conf=confidence, iou=iou)

    def run_batch_inference(self, image_dir, confidence=0.3, iou=0.5):
        if self.inspector is None:
            raise RuntimeError("No model loaded / Aucun modèle chargé")
        return self.inspector.inspect_batch(image_dir, conf=confidence, iou=iou)

    def get_model_info(self):
        if not self.is_loaded:
            return {"status": "No model loaded"}
        return {
            "status": "Loaded",
            "name": Path(self.model_path).name,
            "format": self.model_format,
            "path": self.model_path,
            "performance": "PCB model / Modèle PCB",
        }

    def unload_model(self):
        self.inspector = None
        self.model_path = None
        self.is_loaded = False
        self.model_format = None


def validate_model_path(model_path):
    path = Path(model_path)
    return path.is_file() and path.suffix.lower() == ".pt"
