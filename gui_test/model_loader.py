"""EN: Desktop access to the shared inference core. FR: Accès bureau au moteur partagé."""

from pathlib import Path

from src.detector import PCBInspector

from .config import AUTO_MODEL_PATHS


class ModelManager:
    """EN: Derive load state from the inspector. FR: Déduire l'état de l'inspecteur."""

    def __init__(self):
        self.inspector: PCBInspector | None = None

    @property
    def is_loaded(self) -> bool:
        return self.inspector is not None

    def get_best_model_for_desktop(self):
        return next((str(path) for path in AUTO_MODEL_PATHS if validate_model_path(path)), None)

    def load_model(self, model_path):
        if not validate_model_path(model_path):
            raise ValueError("Choose an existing .pt model / Choisir un modèle .pt existant")
        inspector = PCBInspector(model_path=model_path)
        inspector.detector.ensure_pcb_model()
        self.inspector = inspector
        return True

    def run_inference(self, image_path, confidence=0.3, iou=0.5):
        if self.inspector is None:
            raise RuntimeError("No model loaded / Aucun modèle chargé")
        return self.inspector.inspect(image_path, conf=confidence, iou=iou)


def validate_model_path(model_path):
    path = Path(model_path)
    return path.is_file() and path.suffix.lower() == ".pt"
