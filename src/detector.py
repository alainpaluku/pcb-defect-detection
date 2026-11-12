"""EN: Structured PCB inspection for all interfaces. FR: Inspection PCB pour les interfaces."""

from collections import Counter
from pathlib import Path

from PIL import Image

from src.config import PROJECT_ROOT, Config, normalize_class_name
from src.model import PCBDetector, validate_model_classes
from src.utils import get_all_images


def parse_results(results) -> list[dict]:
    """EN: Use checkpoint labels, including reordered IDs. FR: Respecter l'ordre du modèle."""
    detections = []
    for result in results:
        validate_model_classes(result.names)
        if result.boxes is None:
            continue
        # EN: Transfer once per image. FR: Un seul transfert CPU par image.
        for row in result.boxes.cpu().data.tolist():
            if len(row) != 6:
                raise ValueError("Expected detection boxes / Boîtes de détection attendues")
            x1, y1, x2, y2, confidence, model_class_id = row
            name = normalize_class_name(result.names[int(model_class_id)])
            detections.append(
                {
                    "class_id": Config.get_class_id(name),
                    "class_name": name,
                    "confidence": float(confidence),
                    "bbox": dict(zip(("x1", "y1", "x2", "y2"), (x1, y1, x2, y2), strict=True)),
                }
            )
    return detections


class PCBInspector:
    """EN: Load trained weights without a COCO fallback. FR: Exiger des poids PCB entraînés."""

    def __init__(self, model_path=None, config: Config | None = None):
        self.config = config or Config()
        self.model_path = model_path if model_path is not None else self._find_model()
        self._detector: PCBDetector | None = None

    def _find_model(self) -> Path:
        candidates = (
            PROJECT_ROOT / "models" / "pcb_model.pt",
            self.config.output_path / "pcb_model.pt",
            PROJECT_ROOT / "pcb_model.pt",
        )
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError("No trained PCB weights / Aucun poids PCB entraîné trouvé.")

    @property
    def detector(self) -> PCBDetector:
        if self._detector is None:
            self._detector = PCBDetector(self.model_path, self.config)
        return self._detector

    def inspect(self, image_path, conf=None, save=False, iou=None) -> list[dict]:
        return parse_results(self.detector.predict(image_path, conf=conf, iou=iou, save=save))

    def inspect_batch(self, image_dir, conf=None, save=False, iou=None) -> dict[str, list[dict]]:
        root = Path(image_dir)
        # EN: Relative paths avoid basename collisions. FR: Éviter les collisions de noms.
        return {
            path.relative_to(root).as_posix(): self.inspect(path, conf, save, iou)
            for path in get_all_images(root, recursive=True)
        }

    @staticmethod
    def get_summary(detections: list[dict]) -> dict:
        return {
            "status": "DEFECT" if detections else "NO_DETECTIONS",
            "defect_count": len(detections),
            "defects": dict(Counter(item["class_name"] for item in detections)),
            "max_confidence": max((item["confidence"] for item in detections), default=0.0),
        }

    def visualize(self, image_path, output_path=None):
        result = self.detector.predict(image_path, save=False)[0]
        annotated = result.plot()
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(annotated[..., ::-1]).save(destination)
        return annotated
