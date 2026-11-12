"""EN: Training orchestration and truthful artifacts. FR: Entraînement et résultats traçables."""

import json
import shutil
from dataclasses import asdict
from pathlib import Path

from src.config import Config
from src.data_ingestion import DataIngestion, DatasetError
from src.model import PCBDetector
from src.utils import get_logger

logger = get_logger(__name__)
__all__ = ["DatasetError", "TrainingManager"]


class TrainingManager:
    """EN: Evaluate and publish the actual best checkpoint. FR: Évaluer le meilleur modèle."""

    def __init__(self, data_path=None, config: Config | None = None):
        self.data_path = data_path
        self.config = config or Config()
        self.output_path = self.config.output_path
        self.data: DataIngestion | None = None
        self.model: PCBDetector | None = None
        self.metrics: dict[str, float] = {}
        self.training_results = None
        self.run_dir: Path | None = None
        self.best_path: Path | None = None

    def setup_data(self) -> DataIngestion:
        self.data = DataIngestion(self.data_path, self.config)
        if not self.data.find_data_structure():
            raise DatasetError(
                f"Annotated dataset not found / Dataset annoté introuvable: {self.data.data_path}"
            )
        self.data.collect_images()
        train, val = self.data.create_yolo_dataset()
        logger.info("Dataset: %s train, %s validation", train, val)
        return self.data

    def setup_model(self) -> PCBDetector:
        self.model = PCBDetector(config=self.config)
        return self.model

    def train(self, epochs=None):
        if self.data is None or self.model is None:
            raise RuntimeError("Call setup_data and setup_model first")
        if epochs is not None:
            self.config.model.epochs = epochs
        self.training_results = self.model.train(self.data.get_yaml_path())
        # EN: Ultralytics increments run paths. FR: Ultralytics incrémente les répertoires.
        trainer = self.model.model.trainer
        self.run_dir = Path(trainer.save_dir)
        self.best_path = Path(trainer.best)
        if not self.best_path.is_file():
            raise FileNotFoundError(self.best_path)
        self.model = PCBDetector(self.best_path, self.config)
        return self.training_results

    def evaluate(self) -> dict[str, float]:
        if self.data is None or self.model is None or self.best_path is None:
            raise RuntimeError("Train before evaluation / Entraîner avant évaluation")
        self.metrics = self.model.extract_metrics(self.model.validate(self.data.get_yaml_path()))
        return self.metrics

    def save_model(self) -> Path:
        if self.best_path is None or not self.best_path.is_file():
            raise FileNotFoundError("No trained checkpoint / Aucun modèle entraîné")
        self.output_path.mkdir(parents=True, exist_ok=True)
        destination = self.output_path / "pcb_model.pt"
        shutil.copy2(self.best_path, destination)
        report = {
            "model": str(self.best_path),
            "run_dir": str(self.run_dir),
            "configuration": asdict(self.config),
            "metrics": self.metrics,
        }
        (self.output_path / "training_summary.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        return destination

    def generate_graphs(self) -> None:
        """EN: Reuse Ultralytics plots; do not redraw them. FR: Réutiliser les graphiques générés."""
        if self.run_dir is not None:
            source = self.run_dir / "results.png"
            if source.is_file():
                shutil.copy2(source, self.output_path / "training_results.png")

    def display_sample_predictions(self) -> None:
        if self.run_dir is not None:
            for source in sorted(self.run_dir.glob("val_batch*_pred.jpg")):
                shutil.copy2(source, self.output_path / "sample_predictions.jpg")
                break

    def run_pipeline(self, epochs=None) -> dict[str, float]:
        self.setup_data()
        self.setup_model()
        self.train(epochs)
        self.evaluate()
        self.save_model()
        self.generate_graphs()
        self.display_sample_predictions()
        return self.metrics
