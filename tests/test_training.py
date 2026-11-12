"""EN: Training output paths and failures. FR: Chemins et erreurs d'entraînement."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.config import Config
from src.trainer import TrainingManager


def test_actual_incremented_run_is_evaluated_and_saved(monkeypatch, tmp_path):
    best = tmp_path / "pcb_yolo7" / "weights" / "best.pt"
    best.parent.mkdir(parents=True)
    best.write_bytes(b"checkpoint")
    manager = TrainingManager(config=Config(output_path=tmp_path))
    manager.data = MagicMock()
    manager.data.get_yaml_path.return_value = tmp_path / "data.yaml"
    manager.model = MagicMock()
    manager.model.model.trainer = SimpleNamespace(save_dir=best.parents[1], best=best)
    factory = MagicMock()
    factory.return_value.extract_metrics.return_value = {"map50": 0.9}
    monkeypatch.setattr("src.trainer.PCBDetector", factory)
    manager.train(epochs=3)
    factory.assert_called_once_with(best, manager.config)
    assert manager.evaluate() == {"map50": 0.9}
    assert manager.save_model().read_bytes() == b"checkpoint"
    report = json.loads((tmp_path / "training_summary.json").read_text())
    assert report["model"] == str(best)
    assert report["configuration"]["model"]["epochs"] == 3


def test_missing_checkpoint_does_not_report_success(tmp_path):
    manager = TrainingManager(config=Config(output_path=tmp_path))
    with pytest.raises(FileNotFoundError):
        manager.save_model()
    with pytest.raises(RuntimeError):
        manager.evaluate()
