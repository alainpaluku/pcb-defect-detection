"""EN: Actual YOLO integration with random weights. FR: Intégration YOLO à poids aléatoires.

These checks exercise software compatibility, not PCB detection accuracy.
Ces tests vérifient la compatibilité logicielle, pas la précision de détection.
"""

from pathlib import Path

import yaml

from src.config import Config, InferenceConfig, ModelConfig
from src.detector import parse_results
from src.model import PCBDetector


def test_real_yolo_inference_and_torchscript_export(tmp_path, image_path, monkeypatch):
    monkeypatch.setenv("YOLO_AUTOINSTALL", "False")
    import torch
    import ultralytics
    from ultralytics import YOLO

    torch.set_num_threads(2)
    cfg_path = Path(ultralytics.__file__).parent / "cfg" / "models" / "11" / "yolo11.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["nc"] = Config.NUM_CLASSES
    local_cfg = tmp_path / "yolo11n.yaml"
    local_cfg.write_text(yaml.safe_dump(cfg))
    model = YOLO(str(local_cfg), task="detect")
    model.model.names = dict(enumerate(Config.CLASS_NAMES))
    weights = tmp_path / "synthetic.pt"
    model.save(str(weights))
    config = Config(model=ModelConfig(img_size=64), inference=InferenceConfig(img_size=64))
    detector = PCBDetector(weights, config)
    results = detector.predict(image_path, conf=1)
    assert len(results) == 1
    assert parse_results(results) == []
    artifact = detector.export("torchscript", output=tmp_path / "exports")
    assert artifact.is_file() and artifact.stat().st_size > 0
    reloaded = PCBDetector(artifact, config)
    assert parse_results(reloaded.predict(image_path, conf=1)) == []
