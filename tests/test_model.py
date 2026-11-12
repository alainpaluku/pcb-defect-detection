"""EN: Regressions for configuration, inference and exports. FR: Tests de non-régression du moteur."""

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from time import sleep
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from src.config import Config, DataConfig, InferenceConfig, ModelConfig
from src.detector import PCBInspector, parse_results
from src.model import ModelLoadError, PCBDetector, YOLOWrapper
from src.utils import get_all_images


def test_config_instances_are_independent():
    custom = Config(model=ModelConfig(epochs=100), inference=InferenceConfig(conf_threshold=0))
    default = Config()
    custom.data.val_split = 0.4
    assert default.model.epochs == 50
    assert default.inference.conf_threshold == 0.3
    assert default.data.val_split == 0.2


@pytest.mark.parametrize("name", ["Missing Hole", "MISSING_HOLE", "missing-hole", " missing_hole "])
def test_class_normalization(name):
    assert Config.get_class_id(name) == 0


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (ModelConfig, {"epochs": 0}),
        (ModelConfig, {"img_size": 33}),
        (ModelConfig, {"learning_rate": float("nan")}),
        (ModelConfig, {"workers": -1}),
        (DataConfig, {"val_split": 1}),
        (InferenceConfig, {"max_det": 0}),
        (InferenceConfig, {"conf_threshold": float("nan")}),
    ],
)
def test_invalid_configuration(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)


def fake_detector():
    detector = PCBDetector("unused.pt")
    detector._model = MagicMock(task="detect", names=dict(enumerate(Config.CLASS_NAMES)))
    return detector


def test_zero_thresholds_and_in_memory_source_are_preserved():
    detector = fake_detector()
    image = Image.new("RGB", (16, 16))
    detector.predict(image, conf=0, iou=0)
    options = detector.model.predict.call_args.kwargs
    assert options["source"] is image
    assert options["conf"] == options["iou"] == 0
    assert options["save"] is False
    assert options["max_det"] == 300
    assert options["imgsz"] == 640
    assert options["device"] == "cpu"


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_invalid_threshold_rejected_before_loading(threshold):
    with pytest.raises(ValueError):
        PCBDetector("missing.pt").predict("image.jpg", conf=threshold)


def test_generic_model_cannot_mislabel_people_as_pcb():
    detector = fake_detector()
    detector.model.names = {0: "person"}
    with pytest.raises(ModelLoadError, match="six"):
        detector.predict("image.jpg")
    detector.model.predict.assert_not_called()


def test_missing_file_fails_without_download():
    with pytest.raises(ModelLoadError, match="introuvable"):
        YOLOWrapper("absent/custom.onnx")


def test_labels_follow_checkpoint_order(result):
    result.names = dict(enumerate(reversed(Config.CLASS_NAMES)))
    detection = parse_results([result])[0]
    assert detection["class_name"] == "open_circuit"
    assert detection["class_id"] == 2
    assert detection["bbox"] == {"x1": 10, "y1": 20, "x2": 40, "y2": 60}


def test_no_boxes_is_valid(result):
    result.boxes = None
    assert parse_results([result]) == []
    assert PCBInspector.get_summary([])["status"] == "NO_DETECTIONS"


def test_recursive_batch_preserves_duplicate_basenames(tmp_path):
    for folder in ("a", "b"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "same.PNG").touch()
    (tmp_path / "directory.jpg").mkdir()
    inspector = PCBInspector("unused.pt")
    inspector.inspect = MagicMock(return_value=[])
    assert set(inspector.inspect_batch(tmp_path, iou=0)) == {"a/same.PNG", "b/same.PNG"}
    assert len(get_all_images(tmp_path, recursive=True)) == 2
    assert all(call.args[-1] == 0 for call in inspector.inspect.call_args_list)


def test_visualize_saves_requested_path(result, tmp_path):
    import numpy as np

    result.plot.return_value = np.array([[[0, 0, 255]]], dtype=np.uint8)
    inspector = PCBInspector("unused.pt")
    inspector._detector = MagicMock()
    inspector.detector.predict.return_value = [result]
    path = tmp_path / "nested" / "annotated.png"
    inspector.visualize("image.png", path)
    with Image.open(path) as image:
        assert image.getpixel((0, 0)) == (255, 0, 0)
    assert inspector.detector.predict.call_args.kwargs["save"] is False


def test_exports_requested_formats_and_destination(tmp_path):
    detector = fake_detector()
    artifact = tmp_path / "model.onnx"
    artifact.write_bytes(b"artifact")
    detector.model.export.return_value = str(artifact)
    destination = tmp_path / "exported"
    exported = detector.export_multiple_formats(["onnx"], output=destination)
    assert exported == {"onnx": destination / "model.onnx"}
    assert exported["onnx"].read_bytes() == b"artifact"
    with pytest.raises(FileExistsError):
        detector.export("onnx", output=destination)


def test_predictions_are_serialized():
    detector = fake_detector()
    counts = {"active": 0, "maximum": 0}
    lock = Lock()

    def predict(**kwargs):
        with lock:
            counts["active"] += 1
            counts["maximum"] = max(counts["maximum"], counts["active"])
        sleep(0.02)
        with lock:
            counts["active"] -= 1
        return []

    detector.model.predict.side_effect = predict
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: detector.predict("image.png"), range(4)))
    assert counts["maximum"] == 1


def test_training_passes_only_detection_options(tmp_path):
    detector = fake_detector()
    detector.config.model.device = "0,1"
    detector.train(tmp_path / "dataset.yaml", epochs=2)
    kwargs = detector.model.train.call_args.kwargs
    assert kwargs["epochs"] == 2
    assert kwargs["device"] == "0,1"
    assert kwargs["exist_ok"] is False
    assert kwargs["seed"] == 42
    assert not {"label_smoothing", "auto_augment", "erasing", "overlap_mask"} & kwargs.keys()


def test_metrics_use_unambiguous_names():
    result = SimpleNamespace(box=SimpleNamespace(map50=0.9, map=0.6, mp=0.8, mr=0.7))
    assert PCBDetector.extract_metrics(result) == {
        "map50": 0.9,
        "map50_95": 0.6,
        "precision": 0.8,
        "recall": 0.7,
    }


def test_validation_keeps_outputs_in_configured_directory(tmp_path):
    detector = fake_detector()
    detector.config.output_path = tmp_path
    detector.validate(tmp_path / "dataset.yaml")
    options = detector.model.val.call_args.kwargs
    assert options["project"] == str(tmp_path)
    assert options["name"] == "pcb_validation"
    assert options["exist_ok"] is False
