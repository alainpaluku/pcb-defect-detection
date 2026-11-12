"""EN: Web startup, model sourcing and bounded images. FR: Démarrage web, poids et images bornées."""

from dataclasses import replace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from src.config import PROJECT_ROOT
from src.model import ModelLoadError
from src.space import InferenceService, SpaceSettings, load_image, resolve_model_path


def test_missing_explicit_path_does_not_fall_back(monkeypatch):
    download = MagicMock()
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    with pytest.raises(ModelLoadError):
        resolve_model_path(SpaceSettings(model_path="absent.pt", model_repo="user/model"))
    download.assert_not_called()


def test_hub_download_uses_configured_revision(monkeypatch, tmp_path):
    weights = tmp_path / "weights.pt"
    weights.touch()
    download = MagicMock(return_value=str(weights))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download)
    settings = SpaceSettings(
        model_repo="user/model", model_filename="weights.pt", model_revision="abc123"
    )
    assert resolve_model_path(settings) == weights
    download.assert_called_once_with(repo_id="user/model", filename="weights.pt", revision="abc123")


def test_image_limit_before_model_load(image_path):
    service = InferenceService(SpaceSettings(max_pixels=10))
    service._get_detector = MagicMock()
    with pytest.raises(ValueError, match="pixels"):
        service.predict(image_path)
    service._get_detector.assert_not_called()


def test_file_size_limit(image_path):
    image_path.write_bytes(b"X" * (1024 * 1024 + 1))
    with pytest.raises(ValueError, match="MB"):
        load_image(image_path, SpaceSettings(max_file_mb=1))


def test_corrupt_image(image_path):
    image_path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="invalide"):
        load_image(image_path, SpaceSettings())


def test_exif_orientation_and_rgb(tmp_path):
    path = tmp_path / "image.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("L", (20, 10), 100).save(path, exif=exif)
    image = load_image(path, SpaceSettings())
    assert image.size == (10, 20)
    assert image.mode == "RGB"
    assert not image.info


def test_model_is_lazy_and_loaded_once(monkeypatch, tmp_path):
    weights = tmp_path / "pcb.pt"
    weights.touch()
    factory = MagicMock()
    monkeypatch.setattr("src.space.PCBDetector", factory)
    service = InferenceService(SpaceSettings(model_path=str(weights)))
    factory.assert_not_called()
    assert service._get_detector() is service._get_detector()
    factory.assert_called_once()
    factory.return_value.ensure_pcb_model.assert_called_once()


def test_service_outputs_rgb_table_and_json(image_path, result):
    import numpy as np

    result.plot.return_value = np.zeros((80, 100, 3), dtype=np.uint8)
    result.plot.return_value[..., 2] = 255
    service = InferenceService()
    service._detector = MagicMock()
    service._detector.predict.return_value = [result]
    image, rows, summary, payload = service.predict(image_path, confidence=0, iou=0)
    assert image.getpixel((0, 0)) == (255, 0, 0)
    assert rows[0] == ["short", 0.9, 10, 20, 40, 60]
    assert payload["confidence_threshold"] == 0
    assert payload["image_size"] == [100, 80]
    assert "1 défaut" in summary
    assert service._detector.predict.call_args.kwargs["save"] is False


def test_app_builds_without_model_download(monkeypatch):
    from app import create_app

    service = InferenceService(SpaceSettings(model_path="absent.pt"))
    service._get_detector = MagicMock(side_effect=AssertionError("startup must not load weights"))
    demo = create_app(service)
    assert demo.config["mode"] == "blocks"
    assert demo._queue.max_size == 16
    service._get_detector.assert_not_called()
    demo.close()


def test_space_metadata_matches_runtime():
    import importlib.metadata

    import yaml

    readme = (PROJECT_ROOT / "README.md").read_text()
    metadata = yaml.safe_load(readme.split("---", 2)[1])
    assert metadata["sdk"] == "gradio"
    assert metadata["app_file"] == "app.py"
    assert metadata["sdk_version"] == importlib.metadata.version("gradio")


@pytest.mark.parametrize("field", ["cpu_threads", "max_file_mb", "max_pixels"])
def test_invalid_resource_limits(field):
    with pytest.raises(ValueError):
        replace(SpaceSettings(), **{field: 0})
