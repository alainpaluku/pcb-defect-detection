"""EN: Desktop logic without a display server. FR: Logique bureau sans serveur graphique."""

import csv
import json
from unittest.mock import MagicMock

import pytest

from gui_test.model_loader import ModelManager
from gui_test.utils import export_results, format_detection_results


def test_model_is_not_marked_loaded_before_validation(monkeypatch, tmp_path):
    weights = tmp_path / "bad.pt"
    weights.touch()
    factory = MagicMock()
    factory.return_value.detector.ensure_pcb_model.side_effect = ValueError("incompatible")
    monkeypatch.setattr("gui_test.model_loader.PCBInspector", factory)
    manager = ModelManager()
    with pytest.raises(ValueError):
        manager.load_model(weights)
    assert manager.is_loaded is False
    assert manager.inspector is None


def test_desktop_forwards_iou_and_zero_confidence():
    manager = ModelManager()
    manager.inspector = MagicMock()
    manager.run_inference("image.png", confidence=0, iou=0.7)
    manager.inspector.inspect.assert_called_once_with("image.png", conf=0, iou=0.7)


def test_empty_results_can_be_exported(tmp_path):
    json_path, csv_path, text_path = (
        tmp_path / name for name in ("result.json", "result.csv", "result.txt")
    )
    export_results([], json_path, {"image": "échantillon.png"})
    assert json.loads(json_path.read_text())["summary"]["status"] == "NO_DETECTIONS"
    export_results([], csv_path)
    with csv_path.open() as stream:
        assert len(list(csv.reader(stream))) == 1
    export_results([], text_path)
    assert "\\n" not in text_path.read_text()
    assert "\n" in format_detection_results([])


def test_desktop_modules_import_and_load_image(image_path):
    from gui_test.image_handler import ImageManager
    from gui_test.main_window import PCBDetectionGUI
    from gui_test.ui_components import ResultsPanel

    assert PCBDetectionGUI and ResultsPanel
    manager = ImageManager()
    assert manager.load_image(image_path)
    assert manager.get_image_info()["size"] == (100, 80)
    assert manager.get_image_info()["format"] == ".png"


@pytest.mark.parametrize("fail", [False, True])
def test_worker_never_calls_tk_and_delivers_errors_on_main_thread(monkeypatch, fail):
    import threading

    from gui_test.tasks import run_background

    main_thread = threading.get_ident()
    threads = []
    callbacks = []
    delivered = []

    class Parent:
        def after(self, delay, callback):
            assert threading.get_ident() == main_thread
            callbacks.append(callback)

        def winfo_exists(self):
            assert threading.get_ident() == main_thread
            return True

    def make_thread(**kwargs):
        thread = threading.Thread(**kwargs)
        threads.append(thread)
        return thread

    def work(progress):
        assert threading.get_ident() != main_thread
        progress("halfway")
        if fail:
            raise ValueError("failure")
        return "complete"

    def receive(value):
        assert threading.get_ident() == main_thread
        delivered.append(value)

    monkeypatch.setattr("gui_test.tasks.Thread", make_thread)
    run_background(Parent(), work, receive, receive, receive)
    threads[0].join(timeout=2)
    assert not threads[0].is_alive()
    callbacks.pop(0)()
    assert delivered[0] == "halfway"
    if fail:
        assert isinstance(delivered[1], ValueError)
    else:
        assert delivered[1] == "complete"
