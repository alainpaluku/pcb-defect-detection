"""EN: Small deterministic fixtures. FR: Petites données de test déterministes."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from PIL import Image

from src.config import Config


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "board.png"
    Image.new("RGB", (100, 80), "green").save(path)
    return path


@pytest.fixture
def result():
    boxes = MagicMock()
    boxes.cpu.return_value.data.tolist.return_value = [[10, 20, 40, 60, 0.9, 3]]
    return SimpleNamespace(names=dict(enumerate(Config.CLASS_NAMES)), boxes=boxes, plot=MagicMock())


def write_sample(root: Path, name: str, color: int = 0, objects: str | None = None):
    image = root / "images" / f"{name}.png"
    xml = root / "Annotations" / f"{name}.xml"
    image.parent.mkdir(parents=True, exist_ok=True)
    xml.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 80), (color, 30, 50)).save(image)
    if objects is None:
        objects = "<object><name>Missing Hole</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>30</xmax><ymax>60</ymax></bndbox></object>"
    xml.write_text(
        "<annotation><size><width>100</width><height>80</height></size>"
        + objects
        + "</annotation>",
        encoding="utf-8",
    )
    return image, xml
