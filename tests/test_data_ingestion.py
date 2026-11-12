"""EN: Real file conversion and split integrity. FR: Conversion réelle et intégrité des jeux."""

import json
import random

import pytest
import yaml

from src.config import Config
from src.data_ingestion import MARKER, DataIngestion, DatasetError, VOCConverter
from tests.conftest import write_sample


def dataset(tmp_path):
    source = tmp_path / "source"
    for board in range(1, 5):
        for defect in ("short", "spur"):
            write_sample(source, f"{board:02}_{defect}", board * 10 + len(defect))
    ingestion = DataIngestion(source, Config(output_path=tmp_path / "output"))
    ingestion.collect_images()
    return ingestion


def test_voc_normalizes_case_and_coordinates(tmp_path):
    _, xml = write_sample(tmp_path, "01_test")
    assert VOCConverter.convert(xml, 100, 80) == ["0 0.20000000 0.50000000 0.20000000 0.50000000"]


@pytest.mark.parametrize("replacement", ["nan", "-inf", "bad", ""])
def test_invalid_coordinate_is_not_silently_dropped(tmp_path, replacement):
    _, xml = write_sample(tmp_path, "01_test")
    xml.write_text(xml.read_text().replace("<xmin>10</xmin>", f"<xmin>{replacement}</xmin>"))
    with pytest.raises(DatasetError):
        VOCConverter.convert(xml, 100, 80)


def test_unknown_class_and_bad_size_fail(tmp_path):
    _, xml = write_sample(tmp_path, "01_test")
    with pytest.raises(DatasetError, match="dimensions"):
        VOCConverter.convert(xml, 200, 80)
    xml.write_text(xml.read_text().replace("Missing Hole", "person"))
    with pytest.raises(DatasetError, match="Unknown PCB"):
        VOCConverter.convert(xml, 100, 80)


def test_empty_annotation_is_negative(tmp_path):
    _, xml = write_sample(tmp_path, "01_test", objects="")
    assert VOCConverter.convert(xml, 100, 80) == []


def test_group_split_deterministic_no_global_rng_mutation(tmp_path):
    ingestion = dataset(tmp_path)
    state = random.getstate()
    train, val = ingestion.create_yolo_dataset()
    assert train + val == 8 and train > 0 and val > 0
    assert random.getstate() == state
    marker = ingestion.yolo_path / MARKER
    original = marker.read_text()
    manifest = json.loads(original)
    train_groups = {row["group"] for row in manifest["splits"]["train"]}
    val_groups = {row["group"] for row in manifest["splits"]["val"]}
    assert train_groups.isdisjoint(val_groups)
    stale = ingestion.yolo_path / "images" / "train" / "stale.jpg"
    stale.touch()
    ingestion.create_yolo_dataset()
    assert not stale.exists()
    assert marker.read_text() == original
    config = yaml.safe_load(ingestion.get_yaml_path().read_text())
    assert config["path"] == str(ingestion.yolo_path.resolve())


def test_failed_rebuild_preserves_previous_dataset(tmp_path):
    ingestion = dataset(tmp_path)
    ingestion.create_yolo_dataset()
    original = (ingestion.yolo_path / MARKER).read_bytes()
    ingestion.all_images[0].annotation_path.write_text("<broken")
    with pytest.raises(DatasetError):
        ingestion.create_yolo_dataset()
    assert (ingestion.yolo_path / MARKER).read_bytes() == original


def test_unmanaged_output_is_never_deleted(tmp_path):
    ingestion = dataset(tmp_path)
    ingestion.yolo_path.mkdir(parents=True)
    keep = ingestion.yolo_path / "user-data.txt"
    keep.write_text("keep")
    with pytest.raises(DatasetError, match="unmanaged"):
        ingestion.create_yolo_dataset()
    assert keep.read_text() == "keep"


def test_missing_image_fails(tmp_path):
    image, _ = write_sample(tmp_path, "01_test")
    image.unlink()
    with pytest.raises(DatasetError, match="ambiguous"):
        DataIngestion(tmp_path).collect_images()


def test_duplicate_image_content_fails(tmp_path):
    ingestion = dataset(tmp_path)
    ingestion.all_images[1].image_path.write_bytes(ingestion.all_images[0].image_path.read_bytes())
    with pytest.raises(DatasetError, match="Duplicate image"):
        ingestion.create_yolo_dataset()


def test_same_stem_in_class_folders_has_unique_outputs(tmp_path):
    source = tmp_path / "source"
    for index, folder in enumerate(("short", "spur")):
        write_sample(source, f"{folder}/01_duplicate", color=index + 1)
    config = Config(output_path=tmp_path / "output")
    config.data.group_by_board = False
    ingestion = DataIngestion(source, config)
    ingestion.collect_images()
    assert ingestion.create_yolo_dataset() == (1, 1)
    assert len({path.name for path in ingestion.yolo_path.rglob("*.txt")}) == 2
