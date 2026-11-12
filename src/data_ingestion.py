"""EN: Strict VOC conversion and reproducible splits. FR: Conversion VOC et séparation fiable."""

import hashlib
import json
import math
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from defusedxml import ElementTree
from PIL import Image

from src.config import Config
from src.utils import get_all_images

MARKER = ".pcb-dataset.json"


class DatasetError(ValueError):
    """EN: Invalid or ambiguous training data. FR: Données invalides ou ambiguës."""


@dataclass(frozen=True)
class ImageItem:
    """EN: A matched image and annotation. FR: Image associée à son annotation."""

    image_path: Path
    annotation_path: Path | None = None
    class_name: str | None = None
    source_type: str = "xml"


class VOCConverter:
    """EN: Convert pixel corners to normalized YOLO boxes. FR: Normaliser les boîtes VOC."""

    @staticmethod
    def convert(xml_path: Path, img_width: int, img_height: int) -> list[str]:
        if img_width <= 0 or img_height <= 0:
            raise DatasetError("Invalid image dimensions / Dimensions invalides")
        try:
            root = ElementTree.parse(xml_path).getroot()
            size = root.find("size")
            if size is not None:
                dimensions = (int(size.findtext("width", "0")), int(size.findtext("height", "0")))
                if dimensions != (img_width, img_height):
                    raise ValueError(
                        "XML/image dimensions differ / Dimensions XML/image différentes"
                    )
            lines = []
            for obj in root.findall("object"):
                class_id = Config.get_class_id(obj.findtext("name", ""))
                values = [
                    float(obj.findtext(f"bndbox/{key}", "nan"))
                    for key in ("xmin", "ymin", "xmax", "ymax")
                ]
                if not all(math.isfinite(value) for value in values):
                    raise ValueError(
                        "Nonfinite or missing coordinates / Coordonnées manquantes ou non finies"
                    )
                xmin, ymin, xmax, ymax = values
                xmin, xmax = (min(img_width, max(0, value)) for value in (xmin, xmax))
                ymin, ymax = (min(img_height, max(0, value)) for value in (ymin, ymax))
                if xmax <= xmin or ymax <= ymin:
                    raise ValueError("Empty or inverted box / Boîte vide ou inversée")
                coords = (
                    (xmin + xmax) / (2 * img_width),
                    (ymin + ymax) / (2 * img_height),
                    (xmax - xmin) / img_width,
                    (ymax - ymin) / img_height,
                )
                lines.append(f"{class_id} " + " ".join(f"{value:.8f}" for value in coords))
            # EN: An empty XML is a valid negative. FR: Un XML vide constitue un exemple négatif.
            return lines
        except (ValueError, OSError, ElementTree.ParseError) as error:
            raise DatasetError(f"{xml_path}: {error}") from error


class DataIngestion:
    """EN: Index files once and refuse silent data loss. FR: Indexer sans perdre de données."""

    def __init__(self, data_path=None, config: Config | None = None, yolo_path=None):
        self.config = config or Config()
        self.data_path = Path(data_path) if data_path is not None else Config.get_data_path()
        self.yolo_path = (
            Path(yolo_path) if yolo_path is not None else self.config.output_path / "yolo_dataset"
        )
        self.all_images: list[ImageItem] = []

    def find_data_structure(self) -> bool:
        return self.data_path.is_dir() and any(
            p.is_file() and p.suffix.lower() == ".xml" for p in self.data_path.rglob("*")
        )

    def collect_images(self) -> list[ImageItem]:
        index = defaultdict(list)
        for image in get_all_images(self.data_path, recursive=True):
            index[image.stem.casefold()].append(image)
        matched = []
        seen = set()
        annotations = sorted(
            p for p in self.data_path.rglob("*") if p.is_file() and p.suffix.lower() == ".xml"
        )
        for annotation in annotations:
            candidates = index.get(annotation.stem.casefold(), [])
            if len(candidates) > 1:
                # EN: Prefer mirrored class folders. FR: Privilégier les dossiers de classe correspondants.
                candidates = [
                    p
                    for p in candidates
                    if p.parent.name.casefold() == annotation.parent.name.casefold()
                ]
            if len(candidates) != 1:
                raise DatasetError(
                    f"Missing or ambiguous image / Image absente ou ambiguë: {annotation}"
                )
            image = candidates[0]
            if image.resolve() in seen:
                raise DatasetError(f"Duplicate annotation / Annotation en double: {image}")
            seen.add(image.resolve())
            matched.append(ImageItem(image, annotation))
        self.all_images = matched
        return matched

    def _group_key(self, item: ImageItem) -> str:
        if self.config.data.group_by_board:
            match = re.match(r"^(\d+)_", item.image_path.stem)
            if match:
                return f"board:{match.group(1)}"
        return item.image_path.relative_to(self.data_path).as_posix()

    def create_yolo_dataset(self) -> tuple[int, int]:
        """EN: Build in staging before replacing owned data. FR: Préparer avant remplacement."""
        self.config.data.__post_init__()
        target = self.yolo_path.resolve()
        source = self.data_path.resolve()
        if target == source or target in source.parents or source in target.parents:
            raise DatasetError("Source and output must be separate / Séparer source et destination")
        if target.is_symlink() or self.yolo_path.is_symlink():
            raise DatasetError("Output cannot be a symlink / Destination sans lien symbolique")
        if target.exists() and not (target / MARKER).is_file():
            raise DatasetError(
                f"Refusing to replace unmanaged data / Destination non gérée: {target}"
            )
        if len(self.all_images) < 2:
            raise DatasetError(
                "At least two annotated images are required / Deux images annotées requises"
            )
        groups = defaultdict(list)
        labels = {}
        hashes = set()
        for item in self.all_images:
            if item.annotation_path is None:
                raise DatasetError(f"Missing annotation / Annotation absente: {item.image_path}")
            with item.image_path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest in hashes:
                raise DatasetError(f"Duplicate image content / Image identique: {item.image_path}")
            hashes.add(digest)
            try:
                with Image.open(item.image_path) as image:
                    size = image.size
                    image.verify()
            except (OSError, ValueError, ImportError) as error:
                raise DatasetError(f"Invalid image / Image invalide: {item.image_path}") from error
            labels[item.image_path] = VOCConverter.convert(item.annotation_path, *size)
            groups[self._group_key(item)].append(item)
        if len(groups) < 2:
            raise DatasetError(
                "At least two board groups are required; use --split-mode image only deliberately."
            )
        keys = sorted(groups)
        random.Random(self.config.data.random_seed).shuffle(keys)
        desired = max(1, round(len(self.all_images) * self.config.data.val_split))
        val_keys = set()
        count = 0
        for key in keys[:-1]:
            if count >= desired:
                break
            val_keys.add(key)
            count += len(groups[key])
        splits = {"train": [], "val": []}
        for key, items in groups.items():
            splits["val" if key in val_keys else "train"].extend(items)
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".pcb-staging-", dir=target.parent) as staging:
            stage = Path(staging) / "dataset"
            manifest = {
                "seed": self.config.data.random_seed,
                "group_by_board": self.config.data.group_by_board,
                "source": str(source),
                "splits": {},
            }
            for split, items in splits.items():
                (stage / "images" / split).mkdir(parents=True)
                (stage / "labels" / split).mkdir(parents=True)
                manifest["splits"][split] = []
                for item in sorted(items, key=lambda item: str(item.image_path)):
                    relative = item.image_path.relative_to(self.data_path).as_posix()
                    stem = hashlib.sha256(relative.encode()).hexdigest()
                    shutil.copy2(
                        item.image_path,
                        stage / "images" / split / f"{stem}{item.image_path.suffix.lower()}",
                    )
                    (stage / "labels" / split / f"{stem}.txt").write_text(
                        "\n".join(labels[item.image_path]), encoding="utf-8"
                    )
                    manifest["splits"][split].append(
                        {"source": relative, "id": stem, "group": self._group_key(item)}
                    )
            (stage / "dataset.yaml").write_text(
                yaml.safe_dump(
                    {
                        "path": str(target),
                        "train": "images/train",
                        "val": "images/val",
                        "names": dict(enumerate(Config.CLASS_NAMES)),
                        "nc": Config.NUM_CLASSES,
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            (stage / MARKER).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            backup = Path(staging) / "previous"
            if target.exists():
                target.rename(backup)
            try:
                stage.rename(target)
            except OSError:
                if backup.exists():
                    backup.rename(target)
                raise
        return len(splits["train"]), len(splits["val"])

    def get_yaml_path(self) -> Path:
        return self.yolo_path / "dataset.yaml"

    def get_stats(self) -> dict[str, int]:
        return {
            "total_images": len(self.all_images),
            "with_xml": len(self.all_images),
            "from_folders": 0,
        }
