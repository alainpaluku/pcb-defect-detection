"""EN: Small shared helpers without import side effects. FR: Utilitaires sans effet à l'import."""

import logging
from pathlib import Path

from src.config import IMAGE_EXTENSIONS


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def get_all_images(directory: Path, recursive: bool = False) -> list[Path]:
    """EN: Return sorted files only. FR: Retourner uniquement les fichiers, triés."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(p for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def count_images(directory: Path, recursive: bool = False) -> int:
    return len(get_all_images(directory, recursive))


def format_bytes(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def format_metrics(metrics: dict[str, float]) -> str:
    """EN: Keep mAP distinct from precision. FR: Distinguer mAP et précision."""
    return "\n".join(f"{key}: {value:.4f}" for key, value in metrics.items())


def print_detection_results(detections: list[dict]) -> None:
    if not detections:
        print("No defects detected / Aucun défaut détecté.")
    for item in detections:
        print(f"{item['class_name']}: {item['confidence']:.2%} — {item['bbox']}")
