"""EN: Small shared helpers without import side effects. FR: Utilitaires sans effet à l'import."""

from pathlib import Path

from src.config import IMAGE_EXTENSIONS


def get_all_images(directory: Path, recursive: bool = False) -> list[Path]:
    """EN: Return sorted files only. FR: Retourner uniquement les fichiers, triés."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    candidates = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(p for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def print_detection_results(detections: list[dict]) -> None:
    if not detections:
        print("No defects detected / Aucun défaut détecté.")
    for item in detections:
        print(f"{item['class_name']}: {item['confidence']:.2%} — {item['bbox']}")
