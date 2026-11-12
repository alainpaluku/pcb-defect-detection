"""EN: Desktop formatting and UTF-8 exports. FR: Affichage et exports UTF-8 du bureau."""

import csv
import importlib.util
import json
import logging
from pathlib import Path

from src.detector import PCBInspector


def format_detection_results(detections: list[dict]) -> str:
    if not detections:
        return "No defects detected / Aucun défaut détecté.\n"
    lines = []
    for number, detection in enumerate(detections, 1):
        box = detection["bbox"]
        lines.append(
            f"{number}. {detection['class_name']} — {detection['confidence']:.2%}\n"
            f"   ({box['x1']:.0f}, {box['y1']:.0f}) - ({box['x2']:.0f}, {box['y2']:.0f})"
        )
    return "\n".join(lines) + "\n"


def export_results(detections, file_path, metadata=None):
    path = Path(file_path)
    if path.suffix.lower() == ".json":
        payload = {
            "metadata": metadata or {},
            "detections": detections,
            "summary": PCBInspector.get_summary(detections),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    elif path.suffix.lower() == ".csv":
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["class_name", "confidence", "x1", "y1", "x2", "y2"])
            for detection in detections:
                writer.writerow(
                    [
                        detection["class_name"],
                        detection["confidence"],
                        *[detection["bbox"][key] for key in ("x1", "y1", "x2", "y2")],
                    ]
                )
    elif path.suffix.lower() == ".txt":
        header = "\n".join(f"{key}: {value}" for key, value in (metadata or {}).items())
        path.write_text(header + "\n\n" + format_detection_results(detections), encoding="utf-8")
    else:
        raise ValueError("Use .json, .csv or .txt / Utiliser .json, .csv ou .txt")


def setup_logging(log_level="INFO"):
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")


def validate_dependencies():
    return [
        package
        for module, package in (
            ("tkinter", "tkinter (system package / paquet système)"),
            ("PIL", "Pillow"),
            ("ttkbootstrap", "ttkbootstrap"),
            ("ultralytics", "ultralytics-opencv-headless"),
        )
        if importlib.util.find_spec(module) is None
    ]
