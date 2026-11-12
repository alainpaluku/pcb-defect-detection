"""EN: Shared desktop constants. FR: Constantes partagées du bureau."""

from src.config import CLASS_NAMES, PROJECT_ROOT

APP_TITLE = "PCB Defect Detection - Model Tester"
APP_VERSION = "3.0.0"
DEFAULT_WINDOW_SIZE = "1200x800"
MIN_WINDOW_SIZE = (800, 600)


SUPPORTED_IMAGE_FORMATS = [
    ("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
    ("JPEG files", "*.jpg *.jpeg"),
    ("PNG files", "*.png"),
    ("BMP files", "*.bmp"),
    ("TIFF files", "*.tiff"),
    ("All files", "*.*"),
]

SUPPORTED_MODEL_FORMATS = [("PyTorch models", "*.pt"), ("All files", "*.*")]


DEFAULT_MODELS_DIR = "models"
DEFAULT_IMAGES_DIR = "images"


DEFAULT_CONFIDENCE = 0.3
DEFAULT_INTERSECTION_OVER_UNION = 0.5
CONFIDENCE_RANGE = (0.1, 0.9)
INTERSECTION_OVER_UNION_RANGE = (0.1, 0.9)


DEFECT_CLASSES = CLASS_NAMES

DEFECT_COLORS = [
    "#FF0000",
    "#FF8000",
    "#FFFF00",
    "#8000FF",
    "#0080FF",
    "#FF0080",
]

DEFECT_DESCRIPTIONS = {
    "missing_hole": "Missing drill hole",
    "mouse_bite": "Irregular edge",
    "open_circuit": "Broken trace",
    "short": "Short circuit",
    "spur": "Copper protrusion",
    "spurious_copper": "Unwanted copper",
}


CONTROL_PANEL_WIDTH = 300
CANVAS_MIN_SIZE = (400, 300)
RESULTS_PANEL_HEIGHT = 200


ZOOM_FACTOR = 1.2
MIN_ZOOM = 0.1
MAX_ZOOM = 5.0
DEFAULT_ZOOM = 1.0


AUTO_MODEL_PATHS = [
    PROJECT_ROOT / "models/pcb_model.pt",
    PROJECT_ROOT / "pcb_model.pt",
    PROJECT_ROOT / "output/pcb_model.pt",
]


EXPORT_FORMATS = [
    ("JSON files", "*.json"),
    ("Text files", "*.txt"),
    ("CSV files", "*.csv"),
    ("All files", "*.*"),
]
