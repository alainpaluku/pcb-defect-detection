"""
Configuration settings for the GUI test interface.
"""

from pathlib import Path
from typing import Dict, List, Tuple

# Application settings
APP_TITLE = "PCB Defect Detection - Model Tester"
APP_VERSION = "1.0.0"
DEFAULT_WINDOW_SIZE = "1200x800"
MIN_WINDOW_SIZE = (800, 600)

# File extensions
SUPPORTED_IMAGE_FORMATS = [
    ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
    ("JPEG files", "*.jpg *.jpeg"),
    ("PNG files", "*.png"),
    ("BMP files", "*.bmp"),
    ("TIFF files", "*.tiff"),
    ("All files", "*.*")
]

SUPPORTED_MODEL_FORMATS = [
    ("PyTorch models", "*.pt"),
    ("All files", "*.*")
]

# Default directories
DEFAULT_MODELS_DIR = "models"
DEFAULT_IMAGES_DIR = "images"

# Image extensions for batch processing
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG', '.BMP', '.TIFF'}

# Detection settings
DEFAULT_CONFIDENCE = 0.25
DEFAULT_INTERSECTION_OVER_UNION = 0.5
CONFIDENCE_RANGE = (0.1, 0.9)
INTERSECTION_OVER_UNION_RANGE = (0.1, 0.9)

# Defect classes and colors
DEFECT_CLASSES = [
    "missing_hole",
    "mouse_bite", 
    "open_circuit",
    "short",
    "spur",
    "spurious_copper"
]

DEFECT_COLORS = [
    "#FF0000",  # missing_hole - Red
    "#FF8000",  # mouse_bite - Orange  
    "#FFFF00",  # open_circuit - Yellow
    "#8000FF",  # short - Purple
    "#0080FF",  # spur - Blue
    "#FF0080",  # spurious_copper - Pink
]

DEFECT_DESCRIPTIONS = {
    "missing_hole": "Missing drill hole",
    "mouse_bite": "Irregular edge",
    "open_circuit": "Broken trace", 
    "short": "Short circuit",
    "spur": "Copper protrusion",
    "spurious_copper": "Unwanted copper"
}

# UI Layout settings
CONTROL_PANEL_WIDTH = 300
CANVAS_MIN_SIZE = (400, 300)
RESULTS_PANEL_HEIGHT = 200

# Zoom settings
ZOOM_FACTOR = 1.2
MIN_ZOOM = 0.1
MAX_ZOOM = 5.0
DEFAULT_ZOOM = 1.0

# Auto-detection paths for models
AUTO_MODEL_PATHS = [
    "models/pcb_model.pt",
    "pcb_model.pt",
    "output/pcb_model.pt"
]

# Export formats
EXPORT_FORMATS = [
    ("JSON files", "*.json"),
    ("Text files", "*.txt"),
    ("CSV files", "*.csv"),
    ("All files", "*.*")
]