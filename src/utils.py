"""Utility functions for PCB Defect Detection."""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.config import IMAGE_EXTENSIONS

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def setup_kaggle_environment() -> None:
    """Configure Kaggle execution environment."""
    target_dir = Path("/kaggle/working/pcb-defect-detector")
    if Path.cwd().name != "pcb-defect-detector" and target_dir.exists():
        os.chdir(target_dir)
    
    sys.path.insert(0, str(Path.cwd()))


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger."""
    return logging.getLogger(name)


def count_images(
    directory: Path,
    recursive: bool = False
) -> int:
    """Count images in a directory."""
    pattern = "**/*" if recursive else "*"
    return sum(
        1 for f in directory.glob(pattern)
        if f.suffix.lower() in [ext.lower() for ext in IMAGE_EXTENSIONS]
    )


def get_all_images(
    directory: Path,
    recursive: bool = False
) -> List[Path]:
    """Get all image files from a directory."""
    pattern = "**/*" if recursive else "*"
    return [
        f for f in directory.glob(pattern)
        if f.suffix.lower() in [ext.lower() for ext in IMAGE_EXTENSIONS]
    ]


def find_image_file(
    base_name: str,
    search_dirs: List[Path],
    subdirs: Optional[List[str]] = None
) -> Optional[Path]:
    """Search for an image file by its base name."""
    
    for search_dir in search_dirs:
        if not search_dir or not search_dir.exists():
            continue
        
        # Search directly in the folder
        for ext in IMAGE_EXTENSIONS:
            candidate = search_dir / f"{base_name}{ext}"
            if candidate.exists():
                return candidate
        
        # Search in specified subdirectories
        if subdirs:
            for subdir in subdirs:
                if not subdir:
                    continue
                check_dir = search_dir / subdir
                if not check_dir.exists():
                    continue
                for ext in IMAGE_EXTENSIONS:
                    candidate = check_dir / f"{base_name}{ext}"
                    if candidate.exists():
                        return candidate
    
    return None


def format_bytes(size_bytes: int) -> str:
    """Format bytes to readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def print_section_header(title: str, width: int = 60) -> None:
    """Display formatted section header."""
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def print_detection_results(detections: List[Dict]) -> None:
    """Display detection results in formatted way."""
    if not detections:
        print("No defects detected.")
        return
    
    print(f"\n{len(detections)} defect(s) detected:")
    for i, det in enumerate(detections, 1):
        bbox = det["bbox"]
        print(f"  {i}. {det['class_name']}: {det['confidence']:.2%}")
        print(f"     Box: ({bbox['x1']:.0f}, {bbox['y1']:.0f}) - ({bbox['x2']:.0f}, {bbox['y2']:.0f})")


def find_directory_with_files(
    base_path: Path,
    dir_names: List[str],
    file_extensions: List[str],
    recursive: bool = True
) -> Optional[Path]:
    """Generic search for a directory containing specific files.
    
    Args:
        base_path: Base path for search
        dir_names: Directory names to search for
        file_extensions: File extensions to search for
        recursive: If True, recursive search
    
    Returns:
        Path of found directory or None
    """
    if not base_path.exists():
        return None
    
    # Direct search
    for dir_name in dir_names:
        candidate = base_path / dir_name
        if candidate.exists() and candidate.is_dir():
            # Check if it contains the searched files
            for ext in file_extensions:
                if list(candidate.glob(f"*{ext}")):
                    return candidate
                # Search in subdirectories
                if list(candidate.rglob(f"*{ext}")):
                    return candidate
    
    # Recursive search if enabled
    if recursive:
        for dir_name in dir_names:
            for found_dir in base_path.rglob(dir_name):
                if found_dir.is_dir():
                    for ext in file_extensions:
                        if list(found_dir.glob(f"*{ext}")) or list(found_dir.rglob(f"*{ext}")):
                            return found_dir
    
    return None


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics for display."""
    lines = [
        f"Detection precision: {metrics.get('precision_detection', 0):.4f}",
        f"Strict precision:    {metrics.get('precision_stricte', 0):.4f}",
        f"Reliability:         {metrics.get('fiabilite', 0):.4f}",
        f"Detection rate:      {metrics.get('taux_detection', 0):.4f}",
    ]
    return "\n".join(lines)
