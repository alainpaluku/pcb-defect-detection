"""
Utility functions for the GUI application.
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def format_detection_results(detections: List[Dict[str, Any]]) -> str:
    """Format detection results for display."""
    if not detections:
        return "✅ No defects detected in this image.\\n"
    
    result_text = f"🔍 Detection Results ({len(detections)} defects found):\\n\\n"
    
    for i, detection in enumerate(detections, 1):
        bbox = detection['bbox']
        result_text += f"{i}. {detection['class_name']}\\n"
        result_text += f"   Confidence: {detection['confidence']:.2%}\\n"
        result_text += f"   Location: ({bbox['x1']:.0f}, {bbox['y1']:.0f}) - ({bbox['x2']:.0f}, {bbox['y2']:.0f})\\n\\n"
    
    return result_text


def export_results(detections: List[Dict[str, Any]], file_path: str, 
                  metadata: Dict[str, Any] = None) -> None:
    """Export detection results to file."""
    file_path = Path(file_path)
    
    if file_path.suffix.lower() == '.json':
        _export_json(detections, file_path, metadata)
    elif file_path.suffix.lower() == '.csv':
        _export_csv(detections, file_path)
    else:
        _export_text(detections, file_path, metadata)


def _export_json(detections: List[Dict[str, Any]], file_path: Path, 
                metadata: Dict[str, Any] = None) -> None:
    """Export results as JSON."""
    results_data = {
        'metadata': metadata or {},
        'detections': detections,
        'summary': {
            'total_detections': len(detections),
            'defect_counts': _count_defects_by_class(detections)
        }
    }
    
    with open(file_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logger.info(f"Results exported to JSON: {file_path}")


def _export_csv(detections: List[Dict[str, Any]], file_path: Path) -> None:
    """Export results as CSV."""
    if not detections:
        # Create empty CSV with headers
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['class_name', 'confidence', 'x1', 'y1', 'x2', 'y2'])
        return
    
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['class_name', 'confidence', 'x1', 'y1', 'x2', 'y2'])
        
        # Write detections
        for detection in detections:
            bbox = detection['bbox']
            writer.writerow([
                detection['class_name'],
                detection['confidence'],
                bbox['x1'],
                bbox['y1'],
                bbox['x2'],
                bbox['y2']
            ])
    
    logger.info(f"Results exported to CSV: {file_path}")


def _export_text(detections: List[Dict[str, Any]], file_path: Path, 
                metadata: Dict[str, Any] = None) -> None:
    """Export results as text."""
    with open(file_path, 'w') as f:
        f.write("PCB Defect Detection Results\\n")
        f.write("=" * 40 + "\\n\\n")
        
        # Write metadata if available
        if metadata:
            f.write("Image Information:\\n")
            f.write("-" * 20 + "\\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\\n")
            f.write("\\n")
        
        # Write detection summary
        f.write(f"Total Defects: {len(detections)}\\n\\n")
        
        if not detections:
            f.write("No defects detected.\\n")
            return
        
        # Write detailed results
        f.write("Detailed Results:\\n")
        f.write("-" * 20 + "\\n")
        
        for i, detection in enumerate(detections, 1):
            bbox = detection['bbox']
            f.write(f"{i}. {detection['class_name']}\\n")
            f.write(f"   Confidence: {detection['confidence']:.2%}\\n")
            f.write(f"   Location: ({bbox['x1']:.0f}, {bbox['y1']:.0f}) - ({bbox['x2']:.0f}, {bbox['y2']:.0f})\\n\\n")
        
        # Write summary by class
        defect_counts = _count_defects_by_class(detections)
        if defect_counts:
            f.write("Summary by Defect Type:\\n")
            f.write("-" * 25 + "\\n")
            for class_name, count in sorted(defect_counts.items()):
                percentage = (count / len(detections)) * 100
                f.write(f"{class_name}: {count} ({percentage:.1f}%)\\n")
    
    logger.info(f"Results exported to text: {file_path}")


def _count_defects_by_class(detections: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count detections by class."""
    counts = {}
    for detection in detections:
        class_name = detection['class_name']
        counts[class_name] = counts.get(class_name, 0) + 1
    return counts


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("gui_test.log")
        ]
    )


def validate_dependencies() -> List[str]:
    """Check if required dependencies are available."""
    missing_packages = []
    
    required_packages = [
        ('tkinter', 'tkinter'),
        ('PIL', 'Pillow'),
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
    ]
    
    for package_name, pip_name in required_packages:
        try:
            __import__(package_name)
        except ImportError:
            missing_packages.append(pip_name)
    
    return missing_packages