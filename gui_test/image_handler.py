"""
Image handling and processing functionality.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk

from .config import IMAGE_EXTENSIONS, DEFECT_COLORS, DEFECT_CLASSES

logger = logging.getLogger(__name__)


class ImageManager:
    """Manages image loading, processing, and display operations."""
    
    def __init__(self):
        self.current_image: Optional[Image.Image] = None
        self.current_image_path: Optional[str] = None
        self.display_image: Optional[Image.Image] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None
        self.zoom_factor = 1.0
        self.detections = []
        
    def load_image(self, image_path: Union[str, Path]) -> bool:
        """Load an image from the specified path."""
        try:
            image_path = str(image_path)
            image = Image.open(image_path)
            
            # Verify image integrity
            image.verify()
            
            # Reload image after verify (verify closes the file)
            image = Image.open(image_path)
            
            self.current_image = image
            self.current_image_path = image_path
            self.zoom_factor = 1.0
            self.detections = []
            
            logger.info(f"Image loaded: {image_path} ({image.size})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return False
    
    def get_image_info(self) -> dict:
        """Get information about the current image."""
        if self.current_image is None:
            return {"status": "No image loaded"}
            
        return {
            "status": "Loaded",
            "name": Path(self.current_image_path).name if self.current_image_path else "Unknown",
            "size": self.current_image.size,
            "format": self.current_image.format,
            "mode": self.current_image.mode
        }
    
    def set_detections(self, detections: List[dict]):
        """Set detection results for visualization."""
        self.detections = detections
        logger.info(f"Set {len(detections)} detections for visualization")
    
    def prepare_display_image(self, canvas_size: Tuple[int, int]) -> Optional[ImageTk.PhotoImage]:
        """Prepare image for display with zoom and detections."""
        if self.current_image is None:
            return None
            
        try:
            # Apply zoom
            display_image = self.current_image.copy()
            if self.zoom_factor != 1.0:
                new_size = (
                    int(self.current_image.width * self.zoom_factor),
                    int(self.current_image.height * self.zoom_factor)
                )
                display_image = display_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Draw detections if available
            if self.detections:
                display_image = self._draw_detections(display_image)
            
            # Convert to PhotoImage
            self.display_image = display_image
            self.photo_image = ImageTk.PhotoImage(display_image)
            
            return self.photo_image
            
        except Exception as e:
            logger.error(f"Failed to prepare display image: {e}")
            return None
    
    def _draw_detections(self, image: Image.Image) -> Image.Image:
        """Draw detection bounding boxes and labels on image."""
        draw = ImageDraw.Draw(image)
        
        for detection in self.detections:
            try:
                bbox = detection['bbox']
                class_id = detection.get('class_id', 0)
                class_name = detection.get('class_name', 'unknown')
                confidence = detection.get('confidence', 0.0)
                
                # Scale bbox coordinates with zoom
                x1 = int(bbox['x1'] * self.zoom_factor)
                y1 = int(bbox['y1'] * self.zoom_factor)
                x2 = int(bbox['x2'] * self.zoom_factor)
                y2 = int(bbox['y2'] * self.zoom_factor)
                
                # Get color for this defect type
                color = DEFECT_COLORS[class_id % len(DEFECT_COLORS)]
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Draw label
                label = f"{class_name}: {confidence:.2%}"
                
                # Calculate text size and position
                try:
                    bbox_text = draw.textbbox((x1, y1-25), label)
                    # Draw label background
                    draw.rectangle(bbox_text, fill=color)
                    # Draw label text
                    draw.text((x1, y1-25), label, fill="white")
                except:
                    # Fallback for older Pillow versions
                    draw.text((x1, y1-25), label, fill=color)
                    
            except Exception as e:
                logger.warning(f"Failed to draw detection: {e}")
                continue
        
        return image
    
    def zoom_in(self, factor: float = 1.2):
        """Zoom in on the image."""
        self.zoom_factor = min(self.zoom_factor * factor, 5.0)
        logger.debug(f"Zoomed in to {self.zoom_factor:.2f}x")
    
    def zoom_out(self, factor: float = 1.2):
        """Zoom out on the image."""
        self.zoom_factor = max(self.zoom_factor / factor, 0.1)
        logger.debug(f"Zoomed out to {self.zoom_factor:.2f}x")
    
    def reset_zoom(self):
        """Reset zoom to 100%."""
        self.zoom_factor = 1.0
        logger.debug("Zoom reset to 1.0x")
    
    def get_zoom_percentage(self) -> int:
        """Get current zoom as percentage."""
        return int(self.zoom_factor * 100)
    
    def clear_image(self):
        """Clear the current image and detections."""
        self.current_image = None
        self.current_image_path = None
        self.display_image = None
        self.photo_image = None
        self.zoom_factor = 1.0
        self.detections = []
        logger.info("Image cleared")


def find_images_in_directory(directory: Union[str, Path]) -> List[Path]:
    """Find all image files in a directory."""
    directory = Path(directory)
    
    if not directory.exists() or not directory.is_dir():
        return []
    
    image_files = []
    for file_path in directory.iterdir():
        if file_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(file_path)
    
    return sorted(image_files)


def validate_image_path(image_path: Union[str, Path]) -> bool:
    """Validate if the image path exists and has a supported format."""
    path = Path(image_path)
    
    if not path.exists():
        return False
        
    return path.suffix.lower() in IMAGE_EXTENSIONS