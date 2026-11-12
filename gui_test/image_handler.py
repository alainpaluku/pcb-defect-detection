"""EN: Image display and annotation. FR: Affichage et annotation des images."""

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageTk

from src.space import SpaceSettings, load_image

from .config import DEFECT_COLORS

logger = logging.getLogger(__name__)


class ImageManager:
    """Manages image loading, processing, and display operations."""

    def __init__(self):
        self.current_image: Image.Image | None = None
        self.current_image_path: str | None = None
        self.photo_image: ImageTk.PhotoImage | None = None
        self.zoom_factor = 1.0
        self.detections = []

    def load_image(self, image_path: str | Path) -> bool:
        """Load an image from the specified path."""
        try:
            image_path = str(image_path)
            image = load_image(image_path, SpaceSettings())

            self.current_image = image
            self.current_image_path = image_path
            self.zoom_factor = 1.0
            self.detections = []

            logger.info(f"Image loaded: {image_path} ({image.size})")
            return True

        except (OSError, ValueError) as error:
            raise ValueError(f"Cannot load image / Image illisible: {image_path}") from error

    def get_image_info(self) -> dict:
        """Get information about the current image."""
        if self.current_image is None:
            return {"status": "No image loaded"}

        return {
            "status": "Loaded",
            "name": Path(self.current_image_path).name if self.current_image_path else "Unknown",
            "size": self.current_image.size,
            "format": Path(self.current_image_path).suffix.lower(),
            "mode": self.current_image.mode,
        }

    def set_detections(self, detections: list[dict]):
        """Set detection results for visualization."""
        self.detections = detections
        logger.info(f"Set {len(detections)} detections for visualization")

    def prepare_display_image(self) -> ImageTk.PhotoImage | None:
        """Prepare image for display with zoom and detections."""
        if self.current_image is None:
            return None

        try:
            display_image = self.current_image.copy()
            if self.zoom_factor != 1.0:
                new_size = (
                    max(1, int(self.current_image.width * self.zoom_factor)),
                    max(1, int(self.current_image.height * self.zoom_factor)),
                )
                display_image = display_image.resize(new_size, Image.Resampling.LANCZOS)

            if self.detections:
                display_image = self._draw_detections(display_image)

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
                bbox = detection["bbox"]
                class_id = detection.get("class_id", 0)
                class_name = detection.get("class_name", "unknown")
                confidence = detection.get("confidence", 0.0)

                x1 = int(bbox["x1"] * self.zoom_factor)
                y1 = int(bbox["y1"] * self.zoom_factor)
                x2 = int(bbox["x2"] * self.zoom_factor)
                y2 = int(bbox["y2"] * self.zoom_factor)

                color = DEFECT_COLORS[class_id % len(DEFECT_COLORS)]

                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                label = f"{class_name}: {confidence:.2%}"

                try:
                    bbox_text = draw.textbbox((x1, y1 - 25), label)

                    draw.rectangle(bbox_text, fill=color)

                    draw.text((x1, y1 - 25), label, fill="white")
                except (AttributeError, ValueError):
                    draw.text((x1, y1 - 25), label, fill=color)

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
        self.photo_image = None
        self.zoom_factor = 1.0
        self.detections = []
        logger.info("Image cleared")
