"""Data ingestion and conversion for PCB Defect Detection with YOLOv8."""

import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from src.config import Config, IMAGE_EXTENSIONS
from src.utils import find_image_file, get_logger

logger = get_logger(__name__)


@dataclass
class ImageItem:
    """Represents an image with its metadata."""
    image_path: Path
    annotation_path: Optional[Path] = None
    class_name: Optional[str] = None
    source_type: str = "unknown"  # "xml" ou "class_folder"


class VOCConverter:
    """VOC XML to YOLO format converter."""
    
    @staticmethod
    def convert(
        xml_path: Path,
        img_width: int,
        img_height: int
    ) -> List[str]:
        """Convert VOC XML annotation to YOLO format.
        
        Args:
            xml_path: Path to XML file
            img_width: Image width
            img_height: Image height
        
        Returns:
            List of YOLO format lines
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        yolo_lines = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in Config.CLASS_MAP:
                logger.warning(f"Unknown class ignored: {class_name}")
                continue
            
            class_id = Config.CLASS_MAP[class_name]
            bbox = obj.find("bndbox")
            
            # Extract and normalize coordinates
            xmin = VOCConverter._clamp(float(bbox.find("xmin").text), 0, img_width)
            ymin = VOCConverter._clamp(float(bbox.find("ymin").text), 0, img_height)
            xmax = VOCConverter._clamp(float(bbox.find("xmax").text), 0, img_width)
            ymax = VOCConverter._clamp(float(bbox.find("ymax").text), 0, img_height)
            
            # Convert to YOLO format (normalized center + dimensions)
            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            if width > 0 and height > 0:
                yolo_lines.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )
        
        return yolo_lines
    
    @staticmethod
    def _clamp(value: float, min_val: float, max_val: float) -> float:
        """Clamp a value between min and max."""
        return max(min_val, min(value, max_val))


class DataIngestion:
    """Manages data loading and conversion to YOLO format."""
    
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = Path(data_path) if data_path else Config.get_data_path()
        self.yolo_path = Config.get_yolo_dataset_path()
        self.images_dir: Optional[Path] = None
        self.annot_dir: Optional[Path] = None
        self.all_images: List[ImageItem] = []
    
    def find_data_structure(self) -> bool:
        """Search for images and annotations in the dataset."""
        logger.info(f"Searching in: {self.data_path}")
        
        # Debug: list structure
        self._debug_structure()
        
        self._find_annotations_dir()
        self._find_images_dir()
        
        logger.info(f"Images directory: {self.images_dir}")
        logger.info(f"Annotations directory: {self.annot_dir}")
        
        return self.images_dir is not None or self.annot_dir is not None
    
    def _debug_structure(self) -> None:
        """Display structure for debugging."""
        if not self.data_path.exists():
            logger.warning(f"Path not found: {self.data_path}")
            return
        
        logger.info("Dataset structure:")
        for item in self.data_path.rglob("*"):
            if item.is_dir():
                # Count files in this directory
                files = [f for f in item.iterdir() if f.is_file()]
                if files:
                    logger.info(f"  📁 {item.relative_to(self.data_path)} ({len(files)} files)")
    
    def _find_annotations_dir(self) -> None:
        """Search for annotations directory."""
        from src.utils import find_directory_with_files
        
        self.annot_dir = find_directory_with_files(
            self.data_path,
            ["Annotations", "PCB_DATASET/Annotations"],
            [".xml"],
            recursive=True
        )
        
        if self.annot_dir:
            xml_count = len(list(self.annot_dir.rglob("*.xml")))
            logger.info(f"Found {xml_count} XML files in {self.annot_dir}")
    
    def _find_images_dir(self) -> None:
        """Search for images directory."""
        from src.utils import find_directory_with_files
        
        # Search for "images" directory first
        self.images_dir = find_directory_with_files(
            self.data_path,
            ["images", "PCB_DATASET/images"],
            IMAGE_EXTENSIONS,
            recursive=True
        )
        
        if self.images_dir:
            logger.info(f"Found images directory: {self.images_dir}")
            return
        
        # Otherwise search for class directories
        search_dirs = [self.data_path, self.data_path / "PCB_DATASET"]
        
        for search_dir in search_dirs:
            if search_dir.exists() and self._has_class_folders(search_dir):
                self.images_dir = search_dir
                return
    
    def _has_class_folders(self, search_dir: Path) -> bool:
        """Check if directory contains class subdirectories."""
        if not search_dir.exists():
            return False
        
        class_names_lower = {name.lower() for name in Config.CLASS_NAMES}
        
        for subdir in search_dir.iterdir():
            if subdir.is_dir():
                # Case-insensitive check
                subdir_name = subdir.name.lower().replace("-", "_").replace(" ", "_")
                if subdir_name in class_names_lower:
                    return True
        return False
    
    def collect_images(self) -> List[ImageItem]:
        """Collect all images with their annotations."""
        self.all_images = []
        seen_images = set()
        
        # Collect via XML annotations (priority)
        if self.annot_dir and self.annot_dir.exists():
            self._collect_from_xml(seen_images)
        
        # Collect via class directories
        if self.images_dir:
            self._collect_from_class_folders(seen_images)
        
        logger.info(f"Total images collected: {len(self.all_images)}")
        return self.all_images
    
    def _collect_from_xml(self, seen_images: set) -> None:
        """Collect images from XML annotations."""
        # Search XML files recursively (class structure)
        xml_files = list(self.annot_dir.rglob("*.xml"))
        logger.info(f"Found {len(xml_files)} XML annotations")
        
        # Build list of directories to search for images
        search_dirs = []
        if self.images_dir:
            search_dirs.append(self.images_dir)
            # Add all subdirectories
            for subdir in self.images_dir.rglob("*"):
                if subdir.is_dir():
                    search_dirs.append(subdir)
        
        search_dirs.append(self.data_path)
        
        # Add PCB_DATASET and its subdirectories
        pcb_dataset = self.data_path / "PCB_DATASET"
        if pcb_dataset.exists():
            search_dirs.append(pcb_dataset)
            for subdir in pcb_dataset.rglob("*"):
                if subdir.is_dir():
                    search_dirs.append(subdir)
        
        logger.info(f"Searching for images in {len(search_dirs)} directories")
        
        found_count = 0
        for xml_file in xml_files:
            img_path = find_image_file(xml_file.stem, search_dirs)
            
            if img_path:
                self.all_images.append(ImageItem(
                    image_path=img_path,
                    annotation_path=xml_file,
                    source_type="xml"
                ))
                seen_images.add(img_path)
                found_count += 1
        
        logger.info(f"Images found for {found_count}/{len(xml_files)} annotations")
    
    def _collect_from_class_folders(self, seen_images: set) -> None:
        """Collect images from class directories."""
        if not self.images_dir or not self.images_dir.exists():
            return
        
        # Create case-insensitive mapping
        class_name_map = {}
        for cls_name in Config.CLASS_NAMES:
            class_name_map[cls_name.lower()] = cls_name
            class_name_map[cls_name.lower().replace("_", "-")] = cls_name
            class_name_map[cls_name.lower().replace("_", " ")] = cls_name
        
        # Browse subdirectories
        for subdir in self.images_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            # Normalize directory name
            normalized_name = subdir.name.lower().replace("-", "_").replace(" ", "_")
            
            # Find corresponding class name
            cls_name = None
            if normalized_name in class_name_map:
                cls_name = class_name_map[normalized_name]
            else:
                # Try partial matching
                for key, value in class_name_map.items():
                    if key in normalized_name or normalized_name in key:
                        cls_name = value
                        break
            
            if cls_name is None:
                continue
            
            logger.info(f"Collecting from directory: {subdir.name} -> class: {cls_name}")
            
            for ext in IMAGE_EXTENSIONS:
                for img_path in subdir.glob(f"*{ext}"):
                    if img_path not in seen_images:
                        self.all_images.append(ImageItem(
                            image_path=img_path,
                            class_name=cls_name,
                            source_type="class_folder"
                        ))
                        seen_images.add(img_path)
    
    def create_yolo_dataset(self) -> Tuple[int, int]:
        """Create YOLO format dataset."""
        logger.info("Creating YOLO structure...")
        
        # Création des répertoires
        for split in ["train", "val"]:
            (self.yolo_path / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.yolo_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        # Shuffle and split
        random.seed(Config.data.random_seed)
        shuffled = self.all_images.copy()
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * (1 - Config.data.val_split))
        train_images = shuffled[:split_idx]
        val_images = shuffled[split_idx:]
        
        logger.info(f"Train: {len(train_images)}, Val: {len(val_images)}")
        
        # Process images
        train_count = self._process_split(train_images, "train")
        val_count = self._process_split(val_images, "val")
        
        logger.info(f"Processed - Train: {train_count}, Val: {val_count}")
        
        # Create YAML file
        self._create_yaml_config()
        
        return train_count, val_count
    
    def _process_split(self, image_list: List[ImageItem], split: str) -> int:
        """Process images for a split (train/val)."""
        count = 0
        
        for item in image_list:
            # Check if file exists
            if not item.image_path.exists():
                logger.warning(f"File not found: {item.image_path}")
                continue
            
            try:
                img = Image.open(item.image_path)
                img.verify()  # Check image integrity
                img = Image.open(item.image_path)  # Reopen after verify()
                img_width, img_height = img.size
            except Exception as e:
                logger.warning(f"Invalid image {item.image_path}: {e}")
                continue
            
            # Copy image
            dst_img = self.yolo_path / "images" / split / item.image_path.name
            shutil.copy(item.image_path, dst_img)
            
            # Create label
            label_path = self.yolo_path / "labels" / split / f"{item.image_path.stem}.txt"
            
            if item.source_type == "xml" and item.annotation_path:
                yolo_lines = VOCConverter.convert(
                    item.annotation_path, img_width, img_height
                )
                if yolo_lines:  # Only if valid annotations
                    label_path.write_text("\n".join(yolo_lines))
                else:
                    # No valid annotations, skip this image
                    dst_img.unlink(missing_ok=True)
                    continue
            else:
                # Images without XML - ignore for training
                # as full image bbox is not useful for detection
                logger.debug(f"Image without XML annotation ignored: {item.image_path.name}")
                dst_img.unlink(missing_ok=True)
                continue
            
            count += 1
        
        return count
    
    def _create_yaml_config(self) -> Path:
        """Create YOLO dataset YAML configuration."""
        yaml_content = f"""path: {self.yolo_path}
train: images/train
val: images/val

names:
{chr(10).join(f'  {i}: {name}' for i, name in enumerate(Config.CLASS_NAMES))}

nc: {Config.NUM_CLASSES}
"""
        yaml_path = self.yolo_path / "dataset.yaml"
        yaml_path.write_text(yaml_content)
        
        logger.info(f"Dataset config saved: {yaml_path}")
        return yaml_path
    
    def get_yaml_path(self) -> Path:
        """Return path to dataset YAML config."""
        return self.yolo_path / "dataset.yaml"
    
    def get_stats(self) -> Dict[str, int]:
        """Return dataset statistics."""
        return {
            "total_images": len(self.all_images),
            "with_xml": sum(1 for item in self.all_images if item.source_type == "xml"),
            "from_folders": sum(1 for item in self.all_images if item.source_type == "class_folder"),
        }
