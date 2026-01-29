"""Tests for PCB Defect Detection."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config, ModelConfig, InferenceConfig, IMAGE_EXTENSIONS
from src.data_ingestion import ImageItem, VOCConverter


class TestConfig(unittest.TestCase):
    """Test configuration."""
    
    def test_class_names_count(self):
        """Test que 6 classes sont définies."""
        self.assertEqual(len(Config.CLASS_NAMES), 6)
    
    def test_class_names_content(self):
        """Test les noms de classes attendus."""
        expected = {"missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"}
        self.assertEqual(set(Config.CLASS_NAMES), expected)
    
    def test_class_map_case_insensitive(self):
        """Test que le mapping gère différentes casses."""
        self.assertEqual(Config.CLASS_MAP["missing_hole"], 0)
        self.assertEqual(Config.CLASS_MAP["Missing_hole"], 0)
        self.assertEqual(Config.CLASS_MAP["short"], 3)
        self.assertEqual(Config.CLASS_MAP["Short"], 3)
    
    def test_num_classes_matches(self):
        """Test que NUM_CLASSES correspond au nombre de classes."""
        self.assertEqual(Config.NUM_CLASSES, len(Config.CLASS_NAMES))
    
    def test_output_path_is_path(self):
        """Test que get_output_path retourne un Path."""
        output = Config.get_output_path()
        self.assertIsInstance(output, Path)
    
    def test_create_custom_config(self):
        """Test la création de configuration personnalisée."""
        config = Config.create(epochs=100, batch_size=32)
        self.assertEqual(config.model.epochs, 100)
        self.assertEqual(config.model.batch_size, 32)


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig dataclass."""
    
    def test_default_values(self):
        """Test les valeurs par défaut."""
        cfg = ModelConfig()
        self.assertEqual(cfg.name, "yolov8n.pt")
        self.assertEqual(cfg.img_size, 640)
        self.assertEqual(cfg.batch_size, 16)
        self.assertEqual(cfg.epochs, 50)
    
    def test_custom_values(self):
        """Test les valeurs personnalisées."""
        cfg = ModelConfig(epochs=100, batch_size=32)
        self.assertEqual(cfg.epochs, 100)
        self.assertEqual(cfg.batch_size, 32)


class TestInferenceConfig(unittest.TestCase):
    """Test InferenceConfig dataclass."""
    
    def test_default_thresholds(self):
        """Test les seuils par défaut."""
        cfg = InferenceConfig()
        self.assertEqual(cfg.conf_threshold, 0.25)
        self.assertEqual(cfg.iou_threshold, 0.45)


class TestImageExtensions(unittest.TestCase):
    """Test des extensions d'images."""
    
    def test_common_extensions_included(self):
        """Test que les extensions communes sont incluses."""
        for ext in [".jpg", ".png", ".JPG", ".PNG"]:
            self.assertIn(ext, IMAGE_EXTENSIONS)


class TestImageItem(unittest.TestCase):
    """Test ImageItem dataclass."""
    
    def test_creation(self):
        """Test la création d'un ImageItem."""
        item = ImageItem(
            image_path=Path("test.jpg"),
            annotation_path=Path("test.xml"),
            source_type="xml"
        )
        self.assertEqual(item.image_path, Path("test.jpg"))
        self.assertEqual(item.source_type, "xml")
    
    def test_default_values(self):
        """Test les valeurs par défaut."""
        item = ImageItem(image_path=Path("test.jpg"))
        self.assertIsNone(item.annotation_path)
        self.assertIsNone(item.class_name)
        self.assertEqual(item.source_type, "unknown")


class TestVOCConverter(unittest.TestCase):
    """Test VOCConverter."""
    
    def test_clamp(self):
        """Test la fonction clamp."""
        self.assertEqual(VOCConverter._clamp(5, 0, 10), 5)
        self.assertEqual(VOCConverter._clamp(-5, 0, 10), 0)
        self.assertEqual(VOCConverter._clamp(15, 0, 10), 10)


class TestDataIngestion(unittest.TestCase):
    """Test data ingestion module."""
    
    def test_import(self):
        """Test que le module peut être importé."""
        from src.data_ingestion import DataIngestion
        self.assertIsNotNone(DataIngestion)
    
    def test_init_with_path(self):
        """Test l'initialisation avec un chemin."""
        from src.data_ingestion import DataIngestion
        data = DataIngestion(data_path="test_data")
        self.assertEqual(data.data_path, Path("test_data"))
    
    def test_init_default_path(self):
        """Test l'initialisation avec le chemin par défaut."""
        from src.data_ingestion import DataIngestion
        data = DataIngestion()
        self.assertEqual(data.data_path, Config.get_data_path())


class TestModel(unittest.TestCase):
    """Test model module."""
    
    def test_import(self):
        """Test que le module peut être importé."""
        from src.model import PCBDetector, ModelLoadError
        self.assertIsNotNone(PCBDetector)
        self.assertIsNotNone(ModelLoadError)


class TestDetector(unittest.TestCase):
    """Test detector module."""
    
    def test_import(self):
        """Test que le module peut être importé."""
        from src.detector import PCBInspector
        self.assertIsNotNone(PCBInspector)
    
    def test_get_summary_empty(self):
        """Test get_summary avec liste vide."""
        from src.detector import PCBInspector
        summary = PCBInspector.get_summary([])
        self.assertEqual(summary["status"], "OK")
        self.assertEqual(summary["defect_count"], 0)
    
    def test_get_summary_with_detections(self):
        """Test get_summary avec détections."""
        from src.detector import PCBInspector
        detections = [
            {"class_name": "short", "confidence": 0.9},
            {"class_name": "short", "confidence": 0.8},
            {"class_name": "spur", "confidence": 0.7},
        ]
        summary = PCBInspector.get_summary(detections)
        self.assertEqual(summary["status"], "DEFECT")
        self.assertEqual(summary["defect_count"], 3)
        self.assertEqual(summary["defects"]["short"], 2)
        self.assertEqual(summary["defects"]["spur"], 1)
        self.assertEqual(summary["max_confidence"], 0.9)


class TestTrainer(unittest.TestCase):
    """Test trainer module."""
    
    def test_import(self):
        """Test que le module peut être importé."""
        from src.trainer import TrainingManager, DatasetError
        self.assertIsNotNone(TrainingManager)
        self.assertIsNotNone(DatasetError)


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_format_bytes(self):
        """Test format_bytes."""
        from src.utils import format_bytes
        self.assertEqual(format_bytes(500), "500.00 B")
        self.assertEqual(format_bytes(1024), "1.00 KB")
        self.assertEqual(format_bytes(1024 * 1024), "1.00 MB")
    
    def test_format_metrics(self):
        """Test format_metrics."""
        from src.utils import format_metrics
        metrics = {
            "precision_detection": 0.85,
            "precision_stricte": 0.65,
            "fiabilite": 0.9,
            "taux_detection": 0.8
        }
        result = format_metrics(metrics)
        self.assertIn("0.8500", result)
        self.assertIn("0.6500", result)


if __name__ == "__main__":
    unittest.main()
