"""PCB Defect Detection Package."""

from src.config import Config
from src.model import PCBDetector
from src.detector import PCBInspector
from src.trainer import TrainingManager
from src.data_ingestion import DataIngestion

__all__ = [
    "Config",
    "PCBDetector", 
    "PCBInspector",
    "TrainingManager",
    "DataIngestion",
]

__version__ = "2.0.0"
