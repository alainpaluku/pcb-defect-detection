"""EN: Lightweight package entry point. FR: Point d'entrée léger du paquet."""

from importlib import import_module

__version__ = "3.0.0"
__all__ = ["Config", "PCBDetector", "PCBInspector", "TrainingManager", "DataIngestion"]
_MODULES = {
    "Config": "config",
    "PCBDetector": "model",
    "PCBInspector": "detector",
    "TrainingManager": "trainer",
    "DataIngestion": "data_ingestion",
}


def __getattr__(name: str):
    """EN: Keep training imports out of web startup. FR: Différer les imports d'entraînement."""
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(f"src.{_MODULES[name]}"), name)
    globals()[name] = value
    return value
