#!/usr/bin/env python3
"""
PCB Defect Detection - Kaggle Runner

Usage on Kaggle:
    1. Add dataset akhatova/pcb-defects via "+ Add Input"
    2. Enable GPU T4 + Internet in Settings
    3. Execute: !python run_kaggle.py
"""

import subprocess
import sys
from pathlib import Path
from src.utils import setup_kaggle_environment


def install_dependencies() -> None:
    """Install required dependencies."""
    print("Installing ultralytics...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "ultralytics", "-q"],
        check=True
    )


def debug_dataset_structure() -> None:
    """Display dataset structure for debugging."""
    print("\n" + "=" * 60)
    print("DEBUG: Kaggle dataset structure")
    print("=" * 60)
    
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.exists():
        print("Not in Kaggle environment")
        return
    
    # List available datasets
    print(f"\nDatasets in {kaggle_input}:")
    for item in kaggle_input.iterdir():
        print(f"  📁 {item.name}")
        if item.is_dir():
            # Show subdirectories (2 levels)
            for sub in item.iterdir():
                prefix = "    📁" if sub.is_dir() else "    📄"
                print(f"{prefix} {sub.name}")
                if sub.is_dir():
                    # Count files
                    files = list(sub.iterdir())
                    if len(files) <= 10:
                        for f in files:
                            prefix2 = "      📁" if f.is_dir() else "      📄"
                            print(f"{prefix2} {f.name}")
                    else:
                        print(f"      ... ({len(files)} items)")
    print("=" * 60 + "\n")


def is_gpu_available() -> bool:
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


def main() -> None:
    """Main entry point."""
    setup_kaggle_environment()
    install_dependencies()
    
    # Import after setup
    from src.trainer import TrainingManager
    
    # Debug: display dataset structure
    debug_dataset_structure()
    
    print("\n" + "=" * 60)
    print("🚀 STARTING PCB DEFECT DETECTION TRAINING")
    print("=" * 60)
    print(f"   Epochs: 100")
    print(f"   GPU: {'✅ Available' if is_gpu_available() else '❌ Not available'}")
    print("=" * 60 + "\n")
    
    # Standard training
    trainer = TrainingManager()
    trainer.run_pipeline(epochs=100)


if __name__ == "__main__":
    main()
