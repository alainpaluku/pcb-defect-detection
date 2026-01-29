"""
Main application entry point for the GUI test interface.
"""

import tkinter as tk
import sys
import logging
from pathlib import Path
import ttkbootstrap as ttk

from .utils import setup_logging, validate_dependencies
from .main_window import PCBDetectionGUI


def check_dependencies() -> bool:
    """Check if all required dependencies are available."""
    missing = validate_dependencies()
    
    if missing:
        print("❌ Missing required packages:")
        for package in missing:
            print(f"   • {package}")
        print(f"\\n📦 Install missing packages with:")
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True


def check_project_structure() -> bool:
    """Check if we can import the main project modules."""
    try:
        # Try to import main project modules
        sys.path.append(str(Path(__file__).parent.parent))
        from src.detector import PCBInspector
        from src.config import Config
        return True
    except ImportError as e:
        print(f"❌ Cannot import project modules: {e}")
        print("Make sure you're running from the project root directory.")
        print("The 'src' directory should be accessible from the parent directory.")
        return False


def main():
    """Main entry point for the GUI application."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    print("🚀 PCB Defect Detection - GUI Test Interface")
    print("=" * 50)
    
    # Check dependencies
    print("📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ Dependencies OK")
    
    # Check project structure
    print("📁 Checking project structure...")
    if not check_project_structure():
        sys.exit(1)
    print("✅ Project structure OK")
    
    # Create and run GUI with modern theme
    print("🎨 Launching GUI interface with modern theme...")
    try:
        # Create window with modern theme - using 'cosmo' theme (blue, professional)
        root = ttk.Window(themename="cosmo")
        app = PCBDetectionGUI(root)
        
        logger.info("GUI application started with ttkbootstrap theme")
        root.mainloop()
        
    except KeyboardInterrupt:
        print("\n👋 GUI closed by user")
        logger.info("Application closed by user")
    except Exception as e:
        error_msg = f"GUI application error: {e}"
        print(f"❌ {error_msg}")
        logger.error(error_msg, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()