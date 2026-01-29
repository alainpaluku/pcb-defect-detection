"""
Main window and application controller.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import threading
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from .config import (
    APP_TITLE, DEFAULT_WINDOW_SIZE, MIN_WINDOW_SIZE,
    SUPPORTED_IMAGE_FORMATS, SUPPORTED_MODEL_FORMATS, EXPORT_FORMATS
)
from .model_loader import ModelManager
from .image_handler import ImageManager, find_images_in_directory
from .ui_components import ControlPanel, ImageCanvas, ResultsPanel
from .dialogs import BatchProcessingDialog, StatisticsDialog, AboutDialog
from .utils import format_detection_results, export_results

logger = logging.getLogger(__name__)


class PCBDetectionGUI:
    """Main GUI application for PCB defect detection testing."""
    
    def __init__(self, root: ttk.Window):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry(DEFAULT_WINDOW_SIZE)
        self.root.minsize(*MIN_WINDOW_SIZE)
        
        # Managers
        self.model_manager = ModelManager()
        self.image_manager = ImageManager()
        
        # UI Components
        self.control_panel: Optional[ControlPanel] = None
        self.image_canvas: Optional[ImageCanvas] = None
        self.results_panel: Optional[ResultsPanel] = None
        self.status_var = tk.StringVar()
        
        # Detection results
        self.detection_results: List[Dict[str, Any]] = []
        
        # Setup UI
        self._setup_ui()
        self._setup_menu()
        self._setup_status_bar()
        
        # Auto-load model if available
        self._auto_load_model()
        
        # Set initial status
        self.status_var.set("Ready")
    
    def _setup_ui(self):
        """Setup the main UI components."""
        # Main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for responsive layout
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Left side: Control panel
        main_frame.columnconfigure(0, weight=0)  # Fixed width for controls
        # Middle: Image canvas (takes most space)
        main_frame.columnconfigure(1, weight=5)  # Image gets 5x weight (more space)
        # Right side: Results panel (reduced)
        main_frame.columnconfigure(2, weight=1)  # Results gets 1x weight (smaller)
        main_frame.rowconfigure(0, weight=1)
        
        # Create UI components
        self._create_control_panel(main_frame)
        self._create_image_canvas(main_frame)
        self._create_results_panel(main_frame)
    
    def _create_control_panel(self, parent):
        """Create the control panel."""
        callbacks = {
            'load_model': self.load_model,
            'load_image': self.load_image,
            'load_folder': self.load_folder,
            'run_detection': self.run_detection,
            'show_statistics': self.show_statistics,
            'save_results': self.save_results,
            'clear_all': self.clear_all
        }
        
        self.control_panel = ControlPanel(parent, callbacks)
        self.control_panel.frame.grid(
            row=0, column=0, 
            sticky=(tk.W, tk.E, tk.N, tk.S), 
            padx=(0, 10)
        )
    
    def _create_image_canvas(self, parent):
        """Create the image canvas."""
        callbacks = {
            'canvas_click': self.on_canvas_click,
            'mouse_wheel': self.on_mouse_wheel,
            'zoom_in': self.zoom_in,
            'zoom_out': self.zoom_out,
            'reset_zoom': self.reset_zoom
        }
        
        self.image_canvas = ImageCanvas(parent, callbacks)
        self.image_canvas.frame.grid(
            row=0, column=1, 
            sticky=(tk.W, tk.E, tk.N, tk.S),
            padx=(0, 10)
        )
    
    def _create_results_panel(self, parent):
        """Create the results panel."""
        self.results_panel = ResultsPanel(parent)
        self.results_panel.frame.grid(
            row=0, column=2, 
            sticky=(tk.W, tk.E, tk.N, tk.S)
        )
    
    def _setup_menu(self):
        """Setup the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Model...", command=self.load_model)
        file_menu.add_command(label="Load Image...", command=self.load_image)
        file_menu.add_separator()
        file_menu.add_command(label="Save Results...", command=self.save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Zoom In", command=self.zoom_in)
        view_menu.add_command(label="Zoom Out", command=self.zoom_out)
        view_menu.add_command(label="Reset Zoom", command=self.reset_zoom)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
    
    def _setup_status_bar(self):
        """Setup the status bar."""
        self.status_var.set("Ready")
        status_bar = ttk.Label(
            self.root, 
            textvariable=self.status_var,
            bootstyle="inverse-secondary",
            padding="5"
        )
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
    
    def _auto_load_model(self):
        """Try to automatically load a model."""
        try:
            if self.model_manager.auto_load_model():
                model_info = self.model_manager.get_model_info()
                self.control_panel.update_model_info(
                    f"✅ {model_info['name']}", 
                    "green"
                )
                self.status_var.set(f"Model loaded: {model_info['name']}")
                self._update_ui_state()
        except Exception as e:
            logger.warning(f"Auto-load failed: {e}")
    
    def load_model(self):
        """Load a trained model."""
        # Start from models directory if it exists
        initial_dir = Path("models") if Path("models").exists() else Path.cwd()
        
        model_path = filedialog.askopenfilename(
            title="Select Model File",
            initialdir=str(initial_dir),
            filetypes=SUPPORTED_MODEL_FORMATS
        )
        
        if not model_path:
            return
        
        try:
            self.status_var.set("Loading model...")
            self.root.update()
            
            self.model_manager.load_model(model_path)
            
            model_info = self.model_manager.get_model_info()
            self.control_panel.update_model_info(
                f"✅ {model_info['name']}", 
                "green",
                model_info.get('performance', '')
            )
            self.status_var.set(f"Model loaded: {model_info['name']} - {model_info.get('performance', 'Ready')}")
            self._update_ui_state()
            
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.status_var.set("Failed to load model")
            logger.error(error_msg)
    
    def load_image(self):
        """Load an image for detection."""
        # Start from images directory if it exists
        initial_dir = Path("images") if Path("images").exists() else Path.cwd()
        
        image_path = filedialog.askopenfilename(
            title="Select Image File",
            initialdir=str(initial_dir),
            filetypes=SUPPORTED_IMAGE_FORMATS
        )
        
        if not image_path:
            return
        
        try:
            if self.image_manager.load_image(image_path):
                self._display_current_image()
                
                image_info = self.image_manager.get_image_info()
                info_text = f"📷 {image_info['name']}\n{image_info['size'][0]}x{image_info['size'][1]}"
                self.control_panel.update_image_info(info_text)
                
                self.status_var.set(f"Image loaded: {image_info['name']}")
                self._update_ui_state()
                
                # Clear previous results
                self.detection_results = []
                self.results_panel.clear_results()
                
        except Exception as e:
            error_msg = f"Failed to load image: {str(e)}"
            messagebox.showerror("Error", error_msg)
            self.status_var.set("Failed to load image")
            logger.error(error_msg)
    
    def load_folder(self):
        """Load a folder of images for batch processing."""
        # Start from images directory if it exists
        initial_dir = Path("images") if Path("images").exists() else Path.cwd()
        
        folder_path = filedialog.askdirectory(
            title="Select Image Folder",
            initialdir=str(initial_dir)
        )
        
        if not folder_path:
            return
        
        try:
            image_files = find_images_in_directory(folder_path)
            
            if not image_files:
                messagebox.showwarning("Warning", "No image files found in the selected folder.")
                return
            
            # Show batch processing dialog
            dialog = BatchProcessingDialog(
                self.root, 
                image_files, 
                self.model_manager,
                self.control_panel.get_confidence()
            )
            
        except Exception as e:
            error_msg = f"Failed to load folder: {str(e)}"
            messagebox.showerror("Error", error_msg)
            logger.error(error_msg)
    
    def run_detection(self):
        """Run defect detection on current image."""
        if not self.model_manager.is_loaded or not self.image_manager.current_image:
            messagebox.showerror("Error", "Please load both a model and an image.")
            return
        
        try:
            self.status_var.set("Running detection...")
            self.control_panel.set_detect_button_state(False)
            self.root.update()
            
            # Run detection in thread to avoid blocking UI
            def detect():
                try:
                    confidence = self.control_panel.get_confidence()
                    self.detection_results = self.model_manager.run_inference(
                        self.image_manager.current_image_path,
                        confidence
                    )
                    
                    # Update UI in main thread
                    self.root.after(0, self._update_detection_results)
                    
                except Exception as e:
                    self.root.after(0, lambda: self._show_detection_error(str(e)))
            
            threading.Thread(target=detect, daemon=True).start()
            
        except Exception as e:
            self._show_detection_error(str(e))
    
    def _update_detection_results(self):
        """Update UI with detection results."""
        # Update image with detections
        self.image_manager.set_detections(self.detection_results)
        self._display_current_image()
        
        # Update results with color coding
        self.results_panel.update_results("", self.detection_results)
        
        # Update status
        defect_count = len(self.detection_results)
        self.status_var.set(f"Detection complete: {defect_count} defects found")
        self.control_panel.set_detect_button_state(True)
    
    def _show_detection_error(self, error_msg: str):
        """Show detection error."""
        messagebox.showerror("Detection Error", f"Detection failed:\\n{error_msg}")
        self.status_var.set("Detection failed")
        self.control_panel.set_detect_button_state(True)
    
    def show_statistics(self):
        """Show detection statistics."""
        if not self.detection_results:
            messagebox.showinfo("Statistics", "No detection results available.")
            return
        
        StatisticsDialog(self.root, self.detection_results)
    
    def save_results(self):
        """Save detection results to file."""
        if not self.detection_results:
            messagebox.showinfo("Save Results", "No detection results to save.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".json",
            filetypes=EXPORT_FORMATS
        )
        
        if not file_path:
            return
        
        try:
            export_results(
                self.detection_results,
                file_path,
                {
                    'image_path': self.image_manager.current_image_path,
                    'image_size': self.image_manager.get_image_info().get('size'),
                    'confidence_threshold': self.control_panel.get_confidence(),
                    'iou_threshold': self.control_panel.get_iou()
                }
            )
            
            messagebox.showinfo("Save Results", f"Results saved to:\\n{file_path}")
            
        except Exception as e:
            error_msg = f"Failed to save results: {str(e)}"
            messagebox.showerror("Error", error_msg)
            logger.error(error_msg)
    
    def clear_all(self):
        """Clear all loaded data."""
        self.image_manager.clear_image()
        self.detection_results = []
        
        # Clear UI
        self.image_canvas.clear_canvas()
        self.control_panel.update_image_info("No image loaded")
        self.results_panel.clear_results()
        
        self.status_var.set("Cleared all data")
        self._update_ui_state()
    
    def _update_ui_state(self):
        """Update UI state based on loaded data."""
        can_detect = (
            self.model_manager.is_loaded and 
            self.image_manager.current_image is not None
        )
        self.control_panel.set_detect_button_state(can_detect)
    
    def _display_current_image(self):
        """Display the current image on canvas."""
        if not self.image_manager.current_image:
            return
        
        # Get canvas size
        canvas_width = self.image_canvas.canvas.winfo_width()
        canvas_height = self.image_canvas.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(100, self._display_current_image)
            return
        
        # Prepare and display image
        photo_image = self.image_manager.prepare_display_image((canvas_width, canvas_height))
        if photo_image:
            self.image_canvas.update_image(photo_image)
            self.image_canvas.update_zoom_label(self.image_manager.get_zoom_percentage())
    
    def zoom_in(self):
        """Zoom in on image."""
        self.image_manager.zoom_in()
        self._display_current_image()
    
    def zoom_out(self):
        """Zoom out on image."""
        self.image_manager.zoom_out()
        self._display_current_image()
    
    def reset_zoom(self):
        """Reset zoom to 100%."""
        self.image_manager.reset_zoom()
        self._display_current_image()
    
    def on_canvas_click(self, event):
        """Handle canvas click events."""
        # Could be used for selecting specific detections
        pass
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming."""
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
    
    def show_about(self):
        """Show about dialog."""
        AboutDialog(self.root)