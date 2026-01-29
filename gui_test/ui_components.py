"""
UI components and widgets for the GUI application.
"""

import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
from typing import Callable, Optional, Any
import logging

from .config import (
    CONTROL_PANEL_WIDTH, DEFAULT_CONFIDENCE, DEFAULT_INTERSECTION_OVER_UNION,
    CONFIDENCE_RANGE, INTERSECTION_OVER_UNION_RANGE, DEFECT_CLASSES, DEFECT_DESCRIPTIONS
)

logger = logging.getLogger(__name__)


class ControlPanel:
    """Control panel with model loading, settings, and actions."""
    
    def __init__(self, parent: tk.Widget, callbacks: dict):
        self.parent = parent
        self.callbacks = callbacks
        self.frame = None
        self.confidence_var = tk.DoubleVar(value=DEFAULT_CONFIDENCE)
        self.iou_var = tk.DoubleVar(value=DEFAULT_INTERSECTION_OVER_UNION)
        self.model_info_label = None
        self.image_info_label = None
        self.confidence_label = None
        self.iou_label = None
        self.detect_button = None
        
        self._create_panel()
    
    def _create_panel(self):
        """Create the control panel UI."""
        self.frame = ttk.Labelframe(self.parent, text="Controls", padding=10)
        self.frame.configure(width=CONTROL_PANEL_WIDTH)
        
        # Model section
        self._create_model_section()
        
        # Image section
        self._create_image_section()
        
        # Settings section
        self._create_settings_section()
        
        # Actions section
        self._create_actions_section()
        
        # Info section
        self._create_info_section()
    
    def _create_model_section(self):
        """Create model loading section."""
        model_frame = ttk.Labelframe(self.frame, text="Model", padding="10", bootstyle="primary")
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            model_frame, 
            text="Load Model", 
            command=self.callbacks.get('load_model'),
            width=20,
            bootstyle="outline-primary"
        ).pack(pady=5)
        
        self.model_info_label = ttk.Label(
            model_frame, 
            text="No model loaded", 
            foreground="red",
            font=("Segoe UI", 9)
        )
        self.model_info_label.pack(pady=2)
    
    def _create_image_section(self):
        """Create image loading section."""
        image_frame = ttk.Labelframe(self.frame, text="Image", padding="10", bootstyle="info")
        image_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(
            image_frame, 
            text="Load Image", 
            command=self.callbacks.get('load_image'),
            width=20,
            bootstyle="outline-info"
        ).pack(pady=3)
        
        ttk.Button(
            image_frame, 
            text="Load Folder", 
            command=self.callbacks.get('load_folder'),
            width=20,
            bootstyle="outline-info"
        ).pack(pady=3)
        
        self.image_info_label = ttk.Label(
            image_frame, 
            text="No image loaded",
            font=("Segoe UI", 9)
        )
        self.image_info_label.pack(pady=2)
    
    def _create_settings_section(self):
        """Create detection settings section."""
        settings_frame = ttk.Labelframe(self.frame, text="Detection Settings", padding="10", bootstyle="secondary")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Confidence threshold
        ttk.Label(settings_frame, text="Confidence Threshold:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        confidence_scale = ttk.Scale(
            settings_frame, 
            from_=CONFIDENCE_RANGE[0], 
            to=CONFIDENCE_RANGE[1],
            variable=self.confidence_var, 
            orient=tk.HORIZONTAL,
            bootstyle="info"
        )
        confidence_scale.pack(fill=tk.X, pady=2)
        confidence_scale.configure(command=self._update_confidence_label)
        
        self.confidence_label = ttk.Label(
            settings_frame, 
            text=f"{DEFAULT_CONFIDENCE:.2f}",
            font=("Segoe UI", 10, "bold"),
            bootstyle="info"
        )
        self.confidence_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Intersection over Union threshold
        ttk.Label(settings_frame, text="IoU Threshold:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, pady=(0, 5))
        iou_scale = ttk.Scale(
            settings_frame, 
            from_=INTERSECTION_OVER_UNION_RANGE[0], 
            to=INTERSECTION_OVER_UNION_RANGE[1],
            variable=self.iou_var, 
            orient=tk.HORIZONTAL,
            bootstyle="info"
        )
        iou_scale.pack(fill=tk.X, pady=2)
        iou_scale.configure(command=self._update_iou_label)
        
        self.iou_label = ttk.Label(
            settings_frame, 
            text=f"{DEFAULT_INTERSECTION_OVER_UNION:.2f}",
            font=("Segoe UI", 10, "bold"),
            bootstyle="info"
        )
        self.iou_label.pack(anchor=tk.W)
    
    def _create_actions_section(self):
        """Create action buttons section."""
        action_frame = ttk.Labelframe(self.frame, text="Actions", padding="10", bootstyle="success")
        action_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.detect_button = ttk.Button(
            action_frame, 
            text="Detect Defects",
            command=self.callbacks.get('run_detection'),
            width=20, 
            state=tk.DISABLED,
            bootstyle="success"
        )
        self.detect_button.pack(pady=5)
        
        ttk.Button(
            action_frame, 
            text="Show Statistics",
            command=self.callbacks.get('show_statistics'),
            width=20,
            bootstyle="outline-secondary"
        ).pack(pady=3)
        
        ttk.Button(
            action_frame, 
            text="Save Results",
            command=self.callbacks.get('save_results'),
            width=20,
            bootstyle="outline-secondary"
        ).pack(pady=3)
        
        ttk.Button(
            action_frame, 
            text="Clear All",
            command=self.callbacks.get('clear_all'),
            width=20,
            bootstyle="outline-danger"
        ).pack(pady=3)
    
    def _create_info_section(self):
        """Create defect classes info section."""
        classes_frame = ttk.Labelframe(self.frame, text="Defect Classes", padding="10", bootstyle="warning")
        classes_frame.pack(fill=tk.BOTH, expand=True)
        
        classes_text = ScrolledText(
            classes_frame, 
            height=8, 
            width=30, 
            font=("Consolas", 9),
            wrap=tk.WORD
        )
        classes_text.pack(fill=tk.BOTH, expand=True)
        
        # Add defect classes info
        info_text = "Detectable Defects:\n\n"
        for i, class_name in enumerate(DEFECT_CLASSES):
            description = DEFECT_DESCRIPTIONS.get(class_name, "Unknown defect")
            info_text += f"{i}. {class_name}\n   {description}\n\n"
        
        classes_text.insert(tk.END, info_text)
        classes_text.configure(state=tk.DISABLED)
    
    def _update_confidence_label(self, value):
        """Update confidence threshold label."""
        self.confidence_label.config(text=f"{float(value):.2f}")
    
    def _update_iou_label(self, value):
        """Update IoU threshold label."""
        self.iou_label.config(text=f"{float(value):.2f}")
    
    def update_model_info(self, info: str, color: str = "black", performance_info: str = None):
        """Update model information display with performance information."""
        if self.model_info_label:
            # If performance info is provided, show it in a more detailed way
            if performance_info:
                display_text = f"{info}\n{performance_info}"
                self.model_info_label.config(text=display_text, foreground=color)
                logger.info(f"Model performance: {performance_info}")
            else:
                self.model_info_label.config(text=info, foreground=color)
    
    def update_image_info(self, info: str):
        """Update image information display."""
        if self.image_info_label:
            self.image_info_label.config(text=info)
    
    def set_detect_button_state(self, enabled: bool):
        """Enable or disable the detect button."""
        if self.detect_button:
            state = tk.NORMAL if enabled else tk.DISABLED
            self.detect_button.config(state=state)
    
    def get_confidence(self) -> float:
        """Get current confidence threshold."""
        return self.confidence_var.get()
    
    def get_iou(self) -> float:
        """Get current Intersection over Union threshold."""
        return self.iou_var.get()


class ImageCanvas:
    """Canvas for displaying images with zoom and pan functionality."""
    
    def __init__(self, parent: tk.Widget, callbacks: dict):
        self.parent = parent
        self.callbacks = callbacks
        self.frame = None
        self.canvas = None
        self.zoom_label = None
        
        self._create_canvas()
    
    def _create_canvas(self):
        """Create the image canvas UI."""
        self.frame = ttk.Labelframe(self.parent, text="Image Display", padding=10)
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(0, weight=1)
        
        # Canvas for image display with better styling
        self.canvas = tk.Canvas(
            self.frame, 
            bg='#ecf0f1', 
            width=600, 
            height=400,
            relief=tk.FLAT,
            borderwidth=2,
            highlightthickness=1,
            highlightbackground='#bdc3c7'
        )
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.canvas.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(self.frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.canvas.configure(xscrollcommand=h_scrollbar.set)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.callbacks.get('canvas_click', lambda e: None))
        self.canvas.bind("<MouseWheel>", self.callbacks.get('mouse_wheel', lambda e: None))
        
        # Zoom controls
        self._create_zoom_controls()
    
    def _create_zoom_controls(self):
        """Create zoom control buttons."""
        zoom_frame = ttk.Frame(self.frame)
        zoom_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(
            zoom_frame, 
            text="Zoom In", 
            command=self.callbacks.get('zoom_in'),
            width=10,
            bootstyle="outline-primary"
        ).pack(side=tk.LEFT, padx=3)
        
        ttk.Button(
            zoom_frame, 
            text="Zoom Out", 
            command=self.callbacks.get('zoom_out'),
            width=10,
            bootstyle="outline-primary"
        ).pack(side=tk.LEFT, padx=3)
        
        ttk.Button(
            zoom_frame, 
            text="Reset", 
            command=self.callbacks.get('reset_zoom'),
            width=10,
            bootstyle="outline-secondary"
        ).pack(side=tk.LEFT, padx=3)
        
        self.zoom_label = ttk.Label(
            zoom_frame, 
            text="100%", 
            font=('Segoe UI', 10, 'bold'),
            bootstyle="primary"
        )
        self.zoom_label.pack(side=tk.LEFT, padx=15)
    
    def update_image(self, photo_image):
        """Update the displayed image."""
        if photo_image:
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo_image)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def update_zoom_label(self, zoom_percentage: int):
        """Update the zoom percentage label."""
        if self.zoom_label:
            self.zoom_label.config(text=f"{zoom_percentage}%")
    
    def clear_canvas(self):
        """Clear the canvas."""
        if self.canvas:
            self.canvas.delete("all")


class ResultsPanel:
    """Panel for displaying detection results with color-coded defects."""
    
    def __init__(self, parent: tk.Widget):
        self.parent = parent
        self.frame = None
        self.results_text = None
        self.summary_frame = None
        self.total_label = None
        self.status_label = None
        
        self._create_panel()
    
    def _create_panel(self):
        """Create the results panel UI."""
        self.frame = ttk.Labelframe(self.parent, text="Detection Results", padding=10, bootstyle="danger")
        self.frame.columnconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)
        
        # Summary section at top
        self._create_summary_section()
        
        # Results text area with color tags
        self.results_text = ScrolledText(
            self.frame, 
            height=25,
            width=35,
            font=("Consolas", 9),
            bg='#ffffff',
            fg='#2c3e50',
            relief=tk.FLAT,
            borderwidth=1,
            wrap=tk.WORD
        )
        self.results_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # Configure color tags for each defect type
        self._configure_color_tags()
        
        # Initial message
        self.update_results("Load a model and image to start detection...\n")
        self.results_text.configure(state=tk.DISABLED)
    
    def _create_summary_section(self):
        """Create summary section with icons and counts."""
        self.summary_frame = ttk.Frame(self.frame)
        self.summary_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Status indicator
        self.status_label = ttk.Label(
            self.summary_frame,
            text="● Ready",
            font=("Segoe UI", 10, "bold"),
            foreground="#95a5a6"
        )
        self.status_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Total defects count
        self.total_label = ttk.Label(
            self.summary_frame,
            text="Total Defects: 0",
            font=("Segoe UI", 11, "bold"),
            foreground="#2c3e50"
        )
        self.total_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Separator
        ttk.Separator(self.summary_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
    
    def _configure_color_tags(self):
        """Configure color tags for different defect types."""
        # Color scheme for each defect class
        defect_colors = {
            'missing_hole': '#e74c3c',      # Red
            'mouse_bite': '#e67e22',        # Orange
            'open_circuit': '#f39c12',      # Yellow/Gold
            'short': '#9b59b6',             # Purple
            'spur': '#3498db',              # Blue
            'spurious_copper': '#e91e63',   # Pink
            'header': '#2c3e50',            # Dark gray
            'confidence': '#27ae60',        # Green
            'bbox': '#7f8c8d'               # Gray
        }
        
        for tag, color in defect_colors.items():
            self.results_text.tag_configure(tag, foreground=color, font=("Consolas", 9, "bold"))
        
        # Special tag for high confidence
        self.results_text.tag_configure('high_conf', foreground='#27ae60', font=("Consolas", 9, "bold"))
        # Special tag for low confidence
        self.results_text.tag_configure('low_conf', foreground='#e67e22', font=("Consolas", 9, "bold"))
    
    def update_results(self, text: str, detections: list = None):
        """Update the results display with color-coded information."""
        if self.results_text:
            self.results_text.configure(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            
            if detections:
                self._display_colored_results(detections)
            else:
                self.results_text.insert(tk.END, text)
            
            self.results_text.configure(state=tk.DISABLED)
    
    def _display_colored_results(self, detections: list):
        """Display results with color coding."""
        if not detections:
            self.results_text.insert(tk.END, "✓ No defects detected\n", 'confidence')
            self.update_summary(0, "OK")
            return
        
        # Update summary
        self.update_summary(len(detections), "DEFECTS FOUND")
        
        # Group by defect type
        defect_groups = {}
        for det in detections:
            class_name = det.get('class_name', 'unknown')
            if class_name not in defect_groups:
                defect_groups[class_name] = []
            defect_groups[class_name].append(det)
        
        # Display grouped results
        for class_name, dets in defect_groups.items():
            # Header with count
            self.results_text.insert(tk.END, f"\n▼ {class_name.upper()}", class_name)
            self.results_text.insert(tk.END, f" ({len(dets)})\n", 'header')
            
            # Individual detections
            for i, det in enumerate(dets, 1):
                conf = det.get('confidence', 0)
                conf_tag = 'high_conf' if conf > 0.7 else 'low_conf'
                
                self.results_text.insert(tk.END, f"  #{i} ", 'header')
                self.results_text.insert(tk.END, f"Conf: {conf:.2%}", conf_tag)
                
                # Bbox info (compact)
                bbox = det.get('bbox', {})
                if bbox:
                    x1, y1 = int(bbox.get('x1', 0)), int(bbox.get('y1', 0))
                    x2, y2 = int(bbox.get('x2', 0)), int(bbox.get('y2', 0))
                    self.results_text.insert(tk.END, f"\n     Pos: ({x1},{y1})-({x2},{y2})\n", 'bbox')
    
    def update_summary(self, count: int, status: str):
        """Update the summary section."""
        if self.total_label:
            self.total_label.config(text=f"Total Defects: {count}")
        
        if self.status_label:
            if status == "OK":
                self.status_label.config(text="● No Defects", foreground="#27ae60")
            elif status == "DEFECTS FOUND":
                self.status_label.config(text="● Defects Detected", foreground="#e74c3c")
            else:
                self.status_label.config(text=f"● {status}", foreground="#95a5a6")
    
    def clear_results(self):
        """Clear the results display."""
        self.update_results("Load a model and image to start detection...\n")
        self.update_summary(0, "Ready")