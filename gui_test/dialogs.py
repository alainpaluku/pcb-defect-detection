"""
Dialog windows for the GUI application.
"""

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
import threading
from pathlib import Path
from typing import List, Dict, Any
import logging

from .config import APP_TITLE, APP_VERSION, DEFECT_CLASSES

logger = logging.getLogger(__name__)


class BatchProcessingDialog:
    """Dialog for batch processing multiple images."""
    
    def __init__(self, parent: tk.Tk, image_files: List[Path], 
                 model_manager, confidence: float):
        self.parent = parent
        self.image_files = image_files
        self.model_manager = model_manager
        self.confidence = confidence
        
        self.dialog = None
        self.listbox = None
        
        self._create_dialog()
    
    def _create_dialog(self):
        """Create the batch processing dialog."""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("Batch Processing")
        self.dialog.geometry("500x400")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # File list
        ttk.Label(
            self.dialog, 
            text=f"Found {len(self.image_files)} images:"
        ).pack(pady=10)
        
        # Listbox with scrollbar
        listbox_frame = ttk.Frame(self.dialog)
        listbox_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.listbox = tk.Listbox(listbox_frame)
        scrollbar = ttk.Scrollbar(
            listbox_frame, 
            orient=tk.VERTICAL, 
            command=self.listbox.yview
        )
        self.listbox.configure(yscrollcommand=scrollbar.set)
        
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate listbox
        for img_file in self.image_files:
            self.listbox.insert(tk.END, img_file.name)
        
        # Buttons
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(pady=10)
        
        ttk.Button(
            button_frame, 
            text="Process All", 
            command=self._process_batch
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="Cancel", 
            command=self.dialog.destroy
        ).pack(side=tk.LEFT, padx=5)
    
    def _process_batch(self):
        """Process all images in batch."""
        self.dialog.destroy()
        
        if not self.model_manager.is_loaded:
            tk.messagebox.showerror("Error", "Please load a model first.")
            return
        
        # Create progress dialog
        progress_dialog = ProgressDialog(
            self.parent, 
            "Processing Images...",
            len(self.image_files)
        )
        
        def process_images():
            """Process images in background thread."""
            results = {}
            total_defects = 0
            
            for i, img_file in enumerate(self.image_files):
                try:
                    progress_dialog.update_status(f"Processing: {img_file.name}")
                    
                    detections = self.model_manager.run_inference(
                        img_file, 
                        self.confidence
                    )
                    
                    results[img_file.name] = detections
                    total_defects += len(detections)
                    
                    progress_dialog.update_progress(i + 1)
                    
                except Exception as e:
                    logger.error(f"Error processing {img_file}: {e}")
                    results[img_file.name] = []
            
            # Show results in main thread
            self.parent.after(0, lambda: self._show_batch_results(
                progress_dialog, results, total_defects
            ))
        
        # Start processing thread
        threading.Thread(target=process_images, daemon=True).start()
    
    def _show_batch_results(self, progress_dialog, results: Dict[str, List], 
                           total_defects: int):
        """Show batch processing results."""
        progress_dialog.close()
        BatchResultsDialog(self.parent, results, total_defects)


class ProgressDialog:
    """Progress dialog for long-running operations."""
    
    def __init__(self, parent: tk.Tk, title: str, maximum: int):
        self.parent = parent
        self.maximum = maximum
        
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x150")
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Progress label
        self.status_label = ttk.Label(self.dialog, text="")
        self.status_label.pack(pady=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.dialog, 
            variable=self.progress_var, 
            maximum=maximum
        )
        self.progress_bar.pack(fill=tk.X, padx=20, pady=10)
        
        # Status text
        self.progress_text = ttk.Label(self.dialog, text="0 / 0")
        self.progress_text.pack(pady=5)
    
    def update_status(self, status: str):
        """Update status text."""
        self.status_label.config(text=status)
        self.dialog.update()
    
    def update_progress(self, value: int):
        """Update progress bar."""
        self.progress_var.set(value)
        self.progress_text.config(text=f"{value} / {self.maximum}")
        self.dialog.update()
    
    def close(self):
        """Close the dialog."""
        self.dialog.destroy()


class BatchResultsDialog:
    """Dialog showing batch processing results."""
    
    def __init__(self, parent: tk.Tk, results: Dict[str, List], total_defects: int):
        self.parent = parent
        self.results = results
        self.total_defects = total_defects
        
        self._create_dialog()
    
    def _create_dialog(self):
        """Create the results dialog."""
        dialog = tk.Toplevel(self.parent)
        dialog.title("Batch Processing Results")
        dialog.geometry("600x500")
        dialog.transient(self.parent)
        
        # Summary
        summary_frame = ttk.LabelFrame(dialog, text="Summary", padding="10")
        summary_frame.pack(fill=tk.X, padx=10, pady=5)
        
        total_images = len(self.results)
        defective_images = sum(1 for detections in self.results.values() if detections)
        defect_rate = (defective_images / total_images * 100) if total_images > 0 else 0
        
        summary_text = f"""Total Images: {total_images}
Images with Defects: {defective_images}
Defect Rate: {defect_rate:.1f}%
Total Defects Found: {self.total_defects}"""
        
        ttk.Label(
            summary_frame, 
            text=summary_text, 
            font=("Consolas", 10)
        ).pack()
        
        # Detailed results
        details_frame = ttk.LabelFrame(dialog, text="Detailed Results", padding="10")
        details_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        results_text = ScrolledText(details_frame, font=("Consolas", 9))
        results_text.pack(fill=tk.BOTH, expand=True)
        
        # Populate results
        for image_name, detections in self.results.items():
            results_text.insert(tk.END, f"\\n{image_name}:\\n")
            if detections:
                for i, defect in enumerate(detections, 1):
                    results_text.insert(tk.END, 
                        f"  {i}. {defect['class_name']}: {defect['confidence']:.2%}\\n")
            else:
                results_text.insert(tk.END, "  No defects detected\\n")
        
        results_text.configure(state=tk.DISABLED)
        
        # Close button
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)


class StatisticsDialog:
    """Dialog showing detection statistics."""
    
    def __init__(self, parent: tk.Tk, detections: List[Dict[str, Any]]):
        self.parent = parent
        self.detections = detections
        
        self._create_dialog()
    
    def _create_dialog(self):
        """Create the statistics dialog."""
        dialog = tk.Toplevel(self.parent)
        dialog.title("Detection Statistics")
        dialog.geometry("400x300")
        dialog.transient(self.parent)
        
        # Calculate statistics
        defect_counts = {}
        total_confidence = 0
        
        for detection in self.detections:
            class_name = detection['class_name']
            confidence = detection['confidence']
            
            defect_counts[class_name] = defect_counts.get(class_name, 0) + 1
            total_confidence += confidence
        
        avg_confidence = total_confidence / len(self.detections) if self.detections else 0
        
        # Statistics text
        stats_text = ScrolledText(dialog, font=("Consolas", 10))
        stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        stats_content = f"""Detection Statistics
{'='*30}

Total Defects: {len(self.detections)}
Average Confidence: {avg_confidence:.2%}

Defect Breakdown:
{'-'*20}
"""
        
        for class_name, count in sorted(defect_counts.items()):
            percentage = (count / len(self.detections)) * 100 if self.detections else 0
            stats_content += f"{class_name}: {count} ({percentage:.1f}%)\\n"
        
        stats_text.insert(tk.END, stats_content)
        stats_text.configure(state=tk.DISABLED)
        
        # Close button
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)


class AboutDialog:
    """About dialog with application information."""
    
    def __init__(self, parent: tk.Tk):
        self.parent = parent
        self._create_dialog()
    
    def _create_dialog(self):
        """Create the about dialog."""
        about_text = f"""{APP_TITLE}

Version: {APP_VERSION}
Author: Alain Paluku

A GUI application for testing trained PCB defect detection models.
Supports PyTorch (.pt) and ONNX (.onnx) model formats.

Features:
• Real-time defect detection
• Adjustable confidence and IoU thresholds  
• Batch processing capabilities
• Zoom and pan functionality
• Export results to JSON/text

Detectable Defects:
• Missing holes
• Mouse bites
• Open circuits
• Short circuits
• Spurs
• Spurious copper
"""
        
        tk.messagebox.showinfo("About", about_text)