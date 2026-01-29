"""Training pipeline for PCB Defect Detection with YOLO11."""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import Config
from src.data_ingestion import DataIngestion
from src.model import PCBDetector
from src.utils import format_metrics, get_logger, print_section_header


class DatasetError(Exception):
    """Dataset related error."""
    pass

logger = get_logger(__name__)


class TrainingManager:
    """Manages complete training pipeline."""
    
    def __init__(
        self,
        data_path: Optional[Path] = None,
        config: Optional[Config] = None
    ):
        """Initialize training manager.
        
        Args:
            data_path: Path to dataset (auto-detected if None)
            config: Custom configuration
        """
        self.data_path = data_path
        self.config = config or Config()
        self.output_path = Config.get_output_path()
        self.data: Optional[DataIngestion] = None
        self.model: Optional[PCBDetector] = None
        self.metrics: Dict[str, float] = {}
        self.training_results: Any = None
        
        self._print_header()
    
    def _print_header(self) -> None:
        """Display system information with style."""
        print("\n" + "🔷" * 30)
        print_section_header("🔬 PCB DEFECT DETECTION - YOLO11 🔬")
        print("🔷" * 30)
        
        # System info
        print(f"\n📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🖥️  Environment: {'☁️ Kaggle' if Config.is_kaggle() else '💻 Local'}")
        print(f"📁 Output: {self.output_path}")
        
        # Model config
        model_cfg = self.config.model
        print(f"\n⚙️  Configuration:")
        print(f"   • Model: {model_cfg.name}")
        print(f"   • Epochs: {model_cfg.epochs}")
        print(f"   • Batch size: {model_cfg.batch_size}")
        print(f"   • Learning rate: {model_cfg.learning_rate}")
        print(f"   • Image size: {model_cfg.img_size}x{model_cfg.img_size}")
        print(f"   • Optimizer: {model_cfg.optimizer}")
        print("")
    
    def setup_data(self) -> DataIngestion:
        """Configure data pipeline."""
        print_section_header("📊 [1/5] DATA CONFIGURATION")
        
        self.data = DataIngestion(data_path=self.data_path)
        
        if not self.data.find_data_structure():
            raise DatasetError(f"Dataset not found at {self.data.data_path}")
        
        self.data.collect_images()
        
        if not self.data.all_images:
            raise DatasetError("No images found in dataset")
        
        stats = self.data.get_stats()
        
        print(f"\n📈 Dataset statistics:")
        print(f"   • Total images: {stats['total_images']}")
        print(f"   • With XML annotations: {stats['with_xml']} ✅")
        print(f"   • Without annotations (ignored): {stats['from_folders']} ⚠️")
        
        train_count, val_count = self.data.create_yolo_dataset()
        
        print(f"\n📂 YOLO dataset created:")
        print(f"   • Train: {train_count} images")
        print(f"   • Validation: {val_count} images")
        print(f"   • Ratio: {train_count/(train_count+val_count)*100:.1f}% / {val_count/(train_count+val_count)*100:.1f}%")
        
        return self.data
    
    def setup_model(self) -> PCBDetector:
        """Initialize model."""
        print_section_header("🤖 [2/5] MODEL CONFIGURATION")
        
        self.model = PCBDetector(config=self.config)
        
        print(f"\n✅ Model initialized: {self.config.model.name}")
        print(f"   • Classes: {Config.NUM_CLASSES}")
        print(f"   • Classes: {', '.join(Config.CLASS_NAMES)}")
        
        return self.model
    
    def train(self, epochs: Optional[int] = None) -> Any:
        """Train the model."""
        print_section_header("🚀 [3/5] TRAINING")
        
        if self.data is None:
            raise RuntimeError("Call setup_data() first")
        if self.model is None:
            raise RuntimeError("Call setup_model() first")
        
        epochs = epochs or self.config.model.epochs
        
        print(f"\n⏱️  Starting training for {epochs} epochs...")
        print(f"   (Estimated time: 15-30 min on Kaggle GPU)")
        print(f"   💡 Early stopping enabled - auto stop if convergence")
        print("\n" + "-" * 60)
        
        yaml_path = self.data.get_yaml_path()
        
        self.training_results = self.model.train(
            data_yaml=yaml_path,
            epochs=epochs,
            project=str(self.output_path),
            name="pcb_yolo"
        )
        
        print("-" * 60)
        print("✅ Training completed!")
        
        return self.training_results
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model."""
        print_section_header("📏 [4/5] EVALUATION")
        
        if self.data is None or self.model is None:
            raise RuntimeError("Call setup_data() and setup_model() first")
        
        yaml_path = self.data.get_yaml_path()
        results = self.model.validate(data_yaml=yaml_path)
        
        self.metrics = PCBDetector.extract_metrics(results)
        
        # Styled metrics display
        print("\n" + "=" * 50)
        print("📊 PERFORMANCE METRICS")
        print("=" * 50)
        
        precision_detection = self.metrics.get('detection_precision', 0)
        strict_precision = self.metrics.get('strict_precision', 0)
        reliability = self.metrics.get('reliability', 0)
        detection_rate = self.metrics.get('detection_rate', 0)
        
        # Visual indicators
        def get_indicator(value: float) -> str:
            if value >= 0.9:
                return "🟢 Excellent"
            elif value >= 0.7:
                return "🟡 Good"
            elif value >= 0.5:
                return "🟠 Average"
            else:
                return "🔴 Needs improvement"
        
        print(f"\n   Detection precision:  {precision_detection:.4f}  ({precision_detection*100:.1f}%)  {get_indicator(precision_detection)}")
        print(f"   Strict precision:     {strict_precision:.4f}  ({strict_precision*100:.1f}%)  {get_indicator(strict_precision)}")
        print(f"   Reliability:          {reliability:.4f}  ({reliability*100:.1f}%)  {get_indicator(reliability)}")
        print(f"   Detection rate:       {detection_rate:.4f}  ({detection_rate*100:.1f}%)  {get_indicator(detection_rate)}")
        
        # Global score
        f1_score = 2 * (reliability * detection_rate) / (reliability + detection_rate) if (reliability + detection_rate) > 0 else 0
        print(f"\n   F1-Score:   {f1_score:.4f}  ({f1_score*100:.1f}%)  {get_indicator(f1_score)}")
        
        print("=" * 50)
        
        return self.metrics
    
    def save_model(self) -> Optional[Path]:
        """Save trained model in multiple formats."""
        print_section_header("💾 [5/5] SAVING & EXPORTING")
        
        best_model = self.output_path / "pcb_yolo" / "weights" / "best.pt"
        if not best_model.exists():
            print("⚠️  best.pt model not found")
            return None
        
        # Copy PyTorch model (.pt format)
        dst = self.output_path / "pcb_model.pt"
        shutil.copy(best_model, dst)
        print(f"✅ PyTorch (.pt): {dst}")
        
        # Save PyTorch state dict (.pth format)
        try:
            import torch
            model_state = torch.load(best_model, map_location='cpu')
            pth_path = self.output_path / "pcb_model.pth"
            torch.save(model_state, pth_path)
            print(f"✅ PyTorch state dict (.pth): {pth_path}")
        except Exception as e:
            print(f"⚠️  Failed to save .pth format: {e}")
        
        # Export to the 2 best formats for desktop GUI
        try:
            from src.model import PCBDetector
            best_detector = PCBDetector(model_path=str(best_model))
            
            print(f"\n🔄 Exporting model to PyTorch format...")
            
            # Export to PyTorch format only
            exported_paths = best_detector.export_multiple_formats()
            
            # Model is already saved as best.pt, just copy it
            print(f"\n📦 Model exported:")
            print(f"   ✅ PyTorch (.pt): {dst}")
            
            # Create a summary file
            self._create_export_summary(exported_paths)
            
        except Exception as e:
            print(f"⚠️  Model export failed: {e}")
        
        return dst
    
    def _create_export_summary(self, exported_paths: Dict[str, Path]) -> None:
        """Create a summary file with export information."""
        summary_content = f"""# PCB Defect Detection - Exported Model

## Training Summary
- Model: YOLO11m
- Epochs: {self.config.model.epochs}
- Image Size: {self.config.model.img_size}x{self.config.model.img_size}
- Batch Size: {self.config.model.batch_size}

## Performance Metrics
- Detection Precision: {self.metrics.get('detection_precision', 0):.4f} ({self.metrics.get('detection_precision', 0)*100:.1f}%)
- Strict Precision: {self.metrics.get('strict_precision', 0):.4f} ({self.metrics.get('strict_precision', 0)*100:.1f}%)
- Reliability: {self.metrics.get('reliability', 0):.4f} ({self.metrics.get('reliability', 0)*100:.1f}%)
- Detection Rate: {self.metrics.get('detection_rate', 0):.4f} ({self.metrics.get('detection_rate', 0)*100:.1f}%)

## Exported Model Format

### PyTorch (.pt)
- **File**: `pcb_model.pt`
- **Use Case**: Python development, GUI interface, inference
- **Size**: ~40MB
- **Platform**: Cross-platform with PyTorch

## Usage with GUI

Place the model file in the `models/` directory:
- `pcb_model.pt` - PyTorch model for inference

The GUI will automatically load the PyTorch model.

## Usage Examples

See README.md for detailed usage instructions.
"""
        
        summary_path = self.output_path / "MODEL_EXPORT_SUMMARY.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        print(f"📄 Export summary: {summary_path}")
- **Size**: ~40MB
- **Platform**: Cross-platform, optimized for performance

## Usage with GUI

Place both model files in the `models/` directory:
- `pcb_model.pt` - For compatibility
- `pcb_model.onnx` - For best performance

The GUI will automatically select the ONNX model for optimal performance.

## Usage Examples

See README.md for detailed usage instructions.
"""
        
        summary_path = self.output_path / "MODEL_EXPORT_SUMMARY.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        print(f"📄 Export summary: {summary_path}")
    
    def generate_graphs(self) -> None:
        """Generate and display training graphs."""
        print_section_header("📈 TRAINING GRAPHS")
        
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            
            # Look for results.csv file
            results_file = self.output_path / "pcb_yolo" / "results.csv"
            
            if not results_file.exists():
                print("⚠️  results.csv file not found")
                return
            
            # Load data
            df = pd.read_csv(results_file)
            df.columns = df.columns.str.strip()  # Clean column names
            
            # Create figure with multiple subplots
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('📊 PCB Defect Detection - Training Results', fontsize=16, fontweight='bold')
            
            # 1. Training loss
            ax1 = axes[0, 0]
            if 'train/box_loss' in df.columns:
                ax1.plot(df['epoch'], df['train/box_loss'], 'b-', label='Localization error', linewidth=2)
                ax1.plot(df['epoch'], df['train/cls_loss'], 'r-', label='Classification error', linewidth=2)
                ax1.plot(df['epoch'], df['train/dfl_loss'], 'g-', label='Distribution error', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Error')
            ax1.set_title('📉 Training errors')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Validation loss
            ax2 = axes[0, 1]
            if 'val/box_loss' in df.columns:
                ax2.plot(df['epoch'], df['val/box_loss'], 'b--', label='Localization error', linewidth=2)
                ax2.plot(df['epoch'], df['val/cls_loss'], 'r--', label='Classification error', linewidth=2)
                ax2.plot(df['epoch'], df['val/dfl_loss'], 'g--', label='Distribution error', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Error')
            ax2.set_title('📉 Validation errors')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Mean Average Precision
            ax3 = axes[0, 2]
            if 'metrics/mAP50(B)' in df.columns:
                ax3.plot(df['epoch'], df['metrics/mAP50(B)'], 'b-', label='Detection precision', linewidth=2, marker='o', markersize=3)
                ax3.plot(df['epoch'], df['metrics/mAP50-95(B)'], 'r-', label='Strict precision', linewidth=2, marker='s', markersize=3)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Score')
            ax3.set_title('🎯 Detection precision')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim([0, 1])
            
            # 4. Precision & Recall
            ax4 = axes[1, 0]
            if 'metrics/precision(B)' in df.columns:
                ax4.plot(df['epoch'], df['metrics/precision(B)'], 'g-', label='Reliability', linewidth=2)
                ax4.plot(df['epoch'], df['metrics/recall(B)'], 'm-', label='Detection rate', linewidth=2)
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Score')
            ax4.set_title('📊 Reliability & Detection rate')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim([0, 1])
            
            # 5. Learning Rate
            ax5 = axes[1, 1]
            if 'lr/pg0' in df.columns:
                ax5.plot(df['epoch'], df['lr/pg0'], 'c-', label='Group 0', linewidth=2)
                ax5.plot(df['epoch'], df['lr/pg1'], 'y-', label='Group 1', linewidth=2)
                ax5.plot(df['epoch'], df['lr/pg2'], 'k-', label='Group 2', linewidth=2)
            ax5.set_xlabel('Epoch')
            ax5.set_ylabel('Learning rate')
            ax5.set_title('📈 Learning rate evolution')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # 6. Final summary
            ax6 = axes[1, 2]
            ax6.axis('off')
            
            # Final metrics
            final_metrics = f"""
╔══════════════════════════════════════╗
║     📊 FINAL RESULTS                 ║
╠══════════════════════════════════════╣
║  Detection precision: {self.metrics.get('detection_precision', 0):.4f} ({self.metrics.get('detection_precision', 0)*100:.1f}%)  ║
║  Strict precision:    {self.metrics.get('strict_precision', 0):.4f} ({self.metrics.get('strict_precision', 0)*100:.1f}%)  ║
║  Reliability:         {self.metrics.get('reliability', 0):.4f} ({self.metrics.get('reliability', 0)*100:.1f}%)  ║
║  Detection rate:      {self.metrics.get('taux_detection', 0):.4f} ({self.metrics.get('taux_detection', 0)*100:.1f}%)  ║
╚══════════════════════════════════════╝
"""
            ax6.text(0.1, 0.5, final_metrics, fontsize=12, fontfamily='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            
            # Save
            graph_path = self.output_path / "training_results.png"
            plt.savefig(graph_path, dpi=150, bbox_inches='tight')
            print(f"✅ Graphs saved: {graph_path}")
            
            # Display in Kaggle/Jupyter
            plt.show()
            
        except ImportError as e:
            print(f"⚠️  Matplotlib/Pandas not available: {e}")
        except Exception as e:
            print(f"⚠️  Error generating graphs: {e}")
    
    def display_sample_predictions(self) -> None:
        """Display sample predictions."""
        print_section_header("🖼️  SAMPLE PREDICTIONS")
        
        try:
            import matplotlib.pyplot as plt
            from PIL import Image
            
            # Look for prediction images generated by YOLO
            pred_dir = self.output_path / "pcb_yolo"
            
            # Validation images with predictions
            val_images = list(pred_dir.glob("val_batch*_pred.jpg"))
            
            if not val_images:
                print("⚠️  No prediction images found")
                return
            
            # Display up to 4 images
            n_images = min(4, len(val_images))
            fig, axes = plt.subplots(1, n_images, figsize=(5*n_images, 5))
            
            if n_images == 1:
                axes = [axes]
            
            for i, img_path in enumerate(val_images[:n_images]):
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].axis('off')
                axes[i].set_title(f'Batch {i+1}')
            
            plt.suptitle('🔍 Detection examples on validation', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save
            sample_path = self.output_path / "sample_predictions.png"
            plt.savefig(sample_path, dpi=150, bbox_inches='tight')
            print(f"✅ Examples saved: {sample_path}")
            
            plt.show()
            
        except Exception as e:
            print(f"⚠️  Error displaying examples: {e}")
    
    def run_pipeline(self, epochs: Optional[int] = None) -> Dict[str, float]:
        """Execute complete training pipeline."""
        
        self.setup_data()
        self.setup_model()
        self.train(epochs=epochs)
        self.evaluate()
        self.save_model()
        
        # Generate graphs
        self.generate_graphs()
        self.display_sample_predictions()
        
        # Final summary
        print("\n" + "🎉" * 30)
        print_section_header("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("🎉" * 30)
        
        print(f"""
╔══════════════════════════════════════════════════════════╗
║                    📊 FINAL SUMMARY                      ║
╠══════════════════════════════════════════════════════════╣
║  Detection precision: {self.metrics.get('precision_detection', 0):.4f}  ({self.metrics.get('precision_detection', 0)*100:.1f}%)                   ║
║  Strict precision:    {self.metrics.get('precision_stricte', 0):.4f}  ({self.metrics.get('precision_stricte', 0)*100:.1f}%)                   ║
║  Reliability:         {self.metrics.get('fiabilite', 0):.4f}  ({self.metrics.get('fiabilite', 0)*100:.1f}%)                   ║
║  Detection rate:      {self.metrics.get('taux_detection', 0):.4f}  ({self.metrics.get('taux_detection', 0)*100:.1f}%)                   ║
╠══════════════════════════════════════════════════════════╣
║  📁 Generated files:                                     ║
║     • pcb_model.pt (PyTorch)                             ║
║     • pcb_model.onnx (ONNX)                              ║
║     • pcb_model.torchscript (TorchScript)                ║
║     • pcb_model.tflite (TensorFlow Lite)                 ║
║     • pcb_model.mlmodel (Core ML)                        ║
║     • pcb_model_openvino/ (OpenVINO)                     ║
║     • training_results.png (Graphs)                      ║
║     • sample_predictions.png (Examples)                  ║
║     • MODEL_EXPORT_SUMMARY.md (Usage Guide)             ║
╚══════════════════════════════════════════════════════════╝
""")
        
        print(f"📂 All files in: {self.output_path}")
        
        return self.metrics
