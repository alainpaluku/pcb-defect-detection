# PCB Defect Detection

A YOLO11-based Computer Vision solution for automated detection and classification of defects on Printed Circuit Boards (PCBs). This system provides high-precision defect detection using state-of-the-art deep learning techniques.

[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-20BEFF?style=flat-square&logo=kaggle)](https://www.kaggle.com/datasets/akhatova/pcb-defects)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)](https://python.org)
[![YOLO11](https://img.shields.io/badge/YOLO11-Ultralytics-00FFFF?style=flat-square)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

## Features

- **Detection**: Locate defects with bounding boxes and high precision
- **Classification**: Identify 6 types of PCB defects automatically
- **Real-time**: Fast inference with optimized YOLO11 model
- **Export**: Multiple formats for deployment (PyTorch, ONNX, TorchScript, TensorFlow Lite, Core ML, OpenVINO)
- **Scalable**: Batch processing capabilities for industrial use
- **GUI Interface**: User-friendly testing interface with modular architecture

## Defect Classes

The system detects 6 types of PCB defects:

| ID | Defect | Description |
|----|--------|-------------|
| 0 | `missing_hole` | Missing drill hole |
| 1 | `mouse_bite` | Irregular edge |
| 2 | `open_circuit` | Broken trace |
| 3 | `short` | Short circuit |
| 4 | `spur` | Copper protrusion |
| 5 | `spurious_copper` | Unwanted copper |

## Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Detection Precision | **95.5%** | Model correctly locates 95.5% of defects |
| Strict Precision | 51.8% | Precision with stricter criteria |
| Reliability | 95.4% | 95.4% of detections are true defects |
| Detection Rate | 91.0% | Model finds 91% of all defects |

*Results after 150 epochs of training on Kaggle with YOLO11m.*

## Quick Start

### Option 1: GUI Interface (Recommended for Testing)

```bash
# Clone and setup
git clone https://github.com/alainpaluku/pcb-defect-detection.git
cd pcb-defect-detection
pip install -r requirements.txt

# Install GUI dependencies
cd gui_test
pip install -r requirements.txt

# Launch modular GUI interface
python -m gui_test.app

# Or use the launcher
cd ..
python launch_gui_modular.py
```

### Option 2: Command Line Interface

```bash
# Train model
python main.py train --epochs 150

# Detect defects
python main.py detect path/to/image.jpg --save

# Export to all formats
python main.py export --model output/pcb_model.pt --format all
```

### Option 3: Training on Kaggle

**Prerequisites:**
1. Add dataset `akhatova/pcb-defects` via **"+ Add Input"**
2. Enable **GPU T4** and **Internet** in notebook settings

**Method 1 - Direct Commands (Most Reliable):**
```python
!pip install ultralytics -q
!wget -q https://github.com/alainpaluku/pcb-defect-detection/archive/main.zip
!unzip -q main.zip
!mv pcb-defect-detection-main pcb-defect-detector
%cd pcb-defect-detector
!python run_kaggle.py
```

**Method 2 - Simple Setup Script:**
```python
# Upload kaggle_simple_setup.py to your Kaggle notebook, then run:
!python kaggle_simple_setup.py
```

**Method 3 - Robust Setup Script (with fallback):**
```python
# Upload kaggle_setup.py to your Kaggle notebook, then run:
!python kaggle_setup.py
```

**After training, you'll get multiple model formats:**
- `pcb_model.pt` - PyTorch model
- `pcb_model.onnx` - ONNX model  
- `pcb_model.torchscript` - TorchScript model
- `pcb_model.tflite` - TensorFlow Lite model
- `pcb_model.mlmodel` - Core ML model
- `pcb_model_openvino/` - OpenVINO model files

## Download Models from Kaggle

After training on Kaggle, download the models to use locally:

```python
# In Kaggle notebook, after training completes
import zipfile
from IPython.display import FileLink

# Create a zip file with all models
with zipfile.ZipFile('/kaggle/working/pcb_models.zip', 'w') as zipf:
    zipf.write('/kaggle/working/pcb_model.pt', 'pcb_model.pt')
    zipf.write('/kaggle/working/pcb_model.onnx', 'pcb_model.onnx')
    zipf.write('/kaggle/working/pcb_model.torchscript', 'pcb_model.torchscript')
    zipf.write('/kaggle/working/pcb_model.tflite', 'pcb_model.tflite')
    zipf.write('/kaggle/working/pcb_model.mlmodel', 'pcb_model.mlmodel')
    zipf.write('/kaggle/working/MODEL_EXPORT_SUMMARY.md', 'MODEL_EXPORT_SUMMARY.md')
    
    # Add OpenVINO files if they exist
    import os
    openvino_dir = '/kaggle/working/pcb_model_openvino'
    if os.path.exists(openvino_dir):
        for root, dirs, files in os.walk(openvino_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, '/kaggle/working')
                zipf.write(file_path, arcname)

# Download link
FileLink('/kaggle/working/pcb_models.zip')
```

## GUI Test Interface

A modular, maintainable Tkinter-based GUI is provided for testing trained models interactively.

### Features
- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Model Loading**: Support for PyTorch (.pt) and ONNX (.onnx) models
- **Image Processing**: Load single images or batch process folders
- **Real-time Detection**: Adjustable confidence and IoU thresholds
- **Visual Results**: Bounding boxes with color-coded defect types
- **Export Results**: Save results in JSON, CSV, or text format

### Quick Start

```bash
# Install GUI dependencies
cd gui_test
pip install -r requirements.txt

# Launch the modular GUI
python -m gui_test.app

# Or from project root
python launch_gui_modular.py
```

For detailed GUI documentation, see [gui_test/README.md](gui_test/README.md).

## Using Trained Models

### 1. PyTorch Model (.pt)

**Best for:** Python development, research, further training

```python
from src.detector import PCBInspector

# Load and use PyTorch model
inspector = PCBInspector(model_path="pcb_model.pt")
defects = inspector.inspect("pcb_image.jpg", conf=0.25)

# Print results
for defect in defects:
    print(f"Defect: {defect['class_name']}")
    print(f"Confidence: {defect['confidence']:.2%}")
    print(f"Location: {defect['bbox']}")
```

### 2. ONNX Model (.onnx)

**Best for:** Production deployment, cross-platform inference

```python
import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2

# Load ONNX model
session = ort.InferenceSession("pcb_model.onnx")

# Preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 640))
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # HWC to CHW
    image = np.expand_dims(image, axis=0)   # Add batch dimension
    return image

# Run inference
image_array = preprocess_image("pcb_image.jpg")
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: image_array})

# Process outputs
predictions = outputs[0]  # Shape: [1, 25200, 11] (6 classes + 5 bbox params)
```

### 3. TorchScript Model (.torchscript)

**Best for:** C++ deployment, mobile applications

```python
import torch

# Load TorchScript model
model = torch.jit.load("pcb_model.torchscript")
model.eval()

# Prepare input tensor
image_tensor = torch.randn(1, 3, 640, 640)  # Replace with actual image

# Run inference
with torch.no_grad():
    predictions = model(image_tensor)
```

**C++ Usage:**
```cpp
#include <torch/script.h>
#include <iostream>

int main() {
    // Load the model
    torch::jit::script::Module module = torch::jit::load("pcb_model.torchscript");
    
    // Create input tensor
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::randn({1, 3, 640, 640}));
    
    // Run inference
    at::Tensor output = module.forward(inputs).toTensor();
    
    return 0;
}
```

### 4. TensorFlow Lite Model (.tflite)

**Best for:** Mobile apps (Android/iOS), edge devices

```python
import tensorflow as tf
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="pcb_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prepare input data
input_data = np.random.random_sample((1, 640, 640, 3)).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get output
output_data = interpreter.get_tensor(output_details[0]['index'])
```

### 5. Core ML Model (.mlmodel)

**Best for:** iOS/macOS applications

```swift
import CoreML
import Vision

// Load Core ML model
guard let model = try? VNCoreMLModel(for: pcb_model().model) else {
    fatalError("Failed to load model")
}

// Create request
let request = VNCoreMLRequest(model: model) { request, error in
    guard let results = request.results as? [VNRecognizedObjectObservation] else {
        return
    }
    
    // Process results
    for result in results {
        print("Confidence: \(result.confidence)")
        print("Bounds: \(result.boundingBox)")
    }
}

// Perform request on image
let handler = VNImageRequestHandler(cgImage: cgImage)
try? handler.perform([request])
```

### 6. OpenVINO Model

**Best for:** Intel hardware optimization (CPU, GPU, VPU)

```python
from openvino.runtime import Core

# Initialize OpenVINO
ie = Core()

# Load model
model = ie.read_model("pcb_model_openvino/pcb_model.xml")
compiled_model = ie.compile_model(model, "CPU")  # or "GPU", "MYRIAD"

# Get input/output info
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Prepare input
input_data = np.random.random((1, 3, 640, 640)).astype(np.float32)

# Run inference
result = compiled_model([input_data])[output_layer]
```

## Batch Processing

For processing multiple images efficiently:

```python
from src.detector import PCBInspector
import os

# Initialize inspector
inspector = PCBInspector(model_path="pcb_model.onnx")  # Use ONNX for speed

# Process directory of images
image_dir = "path/to/pcb/images"
results = inspector.inspect_batch(image_dir, conf=0.25, save=True)

# Generate report
total_images = len(results)
defective_images = sum(1 for detections in results.values() if detections)

print(f"Processed {total_images} images")
print(f"Found defects in {defective_images} images")
print(f"Defect rate: {defective_images/total_images*100:.1f}%")

# Detailed results
for image_name, detections in results.items():
    if detections:
        print(f"\n{image_name}: {len(detections)} defect(s)")
        for defect in detections:
            print(f"  - {defect['class_name']}: {defect['confidence']:.2%}")
```

## API Reference

### Core Classes

#### PCBDetector
Main model wrapper for YOLO11 operations.

```python
from src.model import PCBDetector

# Initialize detector
detector = PCBDetector(model_path="yolo11m.pt")

# Train model
results = detector.train(
    data_yaml="dataset.yaml",
    epochs=150,
    batch_size=16
)

# Run inference
predictions = detector.predict(
    source="image.jpg",
    conf=0.25
)

# Export model
onnx_path = detector.export(format="onnx")
```

#### PCBInspector
High-level interface for defect inspection.

```python
from src.detector import PCBInspector

# Initialize inspector
inspector = PCBInspector(model_path="trained_model.pt")

# Inspect single image
defects = inspector.inspect("pcb_image.jpg", conf=0.3)

# Batch processing
results = inspector.inspect_batch("images_folder/")

# Get summary
summary = PCBInspector.get_summary(defects)
```

#### TrainingManager
Complete training pipeline management.

```python
from src.trainer import TrainingManager

# Initialize training manager
trainer = TrainingManager(data_path="dataset/")

# Run complete pipeline
metrics = trainer.run_pipeline(epochs=150)
```

## Algorithm Details

### YOLO11 Architecture
- **Model**: YOLO11 Medium (yolo11m.pt) - optimal balance between accuracy and speed
- **Input Size**: 640x640 pixels
- **Parameters**: ~20M parameters
- **Model Size**: ~40MB (ONNX format)

### Training Configuration
- **Epochs**: 150 (reduced from 200 for efficiency)
- **Optimizer**: AdamW with automatic parameter selection
- **Learning Rate**: 0.001 with cosine annealing
- **Batch Size**: 16 (adjustable based on GPU memory)
- **Early Stopping**: Patience of 50 epochs

### Performance Optimization
- **Mixed Precision Training**: Reduces memory usage by 50%
- **ONNX Runtime**: 2-3x faster inference than PyTorch
- **Batch Processing**: Efficient handling of multiple images
- **Multi-threading**: Parallel processing capabilities

## Project Structure

```
pcb-defect-detection/
├── src/                    # Core Python modules
│   ├── config.py           # Configuration management
│   ├── data_ingestion.py   # Data loading & conversion
│   ├── model.py            # YOLO11 model wrapper
│   ├── detector.py         # Inference interface
│   ├── trainer.py          # Training pipeline
│   └── utils.py            # Utility functions
├── gui_test/               # Modular GUI test interface
│   ├── app.py              # Main application entry point
│   ├── main_window.py      # Main window controller
│   ├── ui_components.py    # UI widgets and components
│   ├── model_loader.py     # Model loading and inference
│   ├── image_handler.py    # Image processing and display
│   ├── dialogs.py          # Dialog windows
│   ├── utils.py            # GUI utility functions
│   └── requirements.txt    # GUI-specific dependencies
├── tests/                  # Unit tests
├── main.py                 # Main CLI entry point
├── launch_gui_modular.py   # GUI launcher
├── run_kaggle.py           # Kaggle training script
├── kaggle_setup.py         # Robust Kaggle setup
├── requirements.txt        # Core dependencies
└── README.md               # Project documentation
```

## Deployment Options

### 1. Python Package
```bash
pip install -e .
python -m pcb_defect_detector.main detect image.jpg
```

### 2. Docker Container
```dockerfile
FROM python:3.10-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
COPY main.py .
CMD ["python", "main.py"]
```

### 3. ONNX Runtime
```python
import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession("pcb_model.onnx")

# Run inference
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: image_array})
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in config
   - Enable mixed precision training

2. **Low Detection Accuracy**
   - Increase training epochs
   - Adjust confidence threshold
   - Verify dataset quality

3. **Slow Inference**
   - Use ONNX model instead of PyTorch
   - Enable GPU acceleration

### Performance Tuning

| Parameter | Default | Range | Impact |
|-----------|---------|-------|---------|
| Confidence | 0.25 | 0.1 - 0.5 | Precision vs Recall |
| IoU Threshold | 0.5 | 0.3 - 0.7 | Duplicate detection |
| Batch Size | 16 | 8 - 32 | Memory vs Speed |

## Export Format Dependencies

To use all export formats, install additional dependencies:

```bash
# For TensorFlow Lite export
pip install tensorflow>=2.13.0

# For Core ML export (macOS only)
pip install coremltools>=7.0

# For OpenVINO export
pip install openvino>=2023.0

# Install all at once
pip install tensorflow coremltools openvino
```

### Format Comparison

| Format | File Size | Speed* | Platform | Use Case |
|--------|-----------|--------|----------|----------|
| PyTorch (.pt) | ~40MB | 15ms | Python | Development |
| ONNX (.onnx) | ~40MB | 8ms | Cross-platform | Production |
| TorchScript | ~40MB | 12ms | C++, Mobile | Mobile Apps |
| TensorFlow Lite | ~10MB | 25ms | Mobile, Edge | Android, iOS |
| Core ML | ~40MB | 10ms | Apple only | iOS, macOS |
| OpenVINO | ~40MB | 5ms | Intel hardware | Intel optimization |

*Approximate inference times on Intel i7 CPU

## Dataset

[PCB Defects - Akhatova](https://www.kaggle.com/datasets/akhatova/pcb-defects)
- 1386 images with XML annotations
- 6 defect classes
- VOC format bounding boxes

## Author

**Alain Paluku** - [@alainpaluku](https://github.com/alainpaluku)

## License

MIT License